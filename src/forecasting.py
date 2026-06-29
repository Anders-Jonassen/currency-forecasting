"""Step 4 - Rolling out-of-sample forecasting of the NOK/USD RETURN.

Target
------
We forecast the one-month-ahead simple currency return (no logs):

    r_{t+1} = (y_{t+1} - y_t) / y_t,    where y = NOK/USD (USD per krone)

using information known at time t. Forecasting the return (rather than the level)
is the natural framing for currency predictability: under the efficient-markets /
random-walk view the best forecast of the return is 0, which is exactly our main
benchmark.

Predictors known at time t:
    Level_t, Slope_t, Curvature_t   (the Diebold-Li factors)
    r_t = (y_t - y_{t-1}) / y_{t-1} (last return, for the AR model)

Two out-of-sample windowing schemes (no leakage)
------------------------------------------------
Both with a 60-month window parameter and the same OOS period, so they are
directly comparable:

* "expanding" (recursive): train on ALL data up to t (the window grows).
* "rolling"   (fixed):      train on only the most recent `window` months.

On each step i we (a) train only on data whose target is already realised,
(b) predict step i's target (realised at i+1), (c) roll one step forward.

Models (all predict r_{t+1})
----------------------------
1. Simple linear regression : r ~ Slope
2. Multiple regression      : r ~ Level+Slope+Curv
3. AR(1)                    : r ~ r_t
4. Elastic Net              : r ~ all factors + r_t (CV on training data only)
5. LSTM (PyTorch)           : sequence of W months of factors+return -> r
6. Combination              : inverse-MSE weighting of 1-5

Benchmarks:
    Random walk (no drift): r_hat = 0
    Random walk with drift: r_hat = mean(r in training)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from . import config
from .data_acquisition import load_dataset
from .diebold_li import FACTORS_PATH

WINDOW = 60             # months: initial size (expanding) / fixed size (rolling)
LSTM_WINDOW = 12        # input window (months) for the LSTM
MIN_COMB = 12           # OOS points before inverse-MSE weighting kicks in
SEED = 42

FEATURES = ["Level", "Slope", "Curvature", "r_lag"]
REAL_MODELS = ["Linear", "Multiple", "AR", "ElasticNet", "LSTM"]
SCHEMES = ["expanding", "rolling"]


def pred_path(scheme: str):
    """Path to the saved forecasts for a given windowing scheme."""
    return config.OUTPUT_DIR / f"forecasts_{scheme}.parquet"


# --------------------------------------------------------------------------- #
#  Data preparation
# --------------------------------------------------------------------------- #
def build_supervised() -> pd.DataFrame:
    """Build a supervised table: predictors at time t, target return r_{t+1}."""
    df = load_dataset()
    factors = pd.read_parquet(FACTORS_PATH)
    y = df["NOKUSD"]
    r = y.pct_change()                   # simple return r_t (no logs)

    sup = factors.copy()
    sup["r_lag"] = r                     # r_t  (known at t)
    sup["target_r"] = r.shift(-1)        # r_{t+1}  (realised at t+1)
    sup["next_date"] = df.index.to_series().shift(-1)
    return sup.dropna()


# --------------------------------------------------------------------------- #
#  LSTM (PyTorch)
# --------------------------------------------------------------------------- #
def _make_sequences(F: np.ndarray, target: np.ndarray, window: int):
    """Build (sequence, target) pairs: each X is the last `window` rows of F."""
    Xs, ys = [], []
    for j in range(window - 1, len(F)):
        Xs.append(F[j - window + 1 : j + 1])
        ys.append(target[j])
    return np.asarray(Xs), np.asarray(ys)


def _train_predict_lstm(F_train, y_train, F_pred_seq, window):
    """Train a small LSTM on the training sequences and predict one step.

    The small architecture (1 layer, few units) is deliberate: with ~60-200
    observations a large network would overfit. Features and target are scaled on
    the TRAINING data only (no leakage).
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    fsc = StandardScaler().fit(F_train)
    F_train_s = fsc.transform(F_train)
    y_mu, y_sd = y_train.mean(), y_train.std() + 1e-8
    y_train_s = (y_train - y_mu) / y_sd

    Xs, ys = _make_sequences(F_train_s, y_train_s, window)
    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(ys, dtype=torch.float32).view(-1, 1)

    class LSTMReg(nn.Module):
        def __init__(self, n_feat, hidden=12):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])  # last time step -> forecast

    model = LSTMReg(F_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        seq = fsc.transform(F_pred_seq)[None, :, :]
        pred_s = model(torch.tensor(seq, dtype=torch.float32)).item()
    return pred_s * y_sd + y_mu  # back to original scale


# --------------------------------------------------------------------------- #
#  Rolling OOS loop
# --------------------------------------------------------------------------- #
def run_scheme(scheme: str = "expanding", verbose: bool = True) -> pd.DataFrame:
    """Run the rolling OOS return forecast for one windowing scheme."""
    assert scheme in SCHEMES, f"unknown scheme {scheme!r}"
    sup = build_supervised().reset_index(drop=True)
    N = len(sup)
    Fall = sup[FEATURES].to_numpy(dtype=float)
    target = sup["target_r"].to_numpy(dtype=float)
    dates = pd.to_datetime(sup["next_date"]).to_numpy()

    idx = {name: FEATURES.index(name) for name in FEATURES}
    cols_simple = [idx["Slope"]]
    cols_multi = [idx["Level"], idx["Slope"], idx["Curvature"]]
    cols_ar = [idx["r_lag"]]
    cols_en = [idx["Level"], idx["Slope"], idx["Curvature"], idx["r_lag"]]

    records = []
    past_errs = {m: [] for m in REAL_MODELS}

    for i in range(WINDOW, N):
        # The only difference between the two schemes is the training start.
        start = 0 if scheme == "expanding" else i - WINDOW
        Xtr, ytr = Fall[start:i], target[start:i]
        xq = Fall[i]

        preds = {}
        preds["Linear"] = (
            LinearRegression().fit(Xtr[:, cols_simple], ytr).predict(xq[cols_simple][None])[0]
        )
        preds["Multiple"] = (
            LinearRegression().fit(Xtr[:, cols_multi], ytr).predict(xq[cols_multi][None])[0]
        )
        preds["AR"] = (
            LinearRegression().fit(Xtr[:, cols_ar], ytr).predict(xq[cols_ar][None])[0]
        )
        en = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=5, max_iter=5000)
        sc = StandardScaler().fit(Xtr[:, cols_en])
        # The tiny training folds occasionally trip a harmless ConvergenceWarning;
        # it does not affect the (already small) coefficients, so we silence it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            en.fit(sc.transform(Xtr[:, cols_en]), ytr)
        preds["ElasticNet"] = en.predict(sc.transform(xq[cols_en][None]))[0]

        preds["LSTM"] = _train_predict_lstm(
            Fall[start:i], target[start:i],
            Fall[i - LSTM_WINDOW + 1 : i + 1], LSTM_WINDOW
        )

        # Inverse-MSE combination of the 5 models (truly OOS weights).
        if all(len(past_errs[m]) >= MIN_COMB for m in REAL_MODELS):
            mse = {m: np.mean(np.square(past_errs[m][-36:])) for m in REAL_MODELS}
            w = {m: 1.0 / (mse[m] + 1e-12) for m in REAL_MODELS}
            s = sum(w.values())
            w = {m: w[m] / s for m in REAL_MODELS}
        else:
            w = {m: 1.0 / len(REAL_MODELS) for m in REAL_MODELS}
        preds["Combination"] = sum(w[m] * preds[m] for m in REAL_MODELS)

        preds["RW"] = 0.0            # random walk in the level => zero expected return
        preds["RW_drift"] = ytr.mean()

        rec = {"date": dates[i], "r_true": target[i]}
        rec.update(preds)
        records.append(rec)

        for m in REAL_MODELS:
            past_errs[m].append(preds[m] - target[i])

        if verbose and (i - WINDOW) % 36 == 0:
            print(f"[fc:{scheme}] {pd.Timestamp(dates[i]).date()}  (OOS #{i-WINDOW+1})")

    out = pd.DataFrame(records).set_index("date").sort_index()
    out.to_parquet(pred_path(scheme))
    print(f"[fc:{scheme}] {len(out)} OOS return forecasts saved to {pred_path(scheme)}")
    return out


def run() -> dict[str, pd.DataFrame]:
    """Run BOTH windowing schemes (expanding and rolling)."""
    return {scheme: run_scheme(scheme) for scheme in SCHEMES}


if __name__ == "__main__":
    run()
