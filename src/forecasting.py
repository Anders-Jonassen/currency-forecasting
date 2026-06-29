"""Steg 4 – Rullende out-of-sample prognoser av NOK/USD.

Rammeverk
---------
Vi forutsier NOK/USD ÉN måned fram. For å unngå spuriøs regresjon på trendende
nivåer modellerer vi ENDRINGEN Δy_{t+1} = y_{t+1} − y_t med informasjon kjent på
tidspunkt t, og setter nivåprognosen = y_t + Δŷ. Det gjør sammenligningen mot en
random walk naturlig (RW sier Δ=0).

Prediktorer kjent på tid t:
    Level_t, Slope_t, Curvature_t   (Diebold-Li-faktorene)
    dy_t = y_t − y_{t-1}            (forrige endring, for AR-modellen)

Out-of-sample uten lekkasje (rullende/utvidende vindu)
------------------------------------------------------
Vi starter med MIN_TRAIN måneders treningsdata. På hvert steg i:
    * tren modellen KUN på data der målet allerede er realisert (rad 0..i-1),
    * predikér rad i sitt mål (realiseres på i+1),
    * rull ett steg fram.
Slik etterligner vi en analytiker som hver måned forutsier neste måned med bare
fortidens data. Ingen framtidsinformasjon lekker inn i trening eller skalering.

Modellene
---------
1. Enkel lineær regresjon  : Δy ~ Slope            (én DL-faktor)
2. Multippel regresjon     : Δy ~ Level+Slope+Curv (alle tre)
3. AR(1)                   : Δy ~ dy_t             (autoregressiv på endringen)
4. Elastic Net             : Δy ~ alle faktorer + dy_t, med regularisering
                             (valgt fordi prediktorene er korrelerte og utvalget
                             lite – krymping reduserer overtilpasning og gjør
                             variabelutvalg; α/l1 velges med CV KUN på trening)
5. LSTM (PyTorch)          : sekvens av W måneder med faktorer+dy -> Δy
6. Kombinasjon             : invers-MSE-vekting av 1–5 (mer vekt til modeller som
                             historisk har hatt lavest OOS-feil)

Benchmarks:
    Random walk (uten drift): Δ̂ = 0
    Random walk med drift   : Δ̂ = gj.snitt(Δy i trening)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from . import config
from .data_acquisition import load_dataset
from .diebold_li import FACTORS_PATH

MIN_TRAIN = 60          # måneder før første OOS-prognose
LSTM_WINDOW = 12        # inputvindu (måneder) til LSTM
MIN_COMB = 12           # OOS-punkter før invers-MSE-vekting slår inn
SEED = 42

FEATURES = ["Level", "Slope", "Curvature", "dy"]
REAL_MODELS = ["Linear", "Multiple", "AR", "ElasticNet", "LSTM"]
PRED_PATH = config.OUTPUT_DIR / "forecasts.parquet"


# --------------------------------------------------------------------------- #
#  Datatilrettelegging
# --------------------------------------------------------------------------- #
def build_supervised() -> pd.DataFrame:
    """Bygg en overvåket tabell: prediktorer på tid t, mål Δy_{t+1}."""
    df = load_dataset()
    factors = pd.read_parquet(FACTORS_PATH)
    y = df["NOKUSD"].rename("y")

    sup = factors.copy()
    sup["dy"] = y.diff()                 # dy_t = y_t - y_{t-1}  (kjent på t)
    sup["y_prev"] = y                    # y_t
    sup["target_dy"] = y.shift(-1) - y   # Δy_{t+1}  (realiseres på t+1)
    sup["y_next"] = y.shift(-1)          # y_{t+1}   (fasit)
    sup["next_date"] = df.index.to_series().shift(-1)
    return sup.dropna()


# --------------------------------------------------------------------------- #
#  LSTM (PyTorch)
# --------------------------------------------------------------------------- #
def _make_sequences(F: np.ndarray, target: np.ndarray, window: int):
    """Bygg (sekvens, mål)-par: hver X er de siste `window` radene av F."""
    Xs, ys, end_idx = [], [], []
    for j in range(window - 1, len(F)):
        Xs.append(F[j - window + 1 : j + 1])
        ys.append(target[j])
        end_idx.append(j)
    return np.asarray(Xs), np.asarray(ys), np.asarray(end_idx)


def _train_predict_lstm(F_train, y_train, F_pred_seq, window):
    """Tren en liten LSTM på treningssekvenser og predikér ett steg.

    Liten arkitektur (1 lag, få enheter) er bevisst: med ~150-200 observasjoner
    ville et stort nett overtilpasse. Vi skalerer features og mål på TRENING.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Skalering tilpasset KUN på trening (ingen lekkasje).
    fsc = StandardScaler().fit(F_train)
    F_train_s = fsc.transform(F_train)
    y_mu, y_sd = y_train.mean(), y_train.std() + 1e-8
    y_train_s = (y_train - y_mu) / y_sd

    Xs, ys, _ = _make_sequences(F_train_s, y_train_s, window)
    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(ys, dtype=torch.float32).view(-1, 1)

    class LSTMReg(nn.Module):
        def __init__(self, n_feat, hidden=12):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])  # siste tidssteg -> prognose

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
    return pred_s * y_sd + y_mu  # tilbake til original skala


# --------------------------------------------------------------------------- #
#  Rullende OOS-løkke
# --------------------------------------------------------------------------- #
def run_rolling(verbose: bool = True) -> pd.DataFrame:
    sup = build_supervised().reset_index(drop=True)
    N = len(sup)
    Fall = sup[FEATURES].to_numpy(dtype=float)
    target = sup["target_dy"].to_numpy(dtype=float)
    y_prev = sup["y_prev"].to_numpy(dtype=float)
    y_next = sup["y_next"].to_numpy(dtype=float)
    dates = pd.to_datetime(sup["next_date"]).to_numpy()

    # Kolonneindekser for prediktor-undermengder.
    idx = {name: FEATURES.index(name) for name in FEATURES}
    cols_simple = [idx["Slope"]]
    cols_multi = [idx["Level"], idx["Slope"], idx["Curvature"]]
    cols_ar = [idx["dy"]]
    cols_en = [idx["Level"], idx["Slope"], idx["Curvature"], idx["dy"]]

    records = []
    past_errs = {m: [] for m in REAL_MODELS}  # for invers-MSE-vekting

    for i in range(MIN_TRAIN, N):
        Xtr, ytr = Fall[:i], target[:i]   # rader med realisert mål
        xq = Fall[i]                      # prediktorer for raden vi forutsier

        preds = {}
        # 1-2-3-4: lineære varianter (Δy-prognoser)
        preds["Linear"] = (
            LinearRegression().fit(Xtr[:, cols_simple], ytr).predict(xq[cols_simple][None])[0]
        )
        preds["Multiple"] = (
            LinearRegression().fit(Xtr[:, cols_multi], ytr).predict(xq[cols_multi][None])[0]
        )
        preds["AR"] = (
            LinearRegression().fit(Xtr[:, cols_ar], ytr).predict(xq[cols_ar][None])[0]
        )
        en = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=5, max_iter=5000, n_jobs=None)
        sc = StandardScaler().fit(Xtr[:, cols_en])
        en.fit(sc.transform(Xtr[:, cols_en]), ytr)
        preds["ElasticNet"] = en.predict(sc.transform(xq[cols_en][None]))[0]

        # 5: LSTM (sekvens fram til og med rad i)
        preds["LSTM"] = _train_predict_lstm(
            Fall[:i], target[:i], Fall[i - LSTM_WINDOW + 1 : i + 1], LSTM_WINDOW
        )

        # 6: invers-MSE-kombinasjon av de 5 modellene
        if all(len(past_errs[m]) >= MIN_COMB for m in REAL_MODELS):
            mse = {m: np.mean(np.square(past_errs[m][-36:])) for m in REAL_MODELS}
            w = {m: 1.0 / (mse[m] + 1e-12) for m in REAL_MODELS}
            s = sum(w.values())
            w = {m: w[m] / s for m in REAL_MODELS}
        else:
            w = {m: 1.0 / len(REAL_MODELS) for m in REAL_MODELS}  # likevekt tidlig
        preds["Combination"] = sum(w[m] * preds[m] for m in REAL_MODELS)

        # Benchmarks
        preds["RW"] = 0.0
        preds["RW_drift"] = ytr.mean()

        # Konverter Δy-prognoser til NIVÅ-prognoser, og logg.
        rec = {"date": dates[i], "y_true": y_next[i], "y_prev": y_prev[i]}
        for m, dyhat in preds.items():
            rec[m] = y_prev[i] + dyhat
        records.append(rec)

        # Oppdater feilhistorikk for kombinasjonen (på Δy-nivå, OOS).
        for m in REAL_MODELS:
            past_errs[m].append(preds[m] - target[i])

        if verbose and (i - MIN_TRAIN) % 24 == 0:
            print(f"[fc] {pd.Timestamp(dates[i]).date()}  (OOS #{i-MIN_TRAIN+1})")

    out = pd.DataFrame(records).set_index("date").sort_index()
    out.to_parquet(PRED_PATH)
    print(f"[fc] {len(out)} OOS-prognoser lagret til {PRED_PATH}")
    return out


def run() -> pd.DataFrame:
    return run_rolling()


if __name__ == "__main__":
    run()
