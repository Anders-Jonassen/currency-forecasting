"""Step 5 - Forecast evaluation (target = NOK/USD return).

Contents (run for BOTH windowing schemes: expanding and rolling)
----------------------------------------------------------------
1. True vs. predicted NOK/USD return for each method.
2. RMSE table per method, compared against two benchmarks: random walk with and
   without drift (for returns, "no drift" predicts a zero return).
3. CSSED - cumulative sum of squared error differences vs. benchmark. A rising
   curve means the model beats the benchmark cumulatively over time.
4. Diebold-Mariano test between models (p-values).
5. A cross-scheme comparison so all models are compared the same way under both
   the expanding and the rolling window.

Learning notes
--------------
* RMSE is the average forecast error; but a lower RMSE can be luck.
* CSSED shows WHEN a model is better/worse, not just on average.
* Diebold-Mariano (1995) formally tests whether two models' forecast errors
  differ significantly. We look at the loss differential d_t = e_A^2 - e_B^2 and
  ask whether its mean is significantly different from 0. For 1-step forecasts we
  use the Harvey-Leybourne-Newbold small-sample correction and the t-distribution.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .forecasting import REAL_MODELS, SCHEMES, pred_path
from .utils import savefig, set_style

BENCHMARK = "RW"
TARGET = "r_true"  # the realised NOK/USD return
MODELS = REAL_MODELS + ["Combination"]
ALL_FORECASTS = MODELS + ["RW", "RW_drift"]


def load_forecasts(scheme: str) -> pd.DataFrame:
    return pd.read_parquet(pred_path(scheme))


def rmse_table(pred: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """RMSE (on the return) for each forecast, sorted lowest first."""
    y = pred[TARGET]
    rows = [(m, np.sqrt(np.mean((pred[m] - y) ** 2))) for m in ALL_FORECASTS]
    tab = pd.DataFrame(rows, columns=["model", "RMSE"]).sort_values("RMSE")
    rw_rmse = tab.loc[tab["model"] == "RW", "RMSE"].iloc[0]
    tab["RMSE_rel_RW"] = tab["RMSE"] / rw_rmse
    tab.to_csv(config.OUTPUT_DIR / f"05_rmse_table_{scheme}.csv", index=False)
    print(f"[eval:{scheme}] RMSE table (returns):\n", tab.round(6).to_string(index=False))
    return tab


def plot_pred_vs_true(pred: pd.DataFrame, scheme: str) -> None:
    """True vs. predicted return - one panel per model."""
    set_style()
    y = pred[TARGET]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for ax, m in zip(axes.ravel(), MODELS):
        ax.plot(y.index, y, color="black", lw=1.0, alpha=0.7, label="True")
        ax.plot(pred.index, pred[m], color="#d62728", lw=1.0, label="Predicted")
        rmse = np.sqrt(np.mean((pred[m] - y) ** 2))
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"{m}  (RMSE={rmse:.5f})")
        ax.legend(fontsize=9)
    fig.suptitle(f"True vs. predicted NOK/USD return - {scheme} window (out-of-sample)",
                 fontweight="bold")
    savefig(fig, f"05_pred_vs_true_{scheme}.png")


def cssed(pred: pd.DataFrame, scheme: str, benchmark: str = BENCHMARK) -> pd.DataFrame:
    """Cumulative sum of squared error differences vs. benchmark.

    CSSED_t = sum_{s<=t} (e_benchmark,s^2 - e_model,s^2).
    Positive and rising => the model has lower squared error than the benchmark.
    """
    y = pred[TARGET]
    e_bench2 = (pred[benchmark] - y) ** 2
    cdf = pd.DataFrame(
        {m: (e_bench2 - (pred[m] - y) ** 2).cumsum() for m in MODELS},
        index=pred.index,
    )
    set_style()
    fig, ax = plt.subplots()
    for m in MODELS:
        ax.plot(cdf.index, cdf[m], label=m)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("CSSED")
    ax.set_title(f"CSSED vs. {benchmark} - {scheme} window (rising = better)")
    ax.legend(fontsize=9)
    savefig(fig, f"05_cssed_{scheme}.png")
    return cdf


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test with squared loss and the HLN correction.

    Returns (DM statistic, two-sided p-value). A positive statistic means model 1
    has higher loss (worse) than model 2.
    """
    d = e1**2 - e2**2
    n = len(d)
    dbar = d.mean()
    # Newey-West variance with lags up to h-1 (h=1 => just the variance).
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * (1 - k / h) * gk
    dm = dbar / np.sqrt(var / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN small-sample
    dm_star = dm * corr
    pval = 2 * (1 - stats.t.cdf(abs(dm_star), df=n - 1))
    return float(dm_star), float(pval)


def dm_pvalue_matrix(pred: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Pairwise DM p-value matrix among all forecasts."""
    y = pred[TARGET]
    errs = {m: (pred[m] - y).to_numpy() for m in ALL_FORECASTS}
    mat = pd.DataFrame(np.nan, index=ALL_FORECASTS, columns=ALL_FORECASTS)
    for a in ALL_FORECASTS:
        for b in ALL_FORECASTS:
            if a != b:
                mat.loc[a, b] = dm_test(errs[a], errs[b])[1]
    mat.to_csv(config.OUTPUT_DIR / f"05_dm_pvalues_{scheme}.csv")

    vs_rw = pd.DataFrame(
        {
            "model": MODELS,
            "DM_stat_vs_RW": [dm_test(errs[m], errs["RW"])[0] for m in MODELS],
            "p_value_vs_RW": [dm_test(errs[m], errs["RW"])[1] for m in MODELS],
        }
    )
    vs_rw.to_csv(config.OUTPUT_DIR / f"05_dm_vs_rw_{scheme}.csv", index=False)
    print(f"[eval:{scheme}] DM vs RW (negative stat = better than RW):\n",
          vs_rw.round(4).to_string(index=False))
    return mat


def compare_schemes(rmse_by_scheme: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Side-by-side RMSE comparison of every model across both schemes."""
    merged = None
    for scheme, tab in rmse_by_scheme.items():
        col = tab.set_index("model")["RMSE"].rename(scheme)
        merged = col if merged is None else pd.concat([merged, col], axis=1)
    merged = merged.loc[ALL_FORECASTS]
    merged.to_csv(config.OUTPUT_DIR / "05_rmse_compare.csv")
    print("[eval] RMSE comparison (expanding vs rolling):\n",
          merged.round(6).to_string())

    set_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(merged))
    width = 0.38
    for k, scheme in enumerate(SCHEMES):
        ax.bar(x + (k - 0.5) * width, merged[scheme], width, label=scheme)
    ax.axhline(merged.loc["RW"].mean(), color="black", ls="--", lw=1, label="RW (ref.)")
    ax.set_xticks(x)
    ax.set_xticklabels(merged.index, rotation=30, ha="right")
    ax.set_ylabel("RMSE (return)")
    ax.set_title("RMSE by model (NOK/USD return): expanding vs. rolling window")
    ax.set_ylim(merged.min().min() * 0.98, merged.max().max() * 1.02)
    ax.legend()
    savefig(fig, "05_rmse_compare.png")
    return merged


def run() -> None:
    rmse_by_scheme = {}
    for scheme in SCHEMES:
        pred = load_forecasts(scheme)
        print(f"[eval:{scheme}] {len(pred)} OOS forecasts "
              f"({pred.index.min().date()} -> {pred.index.max().date()})")
        rmse_by_scheme[scheme] = rmse_table(pred, scheme)
        plot_pred_vs_true(pred, scheme)
        cssed(pred, scheme)
        dm_pvalue_matrix(pred, scheme)
    compare_schemes(rmse_by_scheme)


if __name__ == "__main__":
    run()
