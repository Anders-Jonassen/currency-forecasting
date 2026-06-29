"""Steg 5 – Evaluering av prognosene.

Innhold
-------
1. Graf med sann vs. predikert NOK/USD for hver metode.
2. RMSE-tabell per metode, sammenlignet med to benchmarks: random walk uten og
   med drift.
3. CSSED – kumulativ sum av kvadrerte feildifferanser mot benchmark. Stigende
   kurve = modellen slår benchmark akkumulert over tid.
4. Diebold-Mariano-test mellom modellene (p-verdier).

Læringsnoter
------------
* RMSE er gjennomsnittlig prognosefeil; men en lavere RMSE kan skyldes flaks.
* CSSED viser NÅR en modell er bedre/dårligere, ikke bare i snitt.
* Diebold-Mariano (1995) tester formelt om to modellers prognosefeil er
  signifikant forskjellige. Vi ser på tapsdifferansen d_t = e²_A − e²_B og
  spør om gjennomsnittet er signifikant ulikt 0. For 1-stegs prognoser bruker vi
  Harvey-Leybourne-Newbold sin små-utvalgskorreksjon og t-fordeling.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .forecasting import PRED_PATH, REAL_MODELS
from .utils import savefig, set_style

BENCHMARK = "RW"
MODELS = REAL_MODELS + ["Combination"]
ALL_FORECASTS = MODELS + ["RW", "RW_drift"]


def load_forecasts() -> pd.DataFrame:
    return pd.read_parquet(PRED_PATH)


def rmse_table(pred: pd.DataFrame) -> pd.DataFrame:
    """RMSE (på nivå) for hver prognose, sortert lavest først."""
    y = pred["y_true"]
    rows = []
    for m in ALL_FORECASTS:
        err = pred[m] - y
        rmse = np.sqrt(np.mean(err**2))
        rows.append((m, rmse))
    tab = pd.DataFrame(rows, columns=["model", "RMSE"]).sort_values("RMSE")
    # Relativ til RW (verdi < 1 betyr bedre enn random walk).
    rw_rmse = tab.loc[tab["model"] == "RW", "RMSE"].iloc[0]
    tab["RMSE_rel_RW"] = tab["RMSE"] / rw_rmse
    tab.to_csv(config.OUTPUT_DIR / "05_rmse_table.csv", index=False)
    print("[eval] RMSE-tabell:\n", tab.round(6).to_string(index=False))
    return tab


def plot_pred_vs_true(pred: pd.DataFrame) -> None:
    """Sann vs. predikert nivå – ett panel per modell."""
    set_style()
    y = pred["y_true"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for ax, m in zip(axes.ravel(), MODELS):
        ax.plot(y.index, y, color="black", lw=1.3, label="Sann")
        ax.plot(pred.index, pred[m], color="#d62728", lw=1.0, alpha=0.8,
                label="Predikert")
        rmse = np.sqrt(np.mean((pred[m] - y) ** 2))
        ax.set_title(f"{m}  (RMSE={rmse:.5f})")
        ax.legend(fontsize=9)
    fig.suptitle("Sann vs. predikert NOK/USD (out-of-sample)", fontweight="bold")
    savefig(fig, "05_pred_vs_true.png")


def cssed(pred: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.DataFrame:
    """Kumulativ sum av kvadrerte feildifferanser mot benchmark.

    CSSED_t = Σ_{s≤t} (e²_benchmark,s − e²_model,s).
    Positiv og stigende => modellen har lavere kvadrert feil enn benchmark.
    """
    y = pred["y_true"]
    e_bench2 = (pred[benchmark] - y) ** 2
    css = {}
    for m in MODELS:
        e_m2 = (pred[m] - y) ** 2
        css[m] = (e_bench2 - e_m2).cumsum()
    cdf = pd.DataFrame(css, index=pred.index)

    set_style()
    fig, ax = plt.subplots()
    for m in MODELS:
        ax.plot(cdf.index, cdf[m], label=m)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("CSSED")
    ax.set_title(f"CSSED mot {benchmark} (stigende = bedre enn benchmark)")
    ax.legend(fontsize=9)
    savefig(fig, "05_cssed.png")
    return cdf


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano-test med kvadrert tap og HLN-korreksjon.

    Returnerer (DM-statistikk, tosidig p-verdi). Positiv statistikk => modell 1
    har høyere tap (dårligere) enn modell 2.
    """
    d = e1**2 - e2**2
    n = len(d)
    dbar = d.mean()
    # Newey-West-variansestimat med lag opptil h-1 (h=1 => bare variansen).
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * (1 - k / h) * gk
    dm = dbar / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold små-utvalgskorreksjon + t-fordeling.
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = dm * corr
    pval = 2 * (1 - stats.t.cdf(abs(dm_star), df=n - 1))
    return float(dm_star), float(pval)


def dm_pvalue_matrix(pred: pd.DataFrame) -> pd.DataFrame:
    """Parvis DM p-verdimatrise blant alle prognoser."""
    y = pred["y_true"]
    errs = {m: (pred[m] - y).to_numpy() for m in ALL_FORECASTS}
    names = ALL_FORECASTS
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            if a == b:
                continue
            _, p = dm_test(errs[a], errs[b])
            mat.loc[a, b] = p
    mat.to_csv(config.OUTPUT_DIR / "05_dm_pvalues.csv")
    print("[eval] DM p-verdier (rad vs kolonne):\n", mat.round(3).to_string())

    # Fokusert tabell: hver modell mot RW-benchmark.
    vs_rw = pd.DataFrame(
        {
            "model": MODELS,
            "DM_stat_vs_RW": [dm_test(errs[m], errs["RW"])[0] for m in MODELS],
            "p_value_vs_RW": [dm_test(errs[m], errs["RW"])[1] for m in MODELS],
        }
    )
    vs_rw.to_csv(config.OUTPUT_DIR / "05_dm_vs_rw.csv", index=False)
    print("[eval] DM vs RW (negativ stat = bedre enn RW):\n",
          vs_rw.round(4).to_string(index=False))
    return mat


def run() -> None:
    pred = load_forecasts()
    print(f"[eval] {len(pred)} OOS-prognoser "
          f"({pred.index.min().date()} -> {pred.index.max().date()})")
    rmse_table(pred)
    plot_pred_vs_true(pred)
    cssed(pred)
    dm_pvalue_matrix(pred)


if __name__ == "__main__":
    run()
