"""Steg 6 – Enkel handelsstrategi og lønnsomhet.

Idé
---
En prognose er bare nyttig hvis den kan omsettes i lønnsomme beslutninger. Vi
lager en bevisst SIMPEL fortegnsstrategi: hver måned tar vi posisjon i kronen ut
fra FORTEGNET på predikert endring.

    predikert endring > 0  ->  long NOK (vi tror kronen styrker seg mot USD)
    predikert endring < 0  ->  short NOK

Månedlig avkastning = posisjon × faktisk NOK/USD-avkastning. Vi sammenligner mot
"buy & hold" (alltid long NOK).

Måltall
-------
* Treffrate: andel måneder der vi gjettet riktig retning.
* Annualisert avkastning og volatilitet, og Sharpe-rate (rf = 0).
Strategien er illustrativ: ingen transaksjonskostnader eller risikostyring.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .evaluation import MODELS, load_forecasts
from .utils import savefig, set_style


def _returns_frame(pred: pd.DataFrame) -> pd.DataFrame:
    """Faktisk månedlig NOK/USD-avkastning og strategiavkastning per modell."""
    actual_ret = (pred["y_true"] - pred["y_prev"]) / pred["y_prev"]
    out = pd.DataFrame(index=pred.index)
    out["BuyHold"] = actual_ret
    for m in MODELS:
        signal = np.sign(pred[m] - pred["y_prev"])  # +1 long, -1 short
        out[m] = signal * actual_ret
    return out


def _stats(ret: pd.Series, pred: pd.DataFrame, model: str | None) -> dict:
    n_year = 12
    ann_ret = ret.mean() * n_year
    ann_vol = ret.std() * np.sqrt(n_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    total = (1 + ret).prod() - 1
    row = {
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }
    if model is not None:
        actual_dir = np.sign(pred["y_true"] - pred["y_prev"])
        pred_dir = np.sign(pred[model] - pred["y_prev"])
        row["hit_rate"] = float((actual_dir == pred_dir).mean())
    else:
        row["hit_rate"] = np.nan
    return row


def run() -> pd.DataFrame:
    pred = load_forecasts()
    rets = _returns_frame(pred)

    # Kumulativ avkastning (vekstindeks fra 1).
    cum = (1 + rets).cumprod()

    set_style()
    fig, ax = plt.subplots()
    for col in ["BuyHold"] + MODELS:
        lw = 2.2 if col == "Combination" else 1.3
        ax.plot(cum.index, cum[col], label=col, lw=lw,
                color="black" if col == "BuyHold" else None)
    ax.set_ylabel("Vekst av 1 krone investert")
    ax.set_title("Kumulativ avkastning: fortegnsstrategi vs. buy & hold")
    ax.legend(fontsize=9)
    savefig(fig, "06_cumulative_returns.png")

    rows = {"BuyHold": _stats(rets["BuyHold"], pred, None)}
    for m in MODELS:
        rows[m] = _stats(rets[m], pred, m)
    table = pd.DataFrame(rows).T
    table = table.sort_values("sharpe", ascending=False)
    table.to_csv(config.OUTPUT_DIR / "06_strategy_stats.csv")
    print("[trade] Strategistatistikk:\n", table.round(3).to_string())
    return table


if __name__ == "__main__":
    run()
