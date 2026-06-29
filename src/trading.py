"""Step 6 - Simple trading strategy and profitability.

Idea
----
A forecast is only useful if it can be turned into profitable decisions. We build
a deliberately SIMPLE sign strategy: each month we take a position in the krone
based on the SIGN of the predicted change.

    predicted change > 0  ->  long NOK  (we expect the krone to strengthen vs USD)
    predicted change < 0  ->  short NOK

Monthly return = position x actual NOK/USD return. We compare against "buy & hold"
(always long NOK). Run for both windowing schemes (expanding and rolling).

Metrics
-------
* Hit rate: share of months we got the direction right.
* Annualised return and volatility, and the Sharpe ratio (rf = 0).
The strategy is illustrative: no transaction costs or risk management.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .evaluation import MODELS, load_forecasts
from .forecasting import SCHEMES
from .utils import savefig, set_style


def _returns_frame(pred: pd.DataFrame) -> pd.DataFrame:
    """Actual monthly NOK/USD return and per-model strategy return."""
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
    row = {
        "total_return": (1 + ret).prod() - 1,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
    }
    if model is not None:
        actual_dir = np.sign(pred["y_true"] - pred["y_prev"])
        pred_dir = np.sign(pred[model] - pred["y_prev"])
        row["hit_rate"] = float((actual_dir == pred_dir).mean())
    else:
        row["hit_rate"] = np.nan
    return row


def run_scheme(scheme: str) -> pd.DataFrame:
    pred = load_forecasts(scheme)
    rets = _returns_frame(pred)
    cum = (1 + rets).cumprod()

    set_style()
    fig, ax = plt.subplots()
    for col in ["BuyHold"] + MODELS:
        lw = 2.2 if col == "Combination" else 1.3
        ax.plot(cum.index, cum[col], label=col, lw=lw,
                color="black" if col == "BuyHold" else None)
    ax.set_ylabel("Growth of 1 unit invested")
    ax.set_title(f"Cumulative return: sign strategy vs. buy & hold - {scheme} window")
    ax.legend(fontsize=9)
    savefig(fig, f"06_cumulative_returns_{scheme}.png")

    rows = {"BuyHold": _stats(rets["BuyHold"], pred, None)}
    for m in MODELS:
        rows[m] = _stats(rets[m], pred, m)
    table = pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)
    table.to_csv(config.OUTPUT_DIR / f"06_strategy_stats_{scheme}.csv")
    print(f"[trade:{scheme}] Strategy stats:\n", table.round(3).to_string())
    return table


def run() -> dict[str, pd.DataFrame]:
    stats = {scheme: run_scheme(scheme) for scheme in SCHEMES}
    # Cross-scheme comparison of Sharpe and total return.
    comp = pd.concat(
        {s: stats[s][["total_return", "sharpe", "hit_rate"]] for s in SCHEMES},
        axis=1,
    )
    comp.to_csv(config.OUTPUT_DIR / "06_strategy_compare.csv")
    print("[trade] Strategy comparison (expanding vs rolling):\n",
          comp.round(3).to_string())
    return stats


if __name__ == "__main__":
    run()
