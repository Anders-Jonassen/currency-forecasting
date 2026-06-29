"""Step 2 - Exploratory analysis.

What this step does
-------------------
1. Plots NOK/USD against the front-month oil (M1) and reports the correlation.
2. Computes and plots a 60-month ROLLING correlation between the two.
3. Builds a 3D plot of the term structure (time x maturity x price).

Why these three
---------------
We want a picture BEFORE modelling: do the krone and oil move together as the
petrocurrency hypothesis says? Is the relationship stable over time, or does it
shift? And what does the term structure itself look like - is the market usually
in contango or backwardation?

Levels vs. changes
------------------
Correlation between two trending LEVEL series can be artificially high (both may
trend at the same time). So we compute correlation on monthly PERCENT CHANGES
(returns), which is more economically meaningful and avoids that trap. The level
plot is still shown because it is intuitive to look at.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data_acquisition import load_dataset
from .utils import savefig, set_style

ROLL_WINDOW = 60  # months


def _returns(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly percent changes (returns)."""
    return df.pct_change().dropna()


def plot_fx_vs_frontmonth(df: pd.DataFrame) -> float:
    """NOK/USD vs. front-month (M1) on twin axes. Returns the return correlation."""
    set_style()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df.index, df["NOKUSD"], color="#1f77b4", label="NOK/USD")
    ax2.plot(df.index, df["M1"], color="#d62728", alpha=0.8, label="Brent front-month")
    ax1.set_ylabel("NOK/USD (USD per krone)", color="#1f77b4")
    ax2.set_ylabel("Brent M1 (USD/bbl)", color="#d62728")
    ax2.grid(False)

    # Correlation on returns (more meaningful than on levels, see module docstring).
    ret = _returns(df[["NOKUSD", "M1"]])
    corr_ret = ret["NOKUSD"].corr(ret["M1"])
    corr_lvl = df["NOKUSD"].corr(df["M1"])
    ax1.set_title(
        f"NOK/USD vs. Brent front-month\n"
        f"corr(returns)={corr_ret:.2f}  |  corr(levels)={corr_lvl:.2f}"
    )
    savefig(fig, "02_nokusd_vs_frontmonth.png")
    return corr_ret


def plot_rolling_correlation(df: pd.DataFrame) -> pd.Series:
    """60-month rolling correlation of monthly returns."""
    set_style()
    ret = _returns(df[["NOKUSD", "M1"]])
    roll = ret["NOKUSD"].rolling(ROLL_WINDOW).corr(ret["M1"]).dropna()

    fig, ax = plt.subplots()
    ax.plot(roll.index, roll, color="#2ca02c")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(roll.mean(), color="gray", ls="--", lw=1,
               label=f"mean = {roll.mean():.2f}")
    ax.set_ylabel("Correlation")
    ax.set_title(f"{ROLL_WINDOW}-month rolling correlation: NOK/USD vs. Brent returns")
    ax.legend()
    savefig(fig, "02_rolling_corr_60m.png")
    return roll


def plot_term_structure_3d(df: pd.DataFrame) -> None:
    """3D surface: time (x) x maturity (y) x price (z)."""
    set_style()
    mats = config.MATURITY_MONTHS  # 1..12
    cols = [f"M{m}" for m in mats]
    Z = df[cols].to_numpy().T  # shape (maturity, time)
    x = np.arange(df.shape[0])
    X, Y = np.meshgrid(x, mats)

    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)

    # Put a few year labels on the time axis instead of ordinal numbers.
    tick_idx = np.linspace(0, df.shape[0] - 1, 5).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([df.index[i].strftime("%Y") for i in tick_idx])
    ax.set_xlabel("Time")
    ax.set_ylabel("Maturity (months)")
    ax.set_zlabel("Price (USD/bbl)")
    ax.set_title("ICE Brent term structure over time")
    ax.view_init(elev=25, azim=-60)
    savefig(fig, "02_term_structure_3d.png")


def run() -> None:
    df = load_dataset()
    print(f"[eda] {len(df)} monthly observations "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    corr = plot_fx_vs_frontmonth(df)
    print(f"[eda] Correlation (returns) NOK/USD vs Brent M1: {corr:.3f}")
    roll = plot_rolling_correlation(df)
    print(f"[eda] Rolling corr: mean {roll.mean():.2f}, "
          f"min {roll.min():.2f}, max {roll.max():.2f}")
    plot_term_structure_3d(df)


if __name__ == "__main__":
    run()
