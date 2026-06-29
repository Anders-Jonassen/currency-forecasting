"""Step 3 - Diebold-Li factors on the ROLL-RETURN term structure.

What changed vs. raw prices
---------------------------
We do not decompose the raw price curve. Instead we first express the term
structure as roll returns relative to the front month:

    roll_k = (P_k - P_1) / P_1,    k = 2..12

(see data_acquisition.roll_return_curve). This is scale-free and isolates the
contango/backwardation SHAPE, which is what should carry information for the
currency. We then fit the Diebold-Li (Nelson-Siegel) factors to THIS curve.

The method in brief
-------------------
On each date the curve roll(tau) is described by three numbers via fixed
loadings (given lambda):

    roll(tau) = b1 * 1
              + b2 * (1 - e^(-lambda*tau)) / (lambda*tau)
              + b3 * [ (1 - e^(-lambda*tau)) / (lambda*tau) - e^(-lambda*tau) ]

  * Level  (b1): overall magnitude of the roll-return curve (how strongly the
                 market is in contango/backwardation on average across maturities).
  * Slope  (b2): short vs. long end of the roll curve.
  * Curvature (b3): mid-curve hump.

Because the loadings are fixed given lambda, the factors are a projection
b_t = pinv(X) . roll_t (OLS across the 11 maturities 2..12).

Choosing lambda
---------------
lambda is picked by a grid search that minimises the average fit error over the
2..12 month window; the curvature loading then peaks mid-curve.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data_acquisition import load_dataset, roll_return_curve
from .utils import savefig, set_style

FACTOR_NAMES = ["Level", "Slope", "Curvature"]
FACTORS_PATH = config.DATA_DIR / "factors.parquet"


def nelson_siegel_loadings(maturities, lam: float) -> np.ndarray:
    """Design matrix X (n_maturities x 3) with the three NS loadings."""
    tau = np.asarray(maturities, dtype=float)
    z = lam * tau
    slope = (1 - np.exp(-z)) / z
    curv = slope - np.exp(-z)
    level = np.ones_like(tau)
    return np.column_stack([level, slope, curv])


def estimate_factors(prices: pd.DataFrame, maturities, lam: float) -> tuple[pd.DataFrame, float]:
    """Estimate b1,b2,b3 for each date. Returns (factors, average fit RMSE).

    prices: DataFrame (date x maturities), e.g. the roll-return curve M2..M12.
    maturities: the maturities (in months) of those columns, in the same order.
    """
    X = nelson_siegel_loadings(maturities, lam)          # (k x 3)
    Xpinv = np.linalg.pinv(X)                            # (3 x k)
    Y = prices.to_numpy()                               # (T x k)
    B = Y @ Xpinv.T                                      # (T x 3)
    fitted = B @ X.T                                     # (T x k)
    rmse = float(np.sqrt(np.mean((Y - fitted) ** 2)))
    factors = pd.DataFrame(B, index=prices.index, columns=FACTOR_NAMES)
    return factors, rmse


def choose_lambda(prices: pd.DataFrame, maturities, grid=None) -> tuple[float, pd.DataFrame]:
    """Pick lambda by a grid search that minimises the average fit RMSE."""
    if grid is None:
        grid = np.round(np.arange(0.05, 1.01, 0.01), 3)
    rows = []
    for lam in grid:
        _, rmse = estimate_factors(prices, maturities, lam)
        rows.append((lam, rmse, 1.7937 / lam))  # 1.7937/lambda ~ peak maturity
    table = pd.DataFrame(rows, columns=["lambda", "rmse", "curv_peak_months"])
    best = table.loc[table["rmse"].idxmin(), "lambda"]
    return float(best), table


def plot_loadings(lam: float, maturities) -> None:
    set_style()
    mats = np.asarray(maturities)
    X = nelson_siegel_loadings(mats, lam)
    fig, ax = plt.subplots()
    for j, name in enumerate(FACTOR_NAMES):
        ax.plot(mats, X[:, j], marker="o", label=name)
    ax.set_xlabel("Maturity (months)")
    ax.set_ylabel("Factor weight (loading)")
    ax.set_title(f"Nelson-Siegel loadings (lambda = {lam:.2f})")
    ax.legend()
    savefig(fig, "03_factor_loadings.png")


def plot_factors(factors: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    titles = [
        "Level (b1) - overall roll-return level (contango/backwardation)",
        "Slope (b2) - short vs. long end of the roll curve",
        "Curvature (b3) - mid-curve hump",
    ]
    for ax, name, c, t in zip(axes, FACTOR_NAMES, colors, titles):
        ax.plot(factors.index, factors[name], color=c)
        ax.set_title(t)
        ax.axhline(0, color="black", lw=0.6)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Diebold-Li factors of the roll-return curve over time",
                 fontweight="bold")
    savefig(fig, "03_dl_factors.png")


def plot_sample_fit(prices: pd.DataFrame, maturities, lam: float, date=None) -> None:
    """Show actual vs. fitted roll-return curve on one date - a sanity check."""
    set_style()
    if date is None:
        date = prices.index[-1]
    X = nelson_siegel_loadings(maturities, lam)
    y = prices.loc[date].to_numpy()
    beta = np.linalg.pinv(X) @ y
    fig, ax = plt.subplots()
    ax.plot(maturities, y * 100, "o", label="Actual")
    ax.plot(maturities, (X @ beta) * 100, "-", label="Nelson-Siegel fit")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Maturity (months)")
    ax.set_ylabel("Roll return vs front (%)")
    ax.set_title(f"Roll-curve fit {pd.Timestamp(date).date()}")
    ax.legend()
    savefig(fig, "03_sample_fit.png")


def run() -> pd.DataFrame:
    df = load_dataset()
    roll = roll_return_curve(df)               # term structure as roll returns
    maturities = config.ROLL_MATURITY_MONTHS    # 2..12

    lam, table = choose_lambda(roll, maturities)
    peak = 1.7937 / lam
    print(f"[dl] Chosen lambda = {lam:.2f}  ->  curvature peak ~ {peak:.1f} months")
    factors, rmse = estimate_factors(roll, maturities, lam)
    print(f"[dl] Average fit RMSE: {rmse:.5f} (roll-return units; "
          f"curve std ~{roll.to_numpy().std():.4f})")

    plot_loadings(lam, maturities)
    plot_factors(factors)
    plot_sample_fit(roll, maturities, lam)

    factors.to_parquet(FACTORS_PATH)
    print(f"[dl] Saved factors to {FACTORS_PATH}")
    # Level should track the average roll return (overall contango level).
    print(f"[dl] corr(Level, mean roll return) = "
          f"{factors['Level'].corr(roll.mean(axis=1)):.3f}")
    return factors


if __name__ == "__main__":
    run()
