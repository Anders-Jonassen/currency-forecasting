"""Step 3 - Diebold-Li factors (level, slope, curvature).

The method in brief
-------------------
Diebold & Li (2006) take the Nelson-Siegel curve and let its three parameters
vary over time. On EACH date the whole term structure y(tau) (price at maturity
tau) is described by just three numbers:

    y(tau) = b1 * 1
           + b2 * (1 - e^(-lambda*tau)) / (lambda*tau)
           + b3 * [ (1 - e^(-lambda*tau)) / (lambda*tau) - e^(-lambda*tau) ]

The three "loadings" (factor weights) have a fixed shape determined by lambda:
  * Level  (b1): loading = 1 for all maturities -> shifts the WHOLE curve up/down.
  * Slope  (b2): loading 1 at tau->0, falling to 0 for long tau -> separates the
                 short from the long end (contango vs. backwardation).
  * Curvature (b3): loading 0 at both ends, peak in the middle -> a "hump".

We estimate b1,b2,b3 each month with ordinary OLS across the 12 maturities.
Because the loadings are fixed (given lambda), this is a linear regression with a
known design matrix X (12x3); the factors are then just a projection:
b_t = pinv(X) . y_t.

Why this is useful
------------------
Instead of 12 correlated prices we get three INTERPRETABLE, low-dimensional
series that compress the curve's shape. These are used as predictors for NOK/USD
in step 4.

Choosing lambda
---------------
lambda controls where the curvature loading peaks (peak ~ 1.79/lambda months).
Diebold & Li used lambda=0.0609 for yields with maturities up to 120 months (peak
~30 months). OUR window is only 1-12 months, so we pick lambda by a small grid
search that minimises the average fit error - and check the peak lands mid-curve.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data_acquisition import load_dataset
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


def estimate_factors(prices: pd.DataFrame, lam: float) -> tuple[pd.DataFrame, float]:
    """Estimate b1,b2,b3 for each date. Returns (factors, average fit RMSE).

    prices: DataFrame (date x 12 maturities). Columns are assumed to be M1..M12.
    """
    maturities = config.MATURITY_MONTHS
    X = nelson_siegel_loadings(maturities, lam)          # (12 x 3)
    Xpinv = np.linalg.pinv(X)                            # (3 x 12)
    Y = prices.to_numpy()                               # (T x 12)
    B = Y @ Xpinv.T                                      # (T x 3)
    fitted = B @ X.T                                     # (T x 12)
    rmse = float(np.sqrt(np.mean((Y - fitted) ** 2)))
    factors = pd.DataFrame(B, index=prices.index, columns=FACTOR_NAMES)
    return factors, rmse


def choose_lambda(prices: pd.DataFrame, grid=None) -> tuple[float, pd.DataFrame]:
    """Pick lambda by a grid search that minimises the average fit RMSE."""
    if grid is None:
        grid = np.round(np.arange(0.05, 1.01, 0.01), 3)
    rows = []
    for lam in grid:
        _, rmse = estimate_factors(prices, lam)
        rows.append((lam, rmse, 1.7937 / lam))  # 1.7937/lambda ~ peak maturity
    table = pd.DataFrame(rows, columns=["lambda", "rmse", "curv_peak_months"])
    best = table.loc[table["rmse"].idxmin(), "lambda"]
    return float(best), table


def _price_cols() -> list[str]:
    return [f"M{m}" for m in config.MATURITY_MONTHS]


def plot_loadings(lam: float) -> None:
    set_style()
    mats = np.arange(1, 13)
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
        "Level (b1) - overall price level of the curve",
        "Slope (b2) - contango/backwardation",
        "Curvature (b3) - mid-curve hump",
    ]
    for ax, name, c, t in zip(axes, FACTOR_NAMES, colors, titles):
        ax.plot(factors.index, factors[name], color=c)
        ax.set_title(t)
        ax.axhline(0, color="black", lw=0.6)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Diebold-Li factors over time", fontweight="bold")
    savefig(fig, "03_dl_factors.png")


def plot_sample_fit(prices: pd.DataFrame, lam: float, date=None) -> None:
    """Show actual vs. fitted curve on one date - a sanity check on the fit."""
    set_style()
    if date is None:
        date = prices.index[-1]
    maturities = config.MATURITY_MONTHS
    X = nelson_siegel_loadings(maturities, lam)
    y = prices.loc[date].to_numpy()
    beta = np.linalg.pinv(X) @ y
    fig, ax = plt.subplots()
    ax.plot(maturities, y, "o", label="Actual")
    ax.plot(maturities, X @ beta, "-", label="Nelson-Siegel fit")
    ax.set_xlabel("Maturity (months)")
    ax.set_ylabel("Price (USD/bbl)")
    ax.set_title(f"Curve fit {pd.Timestamp(date).date()}")
    ax.legend()
    savefig(fig, "03_sample_fit.png")


def run() -> pd.DataFrame:
    df = load_dataset()
    prices = df[_price_cols()]

    lam, table = choose_lambda(prices)
    peak = 1.7937 / lam
    print(f"[dl] Chosen lambda = {lam:.2f}  ->  curvature peak ~ {peak:.1f} months")
    factors, rmse = estimate_factors(prices, lam)
    print(f"[dl] Average fit RMSE: {rmse:.3f} USD/bbl "
          f"(mean price ~{prices.to_numpy().mean():.0f})")

    plot_loadings(lam)
    plot_factors(factors)
    plot_sample_fit(prices, lam)

    factors.to_parquet(FACTORS_PATH)
    print(f"[dl] Saved factors to {FACTORS_PATH}")
    print(f"[dl] corr(Level, M1) = {factors['Level'].corr(prices['M1']):.3f}")
    return factors


if __name__ == "__main__":
    run()
