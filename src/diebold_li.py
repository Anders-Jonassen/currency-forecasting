"""Steg 3 – Diebold-Li-faktorer (nivå, helning, krumning).

Kort om metoden
---------------
Diebold & Li (2006) tar Nelson-Siegel-kurven og lar de tre parametrene variere
over tid. På HVER dato beskrives hele terminstrukturen y(τ) (pris for maturity τ)
med bare tre tall:

    y(τ) = β1 · 1
         + β2 · (1 - e^(-λτ)) / (λτ)
         + β3 · [ (1 - e^(-λτ)) / (λτ) - e^(-λτ) ]

De tre "loadingene" (faktorvektene) har en fast form bestemt av λ:
  * Nivå  (β1): loading = 1 for alle maturities  -> flytter HELE kurven opp/ned.
  * Helning (β2): loading 1 ved τ→0, faller mot 0 for lange τ -> skiller kort
                 mot lang ende (contango vs. backwardation).
  * Krumning (β3): loading 0 i begge ender, topp på midten -> en "pukkel".

Vi estimerer β1,β2,β3 hver måned med vanlig OLS på tvers av de 12 maturitiene.
Fordi loadingene er faste (gitt λ), er dette en lineær regresjon med kjent
designmatrise X (12×3); faktorene er da bare en projeksjon: β_t = pinv(X) · y_t.

Hvorfor dette er nyttig
-----------------------
I stedet for 12 korrelerte priser får vi tre TOLKBARE, lavdimensjonale serier
som komprimerer kurvens form. Disse brukes som prediktorer for NOK/USD i steg 4.

Valg av λ
---------
λ styrer hvor krumnings-loadingen topper (topp ≈ 1.79/λ måneder). Diebold & Li
brukte λ=0.0609 for renter med maturities opp til 120 mnd (topp ~30 mnd). VÅRT
vindu er bare 1–12 mnd, så vi velger λ ved et lite rutenett-søk som minimerer
gjennomsnittlig tilpasningsfeil – og sjekker at toppen havner midt på kurven.
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
    """Designmatrise X (n_maturities × 3) med de tre NS-loadingene."""
    tau = np.asarray(maturities, dtype=float)
    z = lam * tau
    slope = (1 - np.exp(-z)) / z
    curv = slope - np.exp(-z)
    level = np.ones_like(tau)
    return np.column_stack([level, slope, curv])


def estimate_factors(prices: pd.DataFrame, lam: float) -> tuple[pd.DataFrame, float]:
    """Estimer β1,β2,β3 for hver dato. Returnerer (faktorer, gj.snittlig RMSE).

    prices: DataFrame (dato × 12 maturities). Kolonnene antas M1..M12.
    """
    maturities = config.MATURITY_MONTHS
    X = nelson_siegel_loadings(maturities, lam)          # (12 × 3)
    Xpinv = np.linalg.pinv(X)                            # (3 × 12)
    Y = prices.to_numpy()                               # (T × 12)
    B = Y @ Xpinv.T                                      # (T × 3)
    fitted = B @ X.T                                     # (T × 12)
    rmse = float(np.sqrt(np.mean((Y - fitted) ** 2)))
    factors = pd.DataFrame(B, index=prices.index, columns=FACTOR_NAMES)
    return factors, rmse


def choose_lambda(prices: pd.DataFrame, grid=None) -> tuple[float, pd.DataFrame]:
    """Velg λ ved rutenett-søk som minimerer gjennomsnittlig tilpasnings-RMSE."""
    if grid is None:
        grid = np.round(np.arange(0.05, 1.01, 0.01), 3)
    rows = []
    for lam in grid:
        _, rmse = estimate_factors(prices, lam)
        rows.append((lam, rmse, 1.7937 / lam))  # 1.7937/λ ≈ topp-maturity
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
    ax.set_xlabel("Maturity (mnd)")
    ax.set_ylabel("Faktorvekt (loading)")
    ax.set_title(f"Nelson-Siegel loadings (λ = {lam:.2f})")
    ax.legend()
    savefig(fig, "03_factor_loadings.png")


def plot_factors(factors: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    titles = [
        "Nivå (β1) – samlet prisnivå på kurven",
        "Helning (β2) – contango/backwardation",
        "Krumning (β3) – pukkel på midten",
    ]
    for ax, name, c, t in zip(axes, FACTOR_NAMES, colors, titles):
        ax.plot(factors.index, factors[name], color=c)
        ax.set_title(t)
        ax.axhline(0, color="black", lw=0.6)
    axes[-1].set_xlabel("Tid")
    fig.suptitle("Diebold-Li-faktorer over tid", fontweight="bold")
    savefig(fig, "03_dl_factors.png")


def plot_sample_fit(prices: pd.DataFrame, lam: float, date=None) -> None:
    """Vis faktisk vs. tilpasset kurve på én dato – sanity check på tilpasningen."""
    set_style()
    if date is None:
        date = prices.index[-1]
    maturities = config.MATURITY_MONTHS
    X = nelson_siegel_loadings(maturities, lam)
    y = prices.loc[date].to_numpy()
    beta = np.linalg.pinv(X) @ y
    fig, ax = plt.subplots()
    ax.plot(maturities, y, "o", label="Faktisk")
    ax.plot(maturities, X @ beta, "-", label="Nelson-Siegel-tilpasning")
    ax.set_xlabel("Maturity (mnd)")
    ax.set_ylabel("Pris (USD/fat)")
    ax.set_title(f"Kurvetilpasning {pd.Timestamp(date).date()}")
    ax.legend()
    savefig(fig, "03_sample_fit.png")


def run() -> pd.DataFrame:
    df = load_dataset()
    prices = df[_price_cols()]

    lam, table = choose_lambda(prices)
    peak = 1.7937 / lam
    print(f"[dl] Valgt λ = {lam:.2f}  ->  krumningstopp ≈ {peak:.1f} mnd")
    factors, rmse = estimate_factors(prices, lam)
    print(f"[dl] Gj.snittlig tilpasnings-RMSE: {rmse:.3f} USD/fat "
          f"(snittpris ~{prices.to_numpy().mean():.0f})")

    plot_loadings(lam)
    plot_factors(factors)
    plot_sample_fit(prices, lam)

    factors.to_parquet(FACTORS_PATH)
    print(f"[dl] Lagret faktorer til {FACTORS_PATH}")
    # Liten tolkningshjelp: korrelasjon mellom Level og front-month-pris.
    print(f"[dl] korr(Level, M1) = {factors['Level'].corr(prices['M1']):.3f}")
    return factors


if __name__ == "__main__":
    run()
