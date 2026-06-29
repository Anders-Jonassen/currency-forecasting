"""Steg 2 – Eksplorativ analyse.

Hva steget gjør
---------------
1. Plotter NOK/USD mot front-month-oljen (M1) og rapporterer korrelasjonen.
2. Beregner og plotter en 60-måneders RULLENDE korrelasjon mellom de to.
3. Lager et 3D-plott av terminstrukturen (tid × maturity × pris).

Hvorfor disse tre
-----------------
Vi vil danne oss et bilde FØR vi modellerer: Henger kronen og oljen sammen slik
petrovaluta-hypotesen sier? Er sammenhengen stabil over tid, eller skifter den?
Og hvordan ser selve terminstrukturen ut – er markedet typisk i contango eller
backwardation?

Nivåer vs. endringer
--------------------
Korrelasjon mellom to trendende NIVÅ-serier kan bli kunstig høy (begge kan
tilfeldigvis trende samtidig). Derfor regner vi korrelasjon på månedlige
PROSENTENDRINGER (avkastning), som er mer økonomisk meningsfullt og unngår denne
fellen. Nivå-plottet vises likevel, fordi det er intuitivt å se på.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data_acquisition import load_dataset
from .utils import savefig, set_style

ROLL_WINDOW = 60  # måneder


def _returns(df: pd.DataFrame) -> pd.DataFrame:
    """Månedlige prosentendringer (avkastning)."""
    return df.pct_change().dropna()


def plot_fx_vs_frontmonth(df: pd.DataFrame) -> float:
    """NOK/USD mot front-month (M1) på to akser. Returnerer avkastningskorrelasjon."""
    set_style()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df.index, df["NOKUSD"], color="#1f77b4", label="NOK/USD")
    ax2.plot(df.index, df["M1"], color="#d62728", alpha=0.8, label="Brent front-month")
    ax1.set_ylabel("NOK/USD (USD per krone)", color="#1f77b4")
    ax2.set_ylabel("Brent M1 (USD/fat)", color="#d62728")
    ax2.grid(False)

    # Korrelasjon på avkastning (mer meningsfullt enn på nivåer, se modul-docstring).
    ret = _returns(df[["NOKUSD", "M1"]])
    corr_ret = ret["NOKUSD"].corr(ret["M1"])
    corr_lvl = df["NOKUSD"].corr(df["M1"])
    ax1.set_title(
        f"NOK/USD vs. Brent front-month\n"
        f"korr(avkastning)={corr_ret:.2f}  |  korr(nivå)={corr_lvl:.2f}"
    )
    savefig(fig, "02_nokusd_vs_frontmonth.png")
    return corr_ret


def plot_rolling_correlation(df: pd.DataFrame) -> pd.Series:
    """60-måneders rullende korrelasjon av månedlig avkastning."""
    set_style()
    ret = _returns(df[["NOKUSD", "M1"]])
    roll = ret["NOKUSD"].rolling(ROLL_WINDOW).corr(ret["M1"]).dropna()

    fig, ax = plt.subplots()
    ax.plot(roll.index, roll, color="#2ca02c")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(roll.mean(), color="gray", ls="--", lw=1,
               label=f"snitt = {roll.mean():.2f}")
    ax.set_ylabel("Korrelasjon")
    ax.set_title(f"{ROLL_WINDOW}-mnd rullende korrelasjon: NOK/USD- vs. Brent-avkastning")
    ax.legend()
    savefig(fig, "02_rolling_corr_60m.png")
    return roll


def plot_term_structure_3d(df: pd.DataFrame) -> None:
    """3D-flate: tid (x) × maturity (y) × pris (z)."""
    set_style()
    mats = config.MATURITY_MONTHS  # 1..12
    cols = [f"M{m}" for m in mats]
    Z = df[cols].to_numpy().T  # form (maturity, tid)
    # X = tid som ordinaltall (for jevn flate), Y = maturity i måneder.
    x = np.arange(df.shape[0])
    X, Y = np.meshgrid(x, mats)

    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)

    # Sett noen få datoetiketter på tidsaksen i stedet for ordinaltall.
    tick_idx = np.linspace(0, df.shape[0] - 1, 5).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([df.index[i].strftime("%Y") for i in tick_idx])
    ax.set_xlabel("Tid")
    ax.set_ylabel("Maturity (mnd)")
    ax.set_zlabel("Pris (USD/fat)")
    ax.set_title("ICE Brent terminstruktur over tid")
    ax.view_init(elev=25, azim=-60)
    savefig(fig, "02_term_structure_3d.png")


def run() -> None:
    df = load_dataset()
    print(f"[eda] {len(df)} månedlige observasjoner "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    corr = plot_fx_vs_frontmonth(df)
    print(f"[eda] Korrelasjon (avkastning) NOK/USD vs Brent M1: {corr:.3f}")
    roll = plot_rolling_correlation(df)
    print(f"[eda] Rullende korr: snitt {roll.mean():.2f}, "
          f"min {roll.min():.2f}, max {roll.max():.2f}")
    plot_term_structure_3d(df)


if __name__ == "__main__":
    run()
