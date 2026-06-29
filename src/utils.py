"""Små hjelpefunksjoner brukt på tvers av analysestegene.

Holder figurstil og lagring ett sted, slik at alle figurer i output/ får et
konsistent, profesjonelt utseende.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: vi lagrer figurer til fil, viser ikke vindu
import matplotlib.pyplot as plt

from . import config


def set_style() -> None:
    """Sett en ren, nøytral matplotlib-stil."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5.5),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
        }
    )


def savefig(fig, name: str) -> None:
    """Lagre en figur til output/ med stramt layout, og lukk den."""
    import warnings

    path = config.OUTPUT_DIR / name
    # tight_layout passer ikke alltid for 3D-akser; demp den uskadelige advarselen.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")
