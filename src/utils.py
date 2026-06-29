"""Small helper functions used across the analysis steps.

Keeps figure style and saving in one place so every figure in output/ gets a
consistent, professional look.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: we save figures to file, no window
import matplotlib.pyplot as plt

from . import config


def set_style() -> None:
    """Set a clean, neutral matplotlib style."""
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
    """Save a figure to output/ with tight layout, and close it."""
    import warnings

    path = config.OUTPUT_DIR / name
    # tight_layout does not always suit 3D axes; silence the harmless warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")
