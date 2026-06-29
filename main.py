"""Run the whole analysis from one place.

Usage:
    python main.py          # run all steps 1-6 in order
    python main.py 4        # jump in and run from step 4 onwards

Each step saves figures/tables to output/. Steps 4-6 are run for BOTH windowing
schemes (expanding and rolling). Steps can also be run individually, e.g.
    python -m src.diebold_li
"""
from __future__ import annotations

import sys

from src import (
    data_acquisition,
    diebold_li,
    eda,
    evaluation,
    forecasting,
    trading,
)

STEPS = [
    ("1 - Data acquisition and alignment", data_acquisition.build_dataset),
    ("2 - Exploratory analysis", eda.run),
    ("3 - Diebold-Li factors", diebold_li.run),
    ("4 - Rolling OOS forecasts: expanding + rolling (~1 min)", forecasting.run),
    ("5 - Evaluation (both schemes)", evaluation.run),
    ("6 - Profitability (both schemes)", trading.run),
]


def main(start: int = 1) -> None:
    for n, (name, fn) in enumerate(STEPS, start=1):
        if n < start:
            continue
        print(f"\n{'=' * 64}\n=== STEP {name}\n{'=' * 64}")
        fn()
    print("\nDone. All figures and tables are in output/.")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(start)
