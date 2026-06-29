"""Kjør hele analysen fra ett sted.

Bruk:
    python main.py          # kjør alle steg 1-6 i rekkefølge
    python main.py 4        # hopp inn og kjør fra og med steg 4

Hvert steg lagrer figurer/tabeller til output/. Stegene kan også kjøres
enkeltvis, f.eks.  python -m src.diebold_li
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
    ("1 – Datainnhenting og align", data_acquisition.build_dataset),
    ("2 – Eksplorativ analyse", eda.run),
    ("3 – Diebold-Li-faktorer", diebold_li.run),
    ("4 – Rullende OOS-prognoser (~30 s)", forecasting.run),
    ("5 – Evaluering", evaluation.run),
    ("6 – Lønnsomhet", trading.run),
]


def main(start: int = 1) -> None:
    for n, (name, fn) in enumerate(STEPS, start=1):
        if n < start:
            continue
        print(f"\n{'=' * 64}\n=== STEG {name}\n{'=' * 64}")
        fn()
    print("\n✓ Ferdig. Alle figurer og tabeller ligger i output/.")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(start)
