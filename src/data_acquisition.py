"""Step 1 - Data acquisition and cleaning.

What this step does
-------------------
1. Reads NOK/USD and the ICE Brent term structure (1-12) from the local Excel
   files through the modular data_loader interface.
2. Aligns the two time series on a common (monthly) date index. Here both sources
   are already month-end data for the same period, but we make the alignment
   explicit and robust so the code also tolerates sources with different
   calendars later.
3. Saves the cleaned dataset to data/ so later steps can run independently
   without re-reading.

Why "first nearby" = M1
-----------------------
The shortest contract (TRc1 -> M1) is the front-month and is used as the "first
nearby" throughout the project.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .data_loader import ExcelFXLoader, ExcelTermStructureLoader

DATASET_PATH = config.DATA_DIR / "dataset.parquet"


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Read, align and save the combined dataset.

    Returns a DataFrame indexed by date with columns:
        NOKUSD, M1, M2, ..., M12
    """
    fx = ExcelFXLoader(config.FX_XLSX, name=config.FX_NAME).load()
    ts_loader = ExcelTermStructureLoader(config.OIL_XLSX)
    term = ts_loader.load()
    print(f"[data] Term-structure source: {ts_loader.source_level}")
    print(f"[data] FX: {fx.name}  ({fx.index.min().date()} -> {fx.index.max().date()})")
    print(f"[data] Maturities: {list(term.columns)}")

    # Normalise both to month-end periods so dates from different sources (which
    # may fall on slightly different days in the same month) line up.
    fx_m = fx.copy()
    fx_m.index = fx_m.index.to_period("M")
    term_m = term.copy()
    term_m.index = term_m.index.to_period("M")

    combined = pd.concat([fx_m, term_m], axis=1)
    n_raw = len(combined)
    combined = combined.dropna()  # keep only months where EVERYTHING is present
    combined.index = combined.index.to_timestamp("M")  # back to a tidy DatetimeIndex
    combined = combined.sort_index()

    print(
        f"[data] {n_raw} months before align -> {len(combined)} complete months "
        f"({combined.index.min().date()} -> {combined.index.max().date()})"
    )

    if save:
        combined.to_parquet(DATASET_PATH)
        print(f"[data] Saved to {DATASET_PATH}")
    return combined


def load_dataset() -> pd.DataFrame:
    """Read the cleaned dataset from disk (for later steps)."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run build_dataset() first."
        )
    return pd.read_parquet(DATASET_PATH)


if __name__ == "__main__":
    df = build_dataset()
    print("\n[data] Preview:")
    print(df.head().to_string())
    print(df.describe().round(3).to_string())
