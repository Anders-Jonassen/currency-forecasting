"""Shared configuration: paths and parameters used across the modules.

Keeping this in one place lets each analysis step run independently, and means
changes (e.g. a new time period) are made in a single location.
"""
from __future__ import annotations

from pathlib import Path

# Project root = the folder ONE level above src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
#  Data source: local Excel files (Datastream/Refinitiv extracts)
# --------------------------------------------------------------------------- #
# We landed on the BEST case: real ICE Brent term structure 1-12 (TRc1..TRc12)
# and NOK/USD, both monthly (month-end) for 2001-01..2021-03.

OIL_XLSX = DATA_DIR / "OilFuturesPrices.xlsx"
FX_XLSX = DATA_DIR / "NOKUSD.xlsx"

# FX orientation: the file gives "US $ TO NORWEGIAN KRONE" = USD per 1 NOK (~0.114).
# That is the price of the krone in dollars (NOKUSD). So we expect POSITIVE
# co-movement with oil: higher oil price -> stronger krone -> higher NOKUSD.
FX_NAME = "NOKUSD"

# The data is monthly (month-end). This is also the classic Diebold & Li (2006)
# setup (monthly curves).
FREQ = "ME"  # month-end

# Full term structure: 12 real maturities (TRc1..TRc12 ~ 1..12 month nearby).
MATURITY_MONTHS = list(range(1, 13))
N_MATURITIES = len(MATURITY_MONTHS)

# Diebold-Li decay parameter lambda (per month). It controls where the curvature
# loading peaks (peak ~ 1.79/lambda months). Diebold & Li (2006) use 0.0609 for
# yields (peak ~30 months). Our window is 1-12 months, so we calibrate lambda in
# the Diebold-Li step so curvature peaks mid-curve (~6 months), and justify it there.
DIEBOLD_LI_LAMBDA = 0.0609
