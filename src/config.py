"""Felles konfigurasjon: stier og parametre brukt på tvers av modulene.

Å samle dette ett sted gjør at hvert analysesteg kan kjøres separat og at
endringer (f.eks. ny tidsperiode) bare gjøres ett sted.
"""
from __future__ import annotations

from pathlib import Path

# Prosjektrot = mappen ETT nivå over src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
#  Datakilde: lokale Excel-filer (Datastream/Refinitiv-uttrekk)
# --------------------------------------------------------------------------- #
# Vi landet på BESTE tilfelle: ekte ICE Brent terminstruktur 1-12 (TRc1..TRc12)
# og NOK/USD, begge månedlige (månedsslutt) for 2001-01..2021-03.

OIL_XLSX = DATA_DIR / "OilFuturesPrices.xlsx"
FX_XLSX = DATA_DIR / "NOKUSD.xlsx"

# FX-orientering: filen gir "US $ TO NORWEGIAN KRONE" = USD per 1 NOK (~0.114).
# Altså prisen på kronen målt i dollar (NOKUSD). Da forventer vi POSITIV
# samvariasjon med olje: høyere oljepris -> sterkere krone -> høyere NOKUSD.
FX_NAME = "NOKUSD"

# Dataene er månedlige (månedsslutt). Dette er også Diebold & Li (2006) sitt
# klassiske oppsett (månedlige kurver).
FREQ = "ME"  # month-end

# Full terminstruktur: 12 ekte maturities (TRc1..TRc12 ~ 1..12 mnd nearby).
MATURITY_MONTHS = list(range(1, 13))
N_MATURITIES = len(MATURITY_MONTHS)

# Diebold-Li henfallsparameter lambda (per måned). Styrer hvor krumningsleddet
# topper. Diebold & Li (2006) bruker 0.0609 for renter (topp ~30 mnd). Vårt
# vindu er 1-12 mnd, så vi kalibrerer lambda slik at krumningen topper midt på
# kurven (~6 mnd) i Diebold-Li-steget, og begrunner valget der.
DIEBOLD_LI_LAMBDA = 0.0609
