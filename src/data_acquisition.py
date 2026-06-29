"""Steg 1 – Datainnhenting og opprydding.

Hva steget gjør
---------------
1. Leser NOK/USD og ICE Brent terminstruktur (1-12) fra de lokale Excel-filene
   gjennom det modulære data_loader-grensesnittet.
2. Aligner de to tidsseriene på en felles (månedlig) datoindeks. Kildene her er
   allerede månedsslutt-data for samme periode, men vi gjør alignet eksplisitt
   og robust slik at koden også tåler kilder med ulike kalendere senere.
3. Lagrer det rensede datasettet til data/ slik at senere steg kan kjøres
   uavhengig uten å lese inn på nytt.

Hvorfor "first nearby" = M1
---------------------------
Den korteste kontrakten (TRc1 -> M1) er front-month og brukes som "first nearby"
gjennom hele prosjektet.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .data_loader import ExcelFXLoader, ExcelTermStructureLoader

DATASET_PATH = config.DATA_DIR / "dataset.parquet"


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Les, align og lagre det kombinerte datasettet.

    Returnerer en DataFrame indeksert på dato med kolonner:
        NOKUSD, M1, M2, ..., M12
    """
    fx = ExcelFXLoader(config.FX_XLSX, name=config.FX_NAME).load()
    ts_loader = ExcelTermStructureLoader(config.OIL_XLSX)
    term = ts_loader.load()
    print(f"[data] Terminstruktur-kilde: {ts_loader.source_level}")
    print(f"[data] FX: {fx.name}  ({fx.index.min().date()} -> {fx.index.max().date()})")
    print(f"[data] Maturities: {list(term.columns)}")

    # Normaliser begge til månedsslutt-perioder, slik at datoer fra ulike
    # kilder (som kan ligge på litt ulike dager i samme måned) faller sammen.
    fx_m = fx.copy()
    fx_m.index = fx_m.index.to_period("M")
    term_m = term.copy()
    term_m.index = term_m.index.to_period("M")

    combined = pd.concat([fx_m, term_m], axis=1)
    n_raw = len(combined)
    combined = combined.dropna()  # behold kun måneder der ALT finnes
    # Tilbake til tidsstempel (månedsslutt) for et ryddig DatetimeIndex.
    combined.index = combined.index.to_timestamp("M")
    combined = combined.sort_index()

    print(
        f"[data] {n_raw} måneder før align -> {len(combined)} komplette måneder "
        f"({combined.index.min().date()} -> {combined.index.max().date()})"
    )

    if save:
        combined.to_parquet(DATASET_PATH)
        print(f"[data] Lagret til {DATASET_PATH}")
    return combined


def load_dataset() -> pd.DataFrame:
    """Les det ferdig rensede datasettet fra disk (for senere steg)."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Fant ikke {DATASET_PATH}. Kjør build_dataset() først."
        )
    return pd.read_parquet(DATASET_PATH)


if __name__ == "__main__":
    df = build_dataset()
    print("\n[data] Forhåndsvisning:")
    print(df.head().to_string())
    print(df.describe().round(3).to_string())
