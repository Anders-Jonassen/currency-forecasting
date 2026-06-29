"""Modulært datagrensesnitt.

Designidé
---------
All resten av prosjektet snakker KUN med abstraksjonene under – aldri direkte
med en konkret fil eller et API. Vil du senere bytte til Bloomberg/Refinitiv-API,
skriver du en ny klasse som arver `TermStructureLoader` og returnerer samme
format. Ingenting annet i koden må endres.

Aktiv kilde i dette prosjektet: lokale Excel-filer (Datastream/Refinitiv-uttrekk)
med ekte ICE Brent terminstruktur 1-12 og NOK/USD – se Excel*-klassene nederst.
yfinance/EIA-variantene er beholdt som dokumenterte alternativer.

Felles returformat
------------------
* FX:               pd.Series indeksert på dato (NOKUSD = USD per krone).
* Terminstruktur:   pd.DataFrame indeksert på dato, én kolonne per maturity
                    (M1, M2, ... = 1., 2., ... nearby-kontrakt), verdier = pris.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


# --------------------------------------------------------------------------- #
#  Abstrakte grensesnitt
# --------------------------------------------------------------------------- #
class FXLoader(ABC):
    """Henter valutakursen som en daglig pd.Series (USDNOK)."""

    @abstractmethod
    def load(self) -> pd.Series:  # pragma: no cover - grensesnitt
        ...


class TermStructureLoader(ABC):
    """Henter oljefuturesenes terminstruktur som DataFrame (dato x maturity).

    Implementasjoner SKAL sette `self.source_level` til en kort streng som
    forteller hvilket fallback-nivå dataene representerer, slik at README og
    figurer kan merkes ærlig (f.eks. "Fallback A: EIA CL1-CL4, ekte data").
    """

    source_level: str = "uspesifisert"

    @abstractmethod
    def load(self) -> pd.DataFrame:  # pragma: no cover - grensesnitt
        ...

    @property
    def maturities(self) -> list[str]:
        """Kolonnenavn for maturities, f.eks. ['M1', 'M2', ...]."""
        return list(self.load().columns)


# --------------------------------------------------------------------------- #
#  AKTIV KILDE: lokale Excel-filer (Datastream/Refinitiv-uttrekk)
# --------------------------------------------------------------------------- #
def _read_excel_with_date_index(path) -> pd.DataFrame:
    """Les et Datastream-ark der kol. 0 ('Name') er datoer og resten verdier.

    Felles hjelpefunksjon for begge Excel-lasterne: setter datokolonnen som
    indeks og kaster bort rader uten gyldig dato.
    """
    df = pd.read_excel(path, header=0)
    date_col = df.columns[0]  # 'Name' i Datastream-uttrekket = datoene
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"
    return df


class ExcelFXLoader(FXLoader):
    """NOK/USD fra lokal Excel-fil (Datastream 'US $ TO NORWEGIAN KRONE').

    Returnerer USD per krone (NOKUSD, ~0.11). Filen har én verdikolonne.
    """

    def __init__(self, path, name: str = "NOKUSD"):
        self.path = path
        self.name = name

    def load(self) -> pd.Series:
        df = _read_excel_with_date_index(self.path)
        s = df.iloc[:, 0].astype(float)  # eneste verdikolonne
        s.name = self.name
        return s.dropna()


class ExcelTermStructureLoader(TermStructureLoader):
    """ICE Brent terminstruktur 1-12 fra lokal Excel-fil (Datastream TRc1..TRc12).

    Kolonnene heter 'ICE-BRENT CRUDE OIL TRc{n} - SETT. PRICE'. Vi trekker ut
    nearby-nummeret {n} og navngir kolonnene M1..M12 i stigende rekkefølge, slik
    at M1 = front-month ("first nearby") og M12 = 12. nearby.
    """

    source_level = "Excel: ICE Brent TRc1-TRc12 (ekte, full 1-12 kurve, månedlig)"

    def __init__(self, path):
        self.path = path

    @staticmethod
    def _nearby_number(col: str) -> int | None:
        """Hent nearby-nummeret fra et kolonnenavn, f.eks. 'TRc7' -> 7."""
        import re

        m = re.search(r"TRc(\d+)", str(col))
        return int(m.group(1)) if m else None

    def load(self) -> pd.DataFrame:
        df = _read_excel_with_date_index(self.path)
        # Map hver kolonne til sitt nearby-nummer og behold bare TRc-kolonnene.
        renaming = {}
        for col in df.columns:
            n = self._nearby_number(col)
            if n is not None:
                renaming[col] = f"M{n}"
        if not renaming:
            raise RuntimeError(
                f"Fant ingen 'TRc<n>'-kolonner i {self.path}. Sjekk filformatet."
            )
        out = df[list(renaming)].rename(columns=renaming).astype(float)
        # Sorter kolonnene M1, M2, ... numerisk (ikke leksikografisk).
        out = out[sorted(out.columns, key=lambda c: int(c[1:]))]
        return out


# --------------------------------------------------------------------------- #
#  ALTERNATIV KILDE: yfinance FX (beholdt for modularitet/dokumentasjon)
# --------------------------------------------------------------------------- #
class YFinanceFXLoader(FXLoader):
    """USDNOK fra yfinance. Ekte, lang daglig historikk (fra 2001)."""

    def __init__(self, ticker: str = "NOK=X", start: str | None = None):
        self.ticker = ticker
        self.start = start

    def load(self) -> pd.Series:
        import yfinance as yf

        df = yf.download(
            self.ticker, start=self.start, progress=False, auto_adjust=True
        )
        s = df["Close"]
        # yfinance kan returnere enten Series eller 1-kolonnes DataFrame.
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = "USDNOK"
        return s.dropna()


# --------------------------------------------------------------------------- #
#  Konkret: yfinance front-month (nearby-1) – alltid tilgjengelig
# --------------------------------------------------------------------------- #
class YFinanceFrontMonthLoader(TermStructureLoader):
    """Kun front-month (CL=F / BZ=F). Gir ÉN maturity-kolonne (M1).

    Brukes som "first nearby" og som byggekloss/benchmark uansett hvilket
    fallback-nivå terminstrukturen ellers havner på.
    """

    source_level = "Front-month (ekte, kun M1)"

    def __init__(self, ticker: str = "CL=F", start: str | None = None):
        self.ticker = ticker
        self.start = start

    def load(self) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            self.ticker, start=self.start, progress=False, auto_adjust=True
        )
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        out = close.dropna().to_frame(name="M1")
        return out


# --------------------------------------------------------------------------- #
#  ALTERNATIV KILDE: EIA WTI 1-4 (beholdt for modularitet/dokumentasjon)
# --------------------------------------------------------------------------- #
class EIATermStructureLoader(TermStructureLoader):
    """Alternativ: EIA Cushing WTI Future Contract 1-4 (ekte nearby-data).

    Krever en gratis EIA API-nøkkel (https://www.eia.gov/opendata/register.php).
    Serie-IDer: PET.RCLC1.D ... PET.RCLC4.D (daglig, flere tiår).
    Gir 4 ekte maturities; resten av 1-12 kan ev. interpoleres (Fallback B).
    """

    source_level = "Fallback A: EIA CL1-CL4 (ekte, 4 maturities)"

    # EIA v2-rute for NYMEX-futurespriser, og serie-IDer for WTI-kontrakt 1-4.
    _ENDPOINT = "https://api.eia.gov/v2/petroleum/pri/fut/data/"
    _SERIES = {"RCLC1": "M1", "RCLC2": "M2", "RCLC3": "M3", "RCLC4": "M4"}
    _PAGE = 5000  # EIA returnerer maks 5000 rader per kall -> vi paginerer.

    def __init__(self, api_key: str, start: str | None = None):
        self.api_key = api_key
        self.start = start

    def load(self) -> pd.DataFrame:
        import requests

        params = {
            "api_key": self.api_key,
            "frequency": "daily",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": self._PAGE,
        }
        for i, sid in enumerate(self._SERIES):
            params[f"facets[series][{i}]"] = sid
        if self.start:
            params["start"] = self.start

        rows: list[dict] = []
        offset = 0
        while True:
            params["offset"] = offset
            r = requests.get(self._ENDPOINT, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()["response"]
            batch = payload["data"]
            rows.extend(batch)
            total = int(payload.get("total", len(rows)))
            offset += self._PAGE
            if offset >= total or not batch:
                break

        raw = pd.DataFrame(rows)
        if raw.empty:
            raise RuntimeError("EIA returnerte ingen data – sjekk nøkkel/serie-IDer.")
        raw["period"] = pd.to_datetime(raw["period"])
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        # Pivot: én kolonne per maturity, indeksert på dato.
        wide = (
            raw.pivot_table(index="period", columns="series", values="value")
            .rename(columns=self._SERIES)
            .sort_index()
        )
        # Sørg for konsistent kolonnerekkefølge M1..M4.
        cols = [c for c in ("M1", "M2", "M3", "M4") if c in wide.columns]
        return wide[cols]
