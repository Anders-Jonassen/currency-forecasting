"""Modular data interface.

Design idea
-----------
The rest of the project talks ONLY to the abstractions below - never directly to
a concrete file or API. To switch to a Bloomberg/Refinitiv API later, write a new
class that subclasses `TermStructureLoader` and returns the same format. Nothing
else in the code has to change.

Active source in this project: local Excel files (Datastream/Refinitiv extracts)
with the real ICE Brent term structure 1-12 and NOK/USD - see the Excel* classes
below. The yfinance/EIA variants are kept as documented alternatives.

Common return format
---------------------
* FX:              pd.Series indexed by date (NOKUSD = USD per krone).
* Term structure:  pd.DataFrame indexed by date, one column per maturity
                   (M1, M2, ... = 1st, 2nd, ... nearby contract), values = price.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


# --------------------------------------------------------------------------- #
#  Abstract interfaces
# --------------------------------------------------------------------------- #
class FXLoader(ABC):
    """Loads the exchange rate as a pd.Series (NOKUSD)."""

    @abstractmethod
    def load(self) -> pd.Series:  # pragma: no cover - interface
        ...


class TermStructureLoader(ABC):
    """Loads the oil-futures term structure as a DataFrame (date x maturity).

    Implementations SHOULD set `self.source_level` to a short string describing
    what the data represents, so the README and figures can be labelled honestly
    (e.g. "Excel: ICE Brent TRc1-TRc12 (real, full 1-12 curve)").
    """

    source_level: str = "unspecified"

    @abstractmethod
    def load(self) -> pd.DataFrame:  # pragma: no cover - interface
        ...

    @property
    def maturities(self) -> list[str]:
        """Maturity column names, e.g. ['M1', 'M2', ...]."""
        return list(self.load().columns)


# --------------------------------------------------------------------------- #
#  ACTIVE SOURCE: local Excel files (Datastream/Refinitiv extracts)
# --------------------------------------------------------------------------- #
def _read_excel_with_date_index(path) -> pd.DataFrame:
    """Read a Datastream sheet where col 0 ('Name') holds dates and the rest values.

    Shared helper for both Excel loaders: sets the date column as the index and
    drops rows without a valid date.
    """
    df = pd.read_excel(path, header=0)
    date_col = df.columns[0]  # 'Name' in the Datastream extract = the dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"
    return df


class ExcelFXLoader(FXLoader):
    """NOK/USD from a local Excel file (Datastream 'US $ TO NORWEGIAN KRONE').

    Returns USD per krone (NOKUSD, ~0.11). The file has a single value column.
    """

    def __init__(self, path, name: str = "NOKUSD"):
        self.path = path
        self.name = name

    def load(self) -> pd.Series:
        df = _read_excel_with_date_index(self.path)
        s = df.iloc[:, 0].astype(float)  # the only value column
        s.name = self.name
        return s.dropna()


class ExcelTermStructureLoader(TermStructureLoader):
    """ICE Brent term structure 1-12 from a local Excel file (Datastream TRc1..TRc12).

    Columns are named 'ICE-BRENT CRUDE OIL TRc{n} - SETT. PRICE'. We extract the
    nearby number {n} and name the columns M1..M12 in ascending order, so M1 is
    the front-month ("first nearby") and M12 is the 12th nearby.
    """

    source_level = "Excel: ICE Brent TRc1-TRc12 (real, full 1-12 curve, monthly)"

    def __init__(self, path):
        self.path = path

    @staticmethod
    def _nearby_number(col: str) -> int | None:
        """Extract the nearby number from a column name, e.g. 'TRc7' -> 7."""
        import re

        m = re.search(r"TRc(\d+)", str(col))
        return int(m.group(1)) if m else None

    def load(self) -> pd.DataFrame:
        df = _read_excel_with_date_index(self.path)
        # Map each column to its nearby number and keep only the TRc columns.
        renaming = {}
        for col in df.columns:
            n = self._nearby_number(col)
            if n is not None:
                renaming[col] = f"M{n}"
        if not renaming:
            raise RuntimeError(
                f"No 'TRc<n>' columns found in {self.path}. Check the file format."
            )
        out = df[list(renaming)].rename(columns=renaming).astype(float)
        # Order the columns M1, M2, ... numerically (not lexicographically).
        out = out[sorted(out.columns, key=lambda c: int(c[1:]))]
        return out


# --------------------------------------------------------------------------- #
#  ALTERNATIVE SOURCE: yfinance FX (kept for modularity/documentation)
# --------------------------------------------------------------------------- #
class YFinanceFXLoader(FXLoader):
    """USDNOK from yfinance. Real, long daily history (since 2001)."""

    def __init__(self, ticker: str = "NOK=X", start: str | None = None):
        self.ticker = ticker
        self.start = start

    def load(self) -> pd.Series:
        import yfinance as yf

        df = yf.download(
            self.ticker, start=self.start, progress=False, auto_adjust=True
        )
        s = df["Close"]
        if isinstance(s, pd.DataFrame):  # yfinance may return Series or 1-col frame
            s = s.iloc[:, 0]
        s.name = "USDNOK"
        return s.dropna()


class YFinanceFrontMonthLoader(TermStructureLoader):
    """Front-month only (CL=F / BZ=F). Provides ONE maturity column (M1).

    Useful as "first nearby" and as a benchmark regardless of which source the
    full term structure comes from.
    """

    source_level = "Front-month (real, M1 only)"

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
        return close.dropna().to_frame(name="M1")


# --------------------------------------------------------------------------- #
#  ALTERNATIVE SOURCE: EIA WTI 1-4 (kept for modularity/documentation)
# --------------------------------------------------------------------------- #
class EIATermStructureLoader(TermStructureLoader):
    """Alternative: EIA Cushing WTI Future Contract 1-4 (real nearby data).

    Requires a free EIA API key (https://www.eia.gov/opendata/register.php).
    Series IDs: RCLC1 ... RCLC4 (daily, several decades). Gives 4 real maturities.
    """

    source_level = "EIA CL1-CL4 (real, 4 maturities)"

    _ENDPOINT = "https://api.eia.gov/v2/petroleum/pri/fut/data/"
    _SERIES = {"RCLC1": "M1", "RCLC2": "M2", "RCLC3": "M3", "RCLC4": "M4"}
    _PAGE = 5000  # EIA returns at most 5000 rows per call -> we paginate.

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
            raise RuntimeError("EIA returned no data - check key/series IDs.")
        raw["period"] = pd.to_datetime(raw["period"])
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        wide = (
            raw.pivot_table(index="period", columns="series", values="value")
            .rename(columns=self._SERIES)
            .sort_index()
        )
        cols = [c for c in ("M1", "M2", "M3", "M4") if c in wide.columns]
        return wide[cols]
