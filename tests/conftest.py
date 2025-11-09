# conftest.py (continued)
import datetime as dt
from typing import Any

import pandas as pd
import polars as pl
import pytest


@pytest.fixture(params=["pandas", "polars"])
def backend(request):
    return request.param.strip()

def _int_yyyymmdd_to_date(v: int) -> dt.date:
    y, m, d = v // 10_000, (v // 100) % 100, v % 100
    return dt.date(y, m, d)

class _LocShim:
    """
    Emulates pandas .loc[row_key, col_key] -> scalar
    Only what tests need: (date, column) lookup.
    """
    def __init__(self, df: Any, date_col: str = "date"):
        self._df = df
        self._date_col = date_col

    def __getitem__(self, key):
        row_key, col_key = key  # expecting a 2-tuple
        if isinstance(self._df, pd.DataFrame):
            if isinstance(row_key, int):  # tests pass 20110125 style
                row_key = pd.Timestamp(_int_yyyymmdd_to_date(row_key))
            return self._df.loc[row_key, col_key]

        # polars path
        date_val = (
            _int_yyyymmdd_to_date(row_key) if isinstance(row_key, int) else row_key
        )
        # allow date or datetime column; cast to date for match
        df = self._df
        if "date" in df.columns and df.schema["date"] != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Date))

        out = (
            df.filter(pl.col(self._date_col) == pl.lit(date_val))
              .select(col_key)
        )
        # expect a single value
        return out.item()

class PandasCompat:
    """
    Wraps a polars-backed stock frame to present enough of the pandas API
    that the existing tests keep working (getitem for indicators + .loc lookup).
    Pass-through for anything else.
    """
    def __init__(self, inner, date_col: str = "date"):
        self._inner = inner
        self.loc = _LocShim(inner, date_col=date_col)

    # allow stock['qqe'] style to trigger your indicator logic on the inner
    def __getitem__(self, key):
        return self._inner.__getitem__(key)

    def __getattr__(self, name):
        # delegate to underlying object (DataFrame / StockFrame)
        return getattr(self._inner, name)

def wrap_for_tests(obj):
    """Return obj unchanged for pandas, or a compat wrapper for polars."""
    # Heuristic: pandas DataFrame or anything with real .loc already → no wrap
    if isinstance(obj, pd.DataFrame) or hasattr(obj, "iloc"):
        return obj
    # Polars path or your polars StockFrame: give it a pandas-like face
    return PandasCompat(obj, date_col="date")


class PolarsStockFrameStub:
    """
    Skeleton implementation used for TDD: forwards basic slicing/within to
    an underlying polars DataFrame but raises NotImplemented for indicators.
    """

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def __getattr__(self, name):
        return getattr(self._df, name)

    def copy(self):
        return PolarsStockFrameStub(self._df.clone())

    def within(self, start: int, end: int):
        if "date" not in self._df.columns:
            raise NotImplementedError("Polars stub requires 'date' column")
        mask = (pl.col("date") >= start) & (pl.col("date") <= end)
        return PolarsStockFrameStub(self._df.filter(mask))

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self._df.height
            length = stop - start if stop is not None else None
            return PolarsStockFrameStub(self._df.slice(start, length))
        raise NotImplementedError("Polars indicator retrieval not implemented yet")

    def __setitem__(self, key, value):
        raise NotImplementedError("Polars assignment not implemented yet")

    def __len__(self):
        return self._df.height
