# coding=utf-8
# Copyright (c) 2016, Cedric Zhuang
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of disclaimer nor the names of its contributors may
#       be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations, unicode_literals

import functools
import io
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import polars as pl  # type: ignore
except Exception as exc:  # pragma: no cover - make failure explicit
    raise ModuleNotFoundError(
        "Polars support requires the optional 'polars' dependency."
    ) from exc

from stockstats import (
    Backend,
    PandasBackend,
    SeriesLike,
    StockDataFrame,
    StockStatsCore,
    _lower_col_name,
    _normalize_pandas_frame,
)

__all__ = [
    'PolarsBackend',
    'PolarsStockDataFrame',
    'StockDataFrame',
    'wrap_frame',
    'unwrap',
]


def _pl_from_pandas(df: pd.DataFrame) -> pl.DataFrame:
    namespace = io.StringIO()
    df.to_csv(namespace, index=False)
    namespace.seek(0)
    return pl.read_csv(namespace)


def _pl_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    namespace = io.StringIO()
    df.write_csv(namespace)
    namespace.seek(0)
    return pd.read_csv(namespace)


class PolarsBackend(Backend):
    """Backend implementation backed by a polars.DataFrame."""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def clone(self) -> 'PolarsBackend':
        return PolarsBackend(self._df.clone())

    def data(self) -> pl.DataFrame:
        return self._df

    def has_column(self, name: str) -> bool:
        return name in self._df.columns

    def get_column(self, name: str) -> pl.Series:
        return self._df.get_column(name)

    def set_column(self, name: str, series: SeriesLike) -> None:
        expr = self._coerce_series(name, series)
        self._df = self._df.with_columns(expr)

    def with_columns(self, **exprs: Any) -> 'PolarsBackend':
        cols = [self._coerce_series(key, value) for key, value in exprs.items()]
        return PolarsBackend(self._df.with_columns(*cols))

    def shift(self,
              series: pl.Series,
              periods: int,
              *,
              fill_value: Optional[Any] = None) -> pl.Series:
        shifted = series.shift(periods)
        if fill_value is not None:
            shifted = shifted.fill_null(fill_value)
        return shifted

    def diff(self, series: pl.Series, periods: int) -> pl.Series:
        return series.diff(n=periods)

    def pct_change(self, series: pl.Series) -> pl.Series:
        return series.pct_change()

    def sma(self, series: pl.Series, window: int) -> pl.Series:
        return series.rolling_mean(window_size=window, min_periods=1)

    def ema(self,
            series: pl.Series,
            window: int,
            *,
            adjust: bool = True,
            min_periods: int = 1) -> pl.Series:
        alpha = 2.0 / (window + 1)
        return series.ewm_mean(alpha=alpha,
                               adjust=adjust,
                               min_periods=min_periods,
                               ignore_nulls=True)

    def rolling_min(self, series: pl.Series, window: int) -> pl.Series:
        return series.rolling_min(window_size=window, min_periods=1)

    def rolling_max(self, series: pl.Series, window: int) -> pl.Series:
        return series.rolling_max(window_size=window, min_periods=1)

    def rolling_sum(self, series: pl.Series, window: int) -> pl.Series:
        return series.rolling_sum(window_size=window, min_periods=1)

    def rolling_std(self, series: pl.Series, window: int) -> pl.Series:
        return series.rolling_std(window_size=window, min_periods=1)

    def rolling_var(self, series: pl.Series, window: int) -> pl.Series:
        return series.rolling_var(window_size=window, min_periods=1)

    def concat_cols(self, *cols: pl.Series) -> pl.DataFrame:
        data = {f'col_{idx}': series for idx, series in enumerate(cols)}
        return pl.DataFrame(data)

    def to_numpy(self, series: pl.Series) -> np.ndarray:
        return series.to_numpy()

    def from_numpy(self, array: np.ndarray, like: Optional[pl.Series] = None) -> pl.Series:
        name = like.name if like is not None else None
        return pl.Series(name=name, values=array)

    @staticmethod
    def _coerce_series(name: str, value: SeriesLike) -> pl.Expr | pl.Series:
        if isinstance(value, pl.Expr):
            return value.alias(name)
        if isinstance(value, pl.Series):
            return value.rename(name)
        return pl.Series(name=name, values=value)


class PolarsStockDataFrame(StockStatsCore):
    """Facade that exposes indicator helpers atop a polars backend."""

    _INDEXER_NAMES = {'loc', 'iloc', 'iat', 'at'}

    def __init__(self,
                 backend: Backend,
                 *,
                 index_column: str = 'date',
                 mirror: Optional[StockDataFrame] = None):
        self._backend = backend
        self._index_column = index_column
        self._mirror = mirror
        super().__init__(backend)

    # -- core hooks for StockStatsCore ---------------------------------
    def _frame_getitem(self, item):
        mirror = self._ensure_mirror()
        result = mirror.__getitem__(item)
        self._sync_backend_from_mirror()
        return result

    def _frame_copy(self, deep: bool = True):
        mirror = self._ensure_mirror()
        copied = mirror.copy(deep=deep)
        backend = self._clone_backend(copied)
        return PolarsStockDataFrame(backend,
                                    index_column=self._index_column,
                                    mirror=copied)

    # -- public helpers ------------------------------------------------
    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def index_column(self) -> str:
        return self._index_column

    @classmethod
    def wrap(cls, df: Any, index_column: Optional[str] = None) -> 'PolarsStockDataFrame':
        index = index_column or 'date'
        backend, mirror = cls._coerce_backend(df, index)
        return cls(backend, index_column=index, mirror=mirror)

    # -- Mirror helpers -------------------------------------------------
    def _ensure_mirror(self) -> StockDataFrame:
        if self._mirror is not None:
            return self._mirror
        data = self._backend.data()
        if isinstance(data, pd.DataFrame):
            pdf = data.copy(deep=True)
        elif isinstance(data, pl.DataFrame):
            pdf = _pl_to_pandas(data)
        else:
            raise TypeError(f'Unsupported backend data type: {type(data)!r}')
        if self._index_column in pdf.columns:
            pdf = pdf.set_index(self._index_column)
        self._mirror = StockDataFrame(pdf)
        return self._mirror

    def _sync_backend_from_mirror(self) -> None:
        if self._mirror is None:
            return
        pdf = pd.DataFrame(self._mirror)
        if isinstance(self._backend, PandasBackend):
            self._backend = PandasBackend(pdf)
            return
        reset_df = self._reset_index_frame(pdf)
        self._backend = PolarsBackend(_pl_from_pandas(reset_df))

    def _reset_index_frame(self, pdf: pd.DataFrame) -> pd.DataFrame:
        reset_df = pdf.reset_index()
        index_name = pdf.index.name if pdf.index.name is not None else 'index'
        if self._index_column != index_name:
            reset_df = reset_df.rename(columns={index_name: self._index_column})
        return reset_df

    # -- DataFrame-like surface ----------------------------------------
    def __contains__(self, item: str) -> bool:
        return item in self._ensure_mirror().columns

    def __len__(self) -> int:
        try:
            return len(self._backend.data())
        except TypeError:
            return len(self._ensure_mirror())

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        mirror = self._ensure_mirror()
        mirror[key] = value
        self._sync_backend_from_mirror()

    def set_column(self, name: str, value: SeriesLike) -> None:
        self[name] = value

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._ensure_mirror())

    def to_polars(self) -> pl.DataFrame:
        data = self._backend.data()
        if isinstance(data, pl.DataFrame):
            return data.clone()
        mirror = self._ensure_mirror()
        frame = pd.DataFrame(mirror).reset_index().rename(columns={'index': self._index_column})
        return _pl_from_pandas(frame)

    def copy(self, deep: bool = True) -> 'PolarsStockDataFrame':
        mirror = self._ensure_mirror()
        copied = mirror.copy(deep=deep)
        backend = self._clone_backend(copied)
        return PolarsStockDataFrame(backend,
                                    index_column=self._index_column,
                                    mirror=copied)

    def __getattr__(self, item):
        mirror = self._ensure_mirror()
        attr = getattr(mirror, item)
        if item in self._INDEXER_NAMES:
            return _IndexerProxy(self, attr)
        if callable(attr):
            @functools.wraps(attr)
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                self._sync_backend_from_mirror()
                if isinstance(result, StockDataFrame):
                    if result is mirror:
                        return self
                    backend = PandasBackend(result)
                    polars_backend = self._backend if isinstance(self._backend, PolarsBackend) else None
                    if polars_backend is not None:
                        polars_df = _pl_from_pandas(self._reset_index_frame(pd.DataFrame(result)))
                        backend = PolarsBackend(polars_df)
                        return PolarsStockDataFrame(backend,
                                                    index_column=self._index_column,
                                                    mirror=result)
                    return result
                return result

            return wrapper
        return attr

    # -- internal helpers ----------------------------------------------
    def _clone_backend(self, mirror: StockDataFrame) -> Backend:
        if isinstance(self._backend, PandasBackend):
            return PandasBackend(pd.DataFrame(mirror))
        if isinstance(self._backend, PolarsBackend):
            polars_df = _pl_from_pandas(self._reset_index_frame(pd.DataFrame(mirror)))
            return PolarsBackend(polars_df)
        return self._backend.clone()

    @staticmethod
    def _coerce_backend(df: Any, index_column: str) -> Tuple[Backend, Optional[StockDataFrame]]:
        if isinstance(df, PolarsStockDataFrame):
            mirror = df._mirror if df._mirror is not None else df._ensure_mirror()
            return df.backend, mirror
        if isinstance(df, PandasBackend):
            pdf = df.data().copy(deep=True)
            mirror_df = StockDataFrame(pdf)
            return df, mirror_df
        if isinstance(df, PolarsBackend):
            pdf = _pl_to_pandas(df.data())
            normalized = _normalize_pandas_frame(pdf, index_column)
            mirror_df = StockDataFrame(normalized)
            return df, mirror_df
        if isinstance(df, StockDataFrame):
            return PandasBackend(df), df
        if isinstance(df, pd.DataFrame):
            normalized = _normalize_pandas_frame(df, index_column)
            mirror_df = StockDataFrame(normalized)
            return PandasBackend(mirror_df), mirror_df
        if isinstance(df, pl.DataFrame):
            lowered = df.rename({col: _lower_col_name(col) for col in df.columns})
            if index_column in lowered.columns:
                lowered = lowered.sort(index_column)
            backend = PolarsBackend(lowered)
            mirror_df = StockDataFrame(_normalize_pandas_frame(_pl_to_pandas(lowered), index_column))
            return backend, mirror_df
        raise TypeError(f'Cannot wrap object of type {type(df)!r}')


class _IndexerProxy:
    """Proxy that makes pandas indexers sync back to the StockFrame backend."""

    def __init__(self, owner: PolarsStockDataFrame, target: Any):
        self._owner = owner
        self._target = target

    def _sync(self, result):
        self._owner._sync_backend_from_mirror()
        return result

    def __getitem__(self, item):
        return self._sync(self._target[item])

    def __setitem__(self, key, value):
        self._target[key] = value
        self._sync(None)

    def __call__(self, *args, **kwargs):
        return self._sync(self._target(*args, **kwargs))


def wrap_frame(df, index_column=None) -> PolarsStockDataFrame:
    return PolarsStockDataFrame.wrap(df, index_column=index_column)


def unwrap(frame: PolarsStockDataFrame) -> pd.DataFrame:
    backend_data = frame.backend.data()
    if isinstance(backend_data, pd.DataFrame):
        df = backend_data.copy(deep=True)
    elif isinstance(backend_data, pl.DataFrame):
        df = _pl_to_pandas(backend_data)
    else:
        raise TypeError(f'Unsupported backend type: {type(backend_data)!r}')
    index_col = getattr(frame, 'index_column', None)
    if index_col and index_col in df.columns:
        df = df.set_index(index_col)
    return df
