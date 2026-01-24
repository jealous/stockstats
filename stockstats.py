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

from __future__ import unicode_literals

import functools
import itertools
import re
from collections import deque
from typing import Optional, Callable, Union

import numpy as np
import pandas as pd

__author__ = "Cedric Zhuang"


class StockStatsError(Exception):
    pass


_dft_windows = {
    # sort alphabetically
    "ao": (5, 34),
    "aroon": 25,
    "atr": 14,
    "boll": 20,
    "cci": 14,
    "change": 1,
    "chop": 14,
    "cmo": 14,
    "coppock": (10, 11, 14),
    "cr": 26,
    "cti": 12,
    "dma": (10, 50),
    "eri": 13,
    "eribear": 13,
    "eribull": 13,
    "ichimoku": (9, 26, 52),
    "inertia": (20, 14),
    "ftr": 9,
    "kama": (10, 5, 34),  # window, fast, slow
    "kdjd": 9,
    "kdjj": 9,
    "kdjk": 9,
    "ker": 10,
    "macd": (12, 26, 9),  # short, long, signal
    "mfi": 14,
    "ndi": 14,
    "pdi": 14,
    "pgo": 14,
    "ppo": (12, 26, 9),  # short, long, signal
    "pvo": (12, 26, 9),  # short, long, signal
    "psl": 12,
    "qqe": (14, 5),  # rsi, rsi ema
    "rsi": 14,
    "rsv": 9,
    "rvgi": 14,
    "stochrsi": 14,
    "supertrend": 14,
    "tema": 5,
    "trix": 12,
    "wr": 14,
    "wt": (10, 21),
    "vr": 26,
    "vwma": 14,
    "num": 0,
}


def set_dft_window(name: str, windows: Union[int, tuple[int, ...]]):
    ret = _dft_windows.get(name)
    _dft_windows[name] = windows
    return ret


_dft_column = {
    # sort alphabetically
    "cti": "close",
    "dma": "close",
    "kama": "close",
    "ker": "close",
    "psl": "close",
    "tema": "close",
    "trix": "close",
}


def dft_windows(name: str) -> Optional[str]:
    if name not in _dft_windows:
        return None
    dft = _dft_windows[name]
    if isinstance(dft, int):
        return str(dft)
    return ",".join(map(str, dft))


def dft_column(name: str) -> Optional[str]:
    if name not in _dft_column:
        return None
    return _dft_column[name]


class _Meta:
    def __init__(self, name, *, column=None, windows=None):
        self._name = name
        self._column = column
        self._windows = windows
        self._dft_column = dft_column(name)
        self._dft_windows = dft_windows(name)

    @staticmethod
    def _process_segment(windows):
        if "~" in windows:
            start, end = windows.split("~")
            shifts = range(int(start), int(end) + 1)
        else:
            shifts = [int(windows)]
        return shifts

    @property
    def ints(self) -> list[int]:
        items = map(self._process_segment, self.windows.split(","))
        return list(itertools.chain(*items))

    @property
    def as_int(self) -> int:
        numbers = self.ints
        if len(numbers) != 1:
            raise StockStatsError("only accept 1 number")
        return numbers[0]

    def _get_int(self, i):
        numbers = self.ints
        if len(numbers) < i + 1:
            # try the defaults
            dft_numbers = _dft_windows[self._name]
            if len(dft_numbers) > i:
                return dft_numbers[i]
            raise StockStatsError(f"not enough ints, need {i + 1}")
        return self.ints[i]

    @property
    def int0(self) -> int:
        return self._get_int(0)

    @property
    def int1(self) -> int:
        return self._get_int(1)

    @property
    def int2(self) -> int:
        return self._get_int(2)

    @property
    def positive_int(self) -> int:
        ret = self.as_int
        if ret <= 0:
            raise StockStatsError("window must be greater than 0")
        return ret

    @property
    def windows(self):
        if self._windows is None:
            return self._dft_windows
        return self._windows

    @property
    def column(self):
        if self._column is None:
            return self._dft_column
        return self._column

    @property
    def name(self):
        if self._windows is None and self._column is None:
            return self._name
        if self._column is None:
            return f"{self._name}_{self._windows}"
        return f"{self.column}_{self.windows}_{self._name}"

    def set_name(self, name: str):
        self._name = name
        return self

    def name_ex(self, ex):
        ret = f"{self._name}{ex}"
        if self._windows is None:
            return ret
        return f"{ret}_{self.windows}"


def _call_handler(handler: Callable):
    meta = _Meta(handler.__name__[5:])
    return handler(meta)


def wrap(df, index_column=None):
    """wraps a pandas DataFrame to StockDataFrame

    :param df: pandas DataFrame
    :param index_column: the name of the index column, default to ``date``
    :return: an object of StockDataFrame
    """
    return StockDataFrame.retype(df, index_column)


def unwrap(sdf):
    """convert a StockDataFrame back to a pandas DataFrame"""
    return pd.DataFrame(sdf)


class StockDataFrame(pd.DataFrame):
    # Start of options.
    KDJ_PARAM = (2.0 / 3.0, 1.0 / 3.0)

    BOLL_STD_TIMES = 2

    DX_SMMA = 14
    ADX_EMA = 6
    ADXR_EMA = 6

    CR_MA = (5, 10, 20)

    SUPERTREND_MUL = 3

    # End of options

    @staticmethod
    def _df_to_series(column):
        # if column is data frame, retrieve the first column
        if isinstance(column, pd.DataFrame):
            num_col = column.shape[1]
            if num_col != 1:
                raise ValueError(f"Expected a single column, got {num_col}")
            column = column.iloc[:, 0]
        return column

    @property
    def high(self) -> pd.Series:
        return self._df_to_series(self["high"])

    @property
    def low(self) -> pd.Series:
        return self._df_to_series(self["low"])

    @property
    def close(self) -> pd.Series:
        return self._df_to_series(self["close"])

    @property
    def open(self) -> pd.Series:
        return self._df_to_series(self["open"])

    @property
    def volume(self) -> pd.Series:
        return self._df_to_series(self["volume"])

    @property
    def amount(self) -> pd.Series:
        return self._df_to_series(self["amount"])

    def _get_change(self, meta: _Meta):
        """Get the percentage change column

        It's an alias for ROC

        :return: result series
        """
        self[meta.name] = self.roc(self.close, meta.as_int)

    def _get_p(self, meta: _Meta):
        """get the permutation of specified range

        example:
        index    x   x_-2,-1_p
        0        1         NaN
        1       -1         NaN
        2        3           2  (0.x > 0, and assigned to weight 2)
        3        5           1  (2.x > 0, and assigned to weight 1)
        4        1           3
        """
        # initialize the column if not
        _ = self.get(meta.column)
        shifts = meta.ints[::-1]
        indices: Optional[pd.Series] = None
        count = 0
        for shift in shifts:
            shifted = self.shift(-shift)
            index = (shifted[meta.column] > 0) * (2 ** count)
            if indices is None:
                indices = index
            else:
                indices += index
            count += 1
        if indices is not None:
            cp = indices.copy()
            self.set_nan(cp, shifts)
            self[meta.name] = cp

    @classmethod
    def to_ints(cls, shifts):
        items = map(cls._process_shifts_segment, shifts.split(","))
        return sorted(list(set(itertools.chain(*items))))

    @classmethod
    def to_int(cls, shifts):
        numbers = cls.to_ints(shifts)
        if len(numbers) != 1:
            raise IndexError("only accept 1 number.")
        return numbers[0]

    @staticmethod
    def _process_shifts_segment(shift_segment):
        if "~" in shift_segment:
            start, end = shift_segment.split("~")
            shifts = range(int(start), int(end) + 1)
        else:
            shifts = [int(shift_segment)]
        return shifts

    @classmethod
    def set_nan(cls, pd_obj, shift):
        try:
            iter(shift)
            max_shift = max(shift)
            min_shift = min(shift)
            cls._set_nan_of_single_shift(pd_obj, max_shift)
            cls._set_nan_of_single_shift(pd_obj, min_shift)
        except TypeError:
            # shift is not iterable
            cls._set_nan_of_single_shift(pd_obj, shift)

    @staticmethod
    def _set_nan_of_single_shift(pd_obj, shift):
        val = np.nan
        if shift > 0:
            pd_obj.iloc[-shift:] = val
        elif shift < 0:
            pd_obj.iloc[:-shift] = val

    def _get_r(self, meta: _Meta):
        """Get rate of change of column

        Note this function is different to the roc function.
        negative values meaning data in the past,
        positive values meaning data in the future.
        """
        shift = -meta.as_int
        self[meta.name] = self.roc(self[meta.column], shift)

    @staticmethod
    def _shift_arr(arr: np.ndarray, window: int) -> np.ndarray:
        out = np.empty_like(arr)
        if window < 0:
            k = -window
            out[:k] = arr[0]
            out[k:] = arr[:-k]
        else:
            k = window
            out[:-k] = arr[k:]
            out[-k:] = arr[-1]
        return out

    @classmethod
    def s_shift(cls, series: pd.Series, window: int):
        """Shift the series

        When window is negative, shift the past period to current.
        Fill the gap with the first data available.

        When window is positive, shift the future period to current.
        Fill the gap with last data available.

        :param series: the series to shift
        :param window: number of periods to shift
        :return: the shifted series with filled gap
        """
        if series.empty or window == 0:
            return series.copy()

        out = cls._shift_arr(series.values, window)
        return pd.Series(out, index=series.index, name=series.name)

    def _get_s(self, meta: _Meta):
        """Get the column shifted by periods

        Note this method is different to the shift method of pandas.
        negative values meaning data in the past,
        positive values meaning data in the future.
        """
        self[meta.name] = self.s_shift(self[meta.column], meta.as_int)

    def _get_log_ret(self, _: _Meta):
        close = self.close
        self["log-ret"] = np.log(close / self.s_shift(close, -1))

    @staticmethod
    def _rolling_sum(arr, window):
        """Compute rolling sum with min_periods=1 using numpy."""
        n = len(arr)
        out = np.zeros(n, dtype=float)
        # Use cumsum for efficient rolling sum
        cumsum = np.cumsum(arr)
        # For positions >= window, subtract cumsum[i-window] from cumsum[i]
        out[window:] = cumsum[window:] - cumsum[:-window]
        # For positions < window, just use cumsum (partial window)
        out[:window] = cumsum[:window]
        return out

    def _get_c(self, meta: _Meta) -> pd.Series:
        """get the count of column in range (shifts)

        example: change_20_c
        :return: result series
        """
        series = self[meta.column]
        window = meta.as_int
        arr = series.values.astype(bool).astype(float)
        counts = pd.Series(self._rolling_sum(arr, window), index=series.index)
        self[meta.name] = counts
        return counts

    def _get_fc(self, meta: _Meta) -> pd.Series:
        """get the count of column in range of future (shifts)

        example: change_20_fc
        :return: result series
        """
        series = self[meta.column]
        window = meta.as_int
        arr = series.values.astype(bool).astype(float)
        # Reverse, count, then reverse back
        arr_rev = arr[::-1]
        out_rev = self._rolling_sum(arr_rev, window)
        counts = pd.Series(out_rev[::-1].copy(), index=series.index)
        self[meta.name] = counts
        return counts

    def _shifted_columns(self, column: pd.Series,
                         shifts: list[int]) -> pd.DataFrame:
        # initialize the column if not
        col = self.get(column)
        res = pd.DataFrame()
        for i in shifts:
            res[int(i)] = self.s_shift(col, i).values
        return res

    def _get_max(self, meta: _Meta):
        column = meta.column
        shifts = meta.ints
        cols = self._shifted_columns(column, shifts)
        self[meta.name] = cols.max(axis=1).values

    def _get_min(self, meta: _Meta):
        column = meta.column
        shifts = meta.ints
        cols = self._shifted_columns(column, shifts)
        self[meta.name] = cols.min(axis=1).values

    @staticmethod
    def _divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.zeros_like(a, dtype=float)
        if out.size == 0:
            return out
        np.divide(a, b, out=out, where=b != 0)

        # handle nan / inf in one pass
        np.nan_to_num(out, copy=False)
        return out

    def _rsv(self, window):
        low_min = self.mov_min(self.low, window).values
        high_max = self.mov_max(self.high, window).values
        close = self.close.values
        cv = self._divide(close - low_min, high_max - low_min) * 100
        return self.to_series(cv)

    def _get_rsv(self, meta: _Meta):
        """Calculate the RSV (Raw Stochastic Value) within N periods

        This value is essential for calculating KDJs
        Current day is included in N

        """
        self[meta.name] = self._rsv(meta.as_int)

    @staticmethod
    def _np_diff(arr: np.ndarray) -> np.ndarray:
        diff = np.zeros_like(arr)
        diff[1:] = np.diff(arr)
        return diff

    def _rsi(self, window) -> pd.Series:
        close = self.close.values
        diff = self._np_diff(close)

        up = np.where(diff > 0, diff, 0.0)
        down = np.where(diff < 0, -diff, 0.0)
        up_sma = self.smma(pd.Series(up), window).values
        down_sma = self.smma(pd.Series(down), window).values

        total_chg = up_sma + down_sma
        with np.errstate(divide="ignore", invalid="ignore"):
            rsi = np.where(total_chg != 0, 100 * (up_sma / total_chg), 50.0)

        rsi[0] = 50.0
        return self.to_series(rsi)

    def _get_rsi(self, meta: _Meta):
        """Calculate the RSI (Relative Strength Index) within N periods

        calculated based on the formula at:
        https://en.wikipedia.org/wiki/Relative_strength_index
        """
        self[meta.name] = self._rsi(meta.as_int)

    def _get_stochrsi(self, meta: _Meta):
        """Calculate the Stochastic RSI

        calculated based on the formula at:
        https://www.investopedia.com/terms/s/stochrsi.asp
        """
        window = meta.as_int
        rsi = self._rsi(window)
        rsi_min = self.mov_min(rsi, window)
        rsi_max = self.mov_max(rsi, window)

        rsi_range = rsi_max - rsi_min
        cv = np.where(rsi_range != 0, (rsi - rsi_min) / rsi_range, 0.0)
        self[meta.name] = cv * 100

    def _wt1(self, n1: int, n2: int) -> pd.Series:
        """wave trand 1

        n1: period of EMA on typical price
        n2: period of EMA
        """
        tp = self._tp()
        esa = self.ema(tp, n1)
        d = self.ema((tp - esa).abs(), n1)
        ci = (tp - esa) / (0.015 * d)
        ret = self.ema(ci, n2)
        ret.iloc[0] = 0.0
        return ret

    def _get_wt1(self, meta: _Meta):
        self[meta.name] = self._wt1(meta.int0, meta.int1)

    def _get_wt2(self, meta: _Meta):
        wt1 = self._wt1(meta.int0, meta.int1)
        self[meta.name] = self.sma(wt1, 4)

    def _get_wt(self, meta: _Meta):
        """Calculate LazyBear's Wavetrend

        Check the algorithm described below:
        https://medium.com/@samuel.mcculloch/lets-take-a-look-at-wavetrend-with-crosses-lazybear-s-indicator-2ece1737f72f
        """
        tci = self._wt1(meta.int0, meta.int1)
        self[meta.name_ex("1")] = tci
        self[meta.name_ex("2")] = self.sma(tci, 4)

    @staticmethod
    def smma(series, window):
        return series.ewm(
            ignore_na=False, alpha=1.0 / window, min_periods=0, adjust=True
        ).mean()

    def _get_smma(self, meta: _Meta):
        """get smoothed moving average"""
        self[meta.name] = self.smma(self[meta.column], meta.as_int)

    def _get_trix(self, meta: _Meta):
        """Triple Exponential Average

        https://www.investopedia.com/articles/technical/02/092402.asp
        """
        window = meta.as_int
        single = self.ema(self[meta.column], window)
        double = self.ema(single, window)
        triple = self.ema(double, window)

        triple_values = triple.values
        trix = np.zeros_like(triple_values)
        with np.errstate(divide="ignore", invalid="ignore"):
            trix[1:] = (triple_values[1:] / triple_values[:-1] - 1) * 100
        self[meta.name] = self.to_series(trix)

    def _get_tema(self, meta: _Meta):
        """Another implementation for triple ema

        Check the algorithm described below:
        https://www.forextraders.com/forex-education/forex-technical-analysis/triple-exponential-moving-average-the-tema-indicator/
        """
        window = meta.as_int
        single = self.ema(self[meta.column], window)
        double = self.ema(single, window)
        triple = self.ema(double, window)
        self[meta.name] = 3.0 * single - 3.0 * double + triple

    def _get_wr(self, meta: _Meta):
        """Williams Overbought/Oversold Index

        Definition: https://www.investopedia.com/terms/w/williamsr.asp
        WMS=[(Hn—Ct)/(Hn—Ln)] × -100
        Ct - the close price
        Hn - N periods high
        Ln - N periods low
        """
        window = meta.as_int
        ln = self.mov_min(self.low, window)
        hn = self.mov_max(self.high, window)
        hn_ln = hn - ln
        wr = np.where(hn_ln != 0, (hn - self.close) / hn_ln, 0.0)
        self[meta.name] = wr * -100

    def _get_cci(self, meta: _Meta):
        """Commodity Channel Index

        CCI = (Typical Price  -  20-period SMA of TP) / (.015 x Mean Deviation)
        * when amount is not available:
          Typical Price (TP) = (High + Low + Close)/3
        * when amount is available:
          Typical Price (TP) = Amount / Volume
        TP is also implemented as 'middle'.
        """
        window = meta.as_int
        tp = self._tp()
        tp_sma = self.sma(tp, window)
        mad = self._mad(tp, window)
        divisor = 0.015 * mad
        self[meta.name] = np.where(divisor != 0, (tp - tp_sma) / divisor, 0.0)

    def _tr(self):
        prev_close = self._shift_arr(self.close.values, -1)
        high = self.high.values
        low = self.low.values
        c1 = high - low
        c2 = np.abs(high - prev_close)
        c3 = np.abs(low - prev_close)
        tr = np.maximum(c1, np.maximum(c2, c3))
        np.nan_to_num(tr, copy=False)
        return self.to_series(tr)

    def _get_tr(self, meta: _Meta):
        """True Range of the trading

         TR is a measure of volatility of a High-Low-Close series

        tr = max[(high - low), abs(high - close_prev), abs(low - close_prev)]

        :return: None
        """
        self[meta.name] = self._tr()

    def _get_supertrend(self, meta: _Meta):
        """Supertrend

        Supertrend indicator shows trend direction.
        It provides buy or sell indicators.
        https://medium.com/codex/step-by-step-implementation-of-the-supertrend-indicator-in-python-656aa678c111
        """
        # 1. Vectorized Calculation of Basic Bands
        high = self["high"].values
        low = self["low"].values
        close = self["close"].values
        multiplier = self.SUPERTREND_MUL
        window = meta.as_int

        # Calculate ATR (Assuming get_atr is already optimized/vectorized)
        atr = self._atr(window).values

        hl2 = (high + low) / 2
        basic_ub = hl2 + (multiplier * atr)
        basic_lb = hl2 - (multiplier * atr)

        # 2. Initialize recursive arrays
        final_ub = np.zeros_like(basic_ub)
        final_lb = np.zeros_like(basic_lb)
        supertrend = np.zeros_like(basic_ub)
        direction = np.zeros_like(basic_ub)  # 1 for Up, -1 for Down

        # 3. The Recursive Loop (Optimized by using raw NumPy arrays)
        # Faster than .apply() because it avoids Pandas overhead per row.
        for i in range(1, len(close)):
            # Final Upper Band Logic
            if basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i - 1]

            # Final Lower Band Logic
            if basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i - 1]

            # Determine Trend Direction
            if close[i] > final_ub[i]:
                direction[i] = 1
            elif close[i] < final_lb[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

            # Set Supertrend value
            supertrend[i] = final_lb[i] if direction[i] == 1 else final_ub[i]

        self["supertrend"] = supertrend
        self[f"{meta.name}_ub"] = final_ub
        self[f"{meta.name}_lb"] = final_lb

    @staticmethod
    def _rolling_arg_index(arr, window, mode="max"):
        n = len(arr)
        out = np.full(n, np.nan, dtype=float)
        dq = deque()

        for i in range(n):
            # Remove old indices
            while dq and dq[0] <= i - window:
                dq.popleft()

            # Remove smaller/larger elements depending on mode
            while dq and (
                (mode == "max" and arr[dq[-1]] <= arr[i])
                or (mode == "min" and arr[dq[-1]] >= arr[i])
            ):
                dq.pop()

            dq.append(i)

            if i >= window - 1:
                out[i] = i - dq[0]  # periods since high/low

        return out

    def _get_aroon(self, meta: _Meta):
        """Aroon Oscillator

        The Aroon Oscillator measures the strength of a trend and
        the likelihood that it will continue.

        The default window is 25.

        * Aroon Oscillator = Aroon Up - Aroon Down
        * Aroon Up = 100 * (n - periods since n-period high) / n
        * Aroon Down = 100 * (n - periods since n-period low) / n
        * n = window size
        """

        window = meta.as_int
        high_arr = self.high.values
        low_arr = self.low.values

        # periods since last high / low in window
        high_idx = self._rolling_arg_index(high_arr, window, mode="max")
        low_idx = self._rolling_arg_index(low_arr, window, mode="min")

        # Compute Aroon Up / Down
        aroon_up = (window - high_idx) / window * 100
        aroon_down = (window - low_idx) / window * 100

        self[meta.name] = pd.Series(aroon_up - aroon_down, index=self.index)

    def _get_z(self, meta: _Meta):
        """Z score

        Z-score is a statistical measurement that describes a value's
        relationship to the mean of a group of values.

        The statistical formula for a value's z-score is calculated using
        the following formula:

        z = ( x - μ ) / σ

        Where:

        * z = Z-score
        * x = the value being evaluated
        * μ = the mean
        * σ = the standard deviation
        """
        window = meta.as_int
        col = self[meta.column]
        mean = self.sma(col, window)
        std = self.mov_std(col, window)
        value = (col - mean) / std
        if len(value) > 1:
            value.iloc[0] = value.iloc[1]
        self[meta.name] = value

    def _atr(self, window):
        tr = self._tr()
        return self.smma(tr, window)

    def _get_atr(self, meta: _Meta):
        """Average True Range

        The average true range is an N-day smoothed moving average (SMMA) of
        the true range values.  Default to 14 periods.
        https://en.wikipedia.org/wiki/Average_true_range
        """
        window = meta.as_int
        self[meta.name] = self._atr(window)

    def _get_dma(self, meta: _Meta):
        """Difference of Moving Average

        default to 10 and 50.

        :return: None
        """
        fast = meta.int0
        slow = meta.int1
        col = self[meta.column]
        diff = self.sma(col, fast) - self.sma(col, slow)
        self[meta.name] = diff

    def _get_dmi(self, _: _Meta):
        """get the default setting for DMI

        including:
        +DI: 14 periods SMMA of +DM,
        -DI: 14 periods SMMA of -DM,
        DX: based on +DI and -DI
        ADX: 6 periods SMMA of DX

        :return:
        """
        self["dx"] = self._dx(self.DX_SMMA)
        self["adx"] = self.ema(self["dx"], self.ADX_EMA)
        self["adxr"] = self.ema(self["adx"], self.ADXR_EMA)

    def _get_pdm_ndm(self, window):
        hd = self._np_diff(self.high.values)
        ld = -self._np_diff(self.low.values)

        p = np.where((hd > 0) & (hd > ld), hd, 0.0)
        n = np.where((ld > 0) & (ld > hd), ld, 0.0)

        if window > 1:
            p = self.smma(self.to_series(p), window)
            n = self.smma(self.to_series(n), window)
        else:
            p = self.to_series(p)
            n = self.to_series(n)
        return p, n

    def _pdm(self, window):
        ret, _ = self._get_pdm_ndm(window)
        return ret

    def _ndm(self, window):
        _, ret = self._get_pdm_ndm(window)
        return ret

    def _get_pdm(self, meta: _Meta):
        """+DM, positive directional moving

        If window is not 1, calculate the SMMA of +DM
        """
        self[meta.name] = self._pdm(meta.as_int)

    def _get_ndm(self, meta: _Meta):
        """-DM, negative directional moving accumulation

        If window is not 1, return the SMA of -DM.
        """
        self[meta.name] = self._ndm(meta.as_int)

    def _get_vr(self, meta: _Meta):
        """VR - Volume Variation Index"""
        window = meta.as_int
        change = self["change"].values
        volume = self.volume.values

        gt_zero = np.where(change > 0, volume, 0.0)
        lt_zero = np.where(change < 0, volume, 0.0)
        eq_zero = np.where(change == 0, volume, 0.0)

        avs = self._rolling_sum(gt_zero, window)
        bvs = self._rolling_sum(lt_zero, window)
        cvs = self._rolling_sum(eq_zero, window)

        half_cvs = cvs * 0.5
        divisor = bvs + half_cvs
        vr = np.divide(avs + half_cvs, divisor,
                       out=np.zeros_like(divisor), where=divisor != 0) * 100
        self[meta.name] = pd.Series(vr, index=self.index)

    def _get_pdi_ndi(self, window):
        pdm, ndm = self._get_pdm_ndm(window)
        atr = self._atr(window)
        pdi = pdm / atr * 100
        ndi = ndm / atr * 100
        return pdi, ndi

    def _get_pdi(self, meta: _Meta):
        """+DI, positive directional moving index"""
        pdi, _ = self._get_pdi_ndi(meta.as_int)
        self[meta.name] = pdi
        return pdi

    def _get_ndi(self, meta: _Meta):
        """-DI, negative directional moving index"""
        _, ndi = self._get_pdi_ndi(meta.as_int)
        self[meta.name] = ndi
        return ndi

    def _dx(self, window):
        pdi, mdi = self._get_pdi_ndi(window)
        divisor = pdi + mdi
        return np.where(divisor != 0, abs(pdi - mdi) / divisor, 0.0) * 100

    def _get_dx(self, meta: _Meta):
        self[meta.name] = self._dx(meta.as_int)

    def _get_cr(self, meta: _Meta):
        """Energy Index (Intermediate Willingness Index)

        https://support.futunn.com/en/topic167/?lang=en-us
        Use the relationship between the highest price, the lowest price and
        yesterday's middle price to reflect the market's willingness to buy
        and sell.
        """
        window = meta.as_int
        middle = self._tp()
        last_middle = self.s_shift(middle, -1)
        ym = self.s_shift(middle, -1)
        high = self.high
        low = self.low
        p1_m = pd.concat((last_middle, high), axis=1).min(axis=1)
        p2_m = pd.concat((last_middle, low), axis=1).min(axis=1)
        p1 = self.mov_sum(high - p1_m, window)
        p2 = self.mov_sum(ym - p2_m, window)

        name = meta.name
        self[name] = cr = p1 / p2 * 100
        self[f"{name}-ma1"] = self._shifted_cr_sma(cr, self.CR_MA[0])
        self[f"{name}-ma2"] = self._shifted_cr_sma(cr, self.CR_MA[1])
        self[f"{name}-ma3"] = self._shifted_cr_sma(cr, self.CR_MA[2])

    def _shifted_cr_sma(self, cr, window):
        cr_sma = self.sma(cr, window)
        return self.s_shift(cr_sma, -int(window / 2.5 + 1))

    def _tp(self) -> pd.Series:
        if "amount" in self:
            tp = self.amount.values / self.volume.values
        else:
            total = self.close.values + self.high.values + self.low.values
            tp = total / 3.0
        return self.to_series(tp)

    def _get_tp(self, meta: _Meta):
        self[meta.name] = self._tp()

    def _get_middle(self, meta: _Meta):
        self[meta.name] = self._tp()

    def _calc_kd(self, column):
        param0, param1 = self.KDJ_PARAM
        k = 50.0
        # noinspection PyTypeChecker
        for i in param1 * column:
            k = param0 * k + i
            yield k

    def _get_kdjk(self, meta: _Meta):
        """Get the K of KDJ

        K ＝ 2/3 × (prev. K) +1/3 × (curr. RSV)
        2/3 and 1/3 are the smooth parameters.
        """
        window = meta.as_int
        rsv = self._rsv(window)
        self[meta.name] = list(self._calc_kd(rsv))

    def _get_kdjd(self, meta: _Meta):
        """Get the D of KDJ

        D = 2/3 × (prev. D) +1/3 × (curr. K)
        2/3 and 1/3 are the smooth parameters.
        """
        k_column = meta.name.replace("kdjd", "kdjk")
        self[meta.name] = list(self._calc_kd(self.get(k_column)))

    def _get_kdjj(self, meta: _Meta):
        """Get the J of KDJ

        J = 3K-2D
        """
        k_column = meta.name.replace("kdjj", "kdjk")
        d_column = meta.name.replace("kdjj", "kdjd")
        self[meta.name] = 3 * self[k_column] - 2 * self[d_column]

    @staticmethod
    def _delta(series, window):
        return series.diff(-window).fillna(0.0)

    def _get_d(self, meta: _Meta):
        self[meta.name] = self._delta(self[meta.column], meta.as_int)

    @classmethod
    def mov_min(cls, series, size) -> pd.Series:
        return cls._rolling(series, size).min()

    @classmethod
    def mov_max(cls, series, size) -> pd.Series:
        return cls._rolling(series, size).max()

    @classmethod
    def mov_sum(cls, series, size) -> pd.Series:
        return cls._rolling(series, size).sum()

    @classmethod
    def sma(cls, series, size) -> pd.Series:
        return cls._rolling(series, size).mean()

    @staticmethod
    def roc(series, size):
        x = series.values.astype(np.float64, copy=False)
        n = x.shape[0]

        out = np.zeros(n, dtype=np.float64)

        if size == 0:
            return pd.Series(out, index=series.index)

        if size > 0:
            # x[t] vs x[t-size]
            out[size:] = (x[size:] - x[:-size]) / x[:-size]
        else:
            k = -size
            # x[t] vs x[t+k]
            out[:-k] = (x[:-k] - x[k:]) / x[k:]

        return pd.Series(out * 100.0, index=series.index)

    @classmethod
    def _mad(cls, series, window):
        """Mean Absolute Deviation

        :param series: Series
        :param window: number of periods
        :return: Series
        """

        arr = series.values
        n = len(arr)
        if window > n:
            nan_arr = np.full(n, np.nan)
            return pd.Series(nan_arr, index=series.index, name=series.name)

        sw = np.lib.stride_tricks.sliding_window_view(arr, window)
        means = sw.mean(axis=1)
        mad_vals = np.mean(np.abs(sw - means[:, None]), axis=1)

        out = np.zeros(n, dtype=float)
        out[window - 1:] = mad_vals
        return pd.Series(out, index=series.index, name=series.name)

    def _get_mad(self, meta: _Meta):
        """get mean absolute deviation"""
        window = meta.as_int
        self[meta.name] = self._mad(self[meta.column], window)

    def _get_sma(self, meta: _Meta):
        """get simple moving average"""
        window = meta.as_int
        self[meta.name] = self.sma(self[meta.column], window)

    def _get_lrma(self, meta: _Meta):
        """get linear regression moving average"""
        window = meta.as_int
        self[meta.name] = self.linear_reg(self[meta.column], window)

    def _get_roc(self, meta: _Meta):
        """get Rate of Change (ROC) of a column

        The Price Rate of Change (ROC) is a momentum-based technical indicator
        that measures the percentage change in price between the current price
        and the price a certain number of periods ago.

        https://www.investopedia.com/terms/p/pricerateofchange.asp

        Formular:

        ROC = (PriceP - PricePn) / PricePn * 100

        Where:
        * PriceP: the price of the current period
        * PricePn: the price of the n periods ago
        """
        self[meta.name] = self.roc(self[meta.column], meta.as_int)

    @staticmethod
    def ema(series, window, *, adjust=True, min_periods=1):
        return series.ewm(
            ignore_na=False, span=window,
            min_periods=min_periods, adjust=adjust).mean()

    @staticmethod
    def _rolling(series: pd.Series, window: int):
        return series.rolling(window, min_periods=1, center=False)

    @classmethod
    def linear_wma(cls, series, window):
        """
        Linear Weighted Moving Average (WMA) using vectorized NumPy.
        Returns 0 for first window-1 positions.
        """
        arr = series.values
        n = len(arr)

        if window > n:
            return pd.Series(np.zeros(n), index=series.index, name=series.name)

        # linear weights
        total_weight = 0.5 * window * (window + 1)
        weights = np.arange(1, window + 1) / total_weight  # shape (window,)

        # create sliding windows (shape: n - window + 1, window)
        sw = np.lib.stride_tricks.sliding_window_view(arr, window)

        # compute WMA
        wma_vals = np.dot(sw, weights)

        # initialize output array with 0
        out = np.zeros(n, dtype=float)
        out[window - 1:] = wma_vals

        return pd.Series(out, index=series.index, name=series.name)

    @classmethod
    def linear_reg(cls, series, window, correlation=False):
        window = cls.get_int_positive(window)
        arr = series.values
        n = len(arr)

        if window > n:
            return pd.Series(np.zeros(n), index=series.index)

        x = np.arange(1, window + 1)
        x_sum = x.sum()
        x2_sum = (x ** 2).sum()
        divisor = window * x2_sum - x_sum ** 2

        sw = np.lib.stride_tricks.sliding_window_view(arr, window)
        y_sum = sw.sum(axis=1)
        xy_sum = np.dot(sw, x)

        if correlation:
            y2_sum = np.sum(sw ** 2, axis=1)
            rn = window * xy_sum - x_sum * y_sum
            rd = np.sqrt(divisor * (window * y2_sum - y_sum ** 2))
            ret_vals = rn / rd
        else:
            m = (window * xy_sum - x_sum * y_sum) / divisor
            b = (y_sum * x2_sum - x_sum * xy_sum) / divisor
            ret_vals = m * (window - 1) + b

        out = np.zeros(n, dtype=float)
        out[window - 1:] = ret_vals
        return pd.Series(out, index=series.index)

    def _get_cti(self, meta: _Meta):
        """get correlation trend indicator

        Correlation Trend Indicator is a study that estimates
        the current direction and strength of a trend.
        https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/C-D/CorrelationTrendIndicator
        """
        col = self[meta.column]
        value = self.linear_reg(col, meta.as_int, correlation=True)
        self[meta.name] = value

    def _get_ema(self, meta: _Meta):
        """get exponential moving average"""
        self[meta.name] = self.ema(self[meta.column], meta.as_int)

    def _get_boll(self, meta: _Meta):
        """Get Bollinger bands.

        boll_ub means the upper band of the Bollinger bands
        boll_lb means the lower band of the Bollinger bands
        boll_ub = MA + Kσ
        boll_lb = MA − Kσ
        M = BOLL_PERIOD
        K = BOLL_STD_TIMES
        :return: None
        """
        n = meta.as_int
        moving_avg = self.sma(self.close, n)
        moving_std = self.mov_std(self.close, n)

        self[meta.name] = moving_avg
        width = self.BOLL_STD_TIMES * moving_std
        self[meta.name_ex("_ub")] = moving_avg + width
        self[meta.name_ex("_lb")] = moving_avg - width

    def _get_macd(self, meta: _Meta):
        """Moving Average Convergence Divergence

        This function will initialize all following columns.

        MACD Line (macd): (12-day EMA - 26-day EMA)
        Signal Line (macds): 9-day EMA of MACD Line
        MACD Histogram (macdh): MACD Line - Signal Line
        """
        close = self.close
        short_w, long_w, signal_w = meta.int0, meta.int1, meta.int2
        ema_short = self.ema(close, short_w)
        ema_long = self.ema(close, long_w)
        macd = meta.name
        macds = meta.name_ex("s")
        macdh = meta.name_ex("h")
        self[macd] = ema_short - ema_long
        self[macds] = self.ema(self[macd], signal_w)
        self[macdh] = self[macd] - self[macds]

    def _ppo_and_pvo(self, name: str, ser: pd.Series, meta: _Meta):
        short_w, long_w, signal_w = meta.int0, meta.int1, meta.int2
        pvo_short = self.ema(ser, short_w).values
        pvo_long = self.ema(ser, long_w).values
        self[name] = self.to_series((pvo_short - pvo_long) / pvo_long * 100)
        self[f"{name}s"] = self.ema(self[name], signal_w)
        self[f"{name}h"] = self[name] - self[f"{name}s"]

    def _get_pvo(self, meta: _Meta):
        """Percentage Volume Oscillator

        The Percentage Volume Oscillator (PVO) is a momentum oscillator for
        volume.  The PVO measures the difference between two volume-based
        moving averages as a percentage of the larger moving average.

        https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo

        Percentage Volume Oscillator (PVO):
            {(12_EOV - 26_EOV)/26_EOV} x 100

        Where:
        * 12_EOV is the 12-day EMA of Volume
        * 26_EOV is the 26-day EMA of Volume

        Signal Line: 9-day EMA of PVO

        PVO Histogram: PVO - Signal Line
        """
        return self._ppo_and_pvo("pvo", self.volume, meta)

    def _get_ppo(self, meta: _Meta):
        """Percentage Price Oscillator

        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:price_oscillators_ppo

        Percentage Price Oscillator (PPO):
            {(12-day EMA - 26-day EMA)/26-day EMA} x 100

        Signal Line: 9-day EMA of PPO

        PPO Histogram: PPO - Signal Line
        """
        return self._ppo_and_pvo("ppo", self.close, meta)

    def _eri(self, window):
        ema = self.ema(self.close, window, adjust=False)
        bull = self.high - ema
        bear = self.low - ema
        return bull, bear

    def _get_eribull(self, meta: _Meta):
        """The bull line of Elder-Ray Index"""
        bull, _ = self._eri(meta.as_int)
        self[meta.name] = bull
        return bull

    def _get_eribear(self, meta: _Meta):
        """The bear line of Elder-Ray Index"""
        _, bear = self._eri(meta.as_int)
        self[meta.name] = bear
        return bear

    def _get_eri(self, meta: _Meta):
        """The Elder-Ray Index

        The Elder-Ray Index contains the bull and the bear power.
        Both are calculated based on the EMA of the close price.

        The default window is 13.

        https://admiralmarkets.com/education/articles/forex-indicators/bears-and-bulls-power-indicator

        Formular:
        * Bulls Power = High - EMA
        * Bears Power = Low - EMA
        * EMA is exponential moving average of close of N periods
        """
        bull, bear = self._eri(meta.as_int)
        self[meta.name_ex("bull")] = bull
        self[meta.name_ex("bear")] = bear

    def _get_coppock(self, meta: _Meta):
        """Get Coppock Curve

        Coppock Curve is a momentum indicator that signals
        long-term trend reversals.

        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:coppock_curve
        """
        window, fast, slow = meta.int0, meta.int1, meta.int2
        fast_roc = self.roc(self.close, fast)
        slow_roc = self.roc(self.close, slow)
        roc_ema = self.linear_wma(fast_roc + slow_roc, window)
        self[meta.name] = roc_ema

    @classmethod
    def get_int_positive(cls, windows):
        if isinstance(windows, int):
            window = windows
        else:
            window = cls.to_int(windows)
            if window <= 0:
                raise IndexError("window must be greater than 0")
        return window

    def _hl_mid(self, period):
        ph = self.mov_max(self.high, period)
        pl = self.mov_min(self.low, period)
        return (ph + pl) * 0.5

    def _get_ichimoku(self, meta: _Meta):
        """get Ichimoku Cloud

        The Ichimoku Cloud is a collection of technical indicators
        that show support and resistance levels, as well as momentum
        and trend direction.

        In this implementation, we only calculate the delta between
        lead A and lead B.

        https://www.investopedia.com/terms/i/ichimoku-cloud.asp

        It contains three windows:
        * window for the conversion line, default to 9
        * window for the baseline and the shifts, default to 26
        * window for the leading line, default to 52

        Formular:
        * conversion line = (PH9 + PL9) / 2
        * baseline = (PH26 + PL26) / 2
        * leading span A = (conversion line + baseline) / 2
        * leading span B = (PH52 + PL52) / 2
        * result = leading span A - leading span B

        Where:
        * PH = Period High
        * PL = Period Low

        """
        conv, base, lead = meta.int0, meta.int1, meta.int2
        conv_line = self._hl_mid(conv)
        base_line = self._hl_mid(base)
        lead_a = (conv_line + base_line) * 0.5
        lead_b = self._hl_mid(lead)

        lead_a_s = self.s_shift(lead_a, -base)
        lead_b_s = self.s_shift(lead_b, -base)
        self[meta.name] = lead_a_s - lead_b_s

    @classmethod
    def mov_std(cls, series, window):
        return cls._rolling(series, window).std()

    def _get_mstd(self, meta: _Meta):
        """get moving standard deviation"""
        self[meta.name] = self.mov_std(self[meta.column], meta.as_int)

    @classmethod
    def mov_var(cls, series, window):
        return cls._rolling(series, window).var()

    def _get_mvar(self, meta: _Meta):
        """get moving variance"""
        self[meta.name] = self.mov_var(self[meta.column], meta.as_int)

    def _get_vwma(self, meta: _Meta):
        """get Volume Weighted Moving Average

        The definition is available at:
        https://www.investopedia.com/articles/trading/11/trading-with-vwap-mvwap.asp
        """
        window = meta.as_int
        tpv = self.volume * self._tp()
        rolling_tpv = self.mov_sum(tpv, window)
        rolling_vol = self.mov_sum(self.volume, window)
        self[meta.name] = rolling_tpv / rolling_vol

    def _get_chop(self, meta: _Meta):
        """get Choppiness Index (CHOP)

        See the definition of the index here:
        https://www.tradingview.com/education/choppinessindex/

        Calculation:

        100 * LOG10( SUM(ATR(1), n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
        n = User defined period length.
        LOG10(n) = base-10 LOG of n
        ATR(1) = Average True Range (Period of 1)
        SUM(ATR(1), n) = Sum of the Average True Range over past n bars
        MaxHi(n) = The highest high over past n bars
        """
        window = meta.as_int
        atr = self._atr(1)
        atr_sum = self.mov_sum(atr, window).values
        high = self.mov_max(self.high, window).values
        low = self.mov_min(self.low, window).values
        high_low = high - low
        choppy = np.divide(atr_sum, high_low,
                           out=np.zeros_like(atr_sum), where=high_low != 0)
        numerator = np.log10(choppy) * 100
        denominator = np.log10(window)
        self[meta.name] = self.to_series(numerator / denominator)

    def _get_mfi(self, meta: _Meta):
        """get money flow index

        The definition of money flow index is available at:
        https://www.investopedia.com/terms/m/mfi.asp
        """
        window = meta.as_int
        tp = self._tp().values
        volume = self.volume.values
        raw_money_flow = tp * volume

        tp_diff = np.zeros_like(tp)
        tp_diff[1:] = np.diff(tp)

        pos_flow = np.where(tp_diff > 0, raw_money_flow, 0.0)
        neg_flow = np.where(tp_diff < 0, raw_money_flow, 0.0)

        pos_sum = self._rolling_sum(pos_flow, window)
        neg_sum = self._rolling_sum(neg_flow, window)

        total_flow = pos_sum + neg_sum
        mfi = np.divide(pos_sum, total_flow, out=np.full_like(pos_sum, 0.5),
                        where=total_flow > 0)
        mfi[:window] = 0.5

        self[meta.name] = self.to_series(mfi)

    def _get_ao(self, meta: _Meta):
        """get awesome oscillator

        The AO indicator is a good indicator for measuring the market dynamics,
        it reflects specific changes in the driving force of the market, which
        helps to identify the strength of the trend, including the points of
        its formation and reversal.


        Awesome Oscillator Formula
        * MEDIAN PRICE = (HIGH+LOW)/2
        * AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

        https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator
        """
        fast = meta.int0
        slow = meta.int1
        median_price = (self.high + self.low) * 0.5
        ao = self.sma(median_price, fast) - self.sma(median_price, slow)
        self[meta.name] = ao

    def _get_bop(self, meta: _Meta):
        """get balance of power

        The Balance of Power indicator measures the strength of the bulls.
        https://school.stockcharts.com/doku.php?id=technical_indicators:balance_of_power

        BOP = (close - open) / (high - low)
        """
        dividend = (self.close - self.open).values
        divisor = (self.high - self.low).values
        bop = np.divide(dividend, divisor,
                        out=np.zeros_like(dividend), where=divisor != 0)
        self[meta.name] = bop

    def _get_cmo(self, meta: _Meta):
        """get Chande Momentum Oscillator

        The Chande Momentum Oscillator (CMO) is a technical momentum
        indicator developed by Tushar Chande.
        https://www.investopedia.com/terms/c/chandemomentumoscillator.asp

        CMO = 100 * ((sH - sL) / (sH + sL))

        where:
        * sH=the sum of higher closes over N periods
        * sL=the sum of lower closes of N periods
        """
        window = meta.as_int
        close_diff = self._col_diff("close")
        up = close_diff.clip(lower=0)
        down = close_diff.clip(upper=0).abs()
        sum_up = self.mov_sum(up, window)
        sum_down = self.mov_sum(down, window)
        dividend = (sum_up - sum_down).values
        divisor = (sum_up + sum_down).values
        cmo = np.divide(100 * dividend, divisor,
                        out=np.zeros_like(dividend), where=divisor != 0)
        res = pd.Series(cmo, index=self.index)
        res.iloc[0] = 0.0
        self[meta.name] = res

    def ker(self, column, window):
        val = self[column].values
        net_change = np.zeros_like(val)
        net_change[window:] = np.abs(val[window:] - val[:-window])

        abs_diff = np.zeros_like(val)
        abs_diff[1:] = np.abs(np.diff(val))

        volatility = pd.Series(abs_diff).rolling(window).sum().values
        with np.errstate(divide='ignore', invalid='ignore'):
            er = np.where(volatility > 0, net_change / volatility, 0.0)
        er[:window] = 0.0
        return pd.Series(er, index=self.index)

    def _get_ker(self, meta: _Meta):
        """get Kaufman's efficiency ratio

        The Efficiency Ratio (ER) is calculated by
        dividing the price change over a period by the
        absolute sum of the price movements that occurred
        to achieve that change.
        The resulting ratio ranges between 0 and 1 with
        higher values representing a more efficient or
        trending market.

        The default column is close.
        The default window is 10.

        https://strategyquant.com/codebase/kaufmans-efficiency-ratio-ker/

        Formular:
        window_change = ABS(close - close[n])
        last_change = ABS(close-close[1])
        volatility = moving sum of last_change in n
        KER = window_change / volatility
        """
        self[meta.name] = self.ker(meta.column, meta.as_int)

    def _get_kama(self, meta: _Meta):
        """get Kaufman's Adaptive Moving Average.
        Implemented after
        https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average
        """
        window, fast, slow = meta.int0, meta.int1, meta.int2
        efficiency_ratio = self.ker(meta.column, window)
        fast_ema_smoothing = 2.0 / (fast + 1)
        slow_ema_smoothing = 2.0 / (slow + 1)
        smoothing_2 = fast_ema_smoothing - slow_ema_smoothing
        efficient_smoothing = efficiency_ratio * smoothing_2
        smoothing = (2 * (efficient_smoothing + slow_ema_smoothing)).values

        # start with simple moving average
        col = self[meta.column]
        col_arr = col.values
        kama = self.sma(col, window).values.copy()
        n = len(kama)

        if n >= window:
            last_kama = kama[window - 1]
        else:
            last_kama = 0.0

        for i in range(window, n):
            cur = smoothing[i] * (col_arr[i] - last_kama) + last_kama
            kama[i] = cur
            last_kama = cur

        self[meta.name] = pd.Series(kama, index=col.index)

    def _ftr(self, window: int) -> pd.Series:
        high = self.high.values
        low = self.low.values
        mp = (high + low) * 0.5
        n = len(mp)

        # Use sliding window view for rolling max/min
        sw = np.lib.stride_tricks.sliding_window_view(mp, window)
        highest = np.empty(n, dtype=float)
        lowest = np.empty(n, dtype=float)
        highest[:window - 1] = np.nan
        lowest[:window - 1] = np.nan
        highest[window - 1:] = sw.max(axis=1)
        lowest[window - 1:] = sw.min(axis=1)

        width = highest - lowest
        width = np.maximum(width, 0.001)

        position = ((mp - lowest) / width) - 0.5

        size = high.size
        result = np.zeros(size)
        v = 0.0

        for i in range(window, size):
            # Update smoothed position (v)
            v = 0.66 * position[i] + 0.67 * v

            # Clamp v to stay within log boundaries (-0.999 to 0.999)
            if v < -0.99:
                v = -0.999
            elif v > 0.99:
                v = 0.999

            # Fisher Transform formula
            # result[i-1] is the recursive component
            result[i] = 0.5 * (np.log((1 + v) / (1 - v)) + result[i - 1])

        return self.to_series(result)

    def _get_ftr(self, meta: _Meta):
        """the Gaussian Fisher Transform Price Reversals indicator

        The Gaussian Fisher Transform Price Reversals indicator, dubbed
        FTR for short, is a stat based price reversal detection indicator
        inspired by and based on the work of the electrical engineer
        now private trader John F. Ehlers.

        https://www.tradingview.com/script/ajZT2tZo-Gaussian-Fisher-Transform-Price-Reversals-FTR/

        Implementation reference:

        https://github.com/twopirllc/pandas-ta/blob/084dbe1c4b76082f383fa3029270ea9ac35e4dc7/pandas_ta/momentum/fisher.py#L9

        Formular:
        * Fisher Transform = 0.5 * ln((1 + X) / (1 - X))
        * X is a series whose values are between -1 to 1
        """
        self[meta.name] = self._ftr(meta.as_int)

    @staticmethod
    def _sym_wma4(arr: np.ndarray) -> np.ndarray:
        weights = np.array([1, 2, 2, 1], dtype=float)
        weights /= weights.sum()  # normalize

        # mode='valid' gives length = n - window + 1
        conv = np.convolve(arr, weights, mode="valid")

        # pad beginning with 0s for first window-1 elements
        out = np.zeros(len(arr), dtype=float)
        out[len(weights) - 1:] = conv

        return out

    @classmethod
    def sym_wma4(cls, series: pd.Series) -> pd.Series:
        res = cls._sym_wma4(series.values)
        return pd.Series(res, index=series.index)

    def _rvgi(self, window: int) -> pd.Series:
        """Relative Vigor Index (RVGI)

        The Relative Vigor Index (RVI) is a momentum indicator
        used in technical analysis that measures the strength
        of a trend by comparing a security's closing price to
        its trading range while smoothing the results using a
        simple moving average (SMA).

        https://www.investopedia.com/terms/r/relative_vigor_index.asp

        Formular
        * NUMERATOR= (a+(2×b)+(2×c)+d) / 6
        * DENOMINATOR= (e+(2×f)+(2×g)+h) / 6
        * RVI= SMA-N of DENOMINATOR / SMA-N of NUMERATOR
        * Signal Line = (RVI+(2×i)+(2×j)+k) / 6

        where:
        * a=Close−Open
        * b=Close−Open One Bar Prior to a
        * c=Close−Open One Bar Prior to b
        * d=Close−Open One Bar Prior to c
        * e=High−Low of Bar a
        * f=High−Low of Bar b
        * g=High−Low of Bar c
        * h=High−Low of Bar d
        * i=RVI Value One Bar Prior
        * j=RVI Value One Bar Prior to i
        * k=RVI Value One Bar Prior to j
        * N=Minutes/Hours/Days/Weeks/Months
        """
        co = self.close.values - self.open.values
        hl = self.high.values - self.low.values

        nu = self.to_series(self._sym_wma4(co))
        de = self.to_series(self._sym_wma4(hl))
        ret = self.sma(nu, window) / self.sma(de, window)
        return ret

    def _get_rvgis(self, meta: _Meta):
        self._get_rvgi(meta.set_name("rvgi"))

    def _get_rvgi(self, meta: _Meta):
        rvgi = self._rvgi(meta.as_int).values.copy()
        rvgi[:3] = 0.0
        rvgi_s = self._sym_wma4(rvgi)
        rvgi_s[:6] = 0.0
        self[meta.name] = self.to_series(rvgi)
        self[meta.name_ex("s")] = self.to_series(rvgi_s)

    def _inertia(self, window: int, rvgi_window: int) -> pd.Series:
        """Inertia Indicator

        https://theforexgeek.com/inertia-indicator/

        In financial markets, the concept of inertia was given by Donald Dorsey
        in the 1995 issue of Technical Analysis of Stocks and Commodities
        through the Inertia Indicator. The Inertia Indicator is moment-based
        and is an extension of Dorsey’s Relative Volatility Index (RVI).
        """
        if len(self) < window + rvgi_window:
            return pd.Series(np.zeros(len(self)), index=self.index)
        rvgi = self._rvgi(rvgi_window)
        value = self.linear_reg(rvgi, window)
        value.iloc[: max(window, rvgi_window) + 2] = 0
        return value

    def _get_inertia(self, meta: _Meta):
        value = self._inertia(meta.int0, meta.int1)
        self[meta.name] = value

    def _kst(self) -> pd.Series:
        """Know Sure Thing (kst)

        https://www.investopedia.com/terms/k/know-sure-thing-kst.asp

        The Know Sure Thing (KST) is a momentum oscillator developed by
        Martin Pring to make rate-of-change readings easier for traders
        to interpret.

        Formular:
        * KST=(RCMA #1×1)+(RCMA #2×2) + (RCMA #3×3)+(RCMA #4×4)

        where:
        * RCMA #1=10-period SMA of 10-period ROC
        * RCMA #2=10-period SMA of 15-period ROC
        * RCMA #3=10-period SMA of 20-period ROC
        * RCMA #4=15-period SMA of 30-period ROC
        """
        ma1 = self.sma(self.roc(self.close, 10), 10)
        ma2 = self.sma(self.roc(self.close, 15), 10)
        ma3 = self.sma(self.roc(self.close, 20), 10)
        ma4 = self.sma(self.roc(self.close, 30), 15)
        return ma1 + ma2 * 2 + ma3 * 3 + ma4 * 4

    def _get_kst(self, meta: _Meta):
        self[meta.name] = self._kst()

    def _pgo(self, window: int) -> pd.Series:
        """Pretty Good Oscillator (PGO)

        https://library.tradingtechnologies.com/trade/chrt-ti-pretty-good-oscillator.html

        The Pretty Good Oscillator indicator by Mark Johnson measures the
        distance of the current close from its N-day simple moving average,
        expressed in terms of an average true range over a similar period.

        Formular:
        * PGO = (Close - SMA) / (EMA of TR)

        Where:
        * SMA = Simple Moving Average of Close over N periods
        * EMA of TR = Exponential Moving Average of True Range over N periods
        """
        up = (self.close - self.sma(self.close, window)).values
        down = self.ema(self._tr(), window).values
        return pd.Series(np.divide(up, down,
                                   out=np.zeros_like(up), where=down != 0),
                         index=self.index)

    def _get_pgo(self, meta: _Meta):
        self[meta.name] = self._pgo(meta.as_int)

    def _psl(self, col_name: str, window: int) -> pd.Series:
        """Psychological Line (PSL)

        The Psychological Line indicator is the ratio of the number of
        rising periods over the total number of periods.

        https://library.tradingtechnologies.com/trade/chrt-ti-psychological-line.html

        Formular:
        * PSL = (Number of Rising Periods) / (Total Number of Periods) * 100

        Example:
        * `df['psl']` retrieves the PSL with default window 12.
        * `df['psl_10']` retrieves the PSL with window 10.
        * `df['high_12_psl']` retrieves the PSL of high price with window 10.
        """
        col_diff = self._col_diff(col_name)
        pos = col_diff > 0
        return self.mov_sum(pos, window) / window * 100

    def _get_psl(self, meta: _Meta):
        self[meta.name] = self._psl(meta.column, meta.as_int)

    def _get_qqe(self, meta: _Meta):
        """QQE (Quantitative Qualitative Estimation)

        https://www.tradingview.com/script/0vn4HZ7O-Quantitative-Qualitative-Estimation-QQE/

        The Qualitative Quantitative Estimation (QQE) indicator works like a
        smoother version of the popular Relative Strength Index (RSI)
        indicator. QQE expands on RSI by adding two volatility based trailing
        stop lines. These trailing stop lines are composed of a fast and a
        slow moving Average True Range (ATR).  These ATR lines are smoothed
        making this indicator less susceptible to short term volatility.

        Implementation reference:
        https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/qqe.py

        """
        rsi_window = meta.int0
        rsi_ma_window = meta.int1
        factor = 4.236
        wilder_window = rsi_window * 2 - 1

        rsi = self._rsi(rsi_window)
        rsi.iloc[:rsi_window] = np.nan

        ema = functools.partial(self.ema, adjust=False)

        rsi_ma_ser = ema(rsi, rsi_ma_window)
        tr = rsi_ma_ser.diff().abs()
        tr_ma = ema(tr, wilder_window)
        tr_ma_ma = ema(tr_ma, wilder_window) * factor

        rsi_ma = rsi_ma_ser.values
        upper_band = (rsi_ma_ser + tr_ma_ma).values
        lower_band = (rsi_ma_ser - tr_ma_ma).values
        size = rsi_ma.size

        # Pre-allocate output arrays (NumPy memory allocation is O(1) here)
        out_long = np.full(size, 0.0)
        out_short = np.full(size, 0.0)
        out_trend = np.ones(size, dtype=int)
        out_qqe = np.full(size, rsi_ma[0])
        out_qqe_long = np.full(size, np.nan)
        out_qqe_short = np.full(size, np.nan)

        # Core Recursive Loop - Optimized for CPU cache locality
        for i in range(1, size):
            c_rsi = rsi_ma[i]
            p_rsi = rsi_ma[i - 1]

            # Long Line Logic (Recursive)
            p_long = out_long[i - 1]
            if p_rsi > p_long and c_rsi > p_long:
                out_long[i] = max(p_long, lower_band[i])
            else:
                out_long[i] = lower_band[i]

            # Short Line Logic (Recursive)
            p_short = out_short[i - 1]
            if p_rsi < p_short and c_rsi < p_short:
                out_short[i] = min(p_short, upper_band[i])
            else:
                out_short[i] = upper_band[i]

            # Trend and QQE state machine
            # We need to check crosses against the previous stop lines
            if c_rsi > p_short and p_rsi <= out_short[max(0, i - 2)]:
                out_trend[i] = 1
            elif c_rsi < p_long and p_rsi >= out_long[max(0, i - 2)]:
                out_trend[i] = -1
            else:
                out_trend[i] = out_trend[i - 1]

            # Assign values based on determined trend
            if out_trend[i] == 1:
                out_qqe[i] = out_qqe_long[i] = out_long[i]
            else:
                out_trend[i] = -1
                out_qqe[i] = out_qqe_short[i] = out_short[i]

        # Batch assignment back to self (Pandas overhead happens once here)
        self[meta.name] = self.to_series(out_qqe)
        self[meta.name_ex("l")] = self.to_series(out_qqe_long)
        self[meta.name_ex("s")] = self.to_series(out_qqe_short)

    def _get_num(self, meta):
        split = meta.name.split(",")
        decimal_places = 0.0
        if len(split) > 1:
            decimal_places_str = split[1]
            power = pow(0.1, len(decimal_places_str))
            decimal_places = float(decimal_places_str) * power
        self[meta.name] = meta.int0 + decimal_places

    def to_series(self, arr: Union[list, np.ndarray]) -> pd.Series:
        return pd.Series(arr, index=self.close.index).fillna(0)

    @staticmethod
    def parse_column_name(name):
        m = re.match(r"(.*)_([\d\-+~,.]+)_(\w+)", name)
        ret = (None,)
        if m is None:
            m = re.match(r"(.*)_([\d\-+~,]+)", name)
            if m is not None:
                ret = m.group(1, 2)
        else:
            ret = m.group(1, 2, 3)
        return ret

    CROSS_COLUMN_MATCH_STR = "(.+)_(x|xu|xd)_(.+)"
    COMPARE_COLUMN_MATCH_STR = "(.+)_(le|ge|lt|gt|eq|ne)_(.+)"

    @classmethod
    def is_cross_columns(cls, name):
        return re.match(cls.CROSS_COLUMN_MATCH_STR, name) is not None

    @classmethod
    def is_compare_columns(cls, name):
        return re.match(cls.COMPARE_COLUMN_MATCH_STR, name) is not None

    @classmethod
    def parse_cross_column(cls, name):
        m = re.match(cls.CROSS_COLUMN_MATCH_STR, name)
        ret = [None, None, None]
        if m is not None:
            ret = m.group(1, 2, 3)
        return ret

    @classmethod
    def parse_compare_column(cls, name):
        m = re.match(cls.COMPARE_COLUMN_MATCH_STR, name)
        ret = [None, None, None]
        if m is not None:
            ret = m.group(1, 2, 3)
        return ret

    def _get_rate(self, _: _Meta):
        """same as percent"""
        self["rate"] = self.close.pct_change() * 100

    def _col_diff(self, col):
        ret = self[col].diff()
        ret = self._df_to_series(ret)
        ret.iloc[0] = 0.0
        return ret

    def _get_delta(self, key):
        key_to_delta = key.replace("_delta", "")
        self[key] = self._col_diff(key_to_delta)
        return self[key]

    def _get_cross(self, key):
        left, op, right = StockDataFrame.parse_cross_column(key)
        lt_series = self[left] > self[right]
        # noinspection PyTypeChecker
        different = np.zeros_like(lt_series)
        if len(different) > 1:
            # noinspection PyTypeChecker
            different[1:] = np.diff(lt_series)
            different[0] = False
        if op == "x":
            self[key] = different
        elif op == "xu":
            self[key] = different & lt_series
        elif op == "xd":
            self[key] = different & ~lt_series
        return self[key]

    def _get_compare(self, key):
        left, op, right = StockDataFrame.parse_compare_column(key)
        if op == "le":
            self[key] = self[left] <= self[right]
        elif op == "ge":
            self[key] = self[left] >= self[right]
        elif op == "lt":
            self[key] = self[left] < self[right]
        elif op == "gt":
            self[key] = self[left] > self[right]
        elif op == "eq":
            self[key] = self[left] == self[right]
        elif op == "ne":
            self[key] = self[left] != self[right]
        return self[key]

    def init_all(self):
        """initialize all stats. in the handler"""
        for handler in self.handler.values():
            _call_handler(handler)

    def drop_column(self, names=None, inplace=False):
        """drop column by the name

        multiple names can be supplied in a list
        :return: StockDataFrame
        """
        if self.empty:
            return self
        ret = self.drop(names, axis=1, inplace=inplace)
        if inplace is True:
            return self
        return wrap(ret)

    def drop_tail(self, n, inplace=False):
        """drop n rows from the tail

        :return: StockDataFrame
        """
        tail = self.tail(n).index
        ret = self.drop(tail, inplace=inplace)
        if inplace:
            return self
        return wrap(ret)

    def drop_head(self, n, inplace=False):
        """drop n rows from the beginning

        :return: StockDataFrame
        """
        head = self.head(n).index
        ret = self.drop(head, inplace=inplace)
        if inplace is True:
            return self
        return wrap(ret)

    def _get_handler(self, name: str):
        return getattr(self, f"_get_{name}")

    @property
    def handler(self):
        ret = {
            ("rate",): self._get_rate,
            ("middle",): self._get_middle,
            ("tp",): self._get_tp,
            ("boll", "boll_ub", "boll_lb"): self._get_boll,
            ("macd", "macds", "macdh"): self._get_macd,
            ("pvo", "pvos", "pvoh"): self._get_pvo,
            ("ppo", "ppos", "ppoh"): self._get_ppo,
            ("qqe", "qqel", "qqes"): self._get_qqe,
            ("cr", "cr-ma1", "cr-ma2", "cr-ma3"): self._get_cr,
            ("tr",): self._get_tr,
            ("dx", "adx", "adxr"): self._get_dmi,
            ("log-ret",): self._get_log_ret,
            ("wt1", "wt2"): self._get_wt,
            ("supertrend", "supertrend_lb",
             "supertrend_ub"): self._get_supertrend,
            ("bop",): self._get_bop,
            ("cti",): self._get_cti,
            ("eribull", "eribear"): self._get_eri,
            ("rvgi", "rvgis"): self._get_rvgi,
            ("kst",): self._get_kst,
            ("num",): self._get_num,
        }
        for k in _dft_windows.keys():
            if k not in ret:
                ret[k] = self._get_handler(k)
        return ret

    def __init_not_exist_column(self, key):
        for names, handler in self.handler.items():
            if key in names:
                return _call_handler(handler)

        if key.endswith("_delta"):
            self._get_delta(key)
        elif self.is_cross_columns(key):
            self._get_cross(key)
        elif self.is_compare_columns(key):
            self._get_compare(key)
        else:
            ret = self.parse_column_name(key)
            if len(ret) == 3:
                col, n, name = ret
            elif len(ret) == 2:
                name, n = ret
                col = None
            else:
                raise UserWarning(
                    "Invalid number of return arguments "
                    f"after parsing column name: '{key}'"
                )
            meta = _Meta(name, windows=n, column=col)
            self._get_handler(name)(meta)

    def __init_column(self, key):
        if key not in self:
            if len(self) == 0:
                self[key] = []
            else:
                self.__init_not_exist_column(key)

    def __getitem__(self, item):
        try:
            result = wrap(super(StockDataFrame, self).__getitem__(item))
        except KeyError:
            try:
                if isinstance(item, list):
                    for column in item:
                        self.__init_column(column)
                else:
                    self.__init_column(item)
            except AttributeError:
                pass
            result = wrap(super(StockDataFrame, self).__getitem__(item))
        return result

    def till(self, end_date):
        return self[self.index <= end_date]

    def start_from(self, start_date):
        return self[self.index >= start_date]

    def within(self, start_date, end_date):
        return self.start_from(start_date).till(end_date)

    # noinspection PyFinal
    def copy(self, deep=True):
        return wrap(super(StockDataFrame, self).copy(deep))

    @staticmethod
    def _ensure_type(obj):
        """override the method in pandas, omit the check

        This patch is not the perfect way but could make the lib work.
        """
        return obj

    @staticmethod
    def retype(value, index_column=None):
        """if the input is a `DataFrame`, convert it to this class.

        :param index_column: name of the index column, default to `date`
        :param value: value to convert
        :return: this extended class
        """
        if index_column is None:
            index_column = "date"

        if isinstance(value, StockDataFrame):
            return value
        elif isinstance(value, pd.DataFrame):
            value = value.rename(_lower_col_name, axis="columns")
            if index_column in value.columns:
                value.set_index(index_column, inplace=True)
            ret = StockDataFrame(value)
            return ret
        return value


def _lower_col_name(name):
    candidates = ("open", "close", "high", "low", "volume", "amount")
    if name.lower() != name and name.lower() in candidates:
        return name.lower()
    return name
