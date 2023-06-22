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

import itertools
import re

import numpy as np
import pandas as pd

__author__ = 'Cedric Zhuang'

from numpy.lib.stride_tricks import as_strided


def wrap(df, index_column=None):
    """ wraps a pandas DataFrame to StockDataFrame

    :param df: pandas DataFrame
    :param index_column: the name of the index column, default to ``date``
    :return: an object of StockDataFrame
    """
    return StockDataFrame.retype(df, index_column)


def unwrap(sdf):
    """ convert a StockDataFrame back to a pandas DataFrame """
    return pd.DataFrame(sdf)


class StockDataFrame(pd.DataFrame):
    # Start of options.
    KDJ_PARAM = (2.0 / 3.0, 1.0 / 3.0)
    KDJ_WINDOW = 9

    BOLL_PERIOD = 20
    BOLL_STD_TIMES = 2

    MACD_EMA_SHORT = 12
    MACD_EMA_LONG = 26
    MACD_EMA_SIGNAL = 9

    PPO_EMA_SHORT = 12
    PPO_EMA_LONG = 26
    PPO_EMA_SIGNAL = 9

    PDI_SMMA = 14
    MDI_SMMA = 14
    DX_SMMA = 14
    ADX_EMA = 6
    ADXR_EMA = 6

    CR_MA1 = 5
    CR_MA2 = 10
    CR_MA3 = 20

    TRIX_EMA_WINDOW = 12

    TEMA_EMA_WINDOW = 5

    ATR_SMMA = 14

    SUPERTREND_MUL = 3
    SUPERTREND_WINDOW = 14

    VWMA = 14

    CHOP = 14

    MFI = 14

    CCI = 14

    RSI = 14

    VR = 26

    WR = 14

    CMO = 14

    WAVE_TREND_1 = 10
    WAVE_TREND_2 = 21

    ER = 10

    KAMA_SLOW = 34
    KAMA_FAST = 5

    AO_SLOW = 34
    AO_FAST = 5

    COPPOCK = (10, 11, 14)

    ICHIMOKU = (9, 26, 52)

    CTI = 12

    MULTI_SPLIT_INDICATORS = ("kama",)

    # End of options

    @staticmethod
    def _change(series, window):
        return series.pct_change(periods=-window).fillna(0.0) * 100

    def _get_change(self):
        """ Get the percentage change column

        :return: result series
        """
        self['change'] = self._change(self['close'], -1)

    def _get_p(self, column, shifts):
        """ get the permutation of specified range

        example:
        index    x   x_-2,-1_p
        0        1         NaN
        1       -1         NaN
        2        3           2  (0.x > 0, and assigned to weight 2)
        3        5           1  (2.x > 0, and assigned to weight 1)
        4        1           3

        :param column: the column to calculate p from
        :param shifts: the range to consider
        :return:
        """
        column_name = '{}_{}_p'.format(column, shifts)
        # initialize the column if not
        self.get(column)
        shifts = self.to_ints(shifts)[::-1]
        indices = None
        count = 0
        for shift in shifts:
            shifted = self.shift(-shift)
            index = (shifted[column] > 0) * (2 ** count)
            if indices is None:
                indices = index
            else:
                indices += index
            count += 1
        if indices is not None:
            cp = indices.copy()
            self.set_nan(cp, shifts)
            self[column_name] = cp

    @classmethod
    def to_ints(cls, shifts):
        items = map(cls._process_shifts_segment, shifts.split(','))
        return sorted(list(set(itertools.chain(*items))))

    @classmethod
    def to_int(cls, shifts):
        numbers = cls.to_ints(shifts)
        if len(numbers) != 1:
            raise IndexError("only accept 1 number.")
        return numbers[0]

    @staticmethod
    def _process_shifts_segment(shift_segment):
        if '~' in shift_segment:
            start, end = shift_segment.split('~')
            shifts = range(int(start), int(end) + 1)
        else:
            shifts = [int(shift_segment)]
        return shifts

    @staticmethod
    def set_nan(pd_obj, shift):
        try:
            iter(shift)
            max_shift = max(shift)
            min_shift = min(shift)
            StockDataFrame._set_nan_of_single_shift(pd_obj, max_shift)
            StockDataFrame._set_nan_of_single_shift(pd_obj, min_shift)
        except TypeError:
            # shift is not iterable
            StockDataFrame._set_nan_of_single_shift(pd_obj, shift)

    @staticmethod
    def _set_nan_of_single_shift(pd_obj, shift):
        val = np.nan
        if shift > 0:
            pd_obj.iloc[-shift:] = val
        elif shift < 0:
            pd_obj.iloc[:-shift] = val

    def _get_r(self, column, shifts):
        """ Get rate of change of column

        :param column: column name of the rate to calculate
        :param shifts: periods to shift, accept one shift only
        :return: None
        """
        shift = self.to_int(shifts)
        rate_key = '{}_{}_r'.format(column, shift)
        self[rate_key] = self._change(self[column], shift)

    @staticmethod
    def _shift(series, window):
        """ Shift the series

        When window is negative, shift the past period to current.
        Fill the gap with the first data available.

        When window is positive, shift the future period to current.
        Fill the gap with last data available.

        :param series: the series to shift
        :param window: number of periods to shift
        :return: the shifted series with filled gap
        """
        ret = series.shift(-window)
        if window < 0:
            ret.iloc[:-window] = series.iloc[0]
        elif window > 0:
            ret.iloc[-window:] = series.iloc[-1]
        return ret

    def _get_s(self, column, shifts):
        """ Get the column shifted by periods

        :param column: name of the column to shift
        :param shifts: periods to shift, accept one shift only
        :return: None
        """
        shift = self.to_int(shifts)
        shifted_key = "{}_{}_s".format(column, shift)
        self[shifted_key] = self._shift(self[column], shift)

    def _get_log_ret(self):
        close = self['close']
        self['log-ret'] = np.log(close / self._shift(close, -1))

    def _get_c(self, column, window):
        """ get the count of column in range (shifts)

        example: change_20_c

        :param column: column name
        :param window: range to count, only to previous
        :return: result series
        """
        column_name = '{}_{}_c'.format(column, window)
        window = self.get_int_positive(window)
        self[column_name] = self._rolling(
            self[column], window).apply(np.count_nonzero, raw=True)
        return self[column_name]

    def _get_fc(self, column, shifts):
        """ get the count of column in range of future (shifts)

        example: change_20_fc

        :param column: column name
        :param shifts: range to count, only to future
        :return: result series
        """
        column_name = '{}_{}_fc'.format(column, shifts)
        shift = self.get_int_positive(shifts)
        reversed_series = self[column][::-1]
        reversed_counts = self._rolling(
            reversed_series, shift).apply(np.count_nonzero, raw=True)
        counts = reversed_counts[::-1]
        self[column_name] = counts
        return counts

    def _init_shifted_columns(self, column, shifts):
        # initialize the column if not
        self.get(column)
        shifts = self.to_ints(shifts)
        shift_column_names = ['{}_{}_s'.format(column, shift) for shift in
                              shifts]
        [self.get(name) for name in shift_column_names]
        return shift_column_names

    def _get_max(self, column, shifts):
        column_name = '{}_{}_max'.format(column, shifts)
        shift_column_names = self._init_shifted_columns(column, shifts)
        self[column_name] = np.max(self[shift_column_names], axis=1)

    def _get_min(self, column, shifts):
        column_name = '{}_{}_min'.format(column, shifts)
        shift_column_names = self._init_shifted_columns(column, shifts)
        self[column_name] = np.min(self[shift_column_names], axis=1)

    def _get_rsv(self, window):
        """ Calculate the RSV (Raw Stochastic Value) within N periods

        This value is essential for calculating KDJs
        Current day is included in N

        :param window: number of periods
        :return: None
        """
        window = self.get_int_positive(window)
        column_name = 'rsv_{}'.format(window)
        low_min = self.mov_min(self['low'], window)
        high_max = self.mov_max(self['high'], window)

        cv = (self['close'] - low_min) / (high_max - low_min)
        self[column_name] = cv.fillna(0.0) * 100

    def _get_rsi(self, window=None):
        """ Calculate the RSI (Relative Strength Index) within N periods

        calculated based on the formula at:
        https://en.wikipedia.org/wiki/Relative_strength_index

        :param window: number of periods
        :return: None
        """
        if window is None:
            window = self.RSI
            column_name = 'rsi'
        else:
            column_name = 'rsi_{}'.format(window)
        window = self.get_int_positive(window)

        change = self._delta(self['close'], -1)
        close_pm = (change + change.abs()) / 2
        close_nm = (-change + change.abs()) / 2
        p_ema = self.smma(close_pm, window)
        n_ema = self.smma(close_nm, window)

        rs_column_name = 'rs_{}'.format(window)
        self[rs_column_name] = rs = p_ema / n_ema
        self[column_name] = 100 - 100 / (1.0 + rs)

    def _get_stochrsi(self, window=None):
        """ Calculate the Stochastic RSI

        calculated based on the formula at:
        https://www.investopedia.com/terms/s/stochrsi.asp

        :param window: number of periods
        :return: None
        """
        if window is None:
            window = self.RSI
            column_name = 'stochrsi'
        else:
            column_name = 'stochrsi_{}'.format(window)
        window = self.get_int_positive(window)

        rsi = self['rsi_{}'.format(window)]
        rsi_min = self.mov_min(rsi, window)
        rsi_max = self.mov_max(rsi, window)

        cv = (rsi - rsi_min) / (rsi_max - rsi_min)
        self[column_name] = cv * 100

    def _get_wave_trend(self):
        """ Calculate LazyBear's Wavetrend
        Check the algorithm described below:
        https://medium.com/@samuel.mcculloch/lets-take-a-look-at-wavetrend-with-crosses-lazybear-s-indicator-2ece1737f72f

        n1: period of EMA on typical price
        n2: period of EMA

        :return: None
        """
        n1 = self.WAVE_TREND_1
        n2 = self.WAVE_TREND_2

        tp = self._tp()
        esa = self.ema(tp, n1)
        d = self.ema((tp - esa).abs(), n1)
        ci = (tp - esa) / (0.015 * d)
        tci = self.ema(ci, n2)
        self["wt1"] = tci
        self["wt2"] = self.sma(tci, 4)

    @staticmethod
    def smma(series, window):
        return series.ewm(
            ignore_na=False,
            alpha=1.0 / window,
            min_periods=0,
            adjust=True).mean()

    def _get_smma(self, column, windows):
        """ get smoothed moving average.

        :param column: the column to calculate
        :param windows: range
        :return: result series
        """
        window = self.get_int_positive(windows)
        column_name = '{}_{}_smma'.format(column, window)
        self[column_name] = self.smma(self[column], window)

    def _get_trix(self, column=None, windows=None):
        """ Triple Exponential Average

        https://www.investopedia.com/articles/technical/02/092402.asp

        :param column: the column to calculate
        :param windows: range
        :return: result series
        """
        column_name = ""
        if column is None and windows is None:
            column_name = 'trix'
        if column is None:
            column = 'close'
        if windows is None:
            windows = self.TRIX_EMA_WINDOW
        if column_name == "":
            column_name = '{}_{}_trix'.format(column, windows)

        window = self.get_int_positive(windows)

        single = self.ema(self[column], window)
        double = self.ema(single, window)
        triple = self.ema(double, window)
        prev_triple = self._shift(triple, -1)
        triple_change = self._delta(triple, -1)
        self[column_name] = triple_change * 100 / prev_triple

    def _get_tema(self, column=None, windows=None):
        """ Another implementation for triple ema

        Check the algorithm described below:
        https://www.forextraders.com/forex-education/forex-technical-analysis/triple-exponential-moving-average-the-tema-indicator/

        :param column: column to calculate ema
        :param windows: window of the calculation
        :return: result series
        """
        column_name = ""
        if column is None and windows is None:
            column_name = 'tema'
        if column is None:
            column = 'close'
        if windows is None:
            windows = self.TEMA_EMA_WINDOW
        if column_name == "":
            column_name = '{}_{}_tema'.format(column, windows)

        window = self.get_int_positive(windows)

        single = self.ema(self[column], window)
        double = self.ema(single, window)
        triple = self.ema(double, window)
        self[column_name] = 3 * single - 3 * double + triple

    def _get_wr(self, window=None):
        """ Williams Overbought/Oversold Index

        Definition: https://www.investopedia.com/terms/w/williamsr.asp
        WMS=[(Hn—Ct)/(Hn—Ln)] × -100
        Ct - the close price
        Hn - N periods high
        Ln - N periods low

        :param window: number of periods
        :return: None
        """
        if window is None:
            window = self.WR
            column_name = 'wr'
        else:
            column_name = 'wr_{}'.format(window)

        window = self.get_int_positive(window)
        ln = self.mov_min(self['low'], window)

        hn = self.mov_max(self['high'], window)
        self[column_name] = (hn - self['close']) / (hn - ln) * -100

    def _get_cci(self, window=None):
        """ Commodity Channel Index

        CCI = (Typical Price  -  20-period SMA of TP) / (.015 x Mean Deviation)
        * when amount is not available:
          Typical Price (TP) = (High + Low + Close)/3
        * when amount is available:
          Typical Price (TP) = Amount / Volume
        TP is also implemented as 'middle'.

        :param window: number of periods
        :return: None
        """
        if window is None:
            window = self.CCI
            column_name = 'cci'
        else:
            column_name = 'cci_{}'.format(window)
        window = self.get_int_positive(window)

        tp = self._tp()
        tp_sma = self.sma(tp, window)
        mad = self._mad(tp, window)
        self[column_name] = (tp - tp_sma) / (.015 * mad)

    def _tr(self):
        prev_close = self._shift(self['close'], -1)
        high = self['high']
        low = self['low']
        c1 = high - low
        c2 = (high - prev_close).abs()
        c3 = (low - prev_close).abs()
        return pd.concat((c1, c2, c3), axis=1).max(axis=1)

    def _get_tr(self):
        """ True Range of the trading

         TR is a measure of volatility of a High-Low-Close series

        tr = max[(high - low), abs(high - close_prev), abs(low - close_prev)]

        :return: None
        """
        self['tr'] = self._tr()

    def _get_supertrend(self, window=None):
        """ Supertrend

        Supertrend indicator shows trend direction.
        It provides buy or sell indicators.
        https://medium.com/codex/step-by-step-implementation-of-the-supertrend-indicator-in-python-656aa678c111

        :param window: number of periods
        :return: None
        """
        if window is None:
            window = self.SUPERTREND_WINDOW
        window = self.get_int_positive(window)

        high = self['high']
        low = self['low']
        close = self['close']
        m_atr = self.SUPERTREND_MUL * self._atr(window)
        hl_avg = (high + low) / 2.0
        # basic upper band
        b_ub = list(hl_avg + m_atr)
        # basic lower band
        b_lb = list(hl_avg - m_atr)

        size = len(close)
        ub = np.empty(size, dtype=np.float64)
        lb = np.empty(size, dtype=np.float64)
        st = np.empty(size, dtype=np.float64)
        close = list(close)

        for i in range(size):
            if i == 0:
                ub[i] = b_ub[i]
                lb[i] = b_lb[i]
                if close[i] <= ub[i]:
                    st[i] = ub[i]
                else:
                    st[i] = lb[i]
                continue

            last_close = close[i - 1]
            curr_close = close[i]
            last_ub = ub[i - 1]
            last_lb = lb[i - 1]
            last_st = st[i - 1]
            curr_b_ub = b_ub[i]
            curr_b_lb = b_lb[i]

            # calculate current upper band
            if curr_b_ub < last_ub or last_close > last_ub:
                ub[i] = curr_b_ub
            else:
                ub[i] = last_ub

            # calculate current lower band
            if curr_b_lb > last_lb or last_close < last_lb:
                lb[i] = curr_b_lb
            else:
                lb[i] = last_lb

            # calculate supertrend
            if last_st == last_ub:
                if curr_close <= ub[i]:
                    st[i] = ub[i]
                else:
                    st[i] = lb[i]
            elif last_st == last_lb:
                if curr_close > lb[i]:
                    st[i] = lb[i]
                else:
                    st[i] = ub[i]

        self['supertrend_ub'] = ub
        self['supertrend_lb'] = lb
        self['supertrend'] = st

    def _get_aroon(self, window=None):
        """ Aroon Oscillator

        The Aroon Oscillator measures the strength of a trend and
        the likelihood that it will continue.

        The default window is 25.

        * Aroon Oscillator = Aroon Up - Aroon Down
        * Aroon Up = 100 * (n - periods since n-period high) / n
        * Aroon Down = 100 * (n - periods since n-period low) / n
        * n = window size
        """
        if window is None:
            window = 25
            column_name = 'aroon'
        else:
            window = self.get_int_positive(window)
            column_name = 'aroon_{}'.format(window)

        def _window_pct(s):
            n = float(window)
            return (n - (n - (s + 1))) / n * 100

        high_since = self._rolling(
            self['high'], window).apply(np.argmax, raw=True)
        low_since = self._rolling(
            self['low'], window).apply(np.argmin, raw=True)

        aroon_up = _window_pct(high_since)
        aroon_down = _window_pct(low_since)
        self[column_name] = aroon_up - aroon_down

    def _get_z(self, column, window):
        """ Z score

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
        window = self.get_int_positive(window)
        column_name = '{}_{}_z'.format(column, window)
        col = self[column]
        mean = self.sma(col, window)
        std = self.mov_std(col, window)
        self[column_name] = ((col - mean) / std).fillna(0.0)

    def _atr(self, window):
        tr = self._tr()
        return self.smma(tr, window)

    def _get_atr(self, window=None):
        """ Average True Range

        The average true range is an N-day smoothed moving average (SMMA) of
        the true range values.  Default to 14 periods.
        https://en.wikipedia.org/wiki/Average_true_range

        :param window: number of periods
        :return: None
        """
        if window is None:
            window = self.ATR_SMMA
            column_name = 'atr'
        else:
            column_name = 'atr_{}'.format(window)
        window = self.get_int_positive(window)
        self[column_name] = self._atr(window)

    def _get_dma(self):
        """ Difference of Moving Average

        default to 10 and 50.

        :return: None
        """
        self['dma'] = self['close_10_sma'] - self['close_50_sma']

    def _get_dmi(self):
        """ get the default setting for DMI

        including:
        +DI: 14 periods SMMA of +DM,
        -DI: 14 periods SMMA of -DM,
        DX: based on +DI and -DI
        ADX: 6 periods SMMA of DX

        :return:
        """
        self['pdi'] = self._get_pdi(self.PDI_SMMA)
        self['mdi'] = self._get_mdi(self.MDI_SMMA)
        self['dx'] = self._get_dx(self.DX_SMMA)
        self['adx'] = self.ema(self['dx'], self.ADX_EMA)
        self['adxr'] = self.ema(self['adx'], self.ADXR_EMA)

    def _get_pdm_ndm(self, window):
        hd = self._col_diff('high')
        ld = -self._col_diff('low')
        p = ((hd > 0) & (hd > ld)) * hd
        n = ((ld > 0) & (ld > hd)) * ld
        if window > 1:
            p = self.smma(p, window)
            n = self.smma(n, window)
        return p, n

    def _pdm(self, window):
        ret, _ = self._get_pdm_ndm(window)
        return ret

    def _ndm(self, window):
        _, ret = self._get_pdm_ndm(window)
        return ret

    def _get_pdm(self, windows):
        """ +DM, positive directional moving

        If window is not 1, calculate the SMMA of +DM

        :param windows: range
        :return:
        """
        window = self.get_int_positive(windows)
        if window > 1:
            column_name = 'pdm_{}'.format(window)
        else:
            column_name = 'pdm'
        self[column_name] = self._pdm(window)

    def _get_ndm(self, windows):
        """ -DM, negative directional moving accumulation

        If window is not 1, return the SMA of -DM.

        :param windows: range
        :return:
        """
        window = self.get_int_positive(windows)
        if window > 1:
            column_name = 'ndm_{}'.format(window)
        else:
            column_name = 'ndm'
        self[column_name] = self._ndm(window)

    def _get_vr(self, windows=None):
        if windows is None:
            window = self.VR
            column_name = 'vr'
        else:
            window = self.get_int_positive(windows)
            column_name = 'vr_{}'.format(window)

        idx = self.index
        gt_zero = np.where(self['change'] > 0, self['volume'], 0)
        av = pd.Series(gt_zero, index=idx)
        avs = self.mov_sum(av, window)

        lt_zero = np.where(self['change'] < 0, self['volume'], 0)
        bv = pd.Series(lt_zero, index=idx)
        bvs = self.mov_sum(bv, window)

        eq_zero = np.where(self['change'] == 0, self['volume'], 0)
        cv = pd.Series(eq_zero, index=idx)
        cvs = self.mov_sum(cv, window)

        self[column_name] = (avs + cvs / 2) / (bvs + cvs / 2) * 100

    def _get_pdi_ndi(self, window):
        pdm, ndm = self._get_pdm_ndm(window)
        atr = self._atr(window)
        pdi = pdm / atr * 100
        ndi = ndm / atr * 100
        return pdi, ndi

    def _get_pdi(self, windows):
        """ +DI, positive directional moving index

        :param windows: range
        :return:
        """
        window = self.get_int_positive(windows)
        pdi, _ = self._get_pdi_ndi(window)
        pdi_column = 'pdi_{}'.format(window)
        self[pdi_column] = pdi
        return self[pdi_column]

    def _get_mdi(self, windows):
        window = self.get_int_positive(windows)
        _, ndi = self._get_pdi_ndi(window)
        mdi_column = 'mdi_{}'.format(window)
        self[mdi_column] = ndi
        return self[mdi_column]

    def _get_dx(self, windows):
        window = self.get_int_positive(windows)
        dx_column = 'dx_{}'.format(window)
        pdi, mdi = self._get_pdi_ndi(window)
        self[dx_column] = abs(pdi - mdi) / (pdi + mdi) * 100
        return self[dx_column]

    def _get_kdj_default(self):
        """ default KDJ, 9 periods

        :return: None
        """
        self['kdjk'] = self['kdjk_{}'.format(self.KDJ_WINDOW)]
        self['kdjd'] = self['kdjd_{}'.format(self.KDJ_WINDOW)]
        self['kdjj'] = self['kdjj_{}'.format(self.KDJ_WINDOW)]

    def _get_cr(self, windows=None):
        """ Energy Index (Intermediate Willingness Index)

        https://support.futunn.com/en/topic167/?lang=en-us
        Use the relationship between the highest price, the lowest price and
        yesterday's middle price to reflect the market's willingness to buy
        and sell.

        :param windows: window of the moving sum
        :return: None
        """
        if windows is None:
            window = 26
        else:
            window = self.get_int_positive(windows)

        middle = self._tp()
        last_middle = self._shift(middle, -1)
        ym = self._shift(middle, -1)
        high = self['high']
        low = self['low']
        p1_m = pd.concat((last_middle, high), axis=1).min(axis=1)
        p2_m = pd.concat((last_middle, low), axis=1).min(axis=1)
        p1 = self.mov_sum(high - p1_m, window)
        p2 = self.mov_sum(ym - p2_m, window)

        if windows is None:
            cr = 'cr'
            cr_ma1 = 'cr-ma1'
            cr_ma2 = 'cr-ma2'
            cr_ma3 = 'cr-ma3'
        else:
            cr = 'cr_{}'.format(window)
            cr_ma1 = 'cr_{}-ma1'.format(window)
            cr_ma2 = 'cr_{}-ma2'.format(window)
            cr_ma3 = 'cr_{}-ma3'.format(window)

        self[cr] = cr = p1 / p2 * 100
        self[cr_ma1] = self._shifted_cr_sma(cr, self.CR_MA1)
        self[cr_ma2] = self._shifted_cr_sma(cr, self.CR_MA2)
        self[cr_ma3] = self._shifted_cr_sma(cr, self.CR_MA3)

    def _shifted_cr_sma(self, cr, window):
        cr_sma = self.sma(cr, window)
        return self._shift(cr_sma, -int(window / 2.5 + 1))

    def _tp(self):
        if 'amount' in self:
            return self['amount'] / self['volume']
        return (self['close'] + self['high'] + self['low']).divide(3.0)

    def _get_tp(self):
        self['tp'] = self._tp()

    def _get_middle(self):
        self['middle'] = self._tp()

    def _calc_kd(self, column):
        param0, param1 = self.KDJ_PARAM
        k = 50.0
        # noinspection PyTypeChecker
        for i in param1 * column:
            k = param0 * k + i
            yield k

    def _get_kdjk(self, window):
        """ Get the K of KDJ

        K ＝ 2/3 × (prev. K) +1/3 × (curr. RSV)
        2/3 and 1/3 are the smooth parameters.
        :param window: number of periods
        :return: None
        """
        rsv_column = 'rsv_{}'.format(window)
        k_column = 'kdjk_{}'.format(window)
        self[k_column] = list(self._calc_kd(self.get(rsv_column)))

    def _get_kdjd(self, window):
        """ Get the D of KDJ

        D = 2/3 × (prev. D) +1/3 × (curr. K)
        2/3 and 1/3 are the smooth parameters.
        :param window: number of periods
        :return: None
        """
        k_column = 'kdjk_{}'.format(window)
        d_column = 'kdjd_{}'.format(window)
        self[d_column] = list(self._calc_kd(self.get(k_column)))

    def _get_kdjj(self, window):
        """ Get the J of KDJ

        J = 3K-2D
        :param self: data
        :param window: number of periods
        :return: None
        """
        k_column = 'kdjk_{}'.format(window)
        d_column = 'kdjd_{}'.format(window)
        j_column = 'kdjj_{}'.format(window)
        self[j_column] = 3 * self[k_column] - 2 * self[d_column]

    @staticmethod
    def _delta(series, window):
        return series.diff(-window).fillna(0.0)

    def _get_d(self, column, shifts):
        shift = self.to_int(shifts)
        column_name = '{}_{}_d'.format(column, shift)
        self[column_name] = self._delta(self[column], shift)

    @classmethod
    def mov_min(cls, series, size):
        return cls._rolling(series, size).min()

    @classmethod
    def mov_max(cls, series, size):
        return cls._rolling(series, size).max()

    @classmethod
    def mov_sum(cls, series, size):
        return cls._rolling(series, size).sum()

    @classmethod
    def sma(cls, series, size):
        return cls._rolling(series, size).mean()

    @staticmethod
    def roc(series, size):
        ret = series.diff(size) / series.shift(size)
        ret.iloc[:size] = 0
        return ret * 100

    @classmethod
    def _mad(cls, series, window):
        """ Mean Absolute Deviation

        :param series: Series
        :param window: number of periods
        :return: Series
        """

        def f(x):
            return np.fabs(x - x.mean()).mean()

        return cls._rolling(series, window).apply(f, raw=True)

    def _get_mad(self, column, window):
        """ get mean absolute deviation

        :param column: column to calculate
        :param window: number of periods
        :return: None
        """
        window = self.get_int_positive(window)
        column_name = '{}_{}_mad'.format(column, window)
        self[column_name] = self._mad(self[column], window)

    def _get_sma(self, column, windows):
        """ get simple moving average

        :param column: column to calculate
        :param windows: window of simple moving average
        :return: None
        """
        window = self.get_int_positive(windows)
        column_name = '{}_{}_sma'.format(column, window)
        self[column_name] = self.sma(self[column], window)

    def _get_lrma(self, column, window):
        """ get linear regression moving average

        :param column: column to calculate
        :param window: window size
        :return: None
        """
        window = self.get_int_positive(window)
        column_name = '{}_{}_lrma'.format(column, window)
        self[column_name] = self.linear_reg(self[column], window)

    def _get_roc(self, column, window):
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

        :param column: column to calculate
        :param window: window of Rate of Change (ROC)
        :return: None
        """
        window = self.get_int_positive(window)
        column_name = '{}_{}_roc'.format(column, window)
        self[column_name] = self.roc(self[column], window)

    @staticmethod
    def ema(series, window):
        return series.ewm(
            ignore_na=False,
            span=window,
            min_periods=0,
            adjust=True).mean()

    @staticmethod
    def _rolling(series, window):
        return series.rolling(window, min_periods=1, center=False)

    @classmethod
    def linear_wma(cls, series, window):
        total_weight = 0.5 * window * (window + 1)
        weights = np.arange(1, window + 1) / total_weight

        def linear(w):
            def _compute(x):
                try:
                    return np.dot(x, w)
                except ValueError:
                    return 0.0

            return _compute

        rolling = cls._rolling(series, window)
        return rolling.apply(linear(weights), raw=True)

    @classmethod
    def linear_reg(cls,
                   series,
                   window,
                   correlation=False):
        window = cls.get_int_positive(window)

        x = range(1, window + 1)
        x_sum = 0.5 * window * (window + 1)
        x2_sum = x_sum * (2 * window + 1) / 3
        divisor = window * x2_sum - x_sum * x_sum

        def linear_regression(s):
            y_sum = s.sum()
            xy_sum = (x * s).sum()

            m = (window * xy_sum - x_sum * y_sum) / divisor
            b = (y_sum * x2_sum - x_sum * xy_sum) / divisor

            if correlation:
                y2_sum = (s * s).sum()
                rn = window * xy_sum - x_sum * y_sum
                rd = (divisor * (window * y2_sum - y_sum * y_sum)) ** 0.5
                return rn / rd
            return m * (window - 1) + b

        def rolling(arr):
            strides = arr.strides + (arr.strides[-1],)
            shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
            return as_strided(arr, shape=shape, strides=strides)

        value = [linear_regression(_)
                 for _ in rolling(np.array(series))]
        ret = pd.Series([0.0] * (window - 1) + value,
                        index=series.index)
        return ret

    def _get_cti(self, column=None, window=None):
        """ get correlation trend indicator

        Correlation Trend Indicator is a study that estimates
        the current direction and strength of a trend.
        https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/C-D/CorrelationTrendIndicator

        :param column: column to calculate, default to 'close'
        :param window: window of Correlation Trend Indicator
        """
        if column is None and window is None:
            column_name = 'cti'
        else:
            column_name = '{}_{}_cti'.format(column, window)

        if column is None:
            column = 'close'
        if window is None:
            window = self.CTI
        else:
            window = self.get_int_positive(window)

        value = self.linear_reg(
            self[column], window, correlation=True)
        self[column_name] = value

    def _get_ema(self, column, windows):
        """ get exponential moving average

        :param column: column to calculate
        :param windows: collection of window of exponential moving average
        :return: None
        """
        window = self.get_int_positive(windows)
        column_name = '{}_{}_ema'.format(column, window)
        self[column_name] = self.ema(self[column], window)

    def _get_boll(self, window=None):
        """ Get Bollinger bands.

        boll_ub means the upper band of the Bollinger bands
        boll_lb means the lower band of the Bollinger bands
        boll_ub = MA + Kσ
        boll_lb = MA − Kσ
        M = BOLL_PERIOD
        K = BOLL_STD_TIMES
        :return: None
        """
        if window is None:
            n = self.BOLL_PERIOD
            boll = 'boll'
            boll_ub = 'boll_ub'
            boll_lb = 'boll_lb'
        else:
            n = self.get_int_positive(window)
            boll = 'boll_{}'.format(n)
            boll_ub = 'boll_ub_{}'.format(n)
            boll_lb = 'boll_lb_{}'.format(n)
        moving_avg = self.sma(self['close'], n)
        moving_std = self.mov_std(self['close'], n)

        self[boll] = moving_avg
        width = self.BOLL_STD_TIMES * moving_std
        self[boll_ub] = moving_avg + width
        self[boll_lb] = moving_avg - width

    def _get_macd(self):
        """ Moving Average Convergence Divergence

        This function will initialize all following columns.

        MACD Line (macd): (12-day EMA - 26-day EMA)
        Signal Line (macds): 9-day EMA of MACD Line
        MACD Histogram (macdh): MACD Line - Signal Line

        :return: None
        """
        close = self['close']
        ema_short = self.ema(close, self.MACD_EMA_SHORT)
        ema_long = self.ema(close, self.MACD_EMA_LONG)
        self['macd'] = ema_short - ema_long
        self['macds'] = self.ema(self['macd'], self.MACD_EMA_SIGNAL)
        self['macdh'] = self['macd'] - self['macds']

    def _get_ppo(self):
        """ Percentage Price Oscillator

        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:price_oscillators_ppo

        Percentage Price Oscillator (PPO):
            {(12-day EMA - 26-day EMA)/26-day EMA} x 100

        Signal Line: 9-day EMA of PPO

        PPO Histogram: PPO - Signal Line

        :return: None
        """
        close = self['close']
        ppo_short = self.ema(close, self.PPO_EMA_SHORT)
        ppo_long = self.ema(close, self.PPO_EMA_LONG)
        self['ppo'] = (ppo_short - ppo_long) / ppo_long * 100
        self['ppos'] = self.ema(self['ppo'], self.PPO_EMA_SIGNAL)
        self['ppoh'] = self['ppo'] - self['ppos']

    def _get_coppock(self, windows=None):
        """ Get Coppock Curve

        Coppock Curve is a momentum indicator that signals
        long-term trend reversals.

        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:coppock_curve

        :param windows: collection of window of Coppock Curve
        :return: None
        """
        if windows is None:
            window = self.COPPOCK[0]
            fast = self.COPPOCK[1]
            slow = self.COPPOCK[2]
            column_name = 'coppock'
        else:
            periods = self.to_ints(windows)
            window = periods[0]
            fast = periods[1]
            slow = periods[2]
            column_name = 'coppock_{}'.format(windows)

        fast_roc = self.roc(self['close'], fast)
        slow_roc = self.roc(self['close'], slow)
        roc_ema = self.linear_wma(fast_roc + slow_roc, window)
        self[column_name] = roc_ema

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
        ph = self.mov_max(self['high'], period)
        pl = self.mov_min(self['low'], period)
        return (ph + pl) * 0.5

    def _get_ichimoku(self, windows=None):
        """ get Ichimoku Cloud

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
        if windows is None:
            conv = self.ICHIMOKU[0]
            base = self.ICHIMOKU[1]
            lead = self.ICHIMOKU[2]
            column_name = 'ichimoku'
        else:
            periods = self.to_ints(windows)
            conv = periods[0]
            base = periods[1]
            lead = periods[2]
            column_name = 'ichimoku_{}'.format(windows)

        conv_line = self._hl_mid(conv)
        base_line = self._hl_mid(base)
        lead_a = (conv_line + base_line) * 0.5
        lead_b = self._hl_mid(lead)

        lead_a_s = lead_a.shift(base, fill_value=lead_a.iloc[0])
        lead_b_s = lead_b.shift(base, fill_value=lead_b.iloc[0])
        self[column_name] = lead_a_s - lead_b_s

    @classmethod
    def mov_std(cls, series, window):
        return cls._rolling(series, window).std()

    def _get_mstd(self, column, windows):
        """ get moving standard deviation

        :param column: column to calculate
        :param windows: collection of window of moving standard deviation
        :return: None
        """
        window = self.get_int_positive(windows)
        column_name = '{}_{}_mstd'.format(column, window)
        self[column_name] = self.mov_std(self[column], window)

    @classmethod
    def mov_var(cls, series, window):
        return cls._rolling(series, window).var()

    def _get_mvar(self, column, windows):
        """ get moving variance

        :param column: column to calculate
        :param windows: collection of window of moving variance
        :return: None
        """
        window = self.get_int_positive(windows)
        column_name = '{}_{}_mvar'.format(column, window)
        self[column_name] = self.mov_var(self[column], window)

    def _get_vwma(self, window=None):
        """ get Volume Weighted Moving Average

        The definition is available at:
        https://www.investopedia.com/articles/trading/11/trading-with-vwap-mvwap.asp

        :param window: number of periods relevant for the indicator
        :return: None
        """
        if window is None:
            window = self.VWMA
            column_name = 'vwma'
        else:
            column_name = 'vwma_{}'.format(window)
        window = self.get_int_positive(window)

        tpv = self['volume'] * self._tp()
        rolling_tpv = self.mov_sum(tpv, window)
        rolling_vol = self.mov_sum(self['volume'], window)
        self[column_name] = rolling_tpv / rolling_vol

    def _get_chop(self, window=None):
        """ get Choppiness Index (CHOP)

        See the definition of the index here:
        https://www.tradingview.com/education/choppinessindex/

        Calculation:

        100 * LOG10( SUM(ATR(1), n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
        n = User defined period length.
        LOG10(n) = base-10 LOG of n
        ATR(1) = Average True Range (Period of 1)
        SUM(ATR(1), n) = Sum of the Average True Range over past n bars
        MaxHi(n) = The highest high over past n bars

        :param window: number of periods relevant for the indicator
        :return: None
        """
        if window is None:
            window = self.CHOP
            column_name = 'chop'
        else:
            column_name = 'chop_{}'.format(window)
        window = self.get_int_positive(window)
        atr = self._atr(1)
        atr_sum = self.mov_sum(atr, window)
        high = self.mov_max(self['high'], window)
        low = self.mov_min(self['low'], window)
        choppy = atr_sum / (high - low)
        numerator = np.log10(choppy) * 100
        denominator = np.log10(window)
        self[column_name] = numerator / denominator

    def _get_mfi(self, window=None):
        """ get money flow index

        The definition of money flow index is available at:
        https://www.investopedia.com/terms/m/mfi.asp

        :param window: number of periods relevant for the indicator
        :return: None
        """
        if window is None:
            window = self.MFI
            column_name = 'mfi'
        else:
            column_name = 'mfi_{}'.format(window)
        window = self.get_int_positive(window)
        middle = self._tp()
        money_flow = (middle * self["volume"]).fillna(0.0)
        shifted = self._shift(middle, -1)
        delta = (middle - shifted).fillna(0)
        pos_flow = money_flow.mask(delta < 0, 0)
        neg_flow = money_flow.mask(delta >= 0, 0)
        rolling_pos_flow = self.mov_sum(pos_flow, window)
        rolling_neg_flow = self.mov_sum(neg_flow, window)
        money_flow_ratio = rolling_pos_flow / (rolling_neg_flow + 1e-12)
        mfi = (1.0 - 1.0 / (1 + money_flow_ratio))
        mfi.iloc[:window] = 0.5
        self[column_name] = mfi

    def _get_ao(self, windows=None):
        """ get awesome oscillator

        The AO indicator is a good indicator for measuring the market dynamics,
        it reflects specific changes in the driving force of the market, which
        helps to identify the strength of the trend, including the points of
        its formation and reversal.


        Awesome Oscillator Formula
        * MEDIAN PRICE = (HIGH+LOW)/2
        * AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

        https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator
        """
        if windows is None:
            fast = self.AO_FAST
            slow = self.AO_SLOW
            column_name = 'ao'
        else:
            n0, n1 = self.to_ints(windows)
            fast = min(n0, n1)
            slow = max(n0, n1)
            column_name = 'ao_{},{}'.format(fast, slow)

        median_price = (self['high'] + self['low']) * 0.5
        ao = self.sma(median_price, fast) - self.sma(median_price, slow)
        self[column_name] = ao

    def _get_bop(self):
        """ get balance of power

        The Balance of Power indicator measures the strength of the bulls.
        https://school.stockcharts.com/doku.php?id=technical_indicators:balance_of_power

        BOP = (close - open) / (high - low)
        """
        dividend = self['close'] - self['open']
        divisor = self['high'] - self['low']
        self['bop'] = dividend / divisor

    def _get_cmo(self, window=None):
        """ get Chande Momentum Oscillator

        The Chande Momentum Oscillator (CMO) is a technical momentum
        indicator developed by Tushar Chande.
        https://www.investopedia.com/terms/c/chandemomentumoscillator.asp

        CMO = 100 * ((sH - sL) / (sH + sL))

        where:
        * sH=the sum of higher closes over N periods
        * sL=the sum of lower closes of N periods
        """
        if window is None:
            window = self.CMO
            column_name = 'cmo'
        else:
            window = self.get_int_positive(window)
            column_name = 'cmo_{}'.format(window)

        close_diff = self['close'].diff()
        up = close_diff.clip(lower=0)
        down = close_diff.clip(upper=0).abs()
        sum_up = self.mov_sum(up, window)
        sum_down = self.mov_sum(down, window)
        dividend = sum_up - sum_down
        divisor = sum_up + sum_down
        res = 100 * dividend / divisor
        res.iloc[0] = 0
        self[column_name] = res

    def ker(self, column, window):
        col = self[column]
        col_window_s = self._shift(col, -window)
        window_diff = (col - col_window_s).abs()
        diff = self._col_diff(column).abs()
        volatility = self.mov_sum(diff, window)
        ret = window_diff / volatility
        ret.iloc[0] = 0
        return ret

    def _get_ker(self, column=None, window=None):
        """ get Kaufman's efficiency ratio

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
        if column is None and window is None:
            column = 'close'
            window = self.ER
            column_name = 'ker'
        else:
            window = self.get_int_positive(window)
            column_name = '{}_{}_ker'.format(column, window)

        self[column_name] = self.ker(column, window)

    def _get_kama(self, column, windows, fasts=None, slows=None):
        """ get Kaufman's Adaptive Moving Average.
        Implemented after
        https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average

        :param column: column to calculate
        :param windows: collection of window of exponential moving average
        :param fasts: faster EMA constant
        :param slows: slower EMA constant
        :return: None
        """
        window = self.get_int_positive(windows)
        if slows is None or fasts is None:
            slow, fast = self.KAMA_SLOW, self.KAMA_FAST
            column_name = "{}_{}_kama".format(column, window)
        else:
            slow = self.get_int_positive(slows)
            fast = self.get_int_positive(fasts)
            column_name = '{}_{}_kama_{}_{}'.format(column, window, fast, slow)

        col = self[column]
        efficiency_ratio = self.ker(column, window)
        fast_ema_smoothing = 2.0 / (fast + 1)
        slow_ema_smoothing = 2.0 / (slow + 1)
        smoothing_2 = fast_ema_smoothing - slow_ema_smoothing
        efficient_smoothing = efficiency_ratio * smoothing_2
        smoothing = list(2 * (efficient_smoothing + slow_ema_smoothing))

        # start with simple moving average
        kama = list(self.sma(col, window))
        col_list = list(col)
        if len(kama) >= window:
            last_kama = kama[window - 1]
        else:
            last_kama = 0.0
        for i in range(window, len(kama)):
            cur = smoothing[i] * (col_list[i] - last_kama) + last_kama
            kama[i] = cur
            last_kama = cur
        self[column_name] = kama

    @staticmethod
    def parse_column_name(name):
        m = re.match(r'(.*)_([\d\-+~,.]+)_(\w+)', name)
        ret = (None,)
        if m is None:
            m = re.match(r'(.*)_([\d\-+~,]+)', name)
            if m is not None:
                ret = m.group(1, 2)
        else:
            ret = m.group(1, 2, 3)
            if any(map(lambda i: i in ret[0],
                       StockDataFrame.MULTI_SPLIT_INDICATORS)):
                m_prev = re.match(r'(.*)_([\d\-+~,.]+)_(\w+)', ret[0])
                if m_prev is not None:
                    ret = m_prev.group(1, 2, 3) + ret[1:]
        return ret

    CROSS_COLUMN_MATCH_STR = '(.+)_(x|xu|xd)_(.+)'

    @classmethod
    def is_cross_columns(cls, name):
        return re.match(cls.CROSS_COLUMN_MATCH_STR, name) is not None

    @classmethod
    def parse_cross_column(cls, name):
        m = re.match(cls.CROSS_COLUMN_MATCH_STR, name)
        ret = [None, None, None]
        if m is not None:
            ret = m.group(1, 2, 3)
        return ret

    def _get_rate(self):
        """ same as percent

        :return: None
        """
        self['rate'] = self['close'].pct_change() * 100

    def _col_diff(self, col):
        ret = self[col].diff()
        ret.iloc[0] = 0.0
        return ret

    def _get_delta(self, key):
        key_to_delta = key.replace('_delta', '')
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
        if op == 'x':
            self[key] = different
        elif op == 'xu':
            self[key] = different & lt_series
        elif op == 'xd':
            self[key] = different & ~lt_series
        return self[key]

    def init_all(self):
        """ initialize all stats. in the handler """
        for handler in self.handler.values():
            handler()

    def drop_column(self, names=None, inplace=False):
        """ drop column by the name

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
        """ drop n rows from the tail

        :return: StockDataFrame
        """
        tail = self.tail(n).index
        ret = self.drop(tail, inplace=inplace)
        if inplace is True:
            return self
        return wrap(ret)

    def drop_head(self, n, inplace=False):
        """ drop n rows from the beginning

        :return: StockDataFrame
        """
        head = self.head(n).index
        ret = self.drop(head, inplace=inplace)
        if inplace is True:
            return self
        return wrap(ret)

    @property
    def handler(self):
        return {
            ('change',): self._get_change,
            ('rsi',): self._get_rsi,
            ('stochrsi',): self._get_stochrsi,
            ('rate',): self._get_rate,
            ('middle',): self._get_middle,
            ('tp',): self._get_tp,
            ('boll', 'boll_ub', 'boll_lb'): self._get_boll,
            ('macd', 'macds', 'macdh'): self._get_macd,
            ('ppo', 'ppos', 'ppoh'): self._get_ppo,
            ('kdjk', 'kdjd', 'kdjj'): self._get_kdj_default,
            ('cr', 'cr-ma1', 'cr-ma2', 'cr-ma3'): self._get_cr,
            ('cci',): self._get_cci,
            ('tr',): self._get_tr,
            ('atr',): self._get_atr,
            ('pdi', 'mdi', 'dx', 'adx', 'adxr'): self._get_dmi,
            ('trix',): self._get_trix,
            ('tema',): self._get_tema,
            ('vr',): self._get_vr,
            ('dma',): self._get_dma,
            ('vwma',): self._get_vwma,
            ('chop',): self._get_chop,
            ('log-ret',): self._get_log_ret,
            ('mfi',): self._get_mfi,
            ('wt1', 'wt2'): self._get_wave_trend,
            ('wr',): self._get_wr,
            ('supertrend',
             'supertrend_lb',
             'supertrend_ub'): self._get_supertrend,
            ('aroon',): self._get_aroon,
            ('ao',): self._get_ao,
            ('bop',): self._get_bop,
            ('cmo',): self._get_cmo,
            ('coppock',): self._get_coppock,
            ('ichimoku',): self._get_ichimoku,
            ('cti',): self._get_cti,
            ('ker',): self._get_ker,
        }

    def __init_not_exist_column(self, key):
        for names, handler in self.handler.items():
            if key in names:
                handler()
                return

        if key.endswith('_delta'):
            self._get_delta(key)
        elif self.is_cross_columns(key):
            self._get_cross(key)
        else:
            ret = self.parse_column_name(key)
            if len(ret) == 5:
                c, r, t, s, f = ret
                func_name = '_get_{}'.format(t)
                getattr(self, func_name)(c, r, s, f)
            elif len(ret) == 3:
                c, r, t = ret
                func_name = '_get_{}'.format(t)
                getattr(self, func_name)(c, r)
            elif len(ret) == 2:
                c, r = ret
                func_name = '_get_{}'.format(c)
                getattr(self, func_name)(r)
            else:
                raise UserWarning("Invalid number of return arguments "
                                  "after parsing column name: '{}'"
                                  .format(key))

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

    def copy(self, deep=True):
        return wrap(super(StockDataFrame, self).copy(deep))

    @staticmethod
    def _ensure_type(obj):
        """ override the method in pandas, omit the check

        This patch is not the perfect way but could make the lib work.
        """
        return obj

    @staticmethod
    def retype(value, index_column=None):
        """ if the input is a `DataFrame`, convert it to this class.

        :param index_column: name of the index column, default to `date`
        :param value: value to convert
        :return: this extended class
        """
        if index_column is None:
            index_column = 'date'

        if isinstance(value, StockDataFrame):
            return value
        elif isinstance(value, pd.DataFrame):
            name = value.columns.name
            # use all lower case for column name
            value.columns = map(lambda c: c.lower(), value.columns)

            if index_column in value.columns:
                value.set_index(index_column, inplace=True)
            ret = StockDataFrame(value)
            ret.columns.name = name
            return ret
        return value
