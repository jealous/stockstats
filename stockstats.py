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


def wrap(df, index_column=None):
    """ wraps a pandas DataFrame to StockDataFrame

    :param df: pandas DataFrame
    :param index_column: the name of the index column, default to ``date``
    :return: an object of StockDataFrame
    """
    return StockDataFrame.retype(df, index_column)


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

    VWMA = 14

    CHOP = 14

    MFI = 14

    CCI = 14

    RSI = 14

    VR = 26

    WR = 14

    WAVE_TREND_1 = 10
    WAVE_TREND_2 = 21

    KAMA_SLOW = 34
    KAMA_FAST = 5

    MULTI_SPLIT_INDICATORS = ("kama",)

    # End of options

    @classmethod
    def _change(cls, series, window):
        return series.pct_change(periods=-window).fillna(0.0) * 100

    @classmethod
    def _get_change(cls, df):
        """ Get the percentage change column

        :param df: DataFrame object
        :return: result series
        """
        df['change'] = cls._change(df['close'], -1)

    @classmethod
    def _get_p(cls, df, column, shifts):
        """ get the permutation of specified range

        example:
        index    x   x_-2,-1_p
        0        1         NaN
        1       -1         NaN
        2        3           2  (0.x > 0, and assigned to weight 2)
        3        5           1  (2.x > 0, and assigned to weight 1)
        4        1           3

        :param df: data frame
        :param column: the column to calculate p from
        :param shifts: the range to consider
        :return:
        """
        column_name = '{}_{}_p'.format(column, shifts)
        # initialize the column if not
        df.get(column)
        shifts = cls.to_ints(shifts)[::-1]
        indices = None
        count = 0
        for shift in shifts:
            shifted = df.shift(-shift)
            index = (shifted[column] > 0) * (2 ** count)
            if indices is None:
                indices = index
            else:
                indices += index
            count += 1
        if indices is not None:
            cp = indices.copy()
            cls.set_nan(cp, shifts)
            df[column_name] = cp

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

    @classmethod
    def _get_r(cls, df, column, shifts):
        """ Get rate of change of column

        :param df: DataFrame object
        :param column: column name of the rate to calculate
        :param shifts: periods to shift, accept one shift only
        :return: None
        """
        shift = cls.to_int(shifts)
        rate_key = '{}_{}_r'.format(column, shift)
        df[rate_key] = cls._change(df[column], shift)

    @classmethod
    def _shift(cls, series, window):
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

    @classmethod
    def _get_s(cls, df, column, shifts):
        """ Get the column shifted by periods

        :param df: DataFrame object
        :param column: name of the column to shift
        :param shifts: periods to shift, accept one shift only
        :return: None
        """
        shift = cls.to_int(shifts)
        shifted_key = "{}_{}_s".format(column, shift)
        df[shifted_key] = cls._shift(df[column], shift)

    @classmethod
    def _get_log_ret(cls, df):
        close = df['close']
        df['log-ret'] = np.log(close / cls._shift(close, -1))

    @classmethod
    def _get_c(cls, df, column, shifts):
        """ get the count of column in range (shifts)

        example: change_20_c
        :param df: stock data
        :param column: column name
        :param shifts: range to count, only to previous
        :return: result series
        """
        column_name = '{}_{}_c'.format(column, shifts)
        shifts = cls.get_int_positive(shifts)
        df[column_name] = df[column].rolling(
            center=False,
            window=shifts,
            min_periods=0).apply(np.count_nonzero)
        return df[column_name]

    @classmethod
    def _get_fc(cls, df, column, shifts):
        """ get the count of column in range of future (shifts)

        example: change_20_fc
        :param df: stock data
        :param column: column name
        :param shifts: range to count, only to future
        :return: result series
        """
        column_name = '{}_{}_fc'.format(column, shifts)
        shift = cls.get_int_positive(shifts)
        reversed_series = df[column][::-1]
        reversed_counts = reversed_series.rolling(
            center=False,
            window=shift,
            min_periods=0).apply(np.count_nonzero)
        counts = reversed_counts[::-1]
        df[column_name] = counts
        return counts

    @classmethod
    def _init_shifted_columns(cls, column, df, shifts):
        # initialize the column if not
        df.get(column)
        shifts = cls.to_ints(shifts)
        shift_column_names = ['{}_{}_s'.format(column, shift) for shift in
                              shifts]
        [df.get(name) for name in shift_column_names]
        return shift_column_names

    @classmethod
    def _get_max(cls, df, column, shifts):
        column_name = '{}_{}_max'.format(column, shifts)
        shift_column_names = cls._init_shifted_columns(column, df, shifts)
        df[column_name] = np.max(df[shift_column_names], axis=1)

    @classmethod
    def _get_min(cls, df, column, shifts):
        column_name = '{}_{}_min'.format(column, shifts)
        shift_column_names = cls._init_shifted_columns(column, df, shifts)
        df[column_name] = np.min(df[shift_column_names], axis=1)

    @classmethod
    def _get_rsv(cls, df, window):
        """ Calculate the RSV (Raw Stochastic Value) within N periods

        This value is essential for calculating KDJs
        Current day is included in N
        :param df: data
        :param window: number of periods
        :return: None
        """
        window = cls.get_int_positive(window)
        column_name = 'rsv_{}'.format(window)
        low_min = cls._mov_min(df['low'], window)
        high_max = cls._mov_max(df['high'], window)

        cv = (df['close'] - low_min) / (high_max - low_min)
        df[column_name] = cv.fillna(0.0) * 100

    @classmethod
    def _get_rsi(cls, df, window=None):
        """ Calculate the RSI (Relative Strength Index) within N periods

        calculated based on the formula at:
        https://en.wikipedia.org/wiki/Relative_strength_index

        :param df: data
        :param window: number of periods
        :return: None
        """
        if window is None:
            window = cls.RSI
            column_name = 'rsi'
        else:
            column_name = 'rsi_{}'.format(window)
        window = cls.get_int_positive(window)

        change = cls._delta(df['close'], -1)
        close_pm = (change + change.abs()) / 2
        close_nm = (-change + change.abs()) / 2
        p_ema = cls._smma(close_pm, window)
        n_ema = cls._smma(close_nm, window)

        rs_column_name = 'rs_{}'.format(window)
        df[rs_column_name] = rs = p_ema / n_ema
        df[column_name] = 100 - 100 / (1.0 + rs)

    @classmethod
    def _get_stochrsi(cls, df, window=None):
        """ Calculate the Stochastic RSI

        calculated based on the formula at:
        https://www.investopedia.com/terms/s/stochrsi.asp

        :param df: data
        :param window: number of periods
        :return: None
        """
        if window is None:
            window = cls.RSI
            column_name = 'stochrsi'
        else:
            column_name = 'stochrsi_{}'.format(window)
        window = cls.get_int_positive(window)

        rsi = df['rsi_{}'.format(window)]
        rsi_min = cls._mov_min(rsi, window)
        rsi_max = cls._mov_max(rsi, window)

        cv = (rsi - rsi_min) / (rsi_max - rsi_min)
        df[column_name] = cv * 100

    @classmethod
    def _get_wave_trend(cls, df):
        """ Calculate LazyBear's Wavetrend
        Check the algorithm described below:
        https://medium.com/@samuel.mcculloch/lets-take-a-look-at-wavetrend-with-crosses-lazybear-s-indicator-2ece1737f72f

        n1: period of EMA on typical price
        n2: period of EMA

        :param df: data frame
        :return: None
        """
        n1 = cls.WAVE_TREND_1
        n2 = cls.WAVE_TREND_2

        tp = cls._tp(df)
        esa = cls._ema(tp, n1)
        d = cls._ema((tp - esa).abs(), n1)
        ci = (tp - esa) / (0.015 * d)
        tci = cls._ema(ci, n2)
        df["wt1"] = tci
        df["wt2"] = cls._sma(tci, 4)

    @classmethod
    def _smma(cls, series, window):
        return series.ewm(
            ignore_na=False,
            alpha=1.0 / window,
            min_periods=0,
            adjust=True).mean()

    @classmethod
    def _get_smma(cls, df, column, windows):
        """ get smoothed moving average.

        :param df: data
        :param column: the column to calculate
        :param windows: range
        :return: result series
        """
        window = cls.get_int_positive(windows)
        column_name = '{}_{}_smma'.format(column, window)
        df[column_name] = cls._smma(df[column], window)

    @classmethod
    def _get_trix(cls, df, column=None, windows=None):
        """ Triple Exponential Average

        https://www.investopedia.com/articles/technical/02/092402.asp
        :param df: data
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
            windows = cls.TRIX_EMA_WINDOW
        if column_name == "":
            column_name = '{}_{}_trix'.format(column, windows)

        window = cls.get_int_positive(windows)

        single = cls._ema(df[column], window)
        double = cls._ema(single, window)
        triple = cls._ema(double, window)
        prev_triple = cls._shift(triple, -1)
        triple_change = cls._delta(triple, -1)
        df[column_name] = triple_change * 100 / prev_triple

    @classmethod
    def _get_tema(cls, df, column=None, windows=None):
        """ Another implementation for triple ema

        Check the algorithm described below:
        https://www.forextraders.com/forex-education/forex-technical-analysis/triple-exponential-moving-average-the-tema-indicator/
        :param df: data frame
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
            windows = cls.TEMA_EMA_WINDOW
        if column_name == "":
            column_name = '{}_{}_tema'.format(column, windows)

        window = cls.get_int_positive(windows)

        single = cls._ema(df[column], window)
        double = cls._ema(single, window)
        triple = cls._ema(double, window)
        df[column_name] = 3 * single - 3 * double + triple

    @classmethod
    def _get_wr(cls, df, window=None):
        """ Williams Overbought/Oversold Index

        Definition: https://www.investopedia.com/terms/w/williamsr.asp
        WMS=[(Hn—Ct)/(Hn—Ln)] × -100
        Ct - the close price
        Hn - N periods high
        Ln - N periods low

        :param df: data
        :param window: number of periods
        :return: None
        """
        if window is None:
            window = cls.WR
            column_name = 'wr'
        else:
            column_name = 'wr_{}'.format(window)

        window = cls.get_int_positive(window)
        ln = cls._mov_min(df['low'], window)

        hn = cls._mov_max(df['high'], window)
        df[column_name] = (hn - df['close']) / (hn - ln) * -100

    @classmethod
    def _get_cci(cls, df, window=None):
        """ Commodity Channel Index

        CCI = (Typical Price  -  20-period SMA of TP) / (.015 x Mean Deviation)
        Typical Price (TP) = (High + Low + Close)/3
        TP is also implemented as 'middle'.

        :param df: data
        :param window: number of periods
        :return: None
        """
        if window is None:
            window = cls.CCI
            column_name = 'cci'
        else:
            column_name = 'cci_{}'.format(window)
        window = cls.get_int_positive(window)

        tp = cls._tp(df)
        tp_sma = cls._sma(tp, window)
        rolling = tp.rolling(min_periods=1, center=False, window=window)
        md = rolling.apply(lambda x: np.fabs(x - x.mean()).mean())

        df[column_name] = (tp - tp_sma) / (.015 * md)

    @classmethod
    def _tr(cls, df):
        prev_close = cls._shift(df['close'], -1)
        high = df['high']
        low = df['low']
        c1 = high - low
        c2 = (high - prev_close).abs()
        c3 = (low - prev_close).abs()
        return pd.concat((c1, c2, c3), axis=1).max(axis=1)

    @classmethod
    def _get_tr(cls, df):
        """ True Range of the trading

         TR is a measure of volatility of a High-Low-Close series

        tr = max[(high - low), abs(high - close_prev), abs(low - close_prev)]
        :param df: data
        :return: None
        """
        df['tr'] = cls._tr(df)

    @classmethod
    def _atr(cls, df, window):
        tr = cls._tr(df)
        return cls._smma(tr, window)

    @classmethod
    def _get_atr(cls, df, window=None):
        """ Average True Range

        The average true range is an N-day smoothed moving average (SMMA) of
        the true range values.  Default to 14 periods.
        https://en.wikipedia.org/wiki/Average_true_range

        :param df: data
        :return: None
        """
        if window is None:
            window = cls.ATR_SMMA
            column_name = 'atr'
        else:
            column_name = 'atr_{}'.format(window)
        window = cls.get_int_positive(window)
        df[column_name] = cls._atr(df, window)

    @classmethod
    def _get_dma(cls, df):
        """ Difference of Moving Average

        default to 10 and 50.
        :param df: data
        :return: None
        """
        df['dma'] = df['close_10_sma'] - df['close_50_sma']

    @classmethod
    def _get_dmi(cls, df):
        """ get the default setting for DMI

        including:
        +DI: 14 periods SMMA of +DM,
        -DI: 14 periods SMMA of -DM,
        DX: based on +DI and -DI
        ADX: 6 periods SMMA of DX
        :param df: data
        :return:
        """
        df['pdi'] = cls._get_pdi(df, cls.PDI_SMMA)
        df['mdi'] = cls._get_mdi(df, cls.MDI_SMMA)
        df['dx'] = cls._get_dx(df, cls.DX_SMMA)
        df['adx'] = cls._ema(df['dx'], cls.ADX_EMA)
        df['adxr'] = cls._ema(df['adx'], cls.ADXR_EMA)

    @classmethod
    def _get_um_dm(cls, df):
        """ Up move and down move

        initialize up move and down move
        :param df: data
        """
        hd = df['high_delta']
        df['um'] = (hd + hd.abs()) / 2
        ld = -df['low_delta']
        df['dm'] = (ld + ld.abs()) / 2

    @classmethod
    def _get_pdm(cls, df, windows):
        """ +DM, positive directional moving

        If window is not 1, calculate the SMMA of +DM
        :param df: data
        :param windows: range
        :return:
        """
        window = cls.get_int_positive(windows)
        column_name = 'pdm_{}'.format(window)
        um, dm = df['um'], df['dm']
        df['pdm'] = np.where(um > dm, um, 0)
        if window > 1:
            pdm = df['pdm_{}_ema'.format(window)]
        else:
            pdm = df['pdm']
        df[column_name] = pdm

    @classmethod
    def _get_vr(cls, df, windows=None):
        if windows is None:
            window = cls.VR
            column_name = 'vr'
        else:
            window = cls.get_int_positive(windows)
            column_name = 'vr_{}'.format(window)

        idx = df.index
        av = pd.Series(np.where(df['change'] > 0, df['volume'], 0), index=idx)
        avs = cls._mov_sum(av, window)

        bv = pd.Series(np.where(df['change'] < 0, df['volume'], 0), index=idx)
        bvs = cls._mov_sum(bv, window)

        cv = pd.Series(np.where(df['change'] == 0, df['volume'], 0), index=idx)
        cvs = cls._mov_sum(cv, window)

        df[column_name] = (avs + cvs / 2) / (bvs + cvs / 2) * 100

    @classmethod
    def _get_mdm(cls, df, windows):
        """ -DM, negative directional moving accumulation

        If window is not 1, return the SMA of -DM.
        :param df: data
        :param windows: range
        :return:
        """
        window = cls.get_int_positive(windows)
        column_name = 'mdm_{}'.format(window)
        um, dm = df['um'], df['dm']
        df['mdm'] = np.where(dm > um, dm, 0)
        if window > 1:
            mdm = df['mdm_{}_ema'.format(window)]
        else:
            mdm = df['mdm']
        df[column_name] = mdm

    @classmethod
    def _get_pdi(cls, df, windows):
        """ +DI, positive directional moving index

        :param df: data
        :param windows: range
        :return:
        """
        window = cls.get_int_positive(windows)
        pdm_column = 'pdm_{}'.format(window)
        tr_column = 'atr_{}'.format(window)
        pdi_column = 'pdi_{}'.format(window)
        df[pdi_column] = df[pdm_column] / df[tr_column] * 100
        return df[pdi_column]

    @classmethod
    def _get_mdi(cls, df, windows):
        window = cls.get_int_positive(windows)
        mdm_column = 'mdm_{}'.format(window)
        tr_column = 'atr_{}'.format(window)
        mdi_column = 'mdi_{}'.format(window)
        df[mdi_column] = df[mdm_column] / df[tr_column] * 100
        return df[mdi_column]

    @classmethod
    def _get_dx(cls, df, windows):
        window = cls.get_int_positive(windows)
        dx_column = 'dx_{}'.format(window)
        mdi_column = 'mdi_{}'.format(window)
        pdi_column = 'pdi_{}'.format(window)
        mdi, pdi = df[mdi_column], df[pdi_column]
        df[dx_column] = abs(pdi - mdi) / (pdi + mdi) * 100
        return df[dx_column]

    @classmethod
    def _get_kdj_default(cls, df):
        """ default KDJ, 9 periods

        :param df: k line data frame
        :return: None
        """
        df['kdjk'] = df['kdjk_{}'.format(cls.KDJ_WINDOW)]
        df['kdjd'] = df['kdjd_{}'.format(cls.KDJ_WINDOW)]
        df['kdjj'] = df['kdjj_{}'.format(cls.KDJ_WINDOW)]

    @classmethod
    def _get_cr(cls, df, window=26):
        """ Energy Index (Intermediate Willingness Index)

        https://support.futunn.com/en/topic167/?lang=en-us
        Use the relationship between the highest price, the lowest price and
        yesterday's middle price to reflect the market's willingness to buy
        and sell.

        :param df: data
        :param window: window of the moving sum
        :return: None
        """
        middle = cls._tp(df)
        last_middle = cls._shift(middle, -1)
        ym = cls._shift(middle, -1)
        high = df['high']
        low = df['low']
        p1_m = pd.concat((last_middle, high), axis=1).min(axis=1)
        p2_m = pd.concat((last_middle, low), axis=1).min(axis=1)
        p1 = cls._mov_sum(high - p1_m, window)
        p2 = cls._mov_sum(ym - p2_m, window)
        df['cr'] = cr = p1 / p2 * 100
        df['cr-ma1'] = cls._shifted_cr_sma(cr, cls.CR_MA1)
        df['cr-ma2'] = cls._shifted_cr_sma(cr, cls.CR_MA2)
        df['cr-ma3'] = cls._shifted_cr_sma(cr, cls.CR_MA3)

    @classmethod
    def _shifted_cr_sma(cls, cr, window):
        cr_sma = cls._sma(cr, window)
        return cls._shift(cr_sma, -int(window / 2.5 + 1))

    @classmethod
    def _tp(cls, df):
        return (df['close'] + df['high'] + df['low']).divide(3.0)

    @classmethod
    def _get_tp(cls, df):
        df['tp'] = cls._tp(df)

    @classmethod
    def _get_middle(cls, df):
        df['middle'] = cls._tp(df)

    @classmethod
    def _calc_kd(cls, column):
        param0, param1 = cls.KDJ_PARAM
        k = 50.0
        # noinspection PyTypeChecker
        for i in param1 * column:
            k = param0 * k + i
            yield k

    @classmethod
    def _get_kdjk(cls, df, window):
        """ Get the K of KDJ

        K ＝ 2/3 × (prev. K) +1/3 × (curr. RSV)
        2/3 and 1/3 are the smooth parameters.
        :param df: data
        :param window: number of periods
        :return: None
        """
        rsv_column = 'rsv_{}'.format(window)
        k_column = 'kdjk_{}'.format(window)
        df[k_column] = list(cls._calc_kd(df.get(rsv_column)))

    @classmethod
    def _get_kdjd(cls, df, window):
        """ Get the D of KDJ

        D = 2/3 × (prev. D) +1/3 × (curr. K)
        2/3 and 1/3 are the smooth parameters.
        :param df: data
        :param window: number of periods
        :return: None
        """
        k_column = 'kdjk_{}'.format(window)
        d_column = 'kdjd_{}'.format(window)
        df[d_column] = list(cls._calc_kd(df.get(k_column)))

    @staticmethod
    def _get_kdjj(df, window):
        """ Get the J of KDJ

        J = 3K-2D
        :param df: data
        :param window: number of periods
        :return: None
        """
        k_column = 'kdjk_{}'.format(window)
        d_column = 'kdjd_{}'.format(window)
        j_column = 'kdjj_{}'.format(window)
        df[j_column] = 3 * df[k_column] - 2 * df[d_column]

    @classmethod
    def _delta(cls, series, window):
        return series.diff(-window).fillna(0.0)

    @classmethod
    def _get_d(cls, df, column, shifts):
        shift = StockDataFrame.to_int(shifts)
        column_name = '{}_{}_d'.format(column, shift)
        df[column_name] = cls._delta(df[column], shift)

    @staticmethod
    def _mov_min(series, size):
        return series.rolling(min_periods=1, window=size, center=False).min()

    @staticmethod
    def _mov_max(series, size):
        return series.rolling(min_periods=1, window=size, center=False).max()

    @staticmethod
    def _mov_sum(series, size):
        return series.rolling(min_periods=1, window=size, center=False).sum()

    @staticmethod
    def _sma(series, size):
        return series.rolling(min_periods=1, window=size, center=False).mean()

    @classmethod
    def _get_sma(cls, df, column, windows):
        """ get simple moving average

        :param df: data
        :param column: column to calculate
        :param windows: collection of window of simple moving average
        :return: None
        """
        window = cls.get_int_positive(windows)
        column_name = '{}_{}_sma'.format(column, window)
        df[column_name] = cls._sma(df[column], window)

    @classmethod
    def _ema(cls, series, window):
        return series.ewm(
            ignore_na=False,
            span=window,
            min_periods=0,
            adjust=True).mean()

    @classmethod
    def _get_ema(cls, df, column, windows):
        """ get exponential moving average

        :param df: data
        :param column: column to calculate
        :param windows: collection of window of exponential moving average
        :return: None
        """
        window = cls.get_int_positive(windows)
        column_name = '{}_{}_ema'.format(column, window)
        df[column_name] = cls._ema(df[column], window)

    @classmethod
    def _get_boll(cls, df):
        """ Get Bollinger bands.

        boll_ub means the upper band of the Bollinger bands
        boll_lb means the lower band of the Bollinger bands
        boll_ub = MA + Kσ
        boll_lb = MA − Kσ
        M = BOLL_PERIOD
        K = BOLL_STD_TIMES
        :param df: data
        :return: None
        """
        moving_avg = cls._sma(df['close'], cls.BOLL_PERIOD)
        moving_std = cls._mstd(df['close'], cls.BOLL_PERIOD)
        df['boll'] = moving_avg
        width = cls.BOLL_STD_TIMES * moving_std
        df['boll_ub'] = moving_avg + width
        df['boll_lb'] = moving_avg - width

    @classmethod
    def _get_macd(cls, df):
        """ Moving Average Convergence Divergence

        This function will initialize all following columns.

        MACD Line (macd): (12-day EMA - 26-day EMA)
        Signal Line (macds): 9-day EMA of MACD Line
        MACD Histogram (macdh): MACD Line - Signal Line
        :param df: data
        :return: None
        """
        close = df['close']
        ema_short = cls._ema(close, cls.MACD_EMA_SHORT)
        ema_long = cls._ema(close, cls.MACD_EMA_LONG)
        df['macd'] = ema_short - ema_long
        df['macds'] = cls._ema(df['macd'], cls.MACD_EMA_SIGNAL)
        df['macdh'] = df['macd'] - df['macds']

    @classmethod
    def _get_ppo(cls, df):
        """ Percentage Price Oscillator

        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:price_oscillators_ppo

        Percentage Price Oscillator (PPO):
            {(12-day EMA - 26-day EMA)/26-day EMA} x 100

        Signal Line: 9-day EMA of PPO

        PPO Histogram: PPO - Signal Line

        :param df: data
        :return: None
        """
        close = df['close']
        ppo_short = cls._ema(close, cls.PPO_EMA_SHORT)
        ppo_long = cls._ema(close, cls.PPO_EMA_LONG)
        df['ppo'] = (ppo_short - ppo_long) / ppo_long * 100
        df['ppos'] = cls._ema(df['ppo'], cls.PPO_EMA_SIGNAL)
        df['ppoh'] = df['ppo'] - df['ppos']

    @classmethod
    def get_int_positive(cls, windows):
        if isinstance(windows, int):
            window = windows
        else:
            window = cls.to_int(windows)
            if window <= 0:
                raise IndexError("window must be greater than 0")
        return window

    @classmethod
    def _mstd(cls, series, window):
        return series.rolling(min_periods=1, window=window, center=False).std()

    @classmethod
    def _get_mstd(cls, df, column, windows):
        """ get moving standard deviation

        :param df: data
        :param column: column to calculate
        :param windows: collection of window of moving standard deviation
        :return: None
        """
        window = cls.get_int_positive(windows)
        column_name = '{}_{}_mstd'.format(column, window)
        df[column_name] = cls._mstd(df[column], window)

    @classmethod
    def _get_mvar(cls, df, column, windows):
        """ get moving variance

        :param df: data
        :param column: column to calculate
        :param windows: collection of window of moving variance
        :return: None
        """
        window = cls.get_int_positive(windows)
        column_name = '{}_{}_mvar'.format(column, window)
        df[column_name] = df[column].rolling(
            min_periods=1, window=window, center=False).var()

    @classmethod
    def _get_vwma(cls, df, window=None):
        """ get Volume Weighted Moving Average

        The definition is available at:
        https://www.investopedia.com/articles/trading/11/trading-with-vwap-mvwap.asp

        :param df: data
        :param window: number of periods relevant for the indicator
        :return: None
        """
        if window is None:
            window = cls.VWMA
            column_name = 'vwma'
        else:
            column_name = 'vwma_{}'.format(window)
        window = cls.get_int_positive(window)

        tpv = df['volume'] * cls._tp(df)
        rolling_tpv = cls._mov_sum(tpv, window)
        rolling_vol = cls._mov_sum(df['volume'], window)
        df[column_name] = rolling_tpv / rolling_vol

    @classmethod
    def _get_chop(cls, df, window=None):
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

        :param df: data
        :param window: number of periods relevant for the indicator
        :return: None
        """
        if window is None:
            window = cls.CHOP
            column_name = 'chop'
        else:
            column_name = 'chop_{}'.format(window)
        window = cls.get_int_positive(window)
        atr = cls._atr(df, 1)
        atr_sum = cls._mov_sum(atr, window)
        high = cls._mov_max(df['high'], window)
        low = cls._mov_min(df['low'], window)
        choppy = atr_sum / (high - low)
        numerator = np.log10(choppy) * 100
        denominator = np.log10(window)
        df[column_name] = numerator / denominator

    @classmethod
    def _get_mfi(cls, df, window=None):
        """ get money flow index

        The definition of money flow index is available at:
        https://www.investopedia.com/terms/m/mfi.asp

        :param df: data
        :param window: number of periods relevant for the indicator
        :return: None
        """
        if window is None:
            window = cls.MFI
            column_name = 'mfi'
        else:
            column_name = 'mfi_{}'.format(window)
        window = cls.get_int_positive(window)
        middle = cls._tp(df)
        money_flow = (middle * df["volume"]).fillna(0.0)
        shifted = cls._shift(middle, -1)
        delta = (middle - shifted).fillna(0)
        pos_flow = money_flow.mask(delta < 0, 0)
        neg_flow = money_flow.mask(delta >= 0, 0)
        rolling_pos_flow = cls._mov_sum(pos_flow, window)
        rolling_neg_flow = cls._mov_sum(neg_flow, window)
        money_flow_ratio = rolling_pos_flow / (rolling_neg_flow + 1e-12)
        mfi = (1.0 - 1.0 / (1 + money_flow_ratio))
        mfi.iloc[:window] = 0.5
        df[column_name] = mfi

    @classmethod
    def _get_kama(cls, df, column, windows, fasts=None, slows=None):
        """ get Kaufman's Adaptive Moving Average.
        Implemented after
        https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average

        :param df: data
        :param column: column to calculate
        :param windows: collection of window of exponential moving average
        :param fasts: fastest EMA constant
        :param slows: slowest EMA constant
        :return: None
        """
        window = cls.get_int_positive(windows)
        if slows is None or fasts is None:
            slow, fast = cls.KAMA_SLOW, cls.KAMA_FAST
            column_name = "{}_{}_kama".format(column, window)
        else:
            slow = cls.get_int_positive(slows)
            fast = cls.get_int_positive(fasts)
            column_name = '{}_{}_kama_{}_{}'.format(column, window, fast, slow)

        col = df[column]
        col_window_s = cls._shift(col, -window)
        col_last = cls._shift(col, -1)
        change = (col - col_window_s).abs()
        volatility = cls._mov_sum((col - col_last).abs(), window)
        efficiency_ratio = change / volatility
        fast_ema_smoothing = 2.0 / (fast + 1)
        slow_ema_smoothing = 2.0 / (slow + 1)
        smoothing_2 = fast_ema_smoothing - slow_ema_smoothing
        efficient_smoothing = efficiency_ratio * smoothing_2
        smoothing = 2 * (efficient_smoothing + slow_ema_smoothing)

        # start with simple moving average
        kama = cls._sma(col, window)
        last_kama = kama.iloc[window - 1]
        for i in range(window, len(kama)):
            cur = smoothing.iloc[i] * (col.iloc[i] - last_kama) + last_kama
            kama.iloc[i] = cur
            last_kama = cur
        df[column_name] = kama

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

    @staticmethod
    def _get_rate(df):
        """ same as percent

        :param df: data frame
        :return: None
        """
        df['rate'] = df['close'].pct_change() * 100

    @staticmethod
    def _get_delta(df, key):
        key_to_delta = key.replace('_delta', '')
        df[key] = df[key_to_delta].diff()
        return df[key]

    @staticmethod
    def _get_cross(df, key):
        left, op, right = StockDataFrame.parse_cross_column(key)
        lt_series = df[left] > df[right]
        # noinspection PyTypeChecker
        different = np.zeros_like(lt_series)
        if len(different) > 1:
            # noinspection PyTypeChecker
            different[1:] = np.diff(lt_series)
            different[0] = False
        if op == 'x':
            df[key] = different
        elif op == 'xu':
            df[key] = different & lt_series
        elif op == 'xd':
            df[key] = different & ~lt_series
        return df[key]

    @classmethod
    def __init_not_exist_column(cls, df, key):
        handlers = {
            ('change',): cls._get_change,
            ('rsi',): cls._get_rsi,
            ('stochrsi',): cls._get_stochrsi,
            ('rate',): cls._get_rate,
            ('middle',): cls._get_middle,
            ('tp',): cls._get_tp,
            ('boll', 'boll_ub', 'boll_lb'): cls._get_boll,
            ('macd', 'macds', 'macdh'): cls._get_macd,
            ('ppo', 'ppos', 'ppoh'): cls._get_ppo,
            ('kdjk', 'kdjd', 'kdjj'): cls._get_kdj_default,
            ('cr', 'cr-ma1', 'cr-ma2', 'cr-ma3'): cls._get_cr,
            ('cci',): cls._get_cci,
            ('tr',): cls._get_tr,
            ('atr',): cls._get_atr,
            ('um', 'dm'): cls._get_um_dm,
            ('pdi', 'mdi', 'dx', 'adx', 'adxr'): cls._get_dmi,
            ('trix',): cls._get_trix,
            ('tema',): cls._get_tema,
            ('vr',): cls._get_vr,
            ('dma',): cls._get_dma,
            ('vwma',): cls._get_vwma,
            ('chop',): cls._get_chop,
            ('log-ret',): cls._get_log_ret,
            ('mfi',): cls._get_mfi,
            ('wt1', 'wt2'): cls._get_wave_trend,
            ('wr',): cls._get_wr,
        }
        for names, handler in handlers.items():
            if key in names:
                handler(df)
                return

        if key.endswith('_delta'):
            cls._get_delta(df, key)
        elif cls.is_cross_columns(key):
            cls._get_cross(df, key)
        elif key == 'mfi':
            cls._get_mfi(df)
        else:
            ret = cls.parse_column_name(key)
            if len(ret) == 5:
                c, r, t, s, f = ret
                func_name = '_get_{}'.format(t)
                getattr(cls, func_name)(df, c, r, s, f)
            elif len(ret) == 3:
                c, r, t = ret
                func_name = '_get_{}'.format(t)
                getattr(cls, func_name)(df, c, r)
            elif len(ret) == 2:
                c, r = ret
                func_name = '_get_{}'.format(c)
                getattr(cls, func_name)(df, r)
            else:
                raise UserWarning("Invalid number of return arguments "
                                  "after parsing column name: '{}'"
                                  .format(key))

    @staticmethod
    def __init_column(df, key):
        if key not in df:
            if len(df) == 0:
                df[key] = []
            else:
                StockDataFrame.__init_not_exist_column(df, key)

    def __getitem__(self, item):
        try:
            result = wrap(super(StockDataFrame, self).__getitem__(item))
        except KeyError:
            try:
                if isinstance(item, list):
                    for column in item:
                        self.__init_column(self, column)
                else:
                    self.__init_column(self, item)
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

    def _ensure_type(self, obj):
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
            # use all lower case for column name
            value.columns = map(lambda c: c.lower(), value.columns)

            if index_column in value.columns:
                value.set_index(index_column, inplace=True)
            return StockDataFrame(value)
        return value
