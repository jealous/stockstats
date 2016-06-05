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
import logging
import operator
import random
import re

import numpy as np
import pandas as pd
from int_date import get_date_from_diff

__author__ = 'Cedric Zhuang'

log = logging.getLogger(__name__)


class StockDataFrame(pd.DataFrame):
    OPERATORS = ['le', 'ge', 'lt', 'gt', 'eq', 'ne']

    KDJ_PARAM = (2.0 / 3.0, 1.0 / 3.0)

    BOLL_PERIOD = 20
    BOLL_STD_TIMES = 2

    @staticmethod
    def _get_change(df):
        df['change'] = df['close'].pct_change() * 100
        return df['change']

    @staticmethod
    def _get_p(df, column, shifts):
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
        shifts = StockDataFrame.to_ints(shifts)[::-1]
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
        StockDataFrame.set_nan(indices, shifts)
        df[column_name] = indices

    @classmethod
    def to_ints(cls, shifts):
        items = map(cls._process_shifts_segment,
                    shifts.split(','))
        return sorted(list(set(itertools.chain(*items))))

    @classmethod
    def to_int(cls, shifts):
        numbers = cls.to_ints(shifts)
        if len(numbers) != 1:
            raise IndexError("only accept 1 number.")
        return numbers[0]

    @staticmethod
    def to_floats(shifts):
        floats = map(float, shifts.split(','))
        return sorted(list(set(floats)))

    @classmethod
    def to_float(cls, shifts):
        floats = cls.to_floats(shifts)
        if len(floats) != 1:
            raise IndexError('only accept 1 float.')
        return floats[0]

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

    @staticmethod
    def _get_r(df, column, shifts):
        """ Get rate of change of column

        :param df: DataFrame object
        :param column: column name of the rate to calculate
        :param shifts: days to shift, accept one shift only
        :return: None
        """
        shift = StockDataFrame.to_int(shifts)
        rate_key = '{}_{}_r'.format(column, shift)
        df[rate_key] = df[column].pct_change(periods=-shift) * 100

    @classmethod
    def _get_s(cls, df, column, shifts):
        """ Get the column shifted by days

        :param df: DataFrame object
        :param column: name of the column to shift
        :param shifts: days to shift, accept one shift only
        :return: None
        """
        shift = cls.to_int(shifts)
        shifted_key = "{}_{}_s".format(column, shift)
        df[shifted_key] = df[column].shift(-shift)
        StockDataFrame.set_nan(df[shifted_key], shift)

    @classmethod
    def _get_log_ret(cls, df):
        df['log-ret'] = np.log(df['close'] / df['close_-1_s'])

    @classmethod
    def _get_c(cls, df, column, shifts):
        """ get the count of column in range (shifts)

        example: kdjj_0_le_20_c
        :param df: stock data
        :param column: column name
        :param shifts: range to count, only to previous
        :return:
        """
        column_name = '{}_{}_{}'.format(column, shifts, 'c')
        shifts = abs(cls.to_int(shifts))
        df[column_name] = df[column].rolling(
            center=False, window=shifts).apply(np.count_nonzero)

    @classmethod
    def _get_op(cls, df, column, threshold, op):
        column_name = '{}_{}_{}'.format(column, threshold, op)
        threshold = cls.to_float(threshold)
        f = getattr(operator, op)
        df[column_name] = f(df[column], threshold)

    @staticmethod
    def get_diff_convolve_array(shift):
        if shift == 0:
            ret = [1]
        else:
            ret = np.zeros(abs(shift) + 1)
            if shift < 0:
                ret[[0, -1]] = 1, -1
            else:
                ret[[0, -1]] = -1, 1
        return ret

    @staticmethod
    def _init_shifted_columns(column, df, shifts):
        # initialize the column if not
        df.get(column)
        shifts = StockDataFrame.to_ints(shifts)
        shift_column_names = ['{}_{}_s'.format(column, shift) for shift in
                              shifts]
        [df.get(name) for name in shift_column_names]
        return shift_column_names

    @staticmethod
    def _get_max(df, column, shifts):
        column_name = '{}_{}_max'.format(column, shifts)
        shift_column_names = StockDataFrame. \
            _init_shifted_columns(column, df, shifts)
        df[column_name] = np.max(df[shift_column_names], axis=1)

    @staticmethod
    def _get_min(df, column, shifts):
        column_name = '{}_{}_min'.format(column, shifts)
        shift_column_names = StockDataFrame. \
            _init_shifted_columns(column, df, shifts)
        df[column_name] = np.min(df[shift_column_names], axis=1)

    @staticmethod
    def _get_rsv(df, n_days):
        """ Calculate the RSV (Raw Stochastic Value) within N days

        This value is essential for calculating KDJs
        Current day is included in N
        :param df: data
        :param n_days: N days
        :return: None
        """
        n_days = int(n_days)
        column_name = 'rsv_{}'.format(n_days)
        low_min = df['low'].rolling(min_periods=1, window=n_days,
                                    center=False).min()
        high_max = df['high'].rolling(min_periods=1, window=n_days,
                                      center=False).max()

        df[column_name] = ((df['close'] - low_min) /
                           (high_max - low_min).astype('float64') * 100)

    @classmethod
    def _get_kdj_default(cls, df):
        """ default KDJ, 9 days

        :param df: k line data frame
        :return: None
        """
        df['kdjk'] = df['kdjk_9']
        df['kdjd'] = df['kdjd_9']
        df['kdjj'] = df['kdjj_9']

    @classmethod
    def _get_cr(cls, df, window=26):
        ym = df['middle_-1_s']
        h = df['high']
        p1_m = df.loc[:, ['middle_-1_s', 'high']].min(axis=1)
        p2_m = df.loc[:, ['middle_-1_s', 'low']].min(axis=1)
        p1 = (h - p1_m).rolling(
            min_periods=1, window=window, center=False).sum()
        p2 = (ym - p2_m).rolling(
            min_periods=1, window=window, center=False).sum()
        df['cr'] = p1 / p2 * 100
        del df['middle_-1_s']
        df['cr-ma1'] = cls._shifted_cr_sma(df, 5)
        df['cr-ma2'] = cls._shifted_cr_sma(df, 10)
        df['cr-ma3'] = cls._shifted_cr_sma(df, 20)

    @classmethod
    def _shifted_cr_sma(cls, df, window):
        name = cls._temp_name()
        df[name] = df['cr'].rolling(min_periods=1, window=window,
                                    center=False).mean()
        to_shift = '{}_-{}_s'.format(name, int(window / 2.5 + 1))
        ret = df[to_shift]
        del df[name], df[to_shift]
        return ret

    @classmethod
    def _temp_name(cls):
        return 'sdf{}'.format(random.randint(0, 10e8))

    @classmethod
    def _get_middle(cls, df):
        df['middle'] = (df['close'] + df['high'] + df['low']) / 3.0

    @classmethod
    def _get_kdjk(cls, df, n_days):
        """ Get the K of KDJ

        K ＝ 2/3 × (prev. K) +1/3 × (curr. RSV)
        2/3 and 1/3 are the smooth parameters.
        :param df: data
        :param n_days: calculation range
        :return: None
        """
        rsv_column = 'rsv_{}'.format(n_days)
        k_column = 'kdjk_{}'.format(n_days)
        df.get(rsv_column)
        prev_k = 50.0
        kdj_k = []
        for i in df.index:
            record = df.ix[i]
            prev_k = cls.KDJ_PARAM[0] * prev_k + cls.KDJ_PARAM[1] * record[
                rsv_column]
            kdj_k.append(prev_k)
        df[k_column] = kdj_k

    @classmethod
    def _get_kdjd(cls, df, n_days):
        """ Get the D of KDJ

        D = 2/3 × (prev. D) +1/3 × (curr. K)
        2/3 and 1/3 are the smooth parameters.
        :param df: data
        :param n_days: calculation range
        :return: None
        """
        k_column = 'kdjk_{}'.format(n_days)
        d_column = 'kdjd_{}'.format(n_days)
        df.get(k_column)
        prev_d = 50.0
        kdj_d = []
        for i in df.index:
            record = df.ix[i]
            prev_d = cls.KDJ_PARAM[0] * prev_d + cls.KDJ_PARAM[1] * record[
                k_column]
            kdj_d.append(prev_d)
        df[d_column] = kdj_d

    @staticmethod
    def _get_kdjj(df, n_days):
        """ Get the J of KDJ

        J = 3K-2D
        :param df: data
        :param n_days: calculation range
        :return: None
        """
        k_column = 'kdjk_{}'.format(n_days)
        d_column = 'kdjd_{}'.format(n_days)
        j_column = 'kdjj_{}'.format(n_days)
        df[j_column] = 3 * df[k_column] - 2 * df[d_column]

    @staticmethod
    def remove_random_nan(pd_obj):
        return pd_obj.where((pd.notnull(pd_obj)), None)

    @staticmethod
    def _get_d(df, column, shifts):
        shift = StockDataFrame.to_int(shifts)
        shift_column = '{}_{}_s'.format(column, shift)
        column_name = '{}_{}_d'.format(column, shift)
        df[column_name] = df[column] - df[shift_column]
        StockDataFrame.set_nan(df[column_name], shift)

    @staticmethod
    def _get_sma(df, column, windows):
        """ get simple moving average

        :param df: data
        :param column: column to calculate
        :param windows: collection of window of simple moving average
        :return: None
        """
        window = StockDataFrame.get_only_one_positive_int(windows)
        column_name = '{}_{}_sma'.format(column, window)
        df[column_name] = df[column].rolling(min_periods=1, window=window,
                                             center=False).mean()

    @classmethod
    def _get_ema(cls, df, column, windows):
        """ get exponential moving average

        :param df: data
        :param column: column to calculate
        :param windows: collection of window of exponential moving average
        :return: None
        """
        window = StockDataFrame.get_only_one_positive_int(windows)
        column_name = '{}_{}_ema'.format(column, window)
        if len(df[column]) > 0:
            df[column_name] = df[column].ewm(
                ignore_na=False, span=window,
                min_periods=0, adjust=True).mean()
        else:
            df[column_name] = []

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
        moving_avg = df['close_{}_sma'.format(cls.BOLL_PERIOD)]
        moving_std = df['close_{}_mstd'.format(cls.BOLL_PERIOD)]
        df['boll'] = moving_avg
        moving_avg = list(map(np.float64, moving_avg))
        moving_std = list(map(np.float64, moving_std))
        # noinspection PyTypeChecker
        df['boll_ub'] = np.add(moving_avg,
                               np.multiply(cls.BOLL_STD_TIMES, moving_std))
        # noinspection PyTypeChecker
        df['boll_lb'] = np.subtract(moving_avg,
                                    np.multiply(cls.BOLL_STD_TIMES,
                                                moving_std))

    @staticmethod
    def _get_macd(df):
        """ Moving Average Convergence Divergence

        This function will initialize all following columns

        MACD Line (macd): (12-day EMA - 26-day EMA)
        Signal Line (macds): 9-day EMA of MACD Line
        MACD Histogram (macdh): MACD Line - Signal Line
        :param df: data
        :return: None
        """
        fast = df['close_12_ema']
        slow = df['close_26_ema']
        df['macd'] = fast - slow
        df['macds'] = df['macd_9_ema']
        df['macdh'] = 2 * (df['macd'] - df['macds'])
        del df['macd_9_ema']
        del fast
        del slow

    @classmethod
    def get_only_one_positive_int(cls, windows):
        window = cls.to_int(windows)
        if window <= 0:
            raise IndexError("window must be greater than 0")
        return window

    @classmethod
    def _get_mstd(cls, df, column, windows):
        """ get moving standard deviation

        :param df: data
        :param column: column to calculate
        :param windows: collection of window of moving standard deviation
        :return: None
        """
        window = cls.get_only_one_positive_int(windows)
        column_name = '{}_{}_mstd'.format(column, window)
        df[column_name] = df[column].rolling(min_periods=1, window=window,
                                             center=False).std()

    @staticmethod
    def _get_mvar(df, column, windows):
        """ get moving variance

        :param df: data
        :param column: column to calculate
        :param windows: collection of window of moving variance
        :return: None
        """
        window = StockDataFrame.get_only_one_positive_int(windows)
        column_name = '{}_{}_mvar'.format(column, window)
        df[column_name] = df[column].rolling(min_periods=1, window=window,
                                             center=False).var()

    @staticmethod
    def parse_column_name(name):
        m = re.match('(.*)_([\d\-\+~,\.]+)_(\w+)', name)
        ret = [None, None, None]
        if m is None:
            m = re.match('(.*)_([\d\-\+~,]+)', name)
            if m is not None:
                ret = m.group(1, 2)
                ret = ret + (None,)
        else:
            ret = m.group(1, 2, 3)
        return ret

    CROSS_COLUMN_MATCH_STR = '(.+)_(x|xu|xd)_(.+)'

    @staticmethod
    def is_cross_columns(name):
        return re.match(StockDataFrame.CROSS_COLUMN_MATCH_STR,
                        name) is not None

    @staticmethod
    def parse_cross_column(name):
        m = re.match(StockDataFrame.CROSS_COLUMN_MATCH_STR, name)
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

    @staticmethod
    def init_columns(obj, columns):
        if isinstance(columns, list):
            for column in columns:
                StockDataFrame.__init_column(obj, column)
        else:
            StockDataFrame.__init_column(obj, columns)

    @classmethod
    def __init_not_exist_column(cls, df, key):
        if key == 'change':
            cls._get_change(df)
        elif key == 'rate':
            cls._get_rate(df)
        elif key == 'middle':
            cls._get_middle(df)
        elif key in ['boll', 'boll_ub', 'boll_lb']:
            cls._get_boll(df)
        elif key in ['macd', 'macds', 'macdh']:
            cls._get_macd(df)
        elif key in ['kdjk', 'kdjd', 'kdjj']:
            cls._get_kdj_default(df)
        elif key in ['cr', 'cr-ma1', 'cr-ma2', 'cr-ma3']:
            cls._get_cr(df)
        elif key == 'log-ret':
            cls._get_log_ret(df)
        elif key.endswith('_delta'):
            cls._get_delta(df, key)
        elif cls.is_cross_columns(key):
            cls._get_cross(df, key)
        else:
            c, r, t = cls.parse_column_name(key)
            if t is not None:
                if t in cls.OPERATORS:
                    # support all kinds of compare operators
                    cls._get_op(df, c, r, t)
                else:
                    func_name = '_get_{}'.format(t)
                    getattr(cls, func_name)(df, c, r)
            else:
                func_name = '_get_{}'.format(c)
                getattr(cls, func_name)(df, r)

    @staticmethod
    def __init_column(df, key):
        if key not in df:
            if len(df) == 0:
                df[key] = []
            else:
                StockDataFrame.__init_not_exist_column(df, key)

    def __getitem__(self, item):
        try:
            result = self.retype(
                super(StockDataFrame, self).__getitem__(item))
        except KeyError:
            try:
                self.init_columns(self, item)
            except AttributeError:
                log.exception('{} not found.'.format(item))
            result = self.retype(
                super(StockDataFrame, self).__getitem__(item))
        return result

    def in_date_delta(self, delta_day, anchor=None):
        if anchor is None:
            anchor = self.get_today()
        other_day = get_date_from_diff(anchor, delta_day)
        if delta_day > 0:
            start, end = anchor, other_day
        else:
            start, end = other_day, anchor
        return self.retype(self.ix[start:end])

    def till(self, end_date):
        return self[self.index <= end_date]

    def start_from(self, start_date):
        return self[self.index >= start_date]

    def within(self, start_date, end_date):
        return self.start_from(start_date).till(end_date)

    def copy(self, deep=True):
        return self.retype(super(StockDataFrame, self).copy(deep))

    @staticmethod
    def retype(value, index_column=None):
        """ if the input is a `DataFrame`, convert it to this class.

        :param index_column: the column that will be used as index,
                             default to `date`
        :param value: value to convert
        :return: this extended class
        """
        if index_column is None:
            index_column = 'date'

        if isinstance(value, pd.DataFrame):
            if index_column in value.columns:
                value.set_index(index_column, inplace=True)
            value = StockDataFrame(value)
        return value
