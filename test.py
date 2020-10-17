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

import os
from unittest import TestCase

import pandas as pd
from hamcrest import greater_than, assert_that, equal_to, close_to, \
    contains_exactly, \
    none, only_contains, is_not
from numpy import isnan

from stockstats import StockDataFrame as Sdf

__author__ = 'Cedric Zhuang'


def get_file(filename):
    filename = os.path.join('test_data', filename)
    return os.path.join(os.path.split(__file__)[0], filename)


class StockDataFrameTest(TestCase):
    _stock = Sdf.retype(pd.read_csv(get_file('987654.csv')))
    _supor = Sdf.retype(pd.read_csv(get_file('002032.csv')))

    def get_stock_20day(self):
        return self.get_stock().within(20110101, 20110120)

    def get_stock_30day(self):
        return self.get_stock().within(20110101, 20110130)

    def get_stock_90day(self):
        return self.get_stock().within(20110101, 20110331)

    def get_stock(self):
        return Sdf(self._stock.copy())

    def test_delta(self):
        stock = self.get_stock()
        assert_that(len(stock['volume_delta']), greater_than(1))
        assert_that(stock.loc[20141219]['volume_delta'], equal_to(-63383600))

    def test_multiple_columns(self):
        ret = self.get_stock()
        ret = ret[['open', 'close']]
        assert_that(ret.columns, contains_exactly('open', 'close'))

    def test_column_le_count(self):
        stock = self.get_stock_20day()
        c = 'close_13.01_le_5_c'
        stock.get(c)
        assert_that(stock.loc[20110117][c], equal_to(1))
        assert_that(stock.loc[20110119][c], equal_to(3))

    def test_column_ge_future_count(self):
        stock = self.get_stock_20day()
        c = stock['close_12.8_ge_5_fc']
        assert_that(c.loc[20110119], equal_to(1))
        assert_that(c.loc[20110117], equal_to(1))
        assert_that(c.loc[20110113], equal_to(3))
        assert_that(c.loc[20110111], equal_to(4))

    def test_column_delta(self):
        stock = self.get_stock_20day()
        open_d = stock['open_-1_d']
        assert_that(isnan(open_d.loc[20110104]), equal_to(True))
        assert_that(open_d.loc[20110120], close_to(0.07, 0.0001))

    def test_column_delta_p2(self):
        stock = self.get_stock_20day()
        open_d = stock['open_2_d']
        assert_that(isnan(open_d.loc[20110119]), equal_to(True))
        assert_that(open_d.loc[20110118], close_to(-0.2, 0.001))

    def test_column_rate_minus_2(self):
        stock = self.get_stock_20day()
        open_r = stock['open_-2_r']
        assert_that(isnan(open_r.loc[20110105]), equal_to(True))
        assert_that(open_r.loc[20110106], close_to(2.49, 0.01))

    def test_column_rate_prev(self):
        stock = self.get_stock_20day()
        rate = stock['rate']
        assert_that(rate.loc[20110107], close_to(4.41, 0.01))

    def test_column_rate_plus2(self):
        stock = self.get_stock_20day()
        open_r = stock['open_2_r']
        assert_that(open_r.loc[20110118], close_to(-1.566, 0.001))
        assert_that(isnan(open_r.loc[20110119]), equal_to(True))
        assert_that(isnan(open_r.loc[20110120]), equal_to(True))

    def test_middle(self):
        stock = self.get_stock_20day()
        middle = stock['middle']
        assert_that(middle.loc[20110104], close_to(12.53, 0.01))

    def test_cr(self):
        stock = self.get_stock_90day()
        stock.get('cr')
        assert_that(stock['cr'].loc[20110331], close_to(178.2, 0.1))
        assert_that(stock['cr-ma1'].loc[20110331], close_to(120.0, 0.1))
        assert_that(stock['cr-ma2'].loc[20110331], close_to(117.1, 0.1))
        assert_that(stock['cr-ma3'].loc[20110331], close_to(111.5, 0.1))
        assert_that(self._supor.columns,
                    is_not(contains_exactly('middle_-1_s')))

    def test_column_permutation(self):
        stock = self.get_stock_20day()
        amount_p = stock['volume_-1_d_-3,-2,-1_p']
        assert_that(amount_p.loc[20110107:20110112],
                    contains_exactly(2, 5, 2, 4))
        assert_that(isnan(amount_p.loc[20110104]), equal_to(True))
        assert_that(isnan(amount_p.loc[20110105]), equal_to(True))
        assert_that(isnan(amount_p.loc[20110106]), equal_to(True))

    def test_column_max(self):
        stock = self.get_stock_20day()
        volume_max = stock['volume_-3,2,-1_max']
        assert_that(volume_max.loc[20110106], equal_to(166409700))
        assert_that(volume_max.loc[20110120], equal_to(110664100))
        assert_that(volume_max.loc[20110112], equal_to(362436800))

    def test_column_min(self):
        stock = self.get_stock_20day()
        volume_max = stock['volume_-3~1_min']
        assert_that(volume_max.loc[20110106], equal_to(83140300))
        assert_that(volume_max.loc[20110120], equal_to(50888500))
        assert_that(volume_max.loc[20110112], equal_to(72035800))

    def test_column_shift_positive(self):
        stock = self.get_stock_20day()
        close_s = stock['close_2_s']
        assert_that(close_s.loc[20110118], equal_to(12.48))
        assert_that(isnan(close_s.loc[20110119]), equal_to(True))
        assert_that(isnan(close_s.loc[20110120]), equal_to(True))

    def test_column_shift_zero(self):
        stock = self.get_stock_20day()
        close_s = stock['close_0_s']
        assert_that(close_s.loc[20110118:20110120],
                    contains_exactly(12.69, 12.82, 12.48))

    def test_column_shift_negative(self):
        stock = self.get_stock_20day()
        close_s = stock['close_-1_s']
        assert_that(isnan(close_s.loc[20110104]), equal_to(True))
        assert_that(close_s.loc[20110105:20110106],
                    contains_exactly(12.61, 12.71))

    def test_column_rsv(self):
        stock = self.get_stock_20day()
        rsv_3 = stock['rsv_3']
        assert_that(rsv_3.loc[20110106], close_to(60.65, 0.01))

    def test_column_kdj_default(self):
        stock = self.get_stock_20day()
        assert_that(stock['kdjk'].loc[20110104], close_to(60.52, 0.01))
        assert_that(stock['kdjd'].loc[20110104], close_to(53.50, 0.01))
        assert_that(stock['kdjj'].loc[20110104], close_to(74.56, 0.01))

    def test_column_kdjk(self):
        stock = self.get_stock_20day()
        kdjk_3 = stock['kdjk_3']
        assert_that(kdjk_3.loc[20110104], close_to(60.52, 0.01))
        assert_that(kdjk_3.loc[20110120], close_to(31.21, 0.01))

    def test_column_kdjd(self):
        stock = self.get_stock_20day()
        kdjk_3 = stock['kdjd_3']
        assert_that(kdjk_3.loc[20110104], close_to(53.50, 0.01))
        assert_that(kdjk_3.loc[20110120], close_to(43.13, 0.01))

    def test_column_kdjj(self):
        stock = self.get_stock_20day()
        kdjk_3 = stock['kdjj_3']
        assert_that(kdjk_3.loc[20110104], close_to(74.56, 0.01))
        assert_that(kdjk_3.loc[20110120], close_to(7.37, 0.01))

    def test_column_cross(self):
        stock = self.get_stock_30day()
        cross = stock['kdjk_3_x_kdjd_3']
        assert_that(sum(cross), equal_to(2))
        assert_that(cross.loc[20110114], equal_to(True))
        assert_that(cross.loc[20110125], equal_to(True))

    def test_column_cross_up(self):
        stock = self.get_stock_30day()
        cross = stock['kdjk_3_xu_kdjd_3']
        assert_that(sum(cross), equal_to(1))
        assert_that(cross.loc[20110125], equal_to(True))

    def test_column_cross_down(self):
        stock = self.get_stock_30day()
        cross = stock['kdjk_3_xd_kdjd_3']
        assert_that(sum(cross), equal_to(1))
        assert_that(cross.loc[20110114], equal_to(True))

    def test_column_sma(self):
        stock = self.get_stock_20day()
        sma_2 = stock['open_2_sma']
        assert_that(sma_2.loc[20110105], close_to(12.56, 0.001))

    def test_column_ema(self):
        stock = self.get_stock_20day()
        ema_5 = stock['close_5_ema']
        assert_that(isnan(ema_5.loc[20110107]), equal_to(False))
        assert_that(ema_5.loc[20110110], close_to(12.9668, 0.01))

    def test_column_macd(self):
        stock = self.get_stock_90day()
        stock.get('macd')
        record = stock.loc[20110225]
        assert_that(record['macd'], close_to(-0.0382, 0.0001))
        assert_that(record['macds'], close_to(-0.0101, 0.0001))
        assert_that(record['macdh'], close_to(-0.02805, 0.0001))
        fast = 'close_{}_ema'.format(Sdf.MACD_EMA_SHORT)
        slow = 'close_{}_ema'.format(Sdf.MACD_EMA_LONG)
        signal = 'macd_{}_ema'.format(Sdf.MACD_EMA_SIGNAL)
        assert_that(self._supor.columns, is_not(contains_exactly(fast)))
        assert_that(self._supor.columns, is_not(contains_exactly(slow)))
        assert_that(self._supor.columns, is_not(contains_exactly(signal)))

    def test_column_macds(self):
        stock = self.get_stock_90day()
        stock.get('macds')
        record = stock.loc[20110225]
        assert_that(record['macds'], close_to(-0.0101, 0.0001))

    def test_column_macdh(self):
        stock = self.get_stock_90day()
        stock.get('macdh')
        record = stock.loc[20110225]
        assert_that(record['macdh'], close_to(-0.02805, 0.0001))

    def test_column_mstd(self):
        stock = self.get_stock_20day()
        mstd_3 = stock['close_3_mstd']
        assert_that(mstd_3.loc[20110106], close_to(0.05033, 0.001))

    def test_bollinger(self):
        stock = self.get_stock().within(20140930, 20141211)
        boll_ub = stock['boll_ub']
        boll_lb = stock['boll_lb']
        assert_that(stock['boll'].loc[20141103], close_to(9.80, 0.01))
        assert_that(boll_ub.loc[20141103], close_to(10.1310, 0.01))
        assert_that(boll_lb.loc[20141103], close_to(9.48, 0.01))

    def test_bollinger_empty(self):
        stock = self.get_stock().within(18800101, 18900101)
        s = stock['boll_ub']
        assert_that(len(s), equal_to(0))

    def test_column_mvar(self):
        stock = self.get_stock_20day()
        mvar_3 = stock['open_3_mvar']
        assert_that(mvar_3.loc[20110106], close_to(0.0292, 0.001))

    def test_parse_column_name_1(self):
        c, r, t = Sdf.parse_column_name('amount_-5~-1_p')
        assert_that(c, equal_to('amount'))
        assert_that(r, equal_to('-5~-1'))
        assert_that(t, equal_to('p'))

    def test_parse_column_name_2(self):
        c, r, t = Sdf.parse_column_name('open_+2~4_d')
        assert_that(c, equal_to('open'))
        assert_that(r, equal_to('+2~4'))
        assert_that(t, equal_to('d'))

    def test_parse_column_name_stacked(self):
        c, r, t = Sdf.parse_column_name('open_-1_d_-1~-3_p')
        assert_that(c, equal_to('open_-1_d'))
        assert_that(r, equal_to('-1~-3'))
        assert_that(t, equal_to('p'))

    def test_parse_column_name_3(self):
        c, r, t = Sdf.parse_column_name('close_-3,-1,+2_p')
        assert_that(c, equal_to('close'))
        assert_that(r, equal_to('-3,-1,+2'))
        assert_that(t, equal_to('p'))

    def test_parse_column_name_max(self):
        c, r, t = Sdf.parse_column_name('close_-3,-1,+2_max')
        assert_that(c, equal_to('close'))
        assert_that(r, equal_to('-3,-1,+2'))
        assert_that(t, equal_to('max'))

    def test_parse_column_name_float(self):
        c, r, t = Sdf.parse_column_name('close_12.32_le')
        assert_that(c, equal_to('close'))
        assert_that(r, equal_to('12.32'))
        assert_that(t, equal_to('le'))

    def test_parse_column_name_stacked_xu(self):
        c, r, t = Sdf.parse_column_name('cr-ma2_xu_cr-ma1_20_c')
        assert_that(c, equal_to('cr-ma2_xu_cr-ma1'))
        assert_that(r, equal_to('20'))
        assert_that(t, equal_to('c'))

    def test_parse_column_name_rsv(self):
        c, r, t = Sdf.parse_column_name('rsv_9')
        assert_that(c, equal_to('rsv'))
        assert_that(r, equal_to('9'))

    def test_parse_column_name_no_match(self):
        c, r, t = Sdf.parse_column_name('no match')
        assert_that(c, none())
        assert_that(r, none())
        assert_that(t, none())

    def test_to_int_split(self):
        shifts = Sdf.to_ints('5,1,3, -2')
        assert_that(shifts, contains_exactly(-2, 1, 3, 5))

    def test_to_int_continue(self):
        shifts = Sdf.to_ints('3, -3~-1, 5')
        assert_that(shifts, contains_exactly(-3, -2, -1, 3, 5))

    def test_to_int_dedup(self):
        shifts = Sdf.to_ints('3, -3~-1, 5, -2~-1')
        assert_that(shifts, contains_exactly(-3, -2, -1, 3, 5))

    def test_to_floats(self):
        floats = Sdf.to_floats('1.3, 4, -12.5, 4.0')
        assert_that(floats, contains_exactly(-12.5, 1.3, 4))

    def test_to_float(self):
        number = Sdf.to_float('12.3')
        assert_that(number, equal_to(12.3))

    def test_is_cross_columns(self):
        assert_that(Sdf.is_cross_columns('a_x_b'), equal_to(True))
        assert_that(Sdf.is_cross_columns('a_xu_b'), equal_to(True))
        assert_that(Sdf.is_cross_columns('a_xd_b'), equal_to(True))
        assert_that(Sdf.is_cross_columns('a_xx_b'), equal_to(False))
        assert_that(Sdf.is_cross_columns('a_xa_b'), equal_to(False))
        assert_that(Sdf.is_cross_columns('a_x_'), equal_to(False))
        assert_that(Sdf.is_cross_columns('_xu_b'), equal_to(False))
        assert_that(Sdf.is_cross_columns('_xd_'), equal_to(False))

    def test_parse_cross_column(self):
        assert_that(Sdf.parse_cross_column('a_x_b'),
                    contains_exactly('a', 'x', 'b'))

    def test_parse_cross_column_xu(self):
        assert_that(Sdf.parse_cross_column('a_xu_b'),
                    contains_exactly('a', 'xu', 'b'))

    def test_get_shift_convolve_array(self):
        assert_that(Sdf.get_diff_convolve_array(0), contains_exactly(1))
        assert_that(Sdf.get_diff_convolve_array(-1), contains_exactly(1, -1))
        assert_that(Sdf.get_diff_convolve_array(-2),
                    contains_exactly(1, 0, -1))
        assert_that(Sdf.get_diff_convolve_array(2), contains_exactly(-1, 0, 1))

    def test_get_log_ret(self):
        stock = self.get_stock_30day()
        stock.get('log-ret')
        assert_that(stock.loc[20110128]['log-ret'],
                    close_to(-0.010972, 0.000001))

    def test_in_date_delta(self):
        stock = self.get_stock_20day()
        assert_that(stock.in_date_delta(-4, 20110110).index,
                    only_contains(20110106, 20110107, 20110110))
        assert_that(stock.in_date_delta(3, 20110110).index,
                    only_contains(20110110, 20110111, 20110112, 20110113))

    def test_rsv_nan_value(self):
        s = Sdf.retype(pd.read_csv(get_file('asml.as.csv')))
        df = Sdf.retype(s)
        assert_that(df['rsv_9'][0], equal_to(0.0))

    def test_get_rsi(self):
        self._supor.get('rsi_6')
        self._supor.get('rsi_12')
        self._supor.get('rsi_24')
        assert_that(self._supor.loc[20160817, 'rsi_6'], close_to(71.31, 0.01))
        assert_that(self._supor.loc[20160817, 'rsi_12'], close_to(63.11, 0.01))
        assert_that(self._supor.loc[20160817, 'rsi_24'], close_to(61.31, 0.01))
        assert_that(self._supor.columns, is_not(contains_exactly('closepm')))
        assert_that(self._supor.columns,
                    is_not(contains_exactly('closepm_6_smma')))
        assert_that(self._supor.columns, is_not(contains_exactly('closenm')))
        assert_that(self._supor.columns,
                    is_not(contains_exactly('closenm_6_smma')))

    def test_get_wr(self):
        self._supor.get('wr_10')
        self._supor.get('wr_6')
        assert_that(self._supor.loc[20160817, 'wr_10'], close_to(13.06, 0.01))
        assert_that(self._supor.loc[20160817, 'wr_6'], close_to(16.53, 0.01))

    def test_get_cci(self):
        self._supor.get('cci_14')
        self._supor.get('cci')
        assert_that(self._supor.loc[20160817, 'cci'], close_to(50, 0.01))
        assert_that(self._supor.loc[20160817, 'cci_14'], close_to(50, 0.01))
        assert_that(self._supor.loc[20160816, 'cci_14'], close_to(24.8, 0.01))
        assert_that(self._supor.loc[20160815, 'cci_14'],
                    close_to(-26.46, 0.01))

    def test_get_atr(self):
        self._supor.get('atr_14')
        self._supor.get('atr')
        assert_that(self._supor.loc[20160817, 'atr_14'], close_to(1.33, 0.01))
        assert_that(self._supor.loc[20160817, 'atr'], close_to(1.33, 0.01))
        assert_that(self._supor.loc[20160816, 'atr'], close_to(1.32, 0.01))
        assert_that(self._supor.loc[20160815, 'atr'], close_to(1.28, 0.01))
        assert_that(self._supor.columns,
                    is_not(contains_exactly('tr_14_smma')))

    def test_get_sma_tr(self):
        c = self._supor.get('tr_14_sma')
        assert_that(c.loc[20160817], close_to(1.33, 0.01))
        assert_that(c.loc[20160816], close_to(1.37, 0.01))
        assert_that(c.loc[20160815], close_to(1.47, 0.01))

    def test_get_dma(self):
        c = self._supor.get('dma')
        assert_that(c.loc[20160817], close_to(2.08, 0.01))
        assert_that(c.loc[20160816], close_to(2.15, 0.01))
        assert_that(c.loc[20160815], close_to(2.27, 0.01))

    def test_get_pdi(self):
        c = self._supor.get('pdi')
        assert_that(c.loc[20160817], close_to(24.60, 0.01))
        assert_that(c.loc[20160816], close_to(28.60, 0.01))
        assert_that(c.loc[20160815], close_to(21.23, 0.01))

    def test_get_mdi(self):
        c = self._supor.get('mdi')
        assert_that(c.loc[20160817], close_to(13.60, 0.01))
        assert_that(c.loc[20160816], close_to(15.82, 0.01))
        assert_that(c.loc[20160815], close_to(18.85, 0.01))

    def test_dx(self):
        c = self._supor.get('dx')
        assert_that(c.loc[20160817], close_to(28.78, 0.01))
        assert_that(c.loc[20160815], close_to(5.95, 0.01))
        assert_that(c.loc[20160812], close_to(10.05, 0.01))

    def test_adx(self):
        c = self._supor.get('adx')
        assert_that(c.loc[20160817], close_to(20.15, 0.01))
        assert_that(c.loc[20160816], close_to(16.71, 0.01))
        assert_that(c.loc[20160815], close_to(11.88, 0.01))

    def test_adxr(self):
        c = self._supor.get('adxr')
        assert_that(c.loc[20160817], close_to(17.36, 0.01))
        assert_that(c.loc[20160816], close_to(16.24, 0.01))
        assert_that(c.loc[20160815], close_to(16.06, 0.01))

    def test_trix_default(self):
        c = self._supor.get('trix')
        assert_that(c.loc[20160817], close_to(0.20, 0.01))
        assert_that(c.loc[20160816], close_to(0.21, 0.01))
        assert_that(c.loc[20160815], close_to(0.24, 0.01))

        single = 'close_{}_ema'.format(Sdf.TRIX_EMA_WINDOW)
        double = 'close_{w}_ema_{w}_ema'.format(w=Sdf.TRIX_EMA_WINDOW)
        triple = 'close_{w}_ema_{w}_ema_{w}_ema'.format(w=Sdf.TRIX_EMA_WINDOW)
        prev_triple = '{}_-1_s'.format(triple)
        assert_that(self._supor.columns, is_not(contains_exactly(single)))
        assert_that(self._supor.columns, is_not(contains_exactly(double)))
        assert_that(self._supor.columns, is_not(contains_exactly(triple)))
        assert_that(self._supor.columns, is_not(contains_exactly(prev_triple)))

    def test_tema_default(self):
        c = self._supor.get('tema')
        a = self._supor.get('close_5_tema')
        assert_that(c.loc[20160817], equal_to(a.loc[20160817]))
        assert_that(c.loc[20160817], close_to(40.29, 0.01))
        assert_that(c.loc[20160816], close_to(39.63, 0.01))
        assert_that(c.loc[20160815], close_to(39.37, 0.01))

        single = 'close_{}_ema'.format(Sdf.TRIX_EMA_WINDOW)
        double = 'close_{w}_ema_{w}_ema'.format(w=Sdf.TRIX_EMA_WINDOW)
        triple = 'close_{w}_ema_{w}_ema_{w}_ema'.format(w=Sdf.TRIX_EMA_WINDOW)
        assert_that(self._supor.columns, is_not(contains_exactly(single)))
        assert_that(self._supor.columns, is_not(contains_exactly(double)))
        assert_that(self._supor.columns, is_not(contains_exactly(triple)))

    def test_trix_ma(self):
        c = self._supor.get('trix_9_sma')
        assert_that(c.loc[20160817], close_to(0.34, 0.01))
        assert_that(c.loc[20160816], close_to(0.38, 0.01))
        assert_that(c.loc[20160815], close_to(0.42, 0.01))

    def test_vr_default(self):
        c = self._supor['vr']
        assert_that(c.loc[20160817], close_to(153.2, 0.01))
        assert_that(c.loc[20160816], close_to(171.69, 0.01))
        assert_that(c.loc[20160815], close_to(178.78, 0.01))

        c = self._supor['vr_26']
        assert_that(c.loc[20160817], close_to(153.2, 0.01))
        assert_that(c.loc[20160816], close_to(171.69, 0.01))
        assert_that(c.loc[20160815], close_to(178.78, 0.01))

        assert_that(self._supor.columns, is_not(contains_exactly('av')))
        assert_that(self._supor.columns, is_not(contains_exactly('bv')))
        assert_that(self._supor.columns, is_not(contains_exactly('cv')))

    def test_vr_ma(self):
        c = self._supor['vr_6_sma']
        assert_that(c.loc[20160817], close_to(182.77, 0.01))
        assert_that(c.loc[20160816], close_to(190.1, 0.01))
        assert_that(c.loc[20160815], close_to(197.52, 0.01))
