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
    contains_exactly, none, is_not, raises
from numpy import isnan

from stockstats import StockDataFrame as Sdf
from stockstats import wrap

__author__ = 'Cedric Zhuang'


def get_file(filename):
    filename = os.path.join('test_data', filename)
    return os.path.join(os.path.split(__file__)[0], filename)


def near_to(value):
    return close_to(value, 1e-3)


class StockDataFrameTest(TestCase):
    _stock = wrap(pd.read_csv(get_file('987654.csv')))
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

    @staticmethod
    def test_must_have_positive_int():
        def do():
            Sdf.get_int_positive("-54")

        assert_that(do, raises(IndexError))

    def test_multiple_columns(self):
        ret = self.get_stock()
        ret = ret[['open', 'close']]
        assert_that(ret.columns, contains_exactly('open', 'close'))

    def test_column_le_count(self):
        stock = self.get_stock_20day()
        stock['res'] = stock['close'] <= 13.01
        count = stock.get('res_5_c')
        assert_that(count.loc[20110117], equal_to(1))
        assert_that(count.loc[20110119], equal_to(3))

    def test_column_ge_future_count(self):
        stock = self.get_stock_20day()
        stock['res'] = stock['close'] >= 12.8
        count = stock['res_5_fc']
        assert_that(count.loc[20110119], equal_to(1))
        assert_that(count.loc[20110117], equal_to(1))
        assert_that(count.loc[20110113], equal_to(3))
        assert_that(count.loc[20110111], equal_to(4))

    def test_column_delta(self):
        stock = self.get_stock_20day()
        open_d = stock['open_-1_d']
        assert_that(open_d.loc[20110104], equal_to(0.0))
        assert_that(open_d.loc[20110120], near_to(0.07))

    def test_column_delta_p2(self):
        stock = self.get_stock_20day()
        open_d = stock['open_2_d']
        assert_that(open_d.loc[20110104], near_to(-0.31))
        assert_that(open_d.loc[20110119], equal_to(0.0))
        assert_that(open_d.loc[20110118], near_to(-0.2))

    def test_column_rate_minus_2(self):
        stock = self.get_stock_20day()
        open_r = stock['open_-2_r']
        assert_that(open_r.loc[20110105], equal_to(0.0))
        assert_that(open_r.loc[20110106], near_to(2.495))

    def test_column_rate_prev(self):
        stock = self.get_stock_20day()
        rate = stock['rate']
        assert_that(rate.loc[20110107], near_to(4.4198))

    def test_column_rate_plus2(self):
        stock = self.get_stock_20day()
        open_r = stock['open_2_r']
        assert_that(open_r.loc[20110118], near_to(-1.566))
        assert_that(open_r.loc[20110119], equal_to(0.0))
        assert_that(open_r.loc[20110120], equal_to(0.0))

    def test_change(self):
        stock = self.get_stock_20day()
        change = stock['change']
        assert_that(change.loc[20110107], near_to(4.4198))

    def test_middle(self):
        stock = self.get_stock_20day()
        middle = stock['middle']
        tp = stock['tp']
        idx = 20110104
        assert_that(middle.loc[idx], near_to(12.53))
        assert_that(tp.loc[idx], equal_to(middle.loc[idx]))

    def test_cr(self):
        stock = self.get_stock_90day()
        stock.get('cr')
        assert_that(stock['cr'].loc[20110331], near_to(178.1714))
        assert_that(stock['cr-ma1'].loc[20110331], near_to(120.0364))
        assert_that(stock['cr-ma2'].loc[20110331], near_to(117.1488))
        assert_that(stock['cr-ma3'].loc[20110331], near_to(111.5195))

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
        assert_that(close_s.loc[20110119], equal_to(12.48))
        assert_that(close_s.loc[20110120], equal_to(12.48))

    def test_column_shift_zero(self):
        stock = self.get_stock_20day()
        close_s = stock['close_0_s']
        assert_that(close_s.loc[20110118:20110120],
                    contains_exactly(12.69, 12.82, 12.48))

    def test_column_shift_negative(self):
        stock = self.get_stock_20day()
        close_s = stock['close_-2_s']
        assert_that(close_s.loc[20110104], equal_to(12.61))
        assert_that(close_s.loc[20110105], equal_to(12.61))
        assert_that(close_s.loc[20110106], equal_to(12.61))
        assert_that(close_s.loc[20110107], equal_to(12.71))

    def test_column_rsv(self):
        stock = self.get_stock_20day()
        rsv_3 = stock['rsv_3']
        assert_that(rsv_3.loc[20110106], near_to(60.6557))

    def test_column_kdj_default(self):
        stock = self.get_stock_20day()
        assert_that(stock['kdjk'].loc[20110104], near_to(60.5263))
        assert_that(stock['kdjd'].loc[20110104], near_to(53.5087))
        assert_that(stock['kdjj'].loc[20110104], near_to(74.5614))

    def test_column_kdjk(self):
        stock = self.get_stock_20day()
        kdjk_3 = stock['kdjk_3']
        assert_that(kdjk_3.loc[20110104], near_to(60.5263))
        assert_that(kdjk_3.loc[20110120], near_to(31.2133))

    def test_column_kdjd(self):
        stock = self.get_stock_20day()
        kdjk_3 = stock['kdjd_3']
        assert_that(kdjk_3.loc[20110104], near_to(53.5087))
        assert_that(kdjk_3.loc[20110120], near_to(43.1347))

    def test_column_kdjj(self):
        stock = self.get_stock_20day()
        kdjk_3 = stock['kdjj_3']
        assert_that(kdjk_3.loc[20110104], near_to(74.5614))
        assert_that(kdjk_3.loc[20110120], near_to(7.37))

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
        assert_that(sma_2.loc[20110104], near_to(12.42))
        assert_that(sma_2.loc[20110105], near_to(12.56))

    def test_column_smma(self):
        stock = self.get_stock_20day()
        smma = stock['high_5_smma']
        assert_that(smma.loc[20110120], near_to(13.0394))

    def test_column_ema(self):
        stock = self.get_stock_20day()
        ema_5 = stock['close_5_ema']
        assert_that(ema_5.loc[20110107], near_to(12.9026))
        assert_that(ema_5.loc[20110110], near_to(12.9668))

    def test_ema_of_empty_df(self):
        s = Sdf.retype(pd.DataFrame())
        ema = s['close_10_ema']
        assert_that(len(ema), equal_to(0))

    def test_column_macd(self):
        stock = self.get_stock_90day()
        stock.get('macd')
        record = stock.loc[20110225]
        assert_that(record['macd'], near_to(-0.0382))
        assert_that(record['macds'], near_to(-0.0101))
        assert_that(record['macdh'], near_to(-0.02805))

    def test_column_macds(self):
        stock = self.get_stock_90day()
        stock.get('macds')
        record = stock.loc[20110225]
        assert_that(record['macds'], near_to(-0.0101))

    def test_column_macdh(self):
        stock = self.get_stock_90day()
        stock.get('macdh')
        record = stock.loc[20110225]
        assert_that(record['macdh'], near_to(-0.02805))

    def test_ppo(self):
        stock = self.get_stock_90day()
        _ = stock[['ppo', 'ppos', 'ppoh']]
        assert_that(stock['ppo'].loc[20110331], near_to(1.1190))
        assert_that(stock['ppos'].loc[20110331], near_to(0.6840))
        assert_that(stock['ppoh'].loc[20110331], near_to(0.4349))

    def test_column_mstd(self):
        stock = self.get_stock_20day()
        mstd_3 = stock['close_3_mstd']
        assert_that(mstd_3.loc[20110106], near_to(0.05033))

    def test_bollinger(self):
        stock = self.get_stock().within(20140930, 20141211)
        boll_ub = stock['boll_ub']
        boll_lb = stock['boll_lb']
        idx = 20141103
        assert_that(stock['boll'].loc[idx], near_to(9.8035))
        assert_that(boll_ub.loc[idx], near_to(10.1310))
        assert_that(boll_lb.loc[idx], near_to(9.4759))

    def test_bollinger_empty(self):
        stock = self.get_stock().within(18800101, 18900101)
        s = stock['boll_ub']
        assert_that(len(s), equal_to(0))

    def test_column_mvar(self):
        stock = self.get_stock_20day()
        mvar_3 = stock['open_3_mvar']
        assert_that(mvar_3.loc[20110106], near_to(0.0292))

    def test_column_parse_error(self):
        stock = self.get_stock_90day()
        with self.assertRaises(UserWarning):
            _ = stock["foobarbaz"]
        with self.assertRaises(KeyError):
            _ = stock["close_1_foo_3_4"]

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
        c, r = Sdf.parse_column_name('rsv_9')
        assert_that(c, equal_to('rsv'))
        assert_that(r, equal_to('9'))

    def test_parse_column_name_no_match(self):
        ret = Sdf.parse_column_name('no match')
        assert_that(len(ret), equal_to(1))
        assert_that(ret[0], none())

    def test_to_int_split(self):
        shifts = Sdf.to_ints('5,1,3, -2')
        assert_that(shifts, contains_exactly(-2, 1, 3, 5))

    def test_to_int_continue(self):
        shifts = Sdf.to_ints('3, -3~-1, 5')
        assert_that(shifts, contains_exactly(-3, -2, -1, 3, 5))

    def test_to_int_dedup(self):
        shifts = Sdf.to_ints('3, -3~-1, 5, -2~-1')
        assert_that(shifts, contains_exactly(-3, -2, -1, 3, 5))

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

    def test_get_log_ret(self):
        stock = self.get_stock_30day()
        stock.get('log-ret')
        assert_that(stock.loc[20110128]['log-ret'], near_to(-0.010972))

    @staticmethod
    def test_rsv_nan_value():
        s = wrap(pd.read_csv(get_file('asml.as.csv')))
        df = wrap(s)
        assert_that(df['rsv_9'][0], equal_to(0.0))

    def test_get_rsi(self):
        rsi = self._supor.get('rsi')
        rsi_6 = self._supor.get('rsi_6')
        rsi_12 = self._supor.get('rsi_12')
        rsi_14 = self._supor.get('rsi_14')
        rsi_24 = self._supor.get('rsi_24')
        idx = 20160817
        assert_that(rsi_6.loc[idx], near_to(71.3114))
        assert_that(rsi_12.loc[idx], near_to(63.1125))
        assert_that(rsi_24.loc[idx], near_to(61.3064))
        assert_that(rsi.loc[idx], near_to(rsi_14.loc[idx]))

    def test_get_stoch_rsi(self):
        stock = self.get_stock_90day()
        stoch_rsi = stock['stochrsi']
        stoch_rsi_6 = stock['stochrsi_6']
        stoch_rsi_14 = stock['stochrsi_14']
        idx = 20110331
        assert_that(stoch_rsi.loc[idx], near_to(67.0955))
        assert_that(stoch_rsi_6.loc[idx], near_to(27.5693))
        assert_that(stoch_rsi_14.loc[idx], near_to(stoch_rsi.loc[idx]))

    def test_get_wr(self):
        wr = self._supor.get('wr')
        wr_6 = self._supor.get('wr_6')
        wr_14 = self._supor.get('wr_14')
        idx = 20160817
        assert_that(wr_14.loc[idx], near_to(-49.1620))
        assert_that(wr_6.loc[idx], near_to(-16.5322))
        assert_that(wr.loc[idx], equal_to(wr_14.loc[idx]))

    def test_get_cci(self):
        stock = self._supor.within(20160701, 20160831)
        stock.get('cci_14')
        stock.get('cci')
        assert_that(stock.loc[20160817, 'cci'], near_to(50))
        assert_that(stock.loc[20160817, 'cci_14'], near_to(50))
        assert_that(stock.loc[20160816, 'cci_14'], near_to(24.7987))
        assert_that(stock.loc[20160815, 'cci_14'], near_to(-26.46))

    def test_get_atr(self):
        self._supor.get('atr_14')
        self._supor.get('atr')
        assert_that(self._supor.loc[20160817, 'atr_14'], near_to(1.3334))
        assert_that(self._supor.loc[20160817, 'atr'], near_to(1.3334))
        assert_that(self._supor.loc[20160816, 'atr'], near_to(1.3229))
        assert_that(self._supor.loc[20160815, 'atr'], near_to(1.2815))

    def test_get_sma_tr(self):
        c = self._supor.get('tr_14_sma')
        assert_that(c.loc[20160817], near_to(1.3321))
        assert_that(c.loc[20160816], near_to(1.37))
        assert_that(c.loc[20160815], near_to(1.47))

    def test_get_dma(self):
        c = self._supor.get('dma')
        assert_that(c.loc[20160817], near_to(2.078))
        assert_that(c.loc[20160816], near_to(2.15))
        assert_that(c.loc[20160815], near_to(2.2743))

    def test_get_pdi(self):
        c = self._supor.get('pdi')
        assert_that(c.loc[20160817], near_to(24.5989))
        assert_that(c.loc[20160816], near_to(28.6088))
        assert_that(c.loc[20160815], near_to(21.23))

    def test_get_mdi(self):
        c = self._supor.get('mdi')
        assert_that(c.loc[20160817], near_to(13.6049))
        assert_that(c.loc[20160816], near_to(15.8227))
        assert_that(c.loc[20160815], near_to(18.8455))

    def test_dx(self):
        c = self._supor.get('dx')
        assert_that(c.loc[20160817], near_to(28.7771))
        assert_that(c.loc[20160815], near_to(5.95))
        assert_that(c.loc[20160812], near_to(10.05))

    def test_adx(self):
        c = self._supor.get('adx')
        assert_that(c.loc[20160817], near_to(20.1545))
        assert_that(c.loc[20160816], near_to(16.7054))
        assert_that(c.loc[20160815], near_to(11.8767))

    def test_adxr(self):
        c = self._supor.get('adxr')
        assert_that(c.loc[20160817], near_to(17.3630))
        assert_that(c.loc[20160816], near_to(16.2464))
        assert_that(c.loc[20160815], near_to(16.0628))

    def test_trix_default(self):
        c = self._supor.get('trix')
        assert_that(c.loc[20160817], near_to(0.1999))
        assert_that(c.loc[20160816], near_to(0.2135))
        assert_that(c.loc[20160815], near_to(0.24))

    def test_tema_default(self):
        c = self._supor.get('tema')
        a = self._supor.get('close_5_tema')
        assert_that(c.loc[20160817], equal_to(a.loc[20160817]))
        assert_that(c.loc[20160817], near_to(40.2883))
        assert_that(c.loc[20160816], near_to(39.6371))
        assert_that(c.loc[20160815], near_to(39.3778))

    def test_trix_ma(self):
        c = self._supor.get('trix_9_sma')
        assert_that(c.loc[20160817], near_to(0.34))
        assert_that(c.loc[20160816], near_to(0.38))
        assert_that(c.loc[20160815], near_to(0.4238))

    def test_vr_default(self):
        c = self._supor['vr']
        assert_that(c.loc[20160817], near_to(153.1961))
        assert_that(c.loc[20160816], near_to(171.6939))
        assert_that(c.loc[20160815], near_to(178.7854))

        c = self._supor['vr_26']
        assert_that(c.loc[20160817], near_to(153.1961))
        assert_that(c.loc[20160816], near_to(171.6939))
        assert_that(c.loc[20160815], near_to(178.7854))

    def test_vr_ma(self):
        c = self._supor['vr_6_sma']
        assert_that(c.loc[20160817], near_to(182.7714))
        assert_that(c.loc[20160816], near_to(190.0970))
        assert_that(c.loc[20160815], near_to(197.5225))

    def test_mfi(self):
        stock = self.get_stock_90day()
        first = 20110104
        last = 20110331

        mfi = stock['mfi']
        assert_that(mfi.loc[first], near_to(0.5))
        assert_that(mfi.loc[last], near_to(0.7144))

        mfi_3 = stock['mfi_3']
        assert_that(mfi_3.loc[first], near_to(0.5))
        assert_that(mfi_3.loc[last], near_to(0.7874))

        mfi_15 = stock['mfi_15']
        assert_that(mfi_15.loc[first], near_to(0.5))
        assert_that(mfi_15.loc[last], near_to(0.6733))

    def test_column_kama(self):
        stock = self.get_stock_90day()
        idx = 20110331
        kama_10 = stock['close_10_kama_2_30']
        assert_that(kama_10.loc[idx], near_to(13.6648))
        kama_2 = stock['close_2_kama']
        assert_that(kama_2.loc[idx], near_to(13.7326))

    def test_vwma(self):
        stock = self.get_stock_90day()
        vwma = stock['vwma']
        vwma_7 = stock['vwma_7']
        vwma_14 = stock['vwma_14']
        assert_that(vwma.loc[20110330], near_to(13.312679))
        idx = 20110331
        assert_that(vwma.loc[idx], near_to(13.350941))
        assert_that(vwma_14.loc[idx], near_to(vwma.loc[idx]))
        assert_that(vwma_7.loc[idx], is_not(near_to(vwma.loc[idx])))

    def test_chop(self):
        stock = self.get_stock_90day()
        chop = stock['chop']
        chop_7 = stock['chop_7']
        chop_14 = stock['chop_14']
        idx = 20110330
        assert_that(chop.loc[idx], near_to(44.8926))
        assert_that(chop_14.loc[idx], near_to(chop.loc[idx]))
        assert_that(chop_7.loc[idx], is_not(near_to(chop.loc[idx])))

    def test_column_conflict(self):
        stock = self.get_stock_90day()
        res = stock[['close_26_ema', 'macd']]
        idx = 20110331
        assert_that(res['close_26_ema'].loc[idx], near_to(13.2488))
        assert_that(res['macd'].loc[idx], near_to(0.1482))

    def test_wave_trend(self):
        stock = self.get_stock_90day()
        wt1, wt2 = stock['wt1'], stock['wt2']
        idx = 20110331
        assert_that(wt1.loc[idx], near_to(38.9610))
        assert_that(wt2.loc[idx], near_to(31.6997))
