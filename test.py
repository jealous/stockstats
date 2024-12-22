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
import yfinance as yf
from hamcrest import greater_than, assert_that, equal_to, close_to, \
    contains_exactly, none, is_not, raises, has_items, instance_of, \
    not_, has_item, has_length
from numpy import isnan

import stockstats
from stockstats import StockDataFrame as Sdf, StockDataFrame
from stockstats import wrap, unwrap

__author__ = 'Cedric Zhuang'


def get_file(filename):
    filename = os.path.join('test_data', filename)
    return os.path.join(os.path.split(__file__)[0], filename)


def near_to(value):
    return close_to(value, 1e-3)


def not_has(item):
    return not_(has_item(item))


class YFinanceCompatibilityTest(TestCase):
    _stock = wrap(yf.download('002032.SZ'))

    def test_wrap_yfinance(self):
        col = self._stock['close_20_dma']
        assert_that(col.loc['2021-01-04'], near_to(0.6196))

    def test_kdj_of_yfinance(self):
        kdjk = self._stock['kdjk']
        assert_that(kdjk.loc['2021-01-04'], near_to(69.54346))

    def test_get_wr(self):
        wr = self._stock.get('wr')
        assert_that(wr.loc['2016-08-17'], near_to(-49.1621))

    def test_get_adx(self):
        wr = self._stock.get('adx')
        assert_that(wr.loc['2016-08-17'], near_to(15.6078))


class StockDataFrameTest(TestCase):
    _stock = wrap(pd.read_csv(get_file('987654.csv')))
    _supor = Sdf.retype(pd.read_csv(get_file('002032.csv')))

    def get_stock_20days(self):
        return self.get_stock().within(20110101, 20110120)

    def get_stock_30days(self):
        return self.get_stock().within(20110101, 20110130)

    def get_stock_90days(self):
        return self.get_stock().within(20110101, 20110331)

    def get_stock(self):
        return Sdf(self._stock.copy())

    def test_delta(self):
        stock = self.get_stock()
        assert_that(len(stock['volume_delta']), greater_than(1))
        assert_that(stock.loc[20141219]['volume_delta'], equal_to(-63383600))

    def test_must_have_positive_int(self):
        def do():
            self._supor.get_int_positive("-54")

        assert_that(do, raises(IndexError))

    def test_multiple_columns(self):
        ret = self.get_stock()
        ret = ret[['open', 'close']]
        assert_that(ret.columns, contains_exactly('open', 'close'))

    def test_column_le_count(self):
        stock = self.get_stock_20days()
        stock['res'] = stock['close'] <= 13.01
        count = stock.get('res_5_c')
        assert_that(count.loc[20110117], equal_to(1))
        assert_that(count.loc[20110119], equal_to(3))

    def test_column_ge_future_count(self):
        stock = self.get_stock_20days()
        stock['res'] = stock['close'] >= 12.8
        count = stock['res_5_fc']
        assert_that(count.loc[20110119], equal_to(1))
        assert_that(count.loc[20110117], equal_to(1))
        assert_that(count.loc[20110113], equal_to(3))
        assert_that(count.loc[20110111], equal_to(4))

    def test_column_delta(self):
        stock = self.get_stock_20days()
        open_d = stock['open_-1_d']
        assert_that(open_d.loc[20110104], equal_to(0.0))
        assert_that(open_d.loc[20110120], near_to(0.07))

    def test_column_delta_p2(self):
        stock = self.get_stock_20days()
        open_d = stock['open_2_d']
        assert_that(open_d.loc[20110104], near_to(-0.31))
        assert_that(open_d.loc[20110119], equal_to(0.0))
        assert_that(open_d.loc[20110118], near_to(-0.2))

    def test_column_rate_minus_2(self):
        stock = self.get_stock_20days()
        open_r = stock['open_-2_r']
        assert_that(open_r.loc[20110105], equal_to(0.0))
        assert_that(open_r.loc[20110106], near_to(2.495))

    def test_column_rate_prev(self):
        stock = self.get_stock_20days()
        rate = stock['rate']
        assert_that(rate.loc[20110107], near_to(4.4198))

    def test_column_rate_plus2(self):
        stock = self.get_stock_20days()
        open_r = stock['open_2_r']
        assert_that(open_r.loc[20110118], near_to(-1.566))
        assert_that(open_r.loc[20110119], equal_to(0.0))
        assert_that(open_r.loc[20110120], equal_to(0.0))

    def test_change(self):
        stock = self.get_stock_20days()
        change = stock['change']
        assert_that(change.loc[20110104], equal_to(0))
        assert_that(change.loc[20110105], near_to(0.793))
        assert_that(change.loc[20110107], near_to(4.4198))

        change = stock['change_2']
        assert_that(change.loc[20110104], equal_to(0))
        assert_that(change.loc[20110105], equal_to(0))
        assert_that(change.loc[20110106], near_to(0.476))

    def test_middle(self):
        stock = self.get_stock_20days()
        middle = stock['middle']
        tp = stock['tp']
        idx = 20110104
        assert_that(middle.loc[idx], near_to(12.53))
        assert_that(tp.loc[idx], equal_to(middle.loc[idx]))

    def test_typical_price_with_amount(self):
        stock = self._supor[:20]
        tp = stock['tp']
        assert_that(tp[20040817], near_to(11.541))

        middle = stock['middle']
        assert_that(middle[20040817], near_to(11.541))

    def test_cr(self):
        stock = self.get_stock_90days()
        stock.get('cr')
        assert_that(stock['cr'].loc[20110331], near_to(178.1714))
        assert_that(stock['cr-ma1'].loc[20110331], near_to(120.0364))
        assert_that(stock['cr-ma2'].loc[20110331], near_to(117.1488))
        assert_that(stock['cr-ma3'].loc[20110331], near_to(111.5195))

        stock.get('cr_26')
        assert_that(stock['cr_26'].loc[20110331], near_to(178.1714))
        assert_that(stock['cr_26-ma1'].loc[20110331], near_to(120.0364))
        assert_that(stock['cr_26-ma2'].loc[20110331], near_to(117.1488))
        assert_that(stock['cr_26-ma3'].loc[20110331], near_to(111.5195))

    def test_column_permutation(self):
        stock = self.get_stock_20days()
        amount_p = stock['volume_-1_d_-3,-2,-1_p']
        assert_that(amount_p.loc[20110107:20110112],
                    contains_exactly(2, 5, 2, 4))
        assert_that(isnan(amount_p.loc[20110104]), equal_to(True))
        assert_that(isnan(amount_p.loc[20110105]), equal_to(True))
        assert_that(isnan(amount_p.loc[20110106]), equal_to(True))

    def test_column_max(self):
        stock = self.get_stock_20days()
        volume_max = stock['volume_-3,2,-1_max']
        assert_that(volume_max.loc[20110106], equal_to(166409700))
        assert_that(volume_max.loc[20110120], equal_to(110664100))
        assert_that(volume_max.loc[20110112], equal_to(362436800))

    def test_column_min(self):
        stock = self.get_stock_20days()
        volume_max = stock['volume_-3~1_min']
        assert_that(volume_max.loc[20110106], equal_to(83140300))
        assert_that(volume_max.loc[20110120], equal_to(50888500))
        assert_that(volume_max.loc[20110112], equal_to(72035800))

    def test_column_shift_positive(self):
        stock = self.get_stock_20days()
        close_s = stock['close_2_s']
        assert_that(close_s.loc[20110118], equal_to(12.48))
        assert_that(close_s.loc[20110119], equal_to(12.48))
        assert_that(close_s.loc[20110120], equal_to(12.48))

    def test_column_shift_zero(self):
        stock = self.get_stock_20days()
        close_s = stock['close_0_s']
        assert_that(close_s.loc[20110118:20110120],
                    contains_exactly(12.69, 12.82, 12.48))

    def test_column_shift_negative(self):
        stock = self.get_stock_20days()
        close_s = stock['close_-2_s']
        assert_that(close_s.loc[20110104], equal_to(12.61))
        assert_that(close_s.loc[20110105], equal_to(12.61))
        assert_that(close_s.loc[20110106], equal_to(12.61))
        assert_that(close_s.loc[20110107], equal_to(12.71))

    def test_column_rsv(self):
        stock = self.get_stock_20days()
        rsv_3 = stock['rsv_3']
        assert_that(rsv_3.loc[20110106], near_to(60.6557))

    def test_change_single_default_window(self):
        stock = self.get_stock_20days()
        rsv = stock['rsv']
        rsv_9 = stock['rsv_9']
        rsv_5 = stock['rsv_5']
        idx = 20110114
        assert_that(rsv[idx], equal_to(rsv_9[idx]))
        assert_that(rsv[idx], not_(equal_to(rsv_5[idx])))

        orig = stockstats.set_dft_window('rsv', 5)
        assert_that(orig, equal_to(9))
        stock.drop_column('rsv', inplace=True)
        rsv = stock['rsv']
        assert_that(rsv[idx], equal_to(rsv_5[idx]))
        stockstats.set_dft_window('rsv', orig)

    def test_column_kdj_default(self):
        stock = self.get_stock_20days()
        assert_that(stock['kdjk'].loc[20110104], near_to(60.5263))
        assert_that(stock['kdjd'].loc[20110104], near_to(53.5087))
        assert_that(stock['kdjj'].loc[20110104], near_to(74.5614))

    def test_column_kdjk(self):
        stock = self.get_stock_20days()
        kdjk_3 = stock['kdjk_3']
        assert_that(kdjk_3.loc[20110104], near_to(60.5263))
        assert_that(kdjk_3.loc[20110120], near_to(31.2133))

    def test_column_kdjd(self):
        stock = self.get_stock_20days()
        kdjk_3 = stock['kdjd_3']
        assert_that(kdjk_3.loc[20110104], near_to(53.5087))
        assert_that(kdjk_3.loc[20110120], near_to(43.1347))

    def test_column_kdjj(self):
        stock = self.get_stock_20days()
        kdjk_3 = stock['kdjj_3']
        assert_that(kdjk_3.loc[20110104], near_to(74.5614))
        assert_that(kdjk_3.loc[20110120], near_to(7.37))

    def test_z_kdj(self):
        stock = self.get_stock_90days()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            stock[col] = stock[f'{col}_20_z']
        _ = stock[['kdjk', 'kdjd', 'kdjj']]
        row = stock.loc[20110104]
        assert_that(row['kdjk'], near_to(66.666))
        assert_that(row['kdjd'], near_to(55.555))
        assert_that(row['kdjj'], near_to(88.888))

    def test_column_cross(self):
        stock = self.get_stock_30days()
        cross = stock['kdjk_3_x_kdjd_3']
        assert_that(sum(cross), equal_to(2))
        assert_that(cross.loc[20110114], equal_to(True))
        assert_that(cross.loc[20110125], equal_to(True))

    def test_column_cross_up(self):
        stock = self.get_stock_30days()
        cross = stock['kdjk_3_xu_kdjd_3']
        assert_that(sum(cross), equal_to(1))
        assert_that(cross.loc[20110125], equal_to(True))

    def test_column_cross_down(self):
        stock = self.get_stock_30days()
        cross = stock['kdjk_3_xd_kdjd_3']
        assert_that(sum(cross), equal_to(1))
        assert_that(cross.loc[20110114], equal_to(True))

    def test_column_sma(self):
        stock = self.get_stock_20days()
        sma_2 = stock['open_2_sma']
        assert_that(sma_2.loc[20110104], near_to(12.42))
        assert_that(sma_2.loc[20110105], near_to(12.56))

    def test_column_smma(self):
        stock = self.get_stock_20days()
        smma = stock['high_5_smma']
        assert_that(smma.loc[20110120], near_to(13.0394))

    def test_column_ema(self):
        stock = self.get_stock_20days()
        ema_5 = stock['close_5_ema']
        assert_that(ema_5.loc[20110107], near_to(12.9026))
        assert_that(ema_5.loc[20110110], near_to(12.9668))

    @staticmethod
    def test_ema_of_empty_df():
        s = Sdf.retype(pd.DataFrame())
        ema = s['close_10_ema']
        assert_that(len(ema), equal_to(0))

    def test_column_macd(self):
        stock = self.get_stock_90days()
        stock.get('macd')
        record = stock.loc[20110225]
        assert_that(record['macd'], near_to(-0.0382))
        assert_that(record['macds'], near_to(-0.0101))
        assert_that(record['macdh'], near_to(-0.02805))

    def test_column_macds(self):
        stock = self.get_stock_90days()
        stock.get('macds')
        record = stock.loc[20110225]
        assert_that(record['macds'], near_to(-0.0101))

    def test_column_macdh(self):
        stock = self.get_stock_90days()
        stock.get('macdh')
        record = stock.loc[20110225]
        assert_that(record['macdh'], near_to(-0.02805))

    def test_ppo(self):
        stock = self.get_stock_90days()
        _ = stock[['ppo', 'ppos', 'ppoh']]
        assert_that(stock['ppo'].loc[20110331], near_to(1.1190))
        assert_that(stock['ppos'].loc[20110331], near_to(0.6840))
        assert_that(stock['ppoh'].loc[20110331], near_to(0.4349))

    def test_eri(self):
        stock = self.get_stock_90days()
        bull = stock['eribull']
        bear = stock['eribear']
        assert_that(bull[20110104], near_to(0.070))
        assert_that(bear[20110104], near_to(-0.309))
        assert_that(bull[20110222], near_to(0.099))
        assert_that(bear[20110222], near_to(-0.290))

        bull = stock['eribull_13']
        bear = stock['eribear_13']
        assert_that(bull[20110104], near_to(0.070))
        assert_that(bear[20110104], near_to(-0.309))
        assert_that(bull[20110222], near_to(0.099))
        assert_that(bear[20110222], near_to(-0.290))

        bull = stock['eribull_10']
        bear = stock['eribear_10']
        assert_that(bull[20110222], near_to(0.092))
        assert_that(bear[20110222], near_to(-0.297))

    def test_column_mstd(self):
        stock = self.get_stock_20days()
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

    def test_bollinger_with_window(self):
        stock = self.get_stock().within(20140930, 20141211)
        _ = stock['boll_20']
        idx = 20141103
        assert_that(stock['boll_20'].loc[idx], near_to(9.8035))
        assert_that(stock['boll_ub_20'].loc[idx], near_to(10.1310))
        assert_that(stock['boll_lb_20'].loc[idx], near_to(9.4759))
        _ = stock['boll_5']
        assert_that(stock['boll_ub_5'].loc[idx], near_to(10.44107))

    def test_bollinger_empty(self):
        stock = self.get_stock().within(18800101, 18900101)
        s = stock['boll_ub']
        assert_that(len(s), equal_to(0))

    def test_column_mvar(self):
        stock = self.get_stock_20days()
        mvar_3 = stock['open_3_mvar']
        assert_that(mvar_3.loc[20110106], near_to(0.0292))

    def test_column_parse_error(self):
        stock = self.get_stock_90days()
        with self.assertRaises(UserWarning):
            _ = stock["foobarbaz"]
        with self.assertRaises(KeyError):
            _ = stock["close_1_foo_3_4"]

    @staticmethod
    def test_parse_column_name_1():
        c, r, t = Sdf.parse_column_name('amount_-5~-1_p')
        assert_that(c, equal_to('amount'))
        assert_that(r, equal_to('-5~-1'))
        assert_that(t, equal_to('p'))

    @staticmethod
    def test_parse_column_name_2():
        c, r, t = Sdf.parse_column_name('open_+2~4_d')
        assert_that(c, equal_to('open'))
        assert_that(r, equal_to('+2~4'))
        assert_that(t, equal_to('d'))

    @staticmethod
    def test_parse_column_name_stacked():
        c, r, t = Sdf.parse_column_name('open_-1_d_-1~-3_p')
        assert_that(c, equal_to('open_-1_d'))
        assert_that(r, equal_to('-1~-3'))
        assert_that(t, equal_to('p'))

    @staticmethod
    def test_parse_column_name_3():
        c, r, t = Sdf.parse_column_name('close_-3,-1,+2_p')
        assert_that(c, equal_to('close'))
        assert_that(r, equal_to('-3,-1,+2'))
        assert_that(t, equal_to('p'))

    @staticmethod
    def test_parse_column_name_max():
        c, r, t = Sdf.parse_column_name('close_-3,-1,+2_max')
        assert_that(c, equal_to('close'))
        assert_that(r, equal_to('-3,-1,+2'))
        assert_that(t, equal_to('max'))

    @staticmethod
    def test_parse_column_name_float():
        c, r, t = Sdf.parse_column_name('close_12.32_le')
        assert_that(c, equal_to('close'))
        assert_that(r, equal_to('12.32'))
        assert_that(t, equal_to('le'))

    @staticmethod
    def test_parse_column_name_stacked_xu():
        c, r, t = Sdf.parse_column_name('cr-ma2_xu_cr-ma1_20_c')
        assert_that(c, equal_to('cr-ma2_xu_cr-ma1'))
        assert_that(r, equal_to('20'))
        assert_that(t, equal_to('c'))

    @staticmethod
    def test_parse_column_name_rsv():
        c, r = Sdf.parse_column_name('rsv_9')
        assert_that(c, equal_to('rsv'))
        assert_that(r, equal_to('9'))

    @staticmethod
    def test_parse_column_name_no_match():
        ret = Sdf.parse_column_name('no match')
        assert_that(len(ret), equal_to(1))
        assert_that(ret[0], none())

    def test_to_int_split(self):
        shifts = self._supor.to_ints('5,1,3, -2')
        assert_that(shifts, contains_exactly(-2, 1, 3, 5))

    def test_to_int_continue(self):
        shifts = self._supor.to_ints('3, -3~-1, 5')
        assert_that(shifts, contains_exactly(-3, -2, -1, 3, 5))

    def test_to_int_dedup(self):
        shifts = self._supor.to_ints('3, -3~-1, 5, -2~-1')
        assert_that(shifts, contains_exactly(-3, -2, -1, 3, 5))

    @staticmethod
    def test_is_cross_columns():
        assert_that(Sdf.is_cross_columns('a_x_b'), equal_to(True))
        assert_that(Sdf.is_cross_columns('a_xu_b'), equal_to(True))
        assert_that(Sdf.is_cross_columns('a_xd_b'), equal_to(True))
        assert_that(Sdf.is_cross_columns('a_xx_b'), equal_to(False))
        assert_that(Sdf.is_cross_columns('a_xa_b'), equal_to(False))
        assert_that(Sdf.is_cross_columns('a_x_'), equal_to(False))
        assert_that(Sdf.is_cross_columns('_xu_b'), equal_to(False))
        assert_that(Sdf.is_cross_columns('_xd_'), equal_to(False))

    @staticmethod
    def test_parse_cross_column():
        assert_that(Sdf.parse_cross_column('a_x_b'),
                    contains_exactly('a', 'x', 'b'))

    @staticmethod
    def test_parse_cross_column_xu():
        assert_that(Sdf.parse_cross_column('a_xu_b'),
                    contains_exactly('a', 'xu', 'b'))

    def test_get_log_ret(self):
        stock = self.get_stock_30days()
        stock.get('log-ret')
        assert_that(stock.loc[20110128]['log-ret'], near_to(-0.010972))

    @staticmethod
    def test_rsv_nan_value():
        df = wrap(pd.read_csv(get_file('asml.as.csv')))
        assert_that(df['rsv_9'][0], equal_to(0.0))

    def test_unwrap(self):
        _ = self._supor['boll']
        df = unwrap(self._supor)
        assert_that(df, instance_of(pd.DataFrame))
        assert_that(df['boll'].loc[20160817], near_to(39.6120))

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
        stock = self.get_stock_90days()
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
        stock.drop('amount', axis=1, inplace=True)
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

        c = self._supor.get('close_10,50_dma')
        assert_that(c.loc[20160817], near_to(2.078))

        c = self._supor.get('high_5,10_dma')
        assert_that(c.loc[20160817], near_to(0.174))

    def test_pdm_ndm(self):
        c = self.get_stock_90days()

        pdm = c['pdm_14']
        assert_that(pdm.loc[20110104], equal_to(0))
        assert_that(pdm.loc[20110331], near_to(.0842))

        ndm = c['ndm_14']
        assert_that(ndm.loc[20110104], equal_to(0))
        assert_that(ndm.loc[20110331], near_to(0.0432))

    def test_get_pdi(self):
        c = self._supor.get('pdi')
        assert_that(c.loc[20160817], near_to(25.747))
        assert_that(c.loc[20160816], near_to(27.948))
        assert_that(c.loc[20160815], near_to(24.646))

    def test_get_mdi(self):
        c = self._supor.get('ndi')
        assert_that(c.loc[20160817], near_to(16.195))
        assert_that(c.loc[20160816], near_to(17.579))
        assert_that(c.loc[20160815], near_to(19.542))

    def test_dx(self):
        c = self._supor.get('dx')
        assert_that(c.loc[20160817], near_to(22.774))
        assert_that(c.loc[20160815], near_to(11.550))
        assert_that(c.loc[20160812], near_to(4.828))

    def test_adx(self):
        c = self._supor.get('adx')
        assert_that(c.loc[20160817], near_to(15.535))
        assert_that(c.loc[20160816], near_to(12.640))
        assert_that(c.loc[20160815], near_to(8.586))

    def test_adxr(self):
        c = self._supor.get('adxr')
        assert_that(c.loc[20160817], near_to(13.208))
        assert_that(c.loc[20160816], near_to(12.278))
        assert_that(c.loc[20160815], near_to(12.133))

    def test_trix_default(self):
        c = self._supor.get('trix')
        assert_that(c.loc[20160817], near_to(0.1999))
        assert_that(c.loc[20160816], near_to(0.2135))
        assert_that(c.loc[20160815], near_to(0.24))

        c = self._supor.get('close_12_trix')
        assert_that(c.loc[20160815], near_to(0.24))

        c = self._supor.get('high_12_trix')
        assert_that(c.loc[20160815], near_to(0.235))

    def test_tema_default(self):
        c = self._supor.get('tema')
        a = self._supor.get('close_5_tema')
        assert_that(c.loc[20160817], equal_to(a.loc[20160817]))
        assert_that(c.loc[20160817], near_to(40.2883))
        assert_that(c.loc[20160816], near_to(39.6371))
        assert_that(c.loc[20160815], near_to(39.3778))

        c = self._supor.get('high_3_tema')
        assert_that(c.loc[20160815], near_to(39.7315))

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
        stock = self.get_stock_90days()
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

    def test_ker(self):
        stock = self.get_stock_90days()
        k = stock['ker']
        assert_that(k[20110104], equal_to(0))
        assert_that(k[20110105], equal_to(1))
        assert_that(k[20110210], near_to(0.305))

        k = stock['close_10_ker']
        assert_that(k[20110104], equal_to(0))
        assert_that(k[20110105], equal_to(1))
        assert_that(k[20110210], near_to(0.305))

        k = stock['high_5_ker']
        assert_that(k[20110210], near_to(0.399))

    def test_column_kama(self):
        stock = self.get_stock_90days()
        kama_10 = stock['close_10,2,30_kama']
        assert_that(kama_10.loc[20110331], near_to(13.6648))

    def test_kama_with_default_fast_slow(self):
        stock = self.get_stock_90days()
        kama_2 = stock['close_2_kama']
        assert_that(kama_2.loc[20110331], near_to(13.7326))

    def test_vwma(self):
        stock = self.get_stock_90days()
        vwma = stock['vwma']
        vwma_7 = stock['vwma_7']
        vwma_14 = stock['vwma_14']
        assert_that(vwma.loc[20110330], near_to(13.312679))
        idx = 20110331
        assert_that(vwma.loc[idx], near_to(13.350941))
        assert_that(vwma_14.loc[idx], near_to(vwma.loc[idx]))
        assert_that(vwma_7.loc[idx], is_not(near_to(vwma.loc[idx])))

    def test_chop(self):
        stock = self.get_stock_90days()
        chop = stock['chop']
        chop_7 = stock['chop_7']
        chop_14 = stock['chop_14']
        idx = 20110330
        assert_that(chop.loc[idx], near_to(44.8926))
        assert_that(chop_14.loc[idx], near_to(chop.loc[idx]))
        assert_that(chop_7.loc[idx], is_not(near_to(chop.loc[idx])))

    def test_column_conflict(self):
        stock = self.get_stock_90days()
        res = stock[['close_26_ema', 'macd']]
        idx = 20110331
        assert_that(res['close_26_ema'].loc[idx], near_to(13.2488))
        assert_that(res['macd'].loc[idx], near_to(0.1482))

    def test_wave_trend(self):
        stock = self.get_stock_90days()
        wt1, wt2 = stock['wt1'], stock['wt2']
        idx = 20110331
        assert_that(wt1.loc[idx], near_to(38.9610))
        assert_that(wt2.loc[idx], near_to(31.6997))

        wt1, wt2 = stock['wt1_10,21'], stock['wt2_10,21']
        assert_that(wt1.loc[idx], near_to(38.9610))
        assert_that(wt2.loc[idx], near_to(31.6997))

    def test_init_all(self):
        stock = self.get_stock_90days()
        stock.init_all()
        columns = stock.columns
        assert_that(columns, has_items(
            'macd', 'kdjj', 'mfi', 'boll',
            'adx', 'cr-ma2', 'supertrend_lb', 'boll_lb',
            'ao', 'cti', 'ftr', 'psl'))

    def test_supertrend(self):
        stock = self.get_stock_90days()
        st = stock['supertrend']
        st_ub = stock['supertrend_ub']
        st_lb = stock['supertrend_lb']

        idx = 20110302
        assert_that(st[idx], near_to(13.3430))
        assert_that(st_ub[idx], near_to(13.3430))
        assert_that(st_lb[idx], near_to(12.2541))

        idx = 20110331
        assert_that(st[idx], near_to(12.9021))
        assert_that(st_ub[idx], near_to(14.6457))
        assert_that(st_lb[idx], near_to(12.9021))

    def test_ao(self):
        stock = self.get_stock_90days()
        ao = stock['ao']
        ao1 = stock['ao_5,34']
        ao2 = stock['ao_5,10']
        idx = 20110302
        assert_that(ao[idx], near_to(-0.112))
        assert_that(ao1[idx], equal_to(ao[idx]))
        assert_that(ao2[idx], near_to(-0.071))

    def test_bop(self):
        stock = self.get_stock_30days()
        bop = stock['bop']
        assert_that(bop[20110104], near_to(0.5))
        assert_that(bop[20110106], near_to(-0.207))

    def test_cmo(self):
        stock = self.get_stock_30days()
        cmo = stock['cmo']
        assert_that(cmo[20110104], equal_to(0))
        assert_that(cmo[20110126], near_to(7.023))
        assert_that(cmo[20110127], near_to(-16.129))

        cmo_14 = stock['cmo_14']
        assert_that(cmo_14[20110126], near_to(7.023))

        cmo_5 = stock['cmo_5']
        assert_that(cmo_5[20110126], near_to(7.895))

    def test_drop_column_inplace(self):
        stock = self._supor[:20]
        stock.columns.name = 'Luke'
        ret = stock.drop_column(['open', 'close'], inplace=True)

        assert_that(ret.columns.name, equal_to('Luke'))
        assert_that(ret.keys(), has_items('high', 'low'))
        assert_that(ret.keys(), not_has('open'))
        assert_that(ret.keys(), not_has('close'))
        assert_that(stock.keys(), has_items('high', 'low'))
        assert_that(stock.keys(), not_has('open'))
        assert_that(stock.keys(), not_has('close'))

    def test_drop_column(self):
        stock = self._supor[:20]
        stock.columns.name = 'Luke'
        ret = stock.drop_column(['open', 'close'])

        assert_that(ret.columns.name, equal_to('Luke'))
        assert_that(ret.keys(), has_items('high', 'low'))
        assert_that(ret.keys(), not_has('open'))
        assert_that(ret.keys(), not_has('close'))
        assert_that(stock.keys(), has_items('high', 'low', 'open', 'close'))

    def test_drop_head_inplace(self):
        stock = self._supor[:20]
        ret = stock.drop_head(9, inplace=True)
        assert_that(ret, has_length(11))
        assert_that(ret.iloc[0].name, equal_to(20040830))
        assert_that(stock, has_length(11))
        assert_that(stock.iloc[0].name, equal_to(20040830))

    def test_drop_head(self):
        stock = self._supor[:20]
        ret = stock.drop_head(9)
        assert_that(ret, has_length(11))
        assert_that(ret.iloc[0].name, equal_to(20040830))
        assert_that(stock, has_length(20))
        assert_that(stock.iloc[0].name, equal_to(20040817))

    def test_drop_tail_inplace(self):
        stock = self._supor[:20]
        ret = stock.drop_tail(9, inplace=True)
        assert_that(ret, has_length(11))
        assert_that(ret.iloc[-1].name, equal_to(20040831))
        assert_that(stock, has_length(11))
        assert_that(stock.iloc[-1].name, equal_to(20040831))

    def test_drop_tail(self):
        stock = self._supor[:20]
        ret = stock.drop_tail(9)
        assert_that(ret, has_length(11))
        assert_that(ret.iloc[-1].name, equal_to(20040831))
        assert_that(stock, has_length(20))
        assert_that(stock.iloc[-1].name, equal_to(20040913))

    def test_aroon(self):
        stock = self._supor[:50]
        _ = stock['aroon']
        assert_that(stock.loc[20040924, 'aroon'], equal_to(28))

        _ = stock['aroon_25']
        assert_that(stock.loc[20040924, 'aroon_25'], equal_to(28))

        _ = stock['aroon_5']
        assert_that(stock.loc[20040924, 'aroon_5'], equal_to(40))
        assert_that(stock.loc[20041020, 'aroon_5'], equal_to(-80))

    def test_close_z(self):
        stock = self._supor[:100]
        _ = stock['close_14_z']
        assert_that(stock.loc[20040817, 'close_14_z'], near_to(-0.7071))
        assert_that(stock.loc[20040915, 'close_14_z'], near_to(2.005))
        assert_that(stock.loc[20041014, 'close_14_z'], near_to(-2.014))

    def test_roc(self):
        stock = self._supor[:100]
        _ = stock['high_5_roc']
        assert_that(stock.loc[20040817, 'high_5_roc'], equal_to(0))
        assert_that(stock.loc[20040915, 'high_5_roc'], near_to(5.912))
        assert_that(stock.loc[20041014, 'high_5_roc'], near_to(5.009))
        assert_that(stock.loc[20041220, 'high_5_roc'], near_to(-4.776))

        s = StockDataFrame.roc(stock['high'], size=5)
        assert_that(s.loc[20040915], near_to(5.912))

    def test_mad(self):
        stock = self.get_stock_30days()
        s = stock['close_5_mad']
        assert_that(s[20110104], equal_to(0))
        assert_that(s[20110114], near_to(0.146))

    @staticmethod
    def test_mad_raw():
        series = pd.Series([10, 15, 15, 17, 18, 21])
        res = StockDataFrame._mad(series, 6)
        assert_that(res[5], near_to(2.667))

    @staticmethod
    def test_sym_wma4():
        series = pd.Series([4, 2, 2, 4, 8])
        res = StockDataFrame.sym_wma4(series)
        assert_that(res[0], equal_to(0))
        assert_that(res[2], equal_to(0))
        assert_that(res[3], near_to(2.666))
        assert_that(res[4], near_to(3.666))

    def test_ichimoku(self):
        stock = self.get_stock_90days()
        i0 = stock['ichimoku']
        i1 = stock['ichimoku_9,26,52']
        i2 = stock['ichimoku_5,10,20']
        assert_that(i0[20110228], equal_to(0))
        assert_that(i0[20110308], near_to(0.0275))
        assert_that(i0[20110318], near_to(-0.0975))

        assert_that(i1[20110228], equal_to(0))
        assert_that(i1[20110308], near_to(0.0275))
        assert_that(i1[20110318], near_to(-0.0975))

        assert_that(i2[20110228], near_to(-0.11))
        assert_that(i2[20110308], near_to(0.0575))
        assert_that(i2[20110318], near_to(0.0175))

    @staticmethod
    def test_linear_wma():
        series = pd.Series([10, 15, 15, 17, 18, 21])
        res = StockDataFrame.linear_wma(series, 6)
        assert_that(res[0], equal_to(0))
        assert_that(res[5], near_to(17.571))

    def test_coppock(self):
        stock = self.get_stock_90days()
        c0 = stock['coppock']
        assert_that(c0[20110117], equal_to(0))
        assert_that(c0[20110221], near_to(3.293))
        assert_that(c0[20110324], near_to(-2.274))

        c1 = stock['coppock_10,11,14']
        assert_that(c1[20110117], equal_to(0))
        assert_that(c1[20110221], near_to(3.293))
        assert_that(c1[20110324], near_to(-2.274))

        c2 = stock['coppock_5,10,15']
        assert_that(c2[20110117], equal_to(0))
        assert_that(c2[20110221], near_to(4.649))
        assert_that(c2[20110324], near_to(-2.177))

    @staticmethod
    def test_linear_regression_raw():
        arr = [1, 5, 7, 2, 4, 3, 7, 9, 2]
        series = pd.Series(arr)
        lg = StockDataFrame.linear_reg(series, 5)
        assert_that(lg.iloc[3], equal_to(0.0))
        assert_that(lg.iloc[8], equal_to(5.2))

        cr = StockDataFrame.linear_reg(
            series, 5, correlation=True)
        assert_that(cr.iloc[3], equal_to(0.0))
        assert_that(cr.iloc[8], near_to(0.108))

    def test_linear_regression(self):
        stock = self.get_stock_90days()
        lr = stock['close_10_lrma']
        assert_that(lr[20110114], equal_to(0))
        assert_that(lr[20110127], near_to(12.782))

    def test_cti(self):
        stock = self.get_stock_90days()
        cti = stock['cti']
        assert_that(cti[20110118], equal_to(0))
        assert_that(cti[20110131], near_to(-0.113))
        assert_that(cti[20110215], near_to(0.369))

        cti = stock['close_12_cti']
        assert_that(cti[20110118], equal_to(0))
        assert_that(cti[20110131], near_to(-0.113))
        assert_that(cti[20110215], near_to(0.369))

        cti = stock['high_10_cti']
        assert_that(cti[20110118], near_to(-0.006))
        assert_that(cti[20110131], near_to(-0.043))
        assert_that(cti[20110215], near_to(0.5006))

    def test_ftr(self):
        stock = self.get_stock_90days()
        f = stock['ftr']
        f9 = stock['ftr_9']
        assert_that(f[20110114], equal_to(0))
        assert_that(f[20110128], near_to(-1.135))
        assert_that(f9[20110128], equal_to(f[20110128]))

        f = stock['ftr_15']
        assert_that(f[20110128], near_to(-1.005))

    def test_rvgi(self):
        stock = self.get_stock_30days()
        r, s = stock['rvgi'], stock['rvgis']
        r14, s14 = stock['rvgi_14'], stock['rvgis_14']
        assert_that(r[20110128], equal_to(r14[20110128]))
        assert_that(s[20110128], equal_to(s14[20110128]))
        assert_that(r[20110106], equal_to(0))
        assert_that(r[20110107], near_to(0.257))
        assert_that(s[20110111], equal_to(0))
        assert_that(s[20110112], near_to(0.303))

        s10, r10 = stock['rvgis_10'], stock['rvgi_10']
        assert_that(r10[20110128], near_to(-0.056))
        assert_that(s10[20110128], near_to(-0.043))

    def test_change_group_window_defaults(self):
        stock = self.get_stock_90days()
        macd = stock['macd']
        ref = stock['macd_12,26,9']
        i = 20110225
        assert_that(macd[i], equal_to(ref[i]))

        orig = stockstats.set_dft_window('macd', (10, 20, 5))
        assert_that(orig, contains_exactly(12, 26, 9))
        stock.drop_column(['macd', 'macdh', 'macds'], inplace=True)
        macd = stock['macd']
        ref = stock['macd_10,20,5']
        assert_that(macd[i], equal_to(ref[i]))

        stockstats.set_dft_window('macd', orig)

    def test_inertia(self):
        stock = self.get_stock_90days()
        inertia = stock['inertia']
        assert_that(inertia[20110209], equal_to(0))
        assert_that(inertia[20110210], near_to(-0.024856))
        assert_that(inertia[20110304], near_to(0.155576))

        inertia_dft = stock['inertia_20,14']
        assert_that(inertia_dft[20110209], equal_to(0))
        assert_that(inertia_dft[20110210], near_to(-0.024856))
        assert_that(inertia_dft[20110304], near_to(0.155576))

        inertia14 = stock['inertia_20']
        assert_that(inertia14[20110209], equal_to(0))
        assert_that(inertia14[20110210], near_to(-0.024856))
        assert_that(inertia14[20110304], near_to(0.155576))

        inertia10 = stock['inertia_10']
        assert_that(inertia10[20110209], near_to(0.011085))
        assert_that(inertia10[20110210], near_to(-0.014669))

    def test_kst(self):
        stock = self.get_stock_90days()
        kst = stock['kst']
        assert_that(kst[20110117], equal_to(0))
        assert_that(kst[20110118], near_to(0.063442))
        assert_that(kst[20110131], near_to(-2.519985))

    def test_pgo(self):
        stock = self.get_stock_90days()
        pgo = stock['pgo']
        assert_that(pgo[20110117], near_to(-0.968845))
        assert_that(pgo[20110214], near_to(1.292029))

        pgo14 = stock['pgo_14']
        assert_that(pgo14[20110117], near_to(-0.968845))
        assert_that(pgo14[20110214], near_to(1.292029))

        pgo10 = stock['pgo_10']
        assert_that(pgo10[20110117], near_to(-0.959768))
        assert_that(pgo10[20110214], near_to(1.214206))

    def test_psl(self):
        stock = self.get_stock_90days()
        psl = stock['psl']
        assert_that(psl[20110118], near_to(41.666))
        assert_that(psl[20110127], near_to(50))

        psl12 = stock['psl_12']
        assert_that(psl12[20110118], near_to(41.666))
        assert_that(psl12[20110127], near_to(50))

        psl10 = stock['psl_10']
        assert_that(psl10[20110118], near_to(50))
        assert_that(psl10[20110131], near_to(60))

        high_psl12 = stock['high_12_psl']
        assert_that(high_psl12[20110118], near_to(41.666))
        assert_that(high_psl12[20110127], near_to(41.666))

    def test_pvo(self):
        stock = self.get_stock_90days()
        _ = stock[['pvo', 'pvos', 'pvoh']]
        assert_that(stock['pvo'].loc[20110331], near_to(3.4708))
        assert_that(stock['pvos'].loc[20110331], near_to(-3.7464))
        assert_that(stock['pvoh'].loc[20110331], near_to(7.2173))

    def test_qqe(self):
        stock = self.get_stock_90days()
        _ = stock['qqe']
        _ = stock['qqe_14,5']
        _ = stock['qqe_10,4']

        assert_that(stock.loc[20110125, 'qqe'], near_to(44.603))
        assert_that(stock.loc[20110125, 'qqel'], near_to(44.603))
        assert_that(stock.loc[20110125, 'qqes'], near_to(0))

        assert_that(stock.loc[20110223, 'qqe'], near_to(53.26))
        assert_that(stock.loc[20110223, 'qqel'], near_to(0))
        assert_that(stock.loc[20110223, 'qqes'], near_to(53.26))

        assert_that(stock.loc[20110125, 'qqe_14,5'], near_to(44.603))
        assert_that(stock.loc[20110125, 'qqe_10,4'], near_to(39.431))
