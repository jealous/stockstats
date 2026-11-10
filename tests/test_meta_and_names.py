"""Tests for the column-name machinery: ``_Meta``, default windows and
column parsing.

All values below were measured by probing the implementation, not taken
from the documentation. The upstream suite covers a few of these paths;
this file pins the rest, including the ``~`` range syntax and the
partial-window fallback to defaults.
"""

from unittest import TestCase

import stockstats
from hamcrest import (
    assert_that,
    contains_exactly,
    equal_to,
    none,
    raises,
    same_instance,
)
from stockstats import (
    StockDataFrame,
    _Meta,
    dft_column,
    dft_windows,
    set_dft_window,
)


class MetaAndNamesTest(TestCase):
    def test_dft_windows_int_and_tuple(self):
        assert_that(dft_windows("rsi"), equal_to("14"))
        assert_that(dft_windows("macd"), equal_to("12,26,9"))
        assert_that(dft_windows("boll"), equal_to("20"))

    def test_dft_windows_unknown_is_none(self):
        assert_that(dft_windows("no_such_indicator"), none())

    def test_dft_column_known_and_unknown(self):
        assert_that(dft_column("dma"), equal_to("close"))
        assert_that(dft_column("kama"), equal_to("close"))
        assert_that(dft_column("rsi"), none())
        assert_that(dft_column("no_such_indicator"), none())

    def test_meta_single_window(self):
        meta = _Meta("rsi", windows="5")
        assert_that(meta.ints, contains_exactly(5))
        assert_that(meta.as_int, equal_to(5))
        assert_that(meta.name, equal_to("rsi_5"))

    def test_meta_range_window(self):
        meta = _Meta("rsi", windows="1~3")
        assert_that(meta.ints, contains_exactly(1, 2, 3))

    def test_meta_comma_windows(self):
        meta = _Meta("macd", windows="5,10,3")
        assert_that(meta.ints, contains_exactly(5, 10, 3))
        assert_that(meta.name, equal_to("macd_5,10,3"))

    def test_meta_partial_windows_fall_back_to_defaults(self):
        # Only two windows given for MACD; the third falls back to the
        # default signal window (9), and the name keeps the given windows.
        meta = _Meta("macd", windows="5,10")
        assert_that(meta.int0, equal_to(5))
        assert_that(meta.int1, equal_to(10))
        assert_that(meta.int2, equal_to(9))
        assert_that(meta.name, equal_to("macd_5,10"))
        assert_that(meta.name_ex("s"), equal_to("macds_5,10"))

    def test_meta_name_without_windows(self):
        meta = _Meta("rsi")
        assert_that(meta.name, equal_to("rsi"))
        assert_that(meta.name_ex("s"), equal_to("rsis"))

    def test_meta_name_with_column_and_windows(self):
        meta = _Meta("dma", windows="10,50", column="high")
        assert_that(meta.column, equal_to("high"))
        assert_that(meta.name, equal_to("high_10,50_dma"))

    def test_meta_column_defaults_to_dft_column(self):
        meta = _Meta("dma", windows="10,50")
        assert_that(meta.column, equal_to("close"))

    def test_meta_column_none_when_no_default(self):
        meta = _Meta("rsi", windows="5")
        assert_that(meta.column, none())

    def test_meta_as_int_rejects_multiple_numbers(self):
        assert_that(lambda: _Meta("macd", windows="5,10").as_int,
                    raises(stockstats.StockStatsError))

    def test_meta_positive_int_rejects_zero(self):
        assert_that(lambda: _Meta("rsi", windows="0").positive_int,
                    raises(stockstats.StockStatsError))

    def test_meta_positive_int_accepts_positive(self):
        assert_that(_Meta("rsi", windows="7").positive_int, equal_to(7))

    def test_meta_set_name(self):
        meta = _Meta("rvgi")
        assert_that(meta.set_name("rvgi"), same_instance(meta))
        assert_that(meta.name, equal_to("rvgi"))

    def test_set_dft_window_returns_old_and_restores(self):
        old = set_dft_window("rsi", 7)
        try:
            assert_that(old, equal_to(14))
            assert_that(dft_windows("rsi"), equal_to("7"))
        finally:
            set_dft_window("rsi", old)
        assert_that(dft_windows("rsi"), equal_to("14"))

    def test_set_dft_window_new_name_returns_none(self):
        old = set_dft_window("zzz_probe_name", 3)
        try:
            assert_that(old, none())
            assert_that(dft_windows("zzz_probe_name"), equal_to("3"))
        finally:
            # Test cleanup only, so the probe name does not leak into
            # the global defaults for the rest of the suite.
            stockstats._dft_windows.pop("zzz_probe_name", None)
        assert_that(dft_windows("zzz_probe_name"), none())

    def test_parse_column_name_with_range(self):
        assert_that(
            StockDataFrame.parse_column_name("close_1~3_max"),
            contains_exactly("close", "1~3", "max"))

    def test_parse_column_name_plain_window(self):
        assert_that(
            StockDataFrame.parse_column_name("close_5_dma"),
            contains_exactly("close", "5", "dma"))

    def test_parse_column_name_two_part(self):
        assert_that(
            StockDataFrame.parse_column_name("ppo_5,10,3"),
            contains_exactly("ppo", "5,10,3"))

    def test_to_int_single_and_multi(self):
        assert_that(StockDataFrame.to_int("7"), equal_to(7))
        assert_that(lambda: StockDataFrame.to_int("1,2"),
                    raises(IndexError))

    def test_process_shifts_segment_range_and_single(self):
        seg = StockDataFrame._process_shifts_segment
        assert_that(list(seg("2~4")), contains_exactly(2, 3, 4))
        assert_that(seg("9"), contains_exactly(9))

    def test_is_compare_columns(self):
        for op in ("le", "ge", "lt", "gt", "eq", "ne"):
            assert_that(
                StockDataFrame.is_compare_columns(f"close_{op}_macd"),
                equal_to(True))
        assert_that(StockDataFrame.is_compare_columns("close_x_macd"),
                    equal_to(False))

    def test_parse_cross_column_no_match(self):
        # The success cases are covered by the upstream suite; only the
        # no-match return is pinned here.
        assert_that(StockDataFrame.parse_cross_column("plain"),
                    contains_exactly(None, None, None))

    def test_parse_compare_column(self):
        assert_that(
            StockDataFrame.parse_compare_column("close_ne_macd"),
            contains_exactly("close", "ne", "macd"))
        assert_that(StockDataFrame.parse_compare_column("plain"),
                    contains_exactly(None, None, None))

    def test_get_int_positive_int_zero_quirk(self):
        # Quirk, pinned as-is: the int path does not check positivity,
        # only the string path does.
        assert_that(StockDataFrame.get_int_positive(0), equal_to(0))
        assert_that(lambda: StockDataFrame.get_int_positive("0"),
                    raises(IndexError))
        assert_that(StockDataFrame.get_int_positive("5"), equal_to(5))
        assert_that(StockDataFrame.get_int_positive(5), equal_to(5))
