"""Tests for the column-name machinery: ``_Meta``, default windows and
column parsing.

All values below were measured by probing the implementation, not taken
from the documentation. The upstream suite covers a few of these paths;
this file pins the rest, including the ``~`` range syntax and the
partial-window fallback to defaults.
"""

import pytest

import stockstats
from stockstats import (
    StockDataFrame,
    _Meta,
    dft_column,
    dft_windows,
    set_dft_window,
)


def test_dft_windows_int_and_tuple():
    assert dft_windows("rsi") == "14"
    assert dft_windows("macd") == "12,26,9"
    assert dft_windows("boll") == "20"


def test_dft_windows_unknown_is_none():
    assert dft_windows("no_such_indicator") is None


def test_dft_column_known_and_unknown():
    assert dft_column("dma") == "close"
    assert dft_column("kama") == "close"
    assert dft_column("rsi") is None
    assert dft_column("no_such_indicator") is None


def test_meta_single_window():
    meta = _Meta("rsi", windows="5")
    assert meta.ints == [5]
    assert meta.as_int == 5
    assert meta.name == "rsi_5"


def test_meta_range_window():
    meta = _Meta("rsi", windows="1~3")
    assert meta.ints == [1, 2, 3]


def test_meta_comma_windows():
    meta = _Meta("macd", windows="5,10,3")
    assert meta.ints == [5, 10, 3]
    assert meta.name == "macd_5,10,3"


def test_meta_partial_windows_fall_back_to_defaults():
    # Only two windows given for MACD; the third falls back to the
    # default signal window (9), and the name keeps the given windows.
    meta = _Meta("macd", windows="5,10")
    assert meta.int0 == 5
    assert meta.int1 == 10
    assert meta.int2 == 9
    assert meta.name == "macd_5,10"
    assert meta.name_ex("s") == "macds_5,10"


def test_meta_name_without_windows():
    meta = _Meta("rsi")
    assert meta.name == "rsi"
    assert meta.name_ex("s") == "rsis"


def test_meta_name_with_column_and_windows():
    meta = _Meta("dma", windows="10,50", column="high")
    assert meta.column == "high"
    assert meta.name == "high_10,50_dma"


def test_meta_column_defaults_to_dft_column():
    meta = _Meta("dma", windows="10,50")
    assert meta.column == "close"


def test_meta_column_none_when_no_default():
    meta = _Meta("rsi", windows="5")
    assert meta.column is None


def test_meta_as_int_rejects_multiple_numbers():
    with pytest.raises(stockstats.StockStatsError):
        _Meta("macd", windows="5,10").as_int


def test_meta_positive_int_rejects_zero():
    with pytest.raises(stockstats.StockStatsError):
        _Meta("rsi", windows="0").positive_int


def test_meta_positive_int_accepts_positive():
    assert _Meta("rsi", windows="7").positive_int == 7


def test_meta_set_name():
    meta = _Meta("rvgi")
    assert meta.set_name("rvgi") is meta
    assert meta.name == "rvgi"


def test_set_dft_window_returns_old_and_restores():
    old = set_dft_window("rsi", 7)
    try:
        assert old == 14
        assert dft_windows("rsi") == "7"
    finally:
        set_dft_window("rsi", old)
    assert dft_windows("rsi") == "14"


def test_set_dft_window_new_name_returns_none():
    old = set_dft_window("zzz_probe_name", 3)
    try:
        assert old is None
        assert dft_windows("zzz_probe_name") == "3"
    finally:
        stockstats._dft_windows.pop("zzz_probe_name", None)


def test_parse_column_name_with_range():
    assert StockDataFrame.parse_column_name("close_1~3_max") == (
        "close", "1~3", "max")


def test_parse_column_name_plain_window():
    assert StockDataFrame.parse_column_name("close_5_dma") == (
        "close", "5", "dma")


def test_parse_column_name_two_part():
    assert StockDataFrame.parse_column_name("ppo_5,10,3") == (
        "ppo", "5,10,3")


def test_parse_column_name_no_match():
    assert StockDataFrame.parse_column_name("eri") == (None,)
    assert StockDataFrame.parse_column_name("bogus") == (None,)


def test_to_ints_sorted_and_deduped():
    assert StockDataFrame.to_ints("1~3,5,3") == [1, 2, 3, 5]
    assert StockDataFrame.to_ints("2") == [2]


def test_to_int_single_and_multi():
    assert StockDataFrame.to_int("7") == 7
    with pytest.raises(IndexError):
        StockDataFrame.to_int("1,2")


def test_process_shifts_segment_range_and_single():
    seg = StockDataFrame._process_shifts_segment
    assert list(seg("2~4")) == [2, 3, 4]
    assert seg("9") == [9]


def test_is_cross_columns():
    assert StockDataFrame.is_cross_columns("close_x_macd")
    assert StockDataFrame.is_cross_columns("close_xu_macd")
    assert StockDataFrame.is_cross_columns("close_xd_macd")
    assert not StockDataFrame.is_cross_columns("close_5_sma")


def test_is_compare_columns():
    for op in ("le", "ge", "lt", "gt", "eq", "ne"):
        assert StockDataFrame.is_compare_columns(f"close_{op}_macd")
    assert not StockDataFrame.is_compare_columns("close_x_macd")


def test_parse_cross_column():
    # Success returns a tuple; no match returns a list of Nones.
    assert StockDataFrame.parse_cross_column("close_xu_macd") == (
        "close", "xu", "macd")
    assert StockDataFrame.parse_cross_column("plain") == [None, None, None]


def test_parse_compare_column():
    assert StockDataFrame.parse_compare_column("close_ne_macd") == (
        "close", "ne", "macd")
    assert StockDataFrame.parse_compare_column("plain") == [
        None, None, None]


def test_get_int_positive_int_zero_quirk():
    # Quirk, pinned as-is: the int path does not check positivity, only
    # the string path does.
    assert StockDataFrame.get_int_positive(0) == 0
    with pytest.raises(IndexError):
        StockDataFrame.get_int_positive("0")
    assert StockDataFrame.get_int_positive("5") == 5
    assert StockDataFrame.get_int_positive(5) == 5
