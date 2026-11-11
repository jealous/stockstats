"""Tests for the pure static helpers on ``StockDataFrame``.

These helpers are exercised directly (no DataFrame column access), so
each expectation is a plain numeric fact measured by probing.
"""

import math

import numpy as np
import pandas as pd
import pytest

from stockstats import StockDataFrame


def test_roc_size_zero_is_all_zeros():
    out = StockDataFrame.roc(pd.Series([1.0, 2.0, 3.0]), 0)
    assert list(out) == [0.0, 0.0, 0.0]


def test_roc_positive_window():
    out = StockDataFrame.roc(pd.Series([10.0, 20.0, 30.0]), 1)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(100.0, rel=1e-3)
    assert out[2] == pytest.approx(50.0, rel=1e-3)


def test_roc_negative_window_looks_forward():
    out = StockDataFrame.roc(pd.Series([10.0, 20.0, 30.0]), -1)
    assert out[0] == pytest.approx(-50.0, rel=1e-3)
    assert out[1] == pytest.approx(-33.3333, rel=1e-3)
    assert out[2] == 0.0


def test_roc_zero_base_gives_inf_quirk():
    # Quirk, pinned as-is: roc() does not guard division by zero, unlike
    # _divide(); a zero base yields inf (with a RuntimeWarning).
    out = StockDataFrame.roc(pd.Series([0.0, 10.0, 20.0]), 1)
    assert out[0] == 0.0
    assert math.isinf(out[1])
    assert out[2] == pytest.approx(100.0, rel=1e-3)


def test_s_shift_zero_returns_copy():
    series = pd.Series([1.0, 2.0, 3.0])
    out = StockDataFrame.s_shift(series, 0)
    assert list(out) == [1.0, 2.0, 3.0]
    assert out is not series


def test_s_shift_positive_pulls_future_and_repeats_last():
    out = StockDataFrame.s_shift(pd.Series([1.0, 2.0, 3.0, 4.0]), 1)
    assert list(out) == [2.0, 3.0, 4.0, 4.0]


def test_s_shift_negative_pulls_past_and_repeats_first():
    out = StockDataFrame.s_shift(pd.Series([1.0, 2.0, 3.0, 4.0]), -1)
    assert list(out) == [1.0, 1.0, 2.0, 3.0]


def test_s_shift_empty_series():
    out = StockDataFrame.s_shift(pd.Series([], dtype=float), 3)
    assert len(out) == 0


def test_set_nan_positive_shift_nans_tail():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    StockDataFrame.set_nan(series, 2)
    assert series[0] == 1.0 and series[1] == 2.0
    assert math.isnan(series[2]) and math.isnan(series[3])


def test_set_nan_negative_shift_nans_head():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    StockDataFrame.set_nan(series, -2)
    assert math.isnan(series[0]) and math.isnan(series[1])
    assert series[2] == 3.0 and series[3] == 4.0


def test_set_nan_list_nans_both_ends():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    StockDataFrame.set_nan(series, [1, 3])
    assert series[0] == 1.0
    assert math.isnan(series[1])
    assert math.isnan(series[2])
    assert math.isnan(series[3])


def test_divide_zero_divisor_gives_zero_not_inf():
    out = StockDataFrame._divide(
        np.array([1.0, 2.0]), np.array([0.0, 2.0]))
    assert list(out) == [0.0, 1.0]


def test_divide_empty_arrays():
    out = StockDataFrame._divide(np.array([]), np.array([]))
    assert len(out) == 0


def test_rolling_sum_partial_windows():
    out = StockDataFrame._rolling_sum(np.array([1.0, 0.0, 1.0, 1.0]), 2)
    assert list(out) == [1.0, 1.0, 1.0, 2.0]


def test_mad_window_larger_than_series_is_all_nan():
    out = StockDataFrame._mad(pd.Series([1.0, 2.0]), 5)
    assert math.isnan(out[0]) and math.isnan(out[1])


def test_mad_exact_value():
    out = StockDataFrame._mad(pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0]), 6)
    assert out[5] == pytest.approx(0.6667, rel=1e-3)


def test_linear_wma_window_larger_than_series_is_zeros():
    out = StockDataFrame.linear_wma(pd.Series([1.0, 2.0]), 5)
    assert list(out) == [0.0, 0.0]


def test_linear_wma_exact_value_and_wma_alias():
    series = pd.Series([10.0, 15.0, 15.0, 17.0, 18.0, 21.0])
    out = StockDataFrame.linear_wma(series, 6)
    assert out[0] == 0.0
    assert out[5] == pytest.approx(17.5714, rel=1e-3)
    assert list(StockDataFrame.wma(series, 6)) == list(out)


def test_sym_wma4_exact_value():
    out = StockDataFrame.sym_wma4(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert list(out[:3]) == [0.0, 0.0, 0.0]
    assert out[3] == pytest.approx(2.5, rel=1e-3)


def test_hma_window_one_is_identity():
    series = pd.Series([10.0, 15.0, 15.0, 17.0, 18.0, 21.0])
    assert list(StockDataFrame.hma(series, 1)) == list(series)


def test_hma_window_larger_than_series_is_zeros():
    series = pd.Series([10.0, 15.0, 15.0, 17.0, 18.0, 21.0])
    assert list(StockDataFrame.hma(series, 10)) == [0.0] * 6


def test_ssl_channel_two_rows_bullish():
    high = pd.Series([10.0, 12.0])
    low = pd.Series([8.0, 10.0])
    close = pd.Series([11.0, 13.0])
    up, down = StockDataFrame.ssl_channel(high, low, close, 2)
    # SMA(high, 2) = [10, 11], SMA(low, 2) = [8, 9]; close stays above
    # the high SMA, so hlv stays +1 and up is the high SMA.
    assert list(up) == [10.0, 11.0]
    assert list(down) == [8.0, 9.0]


def test_ssl_channel_empty_series():
    up, down = StockDataFrame.ssl_channel(
        pd.Series(dtype=float), pd.Series(dtype=float),
        pd.Series(dtype=float), 5)
    assert len(up) == 0 and len(down) == 0


def test_linear_reg_window_larger_than_series_is_zeros():
    series = pd.Series([1.0, 2.0])
    assert list(StockDataFrame.linear_reg(series, 5)) == [0.0, 0.0]
    assert list(StockDataFrame.linear_reg(series, 5, correlation=True)) == [
        0.0, 0.0]


def test_linear_reg_perfect_line_predicts_next_point():
    # Perfectly linear series 1..6, window 3: the fitted line evaluated
    # at x = window - 1 gives 2.0, 3.0, 4.0, 5.0 at the window ends.
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = StockDataFrame.linear_reg(series, 3)
    assert list(out[:2]) == [0.0, 0.0]
    assert list(out[2:]) == [2.0, 3.0, 4.0, 5.0]


def test_ema_and_sma_are_plain_pandas_rolling():
    series = pd.Series([10.0, 11.0, 12.0, 13.0])
    assert list(StockDataFrame.sma(series, 2)) == [
        10.0, 10.5, 11.5, 12.5]
    ema = StockDataFrame.ema(series, 2)
    assert ema[0] == 10.0
    assert ema.iloc[-1] == pytest.approx(12.55, rel=1e-3)


def test_mov_min_max_std_var():
    series = pd.Series([3.0, 1.0, 2.0])
    assert list(StockDataFrame.mov_min(series, 2)) == [3.0, 1.0, 1.0]
    assert list(StockDataFrame.mov_max(series, 2)) == [3.0, 3.0, 2.0]
    assert StockDataFrame.mov_sum(series, 2)[2] == pytest.approx(3.0, rel=1e-3)
    # Single-value rolling std is NaN (pandas default ddof=1).
    assert math.isnan(StockDataFrame.mov_std(series, 2)[0])
    assert math.isnan(StockDataFrame.mov_var(series, 2)[0])
