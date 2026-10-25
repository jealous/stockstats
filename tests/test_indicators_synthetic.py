"""Indicator behaviour on synthetic flat / rising / falling series.

A flat series (all prices equal) and a perfectly rising series
(close = 10, 11, ..., 29) make most indicators take exact, hand-checkable
values. Every number below was measured by probing the implementation.
"""

import math

import numpy as np
import pandas as pd
import pytest

from stockstats import StockDataFrame, wrap


def make_frame(close, volume=1000.0):
    close = np.asarray(close, dtype=float)
    n = len(close)
    return wrap(pd.DataFrame({
        "date": range(n),
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, float(volume)),
    }))


def flat(n=20, value=10.0):
    # Truly flat candles: open == high == low == close, matching the
    # probe frame (a constant close with a high/low range is NOT flat:
    # it gives WR = -50, RSV = 50 and a non-zero supertrend).
    return wrap(pd.DataFrame({
        "date": range(n),
        "open": [value] * n,
        "high": [value] * n,
        "low": [value] * n,
        "close": [value] * n,
        "volume": np.full(n, 1000.0),
    }))


def rising(n=20):
    return make_frame(np.arange(10.0, 10.0 + n))


def falling(n=20):
    return make_frame(np.arange(29.0, 29.0 - n, -1.0))


def seeded(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return make_frame(10 + np.cumsum(rng.normal(0, 0.3, n)))


# --- flat series -------------------------------------------------------

def test_flat_rsi_is_50_everywhere():
    assert list(flat()["rsi"]) == [50.0] * 20
    assert list(flat()["rsi_5"]) == [50.0] * 20


def test_flat_stochrsi_wr_cci_atr_are_zero():
    frame = flat()
    assert list(frame["stochrsi"]) == [0.0] * 20
    assert list(frame["wr"]) == [-0.0] * 20
    assert list(frame["cci"]) == [0.0] * 20
    assert list(frame["atr"]) == [0.0] * 20


def test_flat_macd_family_is_zero():
    frame = flat()
    assert list(frame["macd"]) == [0.0] * 20
    assert list(frame["macds"]) == [0.0] * 20
    assert list(frame["macdh"]) == [0.0] * 20


def test_flat_bollinger_collapses_to_price_after_first_row():
    frame = flat()
    assert list(frame["boll"]) == [10.0] * 20
    # Quirk: the bands use the rolling std, whose first value is NaN
    # (single-sample std), so the first band values stay NaN.
    assert math.isnan(frame["boll_ub"][0])
    assert math.isnan(frame["boll_lb"][0])
    assert list(frame["boll_ub"][1:]) == [10.0] * 19
    assert list(frame["boll_lb"][1:]) == [10.0] * 19


def test_flat_kdj_decays_geometrically_from_zero_rsv():
    frame = flat()
    # RSV is 0 (high == low), so K = 2/3 * prev K, starting from 50.
    assert frame["kdjk"][0] == pytest.approx(33.3333, rel=1e-3)
    assert frame["kdjk"][1] == pytest.approx(22.2222, rel=1e-3)
    assert frame["kdjd"][0] == pytest.approx(44.4444, rel=1e-3)
    assert frame["kdjj"][0] == pytest.approx(11.1111, rel=1e-3)


def test_flat_momentum_indicators_are_neutral():
    frame = flat()
    assert list(frame["trix"]) == [0.0] * 20
    assert list(frame["tema"]) == [10.0] * 20
    assert list(frame["cmo"]) == [0.0] * 20
    assert list(frame["bop"]) == [0.0] * 20
    assert list(frame["kst"]) == [0.0] * 20
    assert list(frame["chop"]) == [0.0] * 20
    assert list(frame["ao"]) == [0.0] * 20
    assert list(frame["pgo"]) == [0.0] * 20
    assert list(frame["psl"]) == [0.0] * 20
    assert list(frame["rvgi"]) == [0.0] * 20
    assert list(frame["rvgis"]) == [0.0] * 20


def test_flat_mfi_is_half():
    assert list(flat()["mfi"]) == [0.5] * 20


def test_flat_cr_is_all_nan_quirk():
    # Quirk, pinned as-is: with no price movement both CR sums are 0,
    # and the division yields NaN instead of a guarded 0.
    assert all(math.isnan(v) for v in flat()["cr"])


def test_flat_qqe_ends_at_50():
    assert flat()["qqe"].iloc[-1] == pytest.approx(50.0, rel=1e-3)


def test_flat_supertrend_starts_at_zero_then_price():
    frame = flat()
    assert frame["supertrend"][0] == 0.0
    assert list(frame["supertrend"][1:]) == [10.0] * 19


# --- rising series -----------------------------------------------------

def test_rising_rsi_jumps_to_100():
    frame = rising()
    assert frame["rsi"][0] == 50.0
    assert list(frame["rsi"][1:]) == [100.0] * 19
    assert frame["rsi_5"].iloc[-1] == 100.0


def test_rising_kdj_first_and_last_values():
    frame = rising()
    assert frame["kdjk"][0] == pytest.approx(50.0, rel=1e-3)
    assert frame["kdjk"].iloc[-1] == pytest.approx(94.4017, rel=1e-3)
    assert frame["kdjd"].iloc[-1] == pytest.approx(94.1728, rel=1e-3)
    assert frame["kdjj"].iloc[-1] == pytest.approx(94.8594, rel=1e-3)


def test_rising_wr_first_and_last():
    frame = rising()
    assert frame["wr"][0] == -50.0
    assert frame["wr"].iloc[-1] == pytest.approx(-3.5714, rel=1e-3)


def test_rising_bollinger_first_row_and_last():
    frame = rising()
    assert frame["boll"][0] == 10.0
    assert math.isnan(frame["boll_ub"][0])
    assert frame["boll"].iloc[-1] == 19.5
    assert frame["boll_ub"].iloc[-1] == pytest.approx(31.3322, rel=1e-3)
    assert frame["boll_lb"].iloc[-1] == pytest.approx(7.6678, rel=1e-3)


def test_rising_macd_values():
    frame = rising()
    assert frame["macd"][0] == 0.0
    assert frame["macd"].iloc[-1] == pytest.approx(2.2709, rel=1e-3)
    assert frame["macds"].iloc[-1] == pytest.approx(1.6666, rel=1e-3)
    assert frame["macdh"].iloc[-1] == pytest.approx(0.6043, rel=1e-3)


def test_rising_ppo_last_and_constant_volume_pvo_zero():
    frame = rising()
    assert frame["ppo"].iloc[-1] == pytest.approx(10.3396, rel=1e-3)
    assert list(frame["pvo"]) == [0.0] * 20


def test_rising_trix_tema_dma():
    frame = rising()
    assert frame["trix"][0] == 0.0
    assert frame["trix"].iloc[-1] == pytest.approx(3.6812, rel=1e-3)
    assert frame["tema"][0] == 10.0
    assert frame["tema"].iloc[-1] == pytest.approx(29.0198, rel=1e-3)
    # dma default (10, 50): SMA10 - SMA50 at the last row.
    assert frame["dma"].iloc[-1] == pytest.approx(5.0, rel=1e-3)
    assert frame["close_5_dma"].iloc[-1] == pytest.approx(7.5, rel=1e-3)


def test_rising_cci_atr_tr():
    frame = rising()
    assert frame["cci"].iloc[-1] == pytest.approx(123.8095, rel=1e-3)
    assert frame["atr"][0] == 1.0
    assert frame["atr"].iloc[-1] == pytest.approx(1.4887, rel=1e-3)
    assert frame["tr"][0] == 1.0
    assert list(frame["tr"][1:]) == [1.5] * 19


def test_rising_directional_system():
    frame = rising()
    assert frame["dx"].iloc[-1] == 100.0
    assert frame["adx"].iloc[-1] == pytest.approx(99.9521, rel=1e-3)
    assert frame["adxr"].iloc[-1] == pytest.approx(99.6554, rel=1e-3)
    assert frame["pdi"].iloc[-1] == pytest.approx(65.6542, rel=1e-3)
    assert list(frame["ndi"]) == [0.0] * 20


def test_rising_wavetrend_and_wt2_identity():
    frame = rising()
    assert frame["wt1"][0] == 0.0
    assert frame["wt1"].iloc[-1] == pytest.approx(86.0289, rel=1e-3)
    assert list(frame["wt2"]) == list(StockDataFrame.sma(frame["wt1"], 4))


def test_rising_aroon_default_all_nan_short_series():
    # Window 25 over a 20-row series never fills.
    assert all(math.isnan(v) for v in rising()["aroon"])
    assert rising()["aroon_5"].iloc[-1] == 80.0


def test_rising_ssl_channels():
    frame = rising()
    assert frame["ssl"][0] == 10.5
    assert frame["ssl_up"].iloc[-1] == 25.0
    assert frame["ssl_down"].iloc[-1] == 24.0


def test_rising_ichimoku_is_zero_on_linear_series():
    assert list(rising()["ichimoku"]) == [0.0] * 20


def test_rising_coppock_last():
    assert rising()["coppock"].iloc[-1] == pytest.approx(163.8754, rel=1e-3)


def test_rising_inertia_is_zero_for_short_series():
    # len 20 < 20 + 14, so the short-series path returns zeros.
    assert list(rising()["inertia"]) == [0.0] * 20


def test_rising_supertrend_qqe_ftr_rvgi_kst_pgo_psl_cti():
    frame = rising()
    assert frame["supertrend"][0] == 0.0
    assert frame["qqe"].iloc[-1] == 100.0
    assert frame["qqes"].iloc[-1] == 0.0
    assert frame["ftr"][0] == 0.0
    assert frame["ftr"].iloc[-1] == pytest.approx(4.686, rel=1e-3)
    assert frame["rvgi"].iloc[-1] == pytest.approx(0.1, rel=1e-3)
    assert frame["rvgis"].iloc[-1] == pytest.approx(0.1, rel=1e-3)
    assert frame["kst"].iloc[-1] == pytest.approx(198.6554, rel=1e-3)
    assert frame["pgo"].iloc[-1] == pytest.approx(4.3468, rel=1e-3)
    assert frame["psl"].iloc[-1] == 100.0
    assert frame["cti"].iloc[-1] == 1.0


def test_rising_generic_moving_averages():
    frame = rising()
    assert frame["close_10_lrma"].iloc[-1] == 28.0
    assert frame["close_5_mad"].iloc[-1] == pytest.approx(1.2, rel=1e-3)
    assert frame["close_5_mstd"].iloc[-1] == pytest.approx(1.5811, rel=1e-3)
    assert frame["close_5_mvar"].iloc[-1] == pytest.approx(2.5, rel=1e-3)
    assert frame["close_5_sma"].iloc[-1] == 27.0
    assert frame["close_5_ema"].iloc[-1] == pytest.approx(27.006, rel=1e-3)
    assert frame["close_5_wma"].iloc[-1] == pytest.approx(27.6667, rel=1e-3)
    assert frame["close_5_hma"].iloc[-1] == pytest.approx(29.3333, rel=1e-3)


def test_rising_z_score_first_row_copies_second_quirk():
    frame = rising()
    # Quirk, pinned as-is: _get_z overwrites row 0 with row 1.
    assert frame["close_14_z"][0] == frame["close_14_z"][1]
    assert frame["close_14_z"][0] == pytest.approx(0.7071, rel=1e-3)


def test_rising_roc_change_rate_logret():
    frame = rising()
    assert frame["high_5_roc"].iloc[-1] == pytest.approx(20.4082, rel=1e-3)
    assert frame["close_5_roc"].iloc[-1] == pytest.approx(20.8333, rel=1e-3)
    assert frame["change"][0] == 0.0
    assert math.isnan(frame["rate"][0])
    assert frame["log-ret"][0] == 0.0
    assert list(frame["tp"]) == list(frame["middle"])


def test_rising_num_is_zero():
    assert list(rising()["num"]) == [0.0] * 20


# --- falling series ----------------------------------------------------

def test_falling_rsi_kdj_wr_psl_cci_cmo():
    frame = falling()
    assert frame["rsi"][0] == 50.0
    assert list(frame["rsi"][1:]) == [0.0] * 19
    assert frame["kdjk"].iloc[-1] == pytest.approx(5.5983, rel=1e-3)
    assert frame["wr"].iloc[-1] == pytest.approx(-96.4286, rel=1e-3)
    assert list(frame["psl"]) == [0.0] * 20
    assert frame["cci"].iloc[-1] == pytest.approx(-123.8095, rel=1e-3)
    assert frame["cmo"].iloc[-1] == -100.0


# --- identities on a seeded random series --------------------------------

def test_macd_histogram_identity():
    frame = seeded()
    _ = frame["macd"], frame["macds"], frame["macdh"]
    diff = frame["macd"] - frame["macds"] - frame["macdh"]
    assert float(np.abs(diff).max()) == 0.0


def test_kdjj_identity():
    frame = seeded()
    _ = frame["kdjk"], frame["kdjd"], frame["kdjj"]
    diff = 3 * frame["kdjk"] - 2 * frame["kdjd"] - frame["kdjj"]
    assert float(np.abs(diff).max()) == 0.0


def test_bollinger_width_matches_std():
    frame = seeded()
    _ = frame["boll"], frame["boll_ub"], frame["boll_lb"]
    width_quarter = (frame["boll_ub"] - frame["boll_lb"]) / 4
    std = StockDataFrame.mov_std(frame["close"], 20)
    assert float(np.nanmax(np.abs(width_quarter - std))) < 1e-9


def test_ppo_histogram_identity_and_qqe_state():
    frame = seeded()
    _ = frame["ppo"], frame["ppos"], frame["ppoh"]
    diff = frame["ppo"] - frame["ppos"] - frame["ppoh"]
    assert float(np.abs(diff).max()) == 0.0
    _ = frame["qqe"], frame["qqel"], frame["qqes"]
    assert bool(((frame["qqe"] == frame["qqel"])
                 | (frame["qqe"] == frame["qqes"])).all())


def test_supertrend_equals_a_band_or_zero():
    frame = seeded()
    _ = frame["supertrend"], frame["supertrend_ub"], frame["supertrend_lb"]
    ok = ((frame["supertrend"] == frame["supertrend_ub"])
          | (frame["supertrend"] == frame["supertrend_lb"])
          | (frame["supertrend"] == 0))
    assert bool(ok.all())


def test_eri_lines_match_adjust_false_ema():
    frame = seeded()
    _ = frame["eribull"], frame["eribear"]
    ema = StockDataFrame.ema(frame["close"], 13, adjust=False)
    assert float(np.abs(frame["eribull"] - (frame["high"] - ema)).max()) == 0.0
    assert float(np.abs(frame["eribear"] - (frame["low"] - ema)).max()) == 0.0


def test_dx_identity_from_pdi_ndi():
    frame = seeded()
    _ = frame["pdi"], frame["ndi"], frame["dx"]
    total = frame["pdi"] + frame["ndi"]
    expected = np.where(total != 0,
                        abs(frame["pdi"] - frame["ndi"]) / total, 0.0) * 100
    assert float(np.abs(expected - frame["dx"].values).max()) == 0.0


def test_dma_ema_sma_identities():
    frame = seeded()
    _ = frame["dma"]
    expected_dma = (StockDataFrame.sma(frame["close"], 10)
                    - StockDataFrame.sma(frame["close"], 50))
    assert float(np.abs(frame["dma"] - expected_dma).max()) == 0.0
    assert float(np.abs(frame["close_5_ema"]
                        - StockDataFrame.ema(frame["close"], 5)).max()) == 0.0
    assert float(np.abs(frame["close_5_sma"]
                        - StockDataFrame.sma(frame["close"], 5)).max()) == 0.0


def test_bollinger_10_identity():
    frame = seeded()
    _ = frame["boll_10"]
    # name_ex("_ub") appends the suffix before the window: boll_ub_10.
    expected_ub = (StockDataFrame.sma(frame["close"], 10)
                   + 2 * StockDataFrame.mov_std(frame["close"], 10))
    assert float(np.nanmax(
        np.abs(frame["boll_ub_10"] - expected_ub))) < 1e-9
