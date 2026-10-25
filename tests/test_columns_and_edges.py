"""Custom windows, column-syntax access, cross/compare columns and
edge cases (empty / single-row frames, frame helpers).

Several accesses below raise ``KeyError`` for the requested name while
still creating side-effect columns. Those are real quirks of the
name-dispatch code, pinned as current behaviour rather than as ideal
behaviour: the requested column is never created, but its sibling
columns are.
"""

import math

import numpy as np
import pandas as pd
import pytest

from stockstats import wrap


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
    # Truly flat candles: open == high == low == close.
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


# --- custom windows ----------------------------------------------------

def test_custom_window_values_rsv_stochrsi_wr_cci_atr():
    frame = rising()
    assert frame["rsv_5"][0] == 50.0
    assert frame["rsv_5"].iloc[-1] == 90.0
    assert frame["stochrsi_5"][1] == 100.0
    assert frame["stochrsi_5"].iloc[-1] == 0.0
    assert frame["wr_5"].iloc[-1] == -10.0
    assert frame["cci_5"].iloc[-1] == pytest.approx(111.1111, rel=1e-3)
    assert frame["atr_5"].iloc[-1] == pytest.approx(1.4985, rel=1e-3)


def test_custom_window_values_trix_tema_kdjk_macd():
    frame = rising()
    assert frame["trix_5"].iloc[-1] == pytest.approx(4.3936, rel=1e-3)
    assert frame["high_5_trix"].iloc[-1] == pytest.approx(4.2965, rel=1e-3)
    assert frame["tema_5"].iloc[-1] == pytest.approx(29.0198, rel=1e-3)
    assert list(frame["close_5_tema"]) == list(frame["tema_5"])
    assert frame["kdjk_5"].iloc[-1] == pytest.approx(89.975, rel=1e-3)
    assert frame["macd_5,10,3"].iloc[-1] == pytest.approx(2.1379, rel=1e-3)
    assert frame["macds_5,10,3"].iloc[-1] == pytest.approx(2.0706, rel=1e-3)
    assert frame["macdh_5,10,3"].iloc[-1] == pytest.approx(0.0673, rel=1e-3)


def test_custom_window_values_pdm_ndi_dx():
    frame = rising()
    assert frame["pdm_14"][1] == pytest.approx(0.5185, rel=1e-3)
    assert frame["pdm_14"].iloc[-1] == pytest.approx(0.9774, rel=1e-3)
    assert list(frame["ndm_14"]) == [0.0] * 20
    assert frame["pdi_14"].iloc[-1] == pytest.approx(65.6542, rel=1e-3)
    assert list(frame["ndi_14"]) == [0.0] * 20
    assert frame["dx_14"].iloc[-1] == 100.0


def test_custom_window_values_qqe_ao_coppock_ichimoku_kama_ker():
    frame = rising()
    assert frame["qqe_10,4"].iloc[-1] == 100.0
    assert frame["ao_5,10"].iloc[-1] == 2.5
    assert frame["coppock_5,10,15"].iloc[-1] == pytest.approx(176.5665, rel=1e-3)
    assert frame["ichimoku_5,10,20"].iloc[-1] == 1.25
    assert frame["kama_5,3,10"].iloc[-1] == 29.0
    assert frame["ker_5"].iloc[-1] == 1.0
    assert list(frame["close_5_ker"]) == list(frame["ker_5"])


def test_custom_window_values_ftr_rvgi_inertia_psl_pgo_cti():
    frame = rising()
    assert frame["ftr_5"].iloc[-1] == pytest.approx(7.4183, rel=1e-3)
    assert frame["rvgi_5"].iloc[-1] == pytest.approx(0.1, rel=1e-3)
    assert frame["rvgis_5"].iloc[-1] == pytest.approx(0.1, rel=1e-3)
    assert frame["inertia_10,5"].iloc[-1] == pytest.approx(0.1, rel=1e-3)
    assert frame["psl_5"].iloc[-1] == 100.0
    assert frame["high_12_psl"].iloc[-1] == 100.0
    assert frame["pgo_5"].iloc[-1] == pytest.approx(1.3334, rel=1e-3)
    assert frame["cti_5"].iloc[-1] == 1.0
    assert list(frame["close_5_cti"]) == list(frame["cti_5"])


def test_custom_window_values_vr_mfi_chop_vwma():
    frame = rising()
    # No down or flat days: both VR sums are 0, so VR is 0 by the
    # guarded division (not NaN).
    assert frame["vr_5"][0] == 100.0
    assert frame["vr_5"].iloc[-1] == 0.0
    assert frame["mfi_5"].iloc[-1] == 1.0
    assert frame["chop_5"].iloc[-1] == pytest.approx(25.193, rel=1e-3)
    assert frame["vwma_5"].iloc[-1] == 27.0
    assert list(frame["close_5_vwma"]) == list(frame["vwma_5"])
    assert frame["open_5_sma"].iloc[-1] == pytest.approx(26.9, rel=1e-3)


# --- column-syntax access ----------------------------------------------

def test_count_columns_past_and_future():
    frame = rising()
    assert list(frame["close_5_c"][:5]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert frame["close_5_c"].iloc[-1] == 5.0
    assert list(frame["close_5_fc"][-5:]) == [5.0, 4.0, 3.0, 2.0, 1.0]


def test_shift_columns_past_and_future():
    frame = rising()
    # _s: negative pulls the past (first value repeated), positive pulls
    # the future (last value repeated).
    assert list(frame["close_-2_s"][:3]) == [10.0, 10.0, 10.0]
    assert list(frame["close_2_s"][:3]) == [12.0, 13.0, 14.0]
    assert list(frame["close_2_s"][-2:]) == [29.0, 29.0]


def test_range_max_min_columns():
    frame = rising()
    assert list(frame["close_1~3_max"][:3]) == [13.0, 14.0, 15.0]
    assert frame["close_1~3_max"].iloc[-1] == 29.0
    assert list(frame["close_1~3_min"][:3]) == [11.0, 12.0, 13.0]
    assert frame["close_1~3_min"].iloc[-1] == 29.0


def test_permutation_column():
    frame = rising()
    col = frame["close_-2,-1_p"]
    assert math.isnan(col[0]) and math.isnan(col[1])
    assert list(col[2:]) == [3.0] * 18


def test_delta_column():
    frame = rising()
    assert list(frame["close_delta"]) == [0.0] + [1.0] * 19


def test_change_window_equals_roc_window():
    frame = rising()
    assert list(frame["change_5"]) == list(frame["close_5_roc"])


def test_cross_columns():
    frame = rising()
    cross = frame["close_x_close_5_sma"]
    assert list(cross) == [False, True] + [False] * 18
    assert list(frame["close_x_macd"]) == [False] * 20
    assert list(frame["close_xu_macd"]) == [False] * 20
    assert list(frame["close_xd_macd"]) == [False] * 20


def test_compare_columns_against_macd():
    frame = rising()
    assert list(frame["close_le_macd"]) == [False] * 20
    assert list(frame["close_ge_macd"]) == [True] * 20
    assert list(frame["close_lt_macd"]) == [False] * 20
    assert list(frame["close_gt_macd"]) == [True] * 20
    assert list(frame["close_eq_macd"]) == [False] * 20
    assert list(frame["close_ne_macd"]) == [True] * 20


def test_list_access_initialises_all_columns():
    frame = seeded()
    out = frame[["rsi", "macd"]]
    assert list(out.columns) == ["rsi", "macd"]


# --- KeyError quirks -----------------------------------------------------

def test_bare_eri_and_wt_raise_keyerror():
    with pytest.raises(KeyError):
        rising()["eri"]
    with pytest.raises(KeyError):
        rising()["wt"]


def test_bare_pdm_and_vwap_raise_user_warning():
    with pytest.raises(UserWarning):
        rising()["pdm"]
    with pytest.raises(UserWarning):
        rising()["vwap"]


def test_custom_supertrend_raises_but_creates_bands():
    frame = rising()
    with pytest.raises(KeyError):
        frame["supertrend_10"]
    assert "supertrend" in frame.columns
    assert "supertrend_10_ub" in frame.columns
    assert "supertrend_10_lb" in frame.columns


def test_custom_wt_raises_but_creates_wt1_wt2():
    frame = rising()
    with pytest.raises(KeyError):
        frame["wt_10,21"]
    assert "wt1_10,21" in frame.columns
    assert "wt2_10,21" in frame.columns


def test_custom_ppo_raises_but_creates_default_ppo_columns():
    frame = rising()
    with pytest.raises(KeyError):
        frame["ppo_5,10,3"]
    assert "ppo" in frame.columns
    assert "ppos" in frame.columns
    assert "ppoh" in frame.columns


def test_custom_pvo_raises_but_creates_default_pvo_columns():
    frame = rising()
    with pytest.raises(KeyError):
        frame["pvo_5,10,3"]
    assert "pvo" in frame.columns
    assert "pvos" in frame.columns
    assert "pvoh" in frame.columns


# --- empty and single-row frames ----------------------------------------

def test_empty_frame_indicators_are_empty():
    frame = wrap(pd.DataFrame({
        "open": [], "high": [], "low": [], "close": [], "volume": []}))
    for col in ("rsi", "macd", "boll", "kdjk", "supertrend"):
        assert list(frame[col]) == []


def test_single_row_frame_values():
    frame = wrap(pd.DataFrame({
        "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5],
        "volume": [100.0]}))
    assert frame["rsi"][0] == 50.0
    assert frame["kdjk"][0] == pytest.approx(58.3333, rel=1e-3)
    assert frame["wr"][0] == -25.0
    # Quirk: single-row CCI is NaN (mean absolute deviation over a
    # window larger than the series is NaN, and the guard only covers a
    # zero divisor, not NaN).
    assert math.isnan(frame["cci"][0])
    assert frame["atr"][0] == 2.0
    assert frame["tr"][0] == 2.0
    assert frame["trix"][0] == 0.0
    assert frame["tema"][0] == 10.5
    assert frame["stochrsi"][0] == 0.0
    assert math.isnan(frame["aroon"][0])
    assert frame["boll"][0] == 10.5
    assert math.isnan(frame["boll_ub"][0])
    assert math.isnan(frame["boll_lb"][0])


# --- frame helpers ---------------------------------------------------------

def test_wrap_lowercases_price_columns_but_not_date():
    frame = wrap(pd.DataFrame({
        "Date": [0, 1], "Open": [10.0, 11.0], "High": [12.0, 13.0],
        "Low": [9.0, 10.0], "Close": [11.0, 12.0], "Volume": [100.0, 200.0],
        "Amount": [1100.0, 2400.0]}))
    # Quirk: "date" is not in the lower-casing candidate list, so "Date"
    # survives unchanged while the price columns are lower-cased.
    assert "Date" in frame.columns
    assert "open" in frame.columns
    # Typical price uses amount / volume when amount is present.
    assert list(frame["tp"]) == [11.0, 12.0]
    assert list(frame["middle"]) == [11.0, 12.0]


def test_wrap_retype_index_column_and_passthrough():
    df = pd.DataFrame({
        "mydate": [0, 1], "open": [1.0, 2.0], "high": [2.0, 3.0],
        "low": [0.0, 1.0], "close": [1.5, 2.5], "volume": [10.0, 20.0]})
    frame = wrap(df, index_column="mydate")
    assert list(frame.index) == [0, 1]
    assert list(frame["close"]) == [1.5, 2.5]
    assert wrap(frame) is frame


def test_unwrap_returns_plain_dataframe():
    frame = rising()
    from stockstats import unwrap
    plain = unwrap(frame)
    assert type(plain) is pd.DataFrame


def test_till_start_from_within():
    frame = seeded(20)
    assert len(frame.till(9)) == 10
    assert len(frame.start_from(10)) == 10
    assert len(frame.within(5, 9)) == 5


def test_copy_drop_head_tail():
    frame = seeded(20)
    assert type(frame.copy()).__name__ == "StockDataFrame"
    assert type(frame.drop_column(["open"])).__name__ == "StockDataFrame"
    assert len(frame.drop_tail(3)) == 17
    assert len(frame.drop_head(3)) == 17


def test_init_all_creates_core_indicators():
    frame = seeded(30)
    frame.init_all()
    for col in ("rsi", "macd", "boll", "kdjk", "atr", "cci"):
        assert col in frame.columns


def test_set_dft_window_changes_macd_and_restores():
    import stockstats
    old = stockstats.set_dft_window("macd", (5, 10, 3))
    try:
        assert old == (12, 26, 9)
        frame = seeded()
        assert frame["macd"].iloc[-1] == pytest.approx(0.0581, rel=1e-3)
        assert frame["macd_12,26,9"].iloc[-1] == pytest.approx(-0.1141, rel=1e-3)
    finally:
        stockstats.set_dft_window("macd", old)
    assert stockstats.dft_windows("macd") == "12,26,9"


def test_zero_volume_vwma_and_zero_range_bop():
    frame = wrap(pd.DataFrame({
        "date": range(5),
        "open": [10.0] * 5,
        "high": [10.0] * 5,
        "low": [10.0] * 5,
        "close": [10.0] * 5,
        "volume": [0.0] * 5,
    }))
    assert list(frame["vwma"]) == [0.0] * 5
    assert list(frame["bop"]) == [0.0] * 5


def test_chop_window_one_is_zero_on_flat_series():
    # log10(1) == 0 in the denominator; the result is filled to 0.
    assert list(flat()["chop_1"]) == [0.0] * 20


def test_flat_and_falling_helpers_used():
    # Sanity: the shared builders behave as the other files assume.
    assert list(flat()["close"]) == [10.0] * 20
    assert falling()["close"][0] == 29.0
    assert falling()["close"].iloc[-1] == 10.0
