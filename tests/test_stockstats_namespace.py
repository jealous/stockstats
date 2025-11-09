import math
import pathlib
import sys

import polars as pl
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stockstats_polars  # noqa: F401  (registers namespace on import)


@pytest.fixture
def df_ohlcv() -> pl.DataFrame:
    """Two symbols across five days so we can exercise window ops + grouping."""
    return pl.DataFrame(
        {
            "date": [20210101, 20210104, 20210105, 20210106, 20210107] * 2,
            "symbol": ["A"] * 5 + ["B"] * 5,
            "open": [10.0, 10.1, 10.4, 10.2, 10.5, 20.0, 20.1, 20.3, 20.2, 20.4],
            "high": [10.3, 10.6, 10.5, 10.7, 10.8, 20.2, 20.6, 20.5, 20.7, 20.9],
            "low": [9.9, 10.0, 10.2, 10.1, 10.4, 19.8, 19.9, 20.1, 20.0, 20.2],
            "close": [10.2, 10.5, 10.3, 10.6, 10.7, 20.1, 20.5, 20.2, 20.6, 20.7],
            "volume": [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1550, 1650, 1700],
        }
    )


def approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
        return True
    return abs(a - b) <= tol


def test_namespace_rsi_composes_with_columns(df_ohlcv: pl.DataFrame):
    lf = df_ohlcv.lazy().with_columns(
        pl.col("close").stockstats.rsi(14, by="symbol").alias("rsi_14")
    )
    out = lf.collect()
    assert "rsi_14" in out.columns
    tail = out.group_by("symbol").tail(1)
    assert tail["rsi_14"].drop_nulls().len() == 2


def test_namespace_macd_struct_field_selection(df_ohlcv: pl.DataFrame):
    macd_struct = pl.col("close").stockstats.macd()
    lf = df_ohlcv.lazy().with_columns(
        macd_struct.struct.field("macd").alias("macd"),
        macd_struct.struct.field("macds").alias("macds"),
        macd_struct.struct.field("macdh").alias("macdh"),
    )
    out = lf.collect()
    assert {"macd", "macds", "macdh"}.issubset(set(out.columns))
    assert approx_equal((out["macd"] - out["macds"] - out["macdh"]).abs().max(), 0.0)


def test_namespace_bollinger_struct_expansion(df_ohlcv: pl.DataFrame):
    boll_struct = pl.col("close").stockstats.boll(20)
    lf = (
        df_ohlcv.lazy()
        .with_columns(
            boll_struct.struct.field("boll").alias("boll"),
            boll_struct.struct.field("boll_ub").alias("boll_ub"),
            boll_struct.struct.field("boll_lb").alias("boll_lb"),
        )
    )
    out = lf.collect()
    assert (out["boll_ub"] >= out["boll"]).all()
    assert (out["boll"] >= out["boll_lb"]).all()


def test_namespace_atr_with_grouping(df_ohlcv: pl.DataFrame):
    lf = df_ohlcv.lazy().with_columns(
        pl.col("close")
        .stockstats.atr(14, high=pl.col("high"), low=pl.col("low"), by="symbol")
        .alias("atr_14")
    )
    out = lf.collect()
    grouped = (
        out.group_by("symbol")
        .agg(pl.col("atr_14").sum().alias("s"))
        .sort("symbol")
    )
    assert grouped["symbol"].to_list() == ["A", "B"]
    assert grouped["s"][0] > 0 and grouped["s"][1] > 0


def test_namespace_chain_with_native_polars(df_ohlcv: pl.DataFrame):
    lf = (
        df_ohlcv.lazy()
        .with_columns(pl.col("close").stockstats.rsi(14, by="symbol").alias("rsi"))
        .filter(pl.col("rsi") > 50)
        .group_by("symbol")
        .agg(pl.col("rsi").mean().alias("rsi_mean"))
    )
    out = lf.collect()
    assert set(out["symbol"]) == {"A", "B"}


def test_namespace_requires_high_low_for_atr(df_ohlcv: pl.DataFrame):
    with pytest.raises(ValueError):
        (
            df_ohlcv.lazy()
            .with_columns(pl.col("close").stockstats.atr(14).alias("atr_missing"))
            .collect()
        )


def test_struct_roundtrip_equals_split_fields(df_ohlcv: pl.DataFrame):
    macd_struct = pl.col("close").stockstats.macd()
    lf_struct = (
        df_ohlcv.lazy()
        .with_columns(macd_struct.alias("macd_struct"))
        .with_columns(
            pl.col("macd_struct").struct.field("macd").alias("macd"),
            pl.col("macd_struct").struct.field("macds").alias("macds"),
            pl.col("macd_struct").struct.field("macdh").alias("macdh"),
        )
        .drop("macd_struct")
    )
    lf_direct = df_ohlcv.lazy().with_columns(
        macd_struct.struct.field("macd").alias("macd_d"),
        macd_struct.struct.field("macds").alias("macds_d"),
        macd_struct.struct.field("macdh").alias("macdh_d"),
    )
    out_struct = lf_struct.collect()
    out_direct = lf_direct.collect()
    assert approx_equal((out_struct["macd"] - out_direct["macd_d"]).abs().max(), 0.0)
    assert approx_equal((out_struct["macds"] - out_direct["macds_d"]).abs().max(), 0.0)
    assert approx_equal((out_struct["macdh"] - out_direct["macdh_d"]).abs().max(), 0.0)


def test_namespace_lazy_until_collect(df_ohlcv: pl.DataFrame):
    lf = df_ohlcv.lazy().with_columns(
        pl.col("close").stockstats.rsi(14, by="symbol").alias("rsi")
    )
    assert isinstance(lf, pl.LazyFrame)
    out = lf.collect()
    assert isinstance(out, pl.DataFrame)


def test_namespace_multi_indicator_pipeline(df_ohlcv: pl.DataFrame):
    lf = df_ohlcv.lazy().with_columns(
        pl.col("close").stockstats.rsi(6, by="symbol").alias("rsi_6"),
        pl.col("close").stockstats.macd().struct.field("macd").alias("macd"),
        pl.col("close").stockstats.macd().struct.field("macds").alias("macds"),
        pl.col("close").stockstats.macd().struct.field("macdh").alias("macdh"),
        pl.col("close").stockstats.boll().struct.field("boll").alias("boll"),
    )
    out = lf.collect()
    assert {"rsi_6", "macd", "macds", "macdh", "boll"}.issubset(set(out.columns))


def test_namespace_wr_with_custom_columns(df_ohlcv: pl.DataFrame):
    renamed = df_ohlcv.rename({"high": "hi", "low": "lo"})
    lf = renamed.lazy().with_columns(
        pl.col("close")
        .stockstats.wr(3, high=pl.col("hi"), low=pl.col("lo"))
        .alias("wr_3")
    )
    out = lf.collect()
    assert "wr_3" in out.columns
