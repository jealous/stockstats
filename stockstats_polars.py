from __future__ import annotations
import inspect
from functools import wraps
from typing import Union

import polars as pl

from stockstats_model import (
    get_default_windows_tuple,
    register_window_listener,
)

Number = Union[int, float]
By = Union[str, list[str], None]


def _maybe_over(expr: pl.Expr, by: By) -> pl.Expr:
    return expr.over(by) if by else expr


def _ewm_mean(x: pl.Expr, alpha: float, *, adjust: bool = True) -> pl.Expr:
    # Polars’ ewm_mean matches the math we want for Wilder/EMA style smoothing
    return x.ewm_mean(alpha=alpha, adjust=adjust, ignore_nulls=True)


def _ema(x: pl.Expr, n: int, *, adjust: bool = True) -> pl.Expr:
    # span->alpha mapping: alpha = 2 / (n + 1)
    return _ewm_mean(x, alpha=2.0 / (n + 1), adjust=adjust)


def _rolling_mean(x: pl.Expr, n: int) -> pl.Expr:
    return x.rolling_mean(window_size=n, min_samples=1)


def _rolling_std(x: pl.Expr, n: int) -> pl.Expr:
    return x.rolling_std(window_size=n, min_samples=1)


def _rolling_sum(x: pl.Expr, n: int) -> pl.Expr:
    return x.rolling_sum(window_size=n, min_samples=1)


def _rolling_max(x: pl.Expr, n: int) -> pl.Expr:
    return x.rolling_max(window_size=n, min_samples=1)


def _rolling_min(x: pl.Expr, n: int) -> pl.Expr:
    return x.rolling_min(window_size=n, min_samples=1)


def _typical_price(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    return (high + low + close) / 3.0


def _resolve_window_default(name: str, index: int) -> int:
    wins = get_default_windows_tuple(name)
    if not wins or len(wins) <= index:
        raise RuntimeError(f"Missing default window for {name}[{index}]")
    return wins[index]


def _sync_param_default(func, param_name: str, value: int):
    sig = getattr(func, "__signature__", inspect.signature(func))
    updated_params = [
        (param.replace(default=value) if param.name == param_name else param)
        for param in sig.parameters.values()
    ]
    new_sig = sig.replace(parameters=updated_params)
    func.__signature__ = new_sig
    return new_sig


def sync_window_default(indicator: str, param_name: str, *, index: int = 0):
    """
    Decorator to keep a method parameter's default value in sync
    with stockstats_model.
    """

    def decorator(func):
        sig = inspect.signature(func)
        param_names = list(sig.parameters)
        if param_name not in param_names:
            raise ValueError(f"{func.__name__} missing parameter {param_name}")
        param_idx = param_names.index(param_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if param_name not in kwargs and len(args) <= param_idx:
                kwargs[param_name] = _resolve_window_default(indicator, index)
            return func(*args, **kwargs)

        # initialize signature/defaults
        current_value = _resolve_window_default(indicator, index)
        _sync_param_default(wrapper, param_name, current_value)

        def _listener(changed_name: str, _wins):
            if changed_name != indicator:
                return
            new_value = _resolve_window_default(indicator, index)
            _sync_param_default(wrapper, param_name, new_value)

        register_window_listener(_listener)
        return wrapper

    return decorator


@pl.api.register_expr_namespace("stockstats")
class StockStatsNS:
    """
    Usage:
        df.with_columns(
            pl.col("close").stockstats.rsi(14).alias("rsi_14"),
            pl.col("close").stockstats.macd().struct.field("macd").alias("macd"),
        )
    The Expr this namespace is bound to is typically a price column (e.g., "close").
    """

    def __init__(self, expr: pl.Expr):
        self._x = expr  # usually close

    # ---- Momentum / Oscillators ----
    @sync_window_default("rsi", "n")
    def rsi(self, n: int, *, by: By = None) -> pl.Expr:
        """
        Wilder-style RSI. Uses ewm smoothing with alpha = 1/n.
        """
        delta = self._x.diff()
        gain = delta.clip(lower_bound=0.0)
        loss = (-delta).clip(lower_bound=0.0)

        # Wilder's smoothing ~ EWM(alpha=1/n)
        avg_gain = gain.ewm_mean(alpha=1.0 / n, adjust=True, ignore_nulls=True)
        avg_loss = loss.ewm_mean(alpha=1.0 / n, adjust=True, ignore_nulls=True)

        rs = avg_gain / avg_loss
        out = 100 - (100 / (1 + rs))
        return _maybe_over(out, by)

    def sma(self, n: int, *, by: By = None) -> pl.Expr:
        """
        Simple moving average.
        """
        return _maybe_over(_rolling_mean(self._x, n), by)

    def ema(self, n: int, *, adjust: bool = True, by: By = None) -> pl.Expr:
        """
        Exponential moving average (span = n).
        """
        return _maybe_over(_ema(self._x, n, adjust=adjust), by)

    @sync_window_default("tema", "n")
    def tema(self, n: int, *, by: By = None) -> pl.Expr:
        """
        Triple Exponential Moving Average (TEMA).
        """
        ema1 = _ema(self._x, n)
        ema2 = _ema(ema1, n)
        ema3 = _ema(ema2, n)
        tema = 3.0 * ema1 - 3.0 * ema2 + ema3
        return _maybe_over(tema, by)

    # ---- Volatility / Bands ----
    @sync_window_default("boll", "n")
    def boll(self, n: int, *, k: Number = 2.0, by: By = None) -> pl.Expr:
        """
        Bollinger Bands over base expr (typically 'close').
        Returns a struct {ma, ub, lb}
        """
        ma = _rolling_mean(self._x, n)
        sd = _rolling_std(self._x, n)
        ub = ma + k * sd
        lb = ma - k * sd
        res = pl.struct(
            ma.alias("boll"),
            ub.alias("boll_ub"),
            lb.alias("boll_lb"),
        )
        return _maybe_over(res, by)

    # ---- Trend / multi-output ----
    @sync_window_default("macd", "short", index=0)
    @sync_window_default("macd", "long", index=1)
    @sync_window_default("macd", "signal", index=2)
    def macd(self, short: int, long: int, signal: int, *, by: By = None) -> pl.Expr:
        """
        MACD line, signal, histogram. Returns struct {macd, macds, macdh}
        """
        ema_s = _ema(self._x, short)
        ema_l = _ema(self._x, long)
        macd_line = (ema_s - ema_l).alias("macd")
        macd_sig = _ema(macd_line, signal).alias("macds")
        macd_hist = (macd_line - macd_sig).alias("macdh")
        res = pl.struct(macd_line, macd_sig, macd_hist)
        return _maybe_over(res, by)

    def _ppo_struct(
        self, name: str, short: int, long: int, signal: int, *, by: By = None
    ) -> pl.Expr:
        ema_s = _ema(self._x, short)
        ema_l = _ema(self._x, long)
        line = (ema_s - ema_l) / ema_l * 100.0
        sig = _ema(line, signal)
        hist = line - sig
        res = pl.struct(
            line.alias(name),
            sig.alias(f"{name}s"),
            hist.alias(f"{name}h"),
        )
        return _maybe_over(res, by)

    @sync_window_default("ppo", "short", index=0)
    @sync_window_default("ppo", "long", index=1)
    @sync_window_default("ppo", "signal", index=2)
    def ppo(self, short: int, long: int, signal: int, *, by: By = None) -> pl.Expr:
        """
        Percentage Price Oscillator over the base expression.
        """
        return self._ppo_struct("ppo", short, long, signal, by=by)

    @sync_window_default("pvo", "short", index=0)
    @sync_window_default("pvo", "long", index=1)
    @sync_window_default("pvo", "signal", index=2)
    def pvo(self, short: int, long: int, signal: int, *, by: By = None) -> pl.Expr:
        """
        Percentage Volume Oscillator (use on volume column).
        """
        return self._ppo_struct("pvo", short, long, signal, by=by)

    # ---- Volatility (True Range / ATR) ----
    @sync_window_default("atr", "n")
    def atr(
        self,
        n: int,
        *,
        high: pl.Expr | None = None,
        low: pl.Expr | None = None,
        close_prev: pl.Expr | None = None,
        by: By = None,
    ) -> pl.Expr:
        """
        Average True Range (SMMA of TR). You must pass high/low/prev_close
        unless your Expr *is* 'close' and you provide them via constants.
        """
        if high is None or low is None:
            raise ValueError("atr() requires high= and low= expressions")
        # prev close: shift close by 1 (previous)
        if close_prev is None:
            # infer base column name for prev close
            close_prev = self._x.shift(1)

        tr1 = (high - low).abs()
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pl.max_horizontal(tr1, tr2, tr3)

        # SMMA ~ EWM with alpha = 1/n (Wilder)
        out = tr.ewm_mean(alpha=1.0 / n, adjust=True, ignore_nulls=True)
        return _maybe_over(out, by)

    # ---- Example: Williams %R ----
    @sync_window_default("wr", "n")
    def wr(self, n: int, *, high: pl.Expr, low: pl.Expr, by: By = None) -> pl.Expr:
        """
        Williams %R = (Hn - Close) / (Hn - Ln) * -100
        """
        hn = _maybe_over(_rolling_max(high, n), by)
        ln = _maybe_over(_rolling_min(low, n), by)
        out = (hn - self._x) / (hn - ln) * -100.0
        return _maybe_over(out, by)
