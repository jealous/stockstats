from __future__ import annotations

from typing import Union
import polars as pl

Number = Union[int, float]
By = Union[str, list[str], None]

@pl.api.register_expr_namespace("stockstats")
class StockStatsNS:
    def rsi(self, n: int = 14, *, by: By | None = ...) -> pl.Expr: ...

    def sma(self, n: int, *, by: By | None = ...) -> pl.Expr: ...

    def ema(self, n: int, *, adjust: bool = ..., by: By | None = ...) -> pl.Expr: ...

    def tema(self, n: int = 5, *, by: By | None = ...) -> pl.Expr: ...

    def boll(self, n: int = 20, *, k: Number = ..., by: By | None = ...) -> pl.Expr: ...

    def macd(
        self,
        short: int = 12,
        long: int = 26,
        signal: int = 9,
        *,
        by: By | None = ...,
    ) -> pl.Expr: ...

    def ppo(
        self,
        short: int = 12,
        long: int = 26,
        signal: int = 9,
        *,
        by: By | None = ...,
    ) -> pl.Expr: ...

    def pvo(
        self,
        short: int = 12,
        long: int = 26,
        signal: int = 9,
        *,
        by: By | None = ...,
    ) -> pl.Expr: ...

    def atr(
        self,
        n: int = 14,
        *,
        high: pl.Expr | None = ...,
        low: pl.Expr | None = ...,
        close_prev: pl.Expr | None = ...,
        by: By | None = ...,
    ) -> pl.Expr: ...

    def wr(
        self,
        n: int = 14,
        *,
        high: pl.Expr,
        low: pl.Expr,
        by: By | None = ...,
    ) -> pl.Expr: ...
