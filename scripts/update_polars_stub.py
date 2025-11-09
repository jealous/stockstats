#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stockstats_model import get_default_windows_tuple

STUB_TEMPLATE = """from __future__ import annotations

from typing import Union
import polars as pl

Number = Union[int, float]
By = Union[str, list[str], None]

@pl.api.register_expr_namespace(\"stockstats\")
class StockStatsNS:
    def rsi(self, n: int = {rsi}, *, by: By | None = ...) -> pl.Expr: ...

    def sma(self, n: int, *, by: By | None = ...) -> pl.Expr: ...

    def ema(self, n: int, *, adjust: bool = ..., by: By | None = ...) -> pl.Expr: ...

    def tema(self, n: int = {tema}, *, by: By | None = ...) -> pl.Expr: ...

    def boll(self, n: int = {boll}, *, k: Number = ..., by: By | None = ...) -> pl.Expr: ...

    def macd(
        self,
        short: int = {macd_short},
        long: int = {macd_long},
        signal: int = {macd_signal},
        *,
        by: By | None = ...,
    ) -> pl.Expr: ...

    def ppo(
        self,
        short: int = {ppo_short},
        long: int = {ppo_long},
        signal: int = {ppo_signal},
        *,
        by: By | None = ...,
    ) -> pl.Expr: ...

    def pvo(
        self,
        short: int = {pvo_short},
        long: int = {pvo_long},
        signal: int = {pvo_signal},
        *,
        by: By | None = ...,
    ) -> pl.Expr: ...

    def atr(
        self,
        n: int = {atr},
        *,
        high: pl.Expr | None = ...,
        low: pl.Expr | None = ...,
        close_prev: pl.Expr | None = ...,
        by: By | None = ...,
    ) -> pl.Expr: ...

    def wr(
        self,
        n: int = {wr},
        *,
        high: pl.Expr,
        low: pl.Expr,
        by: By | None = ...,
    ) -> pl.Expr: ...
"""


def require_default(name: str, index: int = 0) -> int:
    wins = get_default_windows_tuple(name)
    if not wins or len(wins) <= index:
        raise RuntimeError(f"Missing default window for {name}[{index}]")
    return wins[index]


def render_stub() -> str:
    return STUB_TEMPLATE.format(
        rsi=require_default("rsi"),
        tema=require_default("tema"),
        boll=require_default("boll"),
        macd_short=require_default("macd", 0),
        macd_long=require_default("macd", 1),
        macd_signal=require_default("macd", 2),
        ppo_short=require_default("ppo", 0),
        ppo_long=require_default("ppo", 1),
        ppo_signal=require_default("ppo", 2),
        pvo_short=require_default("pvo", 0),
        pvo_long=require_default("pvo", 1),
        pvo_signal=require_default("pvo", 2),
        atr=require_default("atr"),
        wr=require_default("wr"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or check stockstats_polars.pyi")
    parser.add_argument("--check", action="store_true", help="Return non-zero if stub is outdated")
    args = parser.parse_args()

    stub_path = Path(__file__).resolve().parents[1] / "stockstats_polars.pyi"
    new_content = render_stub()

    if args.check:
        current = stub_path.read_text() if stub_path.exists() else ""
        if current != new_content:
            print("stockstats_polars.pyi is out of date. Run scripts/update_polars_stub.py to regenerate.")
            return 1
        return 0

    stub_path.write_text(new_content)
    print(f"Wrote {stub_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
