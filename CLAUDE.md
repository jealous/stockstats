# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

stockstats is a Python library that provides a `StockDataFrame` wrapper around `pandas.DataFrame` with inline stock statistics/indicators support. Users access indicators by column name (e.g., `df['macd']`, `df['rsi_14']`), and values are calculated on-demand.

## Build and Test Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r test-requirements.txt

# Run all tests with coverage
pytest --cov=stockstats test.py

# Run a single test
pytest test.py::StockDataFrameTest::test_get_rsi -v

# Run linting
flake8 stockstats.py test.py

# Run tox (for multi-version testing)
tox
```

## Architecture

### Core Module: stockstats.py

The library is a single-file module with these key components:

**StockDataFrame class** (extends `pd.DataFrame`):
- Overrides `__getitem__` to intercept column access and auto-calculate indicators
- Uses `__init_column` / `__init_not_exist_column` to trigger calculation when a column doesn't exist
- Handler methods named `_get_<indicator>` compute each indicator (e.g., `_get_rsi`, `_get_macd`)

**_Meta class**: Parses column names like `close_20_sma` into components (column, window, indicator name) and provides access to default windows.

**Column naming patterns**:
- `<indicator>` - uses default column and window (e.g., `rsi`, `macd`)
- `<indicator>_<window>` - uses default column with custom window (e.g., `rsi_6`)
- `<column>_<window>_<indicator>` - full specification (e.g., `close_20_sma`)

**Configuration**:
- `_dft_windows` dict: default window sizes for each indicator
- `set_dft_window(name, value)`: modify defaults at runtime
- Class-level constants like `BOLL_STD_TIMES`, `KDJ_PARAM` for indicator tuning

### Adding New Indicators

1. Add default window to `_dft_windows` dict if needed
2. Add default column to `_dft_column` dict if needed
3. Create `_get_<name>(self, meta: _Meta)` method
4. For multi-column indicators, add entry to `handler` property mapping tuple of names to handler

### Test File: test.py

Uses PyHamcrest matchers. Test data files are in `test_data/`. The `YFinanceCompatibilityTest` class downloads live data from Yahoo Finance.

## Commit Guidelines

- Run `flake8 stockstats.py test.py` before each commit to ensure code style compliance
- Use `git commit --amend` for subsequent changes within a single PR to keep history clean
