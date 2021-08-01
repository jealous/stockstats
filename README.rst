Stock Statistics/Indicators Calculation Helper
==============================================

.. image:: https://travis-ci.org/jealous/stockstats.svg
    :target: https://travis-ci.org/jealous/stockstats

.. image:: https://coveralls.io/repos/jealous/stockstats/badge.svg
    :target: https://coveralls.io/github/jealous/stockstats

.. image:: https://img.shields.io/pypi/v/stockstats.svg
    :target: https://pypi.python.org/pypi/stockstats


VERSION: 0.3.2

Introduction
------------

Supply a wrapper ``StockDataFrame`` based on the ``pandas.DataFrame`` with
inline stock statistics/indicators support.

Supported statistics/indicators are:

- change (in percent)
- delta
- permutation (zero based)
- log return
- max in range
- min in range
- middle = (close + high + low) / 3
- compare: le, ge, lt, gt, eq, ne
- count: both backward(c) and forward(fc)
- SMA: simple moving average
- EMA: exponential moving average
- MSTD: moving standard deviation
- MVAR: moving variance
- RSV: raw stochastic value
- RSI: relative strength index
- KDJ: Stochastic oscillator
- Bolling: including upper band and lower band.
- MACD: moving average convergence divergence.  Including signal and histogram. (see note)
- CR:
- WR: Williams Overbought/Oversold index
- CCI: Commodity Channel Index
- TR: true range
- ATR: average true range
- line cross check, cross up or cross down.
- DMA: Different of Moving Average (10, 50)
- DMI: Directional Moving Index, including

  - +DI: Positive Directional Indicator
  - -DI: Negative Directional Indicator
  - ADX: Average Directional Movement Index
  - ADXR: Smoothed Moving Average of ADX

- TRIX: Triple Exponential Moving Average
- TEMA: Another Triple Exponential Moving Average
- VR: Volatility Volume Ratio
- MFI: Money Flow Index

Installation
------------

``pip install stockstats``

Compatibility
-------------

Please check the `setup.py`_ file.

Note that pandas add some type check after version 1.0.
One type assert is skipped in ``StockDataFrame``.  Check ISSUE-50 for detail.

License
-------

`BSD`_

Tutorial
--------

- Initialize the ``StockDataFrame`` with the ``retype`` function which
  convert a ``pandas.DataFrame`` to a ``StockDataFrame``.

.. code-block:: python

    stock = StockDataFrame.retype(pd.read_csv('stock.csv'))


- Formalize your data.  This package takes for granted that your data is sorted
  by timestamp and contains certain columns.  Please align your column name.

  + ``open``: the open price of the interval

  + ``close``: the close price of the interval

  + ``high``: the highest price of the interval

  + ``low``: the lowest price of the interval

  + ``volume``: the volume of stocks traded during the interval

  + ``amount``: the amount of the stocks during the interval

- There are some shortcuts for frequent used statistics/indicators like
  ``kdjk``, ``boll_hb``, ``macd``, etc.

- The indicators/statistics are generated on the fly when they are accessed.
  If you are accessing through ``Series``, it may return not found error.
  The fix is to explicitly initialize it by accessing it like below:

.. code-block:: python

    _ = stock['macd']
    # or
    stock.get('macd')

- Using get item to access the indicators.  The item name following the
  pattern: ``{columnName_window_statistics}``.
  Some statistics/indicators has their short cut.  See examples below:

.. code-block:: python

    # volume delta against previous day
    stock['volume_delta']

    # open delta against next 2 day
    stock['open_2_d']

    # open price change (in percent) between today and the day before yesterday
    # 'r' stands for rate.
    stock['open_-2_r']

    # CR indicator, including 5, 10, 20 days moving average
    stock['cr']
    stock['cr-ma1']
    stock['cr-ma2']
    stock['cr-ma3']

    # volume max of three days ago, yesterday and two days later
    stock['volume_-3,2,-1_max']

    # volume min between 3 days ago and tomorrow
    stock['volume_-3~1_min']

    # KDJ, default to 9 days
    stock['kdjk']
    stock['kdjd']
    stock['kdjj']

    # three days KDJK cross up 3 days KDJD
    stock['kdj_3_xu_kdjd_3']

    # 2 days simple moving average on open price
    stock['open_2_sma']

    # MACD
    stock['macd']
    # MACD signal line
    stock['macds']
    # MACD histogram
    stock['macdh']

    # bolling, including upper band and lower band
    stock['boll']
    stock['boll_ub']
    stock['boll_lb']

    # close price less than 10.0 in 5 days count
    stock['close_10.0_le_5_c']

    # CR MA2 cross up CR MA1 in 20 days count
    stock['cr-ma2_xu_cr-ma1_20_c']

    # count forward(future) where close price is larger than 10
    stock['close_10.0_ge_5_fc']

    # 6 days RSI
    stock['rsi_6']
    # 12 days RSI
    stock['rsi_12']

    # 10 days WR
    stock['wr_10']
    # 6 days WR
    stock['wr_6']

    # CCI, default to 14 days
    stock['cci']
    # 20 days CCI
    stock['cci_20']

    # TR (true range)
    stock['tr']
    # ATR (Average True Range)
    stock['atr']

    # DMA, difference of 10 and 50 moving average
    stock['dma']

    # DMI
    # +DI, default to 14 days
    stock['pdi']
    # -DI, default to 14 days
    stock['mdi']
    # DX, default to 14 days of +DI and -DI
    stock['dx']
    # ADX, 6 days SMA of DX, same as stock['dx_6_ema']
    stock['adx']
    # ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']
    stock['adxr']

    # TRIX, default to 12 days
    stock['trix']
    # TRIX based on the close price for a window of 3
    stock['close_3_trix']
    # MATRIX is the simple moving average of TRIX
    stock['trix_9_sma']
    # TEMA, another implementation for triple ema
    stock['tema']
    # TEMA based on the close price for a window of 2
    stock['close_2_tema']

    # VR, default to 26 days
    stock['vr']
    # MAVR is the simple moving average of VR
    stock['vr_6_sma']

    # Money flow index, default to 14 days
    stock['mfi']

- Following options are available for tuning.  Note that all of them are class level options and MUST be changed before any calculation happens.
    - KDJ
        - KDJ_WINDOW: default to 9
    - BOLL
        - BOLL_WINDOW: default to 20
        - BOLL_STD_TIMES: default to 2
    - MACD
        - MACD_EMA_SHORT: default to 12
        - MACD_EMA_LONG: default to 26
        - MACD_EMA_SIGNAL: default to 9
    - PDI, MDI, DX & ADX
        - PDI_SMMA: default to 14
        - MDI_SMMA: default to 14
        - DX_SMMA: default to 14
        - ADX_EMA: default to 6
        - ADXR_EMA: default to 6
    - CR
        - CR_MA1: default to 5
        - CR_MA2: default to 10
        - CR_MA3: default to 20
    - Triple EMA
        - TRIX_EMA_WINDOW: default to 12
        - TEMA_EMA_WINDOW: default to 5
    - ATR
        - ATR_SMMA: default to 14
    - MFI
        - MFI: default to 14


To file issue, please visit:

https://github.com/jealous/stockstats


MACDH Note:

In July 2017 the code for MACDH was changed to drop an extra 2x multiplier on the final value to align better with calculation methods used in tools like cryptowatch, tradingview, etc.

Contact author:

- Cedric Zhuang <jealous@163.com>

.. _BSD: LICENSE.txt
.. _setup.py: setup.py
