Stock Statistics/Indicators Calculation Helper
==============================================

.. image:: https://travis-ci.org/jealous/stockstats.svg
    :target: https://travis-ci.org/jealous/stockstats
    
.. image:: https://coveralls.io/repos/jealous/stockstats/badge.svg
    :target: https://coveralls.io/github/jealous/stockstats

.. image:: https://img.shields.io/pypi/v/stockstats.svg
    :target: https://pypi.python.org/pypi/stockstats


VERSION: 0.1.2

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
- SMA: simple moving average
- EMA: exponential moving average
- MSTD: moving standard deviation
- MVAR: moving variance
- RSV: raw stochastic value
- KDJ: Stochastic oscillator
- Bolling: including upper band and lower band.
- MACD: moving average convergence divergence.  Including signal and histogram.
- CR:
- line cross check, cross up or cross down.


Installation
------------

``pip install stockstats``


License
-------

`BSD`_

Tutorial
--------

- Initialize the ``StockDataFrame`` with the ``retype`` function which
  convert a ``pandas.DataFrame`` to a ``StockDataFrame``.

.. code-block:: python

    stock = StockDataFrame.retype(pd.read_csv('stock.csv'))

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


To file issue, please visit:

https://github.com/jealous/stockstats


Contact author:

- Cedric Zhuang <jealous@163.com>

.. _BSD: LICENSE.txt
.. _test.py: test.py