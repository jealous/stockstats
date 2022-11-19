# Stock Statistics/Indicators Calculation Helper

[![build & test](https://github.com/jealous/stockstats/actions/workflows/build-test.yml/badge.svg)](https://github.com/jealous/stockstats/actions/workflows/build-test.yml)
[![codecov](https://codecov.io/gh/jealous/stockstats/branch/master/graph/badge.svg?token=IFMD1pVJ7T)](https://codecov.io/gh/jealous/stockstats)
[![pypi](https://img.shields.io/pypi/v/stockstats.svg)](https://pypi.python.org/pypi/stockstats)

VERSION: 0.5.1

## Introduction

Supply a wrapper ``StockDataFrame`` for ``pandas.DataFrame`` with inline stock
statistics/indicators support.

Supported statistics/indicators are:

* change (in percent)
* delta
* permutation (zero-based)
* log return
* max in range
* min in range
* middle = (close + high + low) / 3
* compare: le, ge, lt, gt, eq, ne
* count: both backward(c) and forward(fc)
* cross: including upward cross and downward cross
* SMA: Simple Moving Average
* EMA: Exponential Moving Average
* MSTD: Moving Standard Deviation
* MVAR: Moving Variance
* RSV: Raw Stochastic Value
* RSI: Relative Strength Index
* KDJ: Stochastic Oscillator
* Bolling: Bollinger Band
* MACD: Moving Average Convergence Divergence
* CR: Energy Index (Intermediate Willingness Index)
* WR: Williams Overbought/Oversold index
* CCI: Commodity Channel Index
* TR: True Range
* ATR: Average True Range
* DMA: Different of Moving Average (10, 50)
* DMI: Directional Moving Index, including
    * +DI: Positive Directional Indicator
    * -DI: Negative Directional Indicator
    * ADX: Average Directional Movement Index
    * ADXR: Smoothed Moving Average of ADX
* TRIX: Triple Exponential Moving Average
* TEMA: Another Triple Exponential Moving Average
* VR: Volume Variation Index
* MFI: Money Flow Index
* VWMA: Volume Weighted Moving Average
* CHOP: Choppiness Index
* KAMA: Kaufman's Adaptive Moving Average
* PPO: Percentage Price Oscillator
* StochRSI: Stochastic RSI
* WT: LazyBear's Wave Trend
* Supertrend: with the Upper Band and Lower Band

## Installation

```pip install stockstats```

## Compatibility

The build checks the compatibility for the last two major releases of python3 and
the last release of python2.

## License

[BSD-3-Clause License](./LICENSE.txt)

## Tutorial

### Initialization

`StockDataFrame` works as a wrapper for the `pandas.DataFrame`. You need to
Initialize the `StockDataFrame` with `wrap` or `StockDataFrame.retype`.

``` python
import pandas as pd
from stockstats import wrap

data = pd.read_csv('stock.csv')
df = wrap(data)
```

Formalize your data. This package takes for granted that your data is sorted by
timestamp and contains certain columns. Please align your column name.

* `date`: timestamp of the record, optional.
* `close`: the close price of the period
* `high`: the highest price of the interval
* `low`: the lowest price of the interval
* `volume`: the volume of stocks traded during the interval

Note these column names are case-insensitive. They are converted to lower case
when you wrap the data frame.

By default, the `date` column is used as the index. Users can also specify the
index column name in the `wrap` or `retype` function.

Example:
`DataFrame` loaded from CSV.

```
          Date      Amount  Close   High    Low   Volume
0     20040817  90923240.0  11.20  12.21  11.03  7877900
1     20040818  52955668.0  10.29  10.90  10.29  5043200
2     20040819  32614676.0  10.53  10.65  10.30  3116800
...        ...         ...    ...    ...    ...      ...
2810  20160815  56416636.0  39.58  39.79  38.38  1436706
2811  20160816  68030472.0  39.66  40.86  39.00  1703600
2812  20160817  62536480.0  40.45  40.59  39.12  1567600
```

After conversion to `StockDataFrame`

```
              amount  close   high    low   volume
date
20040817  90923240.0  11.20  12.21  11.03  7877900
20040818  52955668.0  10.29  10.90  10.29  5043200
20040819  32614676.0  10.53  10.65  10.30  3116800
...              ...    ...    ...    ...      ...
20160815  56416636.0  39.58  39.79  38.38  1436706
20160816  68030472.0  39.66  40.86  39.00  1703600
20160817  62536480.0  40.45  40.59  39.12  1567600 
```

Use `unwrap` to convert it back to a `pandas.DataFrame`.
Note that `unwrap` won't reset the columns and the index.

### Access the Data

`StockDataFrame` is a subclass of `pandas.DataFrame`. All the functions
of `pandas.DataFrame` should work the same as before.

#### Retrieve the data with symbol

We allow the user to access the statistics directly with some specified column
name, such as `kdjk`, `macd`, `rsi`.

The values of these columns are calculated the first time you access
them from the data frame. Please delete those columns first if you want the
lib to re-evaluate them.

#### Retrieve the Series

Use `macd = stock['macd']` or `rsi = stock.get('rsi')` to retrieve the `Series`.

#### Retrieve the symbol with 2 arguments

Some statistics need the column name and the window size,
such as delta, shift, simple moving average, etc. Use this patter to retrieve
them: `<columnName>_<windowSize>_<statistics>`

Examples:

* 5 periods simple moving average of the high price: `high_5_sma`
* 10 periods exponential moving average of the close: `close_10_ema`
* 1 period delta of the high price: `high_-1_d`.
  The minus symbol means looking backward.

#### Retrieve the symbol with 1 argument

Some statistics require the window size but not the column name. Use
this patter to specify your window: `<statistics>_<windowSize>`

Examples:

* 6 periods RSI: `rsi_6`
* 10 periods CCI: `cci_10`
* 13 periods ATR: `atr_13`

Some of them have default windows.  Check their document for detail.

#### Initialize all indicators with shortcuts

Some indicators, such as KDJ, BOLL, MFI, have shortcuts.  Use `df.init_all()`
to initialize all these indicators.

This operation generates lots of columns.  Please use it with caution.

### Statistics/Indicators

Some statistics have configurable parameters. They are class-level fields. Change
of these fields is global. And they won't affect the existing results. Removing
existing columns so that they will be re-evaluated the next time you access them.

#### Change of the Close

`df['change']` is the change of the `close` price in percentage.

#### Delta of Periods

Using pattern `<column>_<window>_d` to retrieve the delta between different periods.

You can also use `<column>_delta` as a shortcut to `<column>_-1_d` 

Examples:
* `df['close_-1_d']` retrieves the close price delta between current and prev. period.
* `df['close_delta']` is the same as `df['close_-1_d']`
* `df['high_2_d']` retrieves the high price delta between current and 2 days later

#### Shift Periods

Shift the column backward or forward. It takes 2 parameters:

* the name of the column to shift
* periods to shift, can be negative

We fill the head and tail with the nearest data.

See the example below:

``` python
In [15]: df[['close', 'close_-1_s', 'close_2_s']]
Out[15]:
          close  close_-1_s  close_2_s
date
20040817  11.20       11.20      10.53
20040818  10.29       11.20      10.55
20040819  10.53       10.29      10.10
20040820  10.55       10.53      10.25
...         ...         ...        ...
20160812  39.10       38.70      39.66
20160815  39.58       39.10      40.45
20160816  39.66       39.58      40.45
20160817  40.45       39.66      40.45

[2813 rows x 3 columns]
```

#### [RSI - Relative Strength Index](https://en.wikipedia.org/wiki/Relative_strength_index)

RSI has a configurable window. The default window size is 14 which is
configurable through `StockDataFrame.RSI`. e.g.

* `df['rsi']`: 14 periods RSI
* `df['rsi_6']`: 6 periods RSI

#### [Log Return of the Close](https://en.wikipedia.org/wiki/Rate_of_return)

Logarithmic return = ln( close / last close)

From wiki:

> For example, if a stock is priced at 3.570 USD per share at the close on
> one day, and at 3.575 USD per share at the close the next day, then the
> logarithmic return is: ln(3.575/3.570) = 0.0014, or 0.14%.

Use `df['log-ret']` to access this column.

#### Count of Non-Zero Value

Count non-zero values of a specific range. It requires a column and a window.

Examples:

* Count how many typical prices are larger than close in the past 10 periods

``` python
In [22]: tp = df['middle']                             
                                                       
In [23]: df['res'] = df['middle'] > df['close']        
                                                       
In [24]: df[['middle', 'close', 'res', 'res_10_c']]    
Out[24]:                                               
             middle  close    res  res_10_c            
date                                                   
20040817  11.480000  11.20   True       1.0            
20040818  10.493333  10.29   True       2.0            
20040819  10.493333  10.53  False       2.0            
20040820  10.486667  10.55  False       2.0            
20040823  10.163333  10.10   True       3.0            
...             ...    ...    ...       ...            
20160811  38.703333  38.70   True       5.0            
20160812  38.916667  39.10  False       5.0            
20160815  39.250000  39.58  False       4.0            
20160816  39.840000  39.66   True       5.0            
20160817  40.053333  40.45  False       5.0            
                                                       
[2813 rows x 4 columns]                                
```

* Count ups in the past 10 periods

``` python
In [26]: df['ups'], df['downs'] = df['change'] > 0, df['change'] < 0 
                                                                     
In [27]: df[['ups', 'ups_10_c', 'downs', 'downs_10_c']]              
Out[27]:                                                             
            ups  ups_10_c  downs  downs_10_c                         
date                                                                 
20040817  False       0.0  False         0.0                         
20040818  False       0.0   True         1.0                         
20040819   True       1.0  False         1.0                         
20040820   True       2.0  False         1.0                         
20040823  False       2.0   True         2.0                         
...         ...       ...    ...         ...                         
20160811  False       3.0   True         7.0                         
20160812   True       3.0  False         7.0                         
20160815   True       4.0  False         6.0                         
20160816   True       5.0  False         5.0                         
20160817   True       5.0  False         5.0                         
                                                                     
[2813 rows x 4 columns]                                              
```

#### Max and Min of the Periods

Retrieve the max/min value of specified periods. They require column and
window.  
Note the window does NOT simply stand for the rolling window.

Examples:

* `close_-3,2_max` stands for the max of 2 periods later and 3 periods ago
* `close_-2~0_min` stands for the min of 2 periods ago till now

#### RSV - Raw Stochastic Value

RSV is essential for calculating KDJ. It takes a window parameter.
Use `df['rsv']` or `df['rsv_6']` to access it.

#### [RSI - Relative Strength Index](https://en.wikipedia.org/wiki/Relative_strength_index)

RSI chart the current and historical strength or weakness of a stock. It takes 
a window parameter.

The default window is 14. Use `StockDataFrame.RSI` to tune it.

Examples:

* `df['rsi']`: retrieve the RSI of 14 periods
* `df['rsi_6']`: retrieve the RSI of 6 periods

#### [Stochastic RSI](https://www.investopedia.com/terms/s/stochrsi.asp)

Stochastic RSI gives traders an idea of whether the current RSI value is 
overbought or oversold. It takes a window parameter.

The default window is 14. Use `StockDataFrame.RSI` to tune it.

Examples:

* `df['stochrsi']`: retrieve the Stochastic RSI of 14 periods
* `df['stochrsi_6']`: retrieve the Stochastic RSI of 6 periods

#### [WT - Wave Trend](https://medium.com/@samuel.mcculloch/lets-take-a-look-at-wavetrend-with-crosses-lazybear-s-indicator-2ece1737f72f)

Retrieve the LazyBear's Wave Trend with `df['wt1']` and `df['wt2']`.

Wave trend uses two parameters. You can tune them with
`StockDataFrame.WAVE_TREND_1` and `StockDataFrame.WAVE_TREND_2`.

#### SMMA - Smoothed Moving Average

It requires column and window.

For example, use `df['close_7_smma']` to retrieve the 7 periods smoothed moving
average of the close price.

#### [TRIX - Triple Exponential Average](https://www.investopedia.com/articles/technical/02/092402.asp)

The triple exponential average is used to identify oversold and overbought 
markets.

The algorithm is:

```
TRIX = (TripleEMA - LastTripleEMA) -  * 100 / LastTripleEMA
TripleEMA = EMA of EMA of EMA
LastTripleEMA =  TripleEMA of the last period
```

It requires column and window. By default, the column is `close`,
the window is 12.

Use `StockDataFrame.TRIX_EMA_WINDOW` to change the default window.

Examples:

* `df['trix']` stands for 12 periods Trix for the close price.
* `df['middle_10_trix']` stands for the 10 periods Trix for the typical price.

#### [TEMA - Another Triple Exponential Average](https://www.forextraders.com/forex-education/forex-technical-analysis/triple-exponential-moving-average-the-tema-indicator/)

Tema is another implementation for the triple exponential moving average.

```
TEMA=(3 x EMA) - (3 x EMA of EMA) + (EMA of EMA of EMA)
```

It takes two parameters, column and window. By default, the column is `close`,
the window is 5.

Use `StockDataFrame.TEMA_EMA_WINDOW` to change the default window.

Examples:

* `df['tema']` stands for 12 periods TEMA for the close price.
* `df['middle_10_tema']` stands for the 10 periods TEMA for the typical price.

#### [VR - Volume Variation Index](https://help.eaglesmarkets.com/hc/en-us/articles/900002867026-Summary-of-volume-variation-index)

It is the strength index of the trading volume.

It has a default window of 26. Change it with `StockDataFrame.VR`.

Examples:
* `df['vr']` retrieves the 26 periods VR.
* `df['vr_6']` retrieves the 6 periods VR.

#### [WR - Williams Overbought/Oversold Index](https://www.investopedia.com/terms/w/williamsr.asp)

Williams Overbought/Oversold index
is a type of momentum indicator that moves between 0 and -100 and measures
overbought and oversold levels.

It takes a window parameter. The default window is 14. Use `StockDataFrame.WR`
to change the default window.

Examples:

* `df['wr']` retrieves the 14 periods WR.
* `df['wr_6']` retrieves the 6 periods WR.

#### [CCI - Commodity Channel Index](https://www.investopedia.com/terms/c/commoditychannelindex.asp)

CCI stands for Commodity Channel Index.

It requires a window parameter. The default window is 14. Use
`StockDataFrame.CCI` to change it.

Examples:

* `df['cci']` retrieves the default 14 periods CCI.
* `df['cci_6']` retrieves the 6 periods CCI.

#### TR - True Range of Trading

TR is a measure of the volatility of a High-Low-Close series. It is used for
calculating the ATR.

#### [ATR - Average True Range](https://en.wikipedia.org/wiki/Average_true_range)

The Average True Range is an
N-period smoothed moving average (SMMA) of the true range value.  
Default to 14 periods.

Users can modify the default window with `StockDataFrame.ATR_SMMA`.

Example:

* `df['atr']` retrieves the 14 periods ATR.
* `df['atr_5']` retrieves the 5 periods ATR.

#### [Supertrend](https://economictimes.indiatimes.com/markets/stocks/news/how-to-use-supertrend-indicator-to-find-buying-and-selling-opportunities-in-market/articleshow/54492970.cms)

Supertrend indicates the current trend.  
We use the [algorithm described here](https://medium.com/codex/step-by-step-implementation-of-the-supertrend-indicator-in-python-656aa678c111).
It includes 3 lines:
* `df['supertrend']` is the trend line.
* `df['supertrend_ub']` is the upper band of the trend
* `df['supertrend_lb']` is the lower band of the trend

It has 2 parameters:
* `StockDataFrame.SUPERTREND_MUL` is the multiplier of the band, default to 3.
* `StockDataFrame.SUPERTREND_WINDOW` is the window size, default to 14.

#### DMA - Difference of Moving Average

`df['dma']` retrieves the difference of 10 periods SMA of the close price and
the 50 periods SMA of the close price.

#### [DMI - Directional Movement Index](https://www.investopedia.com/terms/d/dmi.asp)

The directional movement index (DMI)
identifies in which direction the price of an asset is moving.

It has several lines:

* `df['pdi']` is the positive directional movement line (+DI)
* `df['mdi']` is the negative directional movement line (-DI)
* `df['dx']` is the directional index (DX)
* `df['adx']` is the average directional index (ADX)
* `df['adxr']` is an EMA for ADX

It has several parameters.

* `StockDataFrame.PDI_SMMA` - window for +DI
* `StockDataFrame.MDI_SMMA` - window for -DI
* `StockDataFrame.DX_SMMA` - window for DX
* `StockDataFrame.ADX_EMA` - window for ADX
* `StockDataFrame.ADXR_EMA` - window for ADXR

#### [KDJ Indicator](https://en.wikipedia.org/wiki/Stochastic_oscillator)

The stochastic oscillator is a momentum indicator that uses support and 
resistance levels.

It includes three lines:
* `df['kdjk']` - K series
* `df['kdjd']` - D series
* `df['kdjj']` - J series

The default window is 9.  Use `StockDataFrame.KDJ_WINDOW` to change it.
Use `df['kdjk_6']` to retrieve the K series of 6 periods.

KDJ also has two configurable parameters named `StockDataFrame.KDJ_PARAM`.
The default value is `(2.0/3.0, 1.0/3.0)`

#### [CR - Energy Index](https://support.futunn.com/en/topic167/?lang=en-us)

The Energy Index (Intermediate Willingness Index)
uses the relationship between the highest price, the lowest price and
yesterday's middle price to reflect the market's willingness to buy
and sell.

It contains 4 lines:
* `df['cr']` - the CR line
* `df['cr-ma1']` - `StockDataFrame.CR_MA1` periods of the CR moving average
* `df['cr-ma2']` - `StockDataFrame.CR_MA2` periods of the CR moving average
* `df['cr-ma3']` - `StockDataFrame.CR_MA3` periods of the CR moving average

#### [Typical Price](https://en.wikipedia.org/wiki/Typical_price)

It's the average of `high`, `low` and `close`.
Use `df['middle']` to access this value.

When `amount` is available, `middle = amount / volume`
This should be more accurate because amount represents the total cash flow. 

#### [Bollinger Bands](https://en.wikipedia.org/wiki/Bollinger_Bands)

The Bollinger bands includes three lines
* `df['boll']` is the baseline
* `df['boll_ub']` is the upper band
* `df['boll_lb']` is the lower band

The default period of the Bollinger Band can be changed with
`StockDataFrame.BOLL_PERIOD`.  The width of the bands can be turned with
`StockDataFrame.BOLL_STD_TIMES`.  The default value is 2.

#### [MACD - Moving Average Convergence Divergence](https://en.wikipedia.org/wiki/MACD)

We use the close price to calculate the MACD lines.
* `df['macd']` is the difference between two exponential moving averages.
* `df['macds]` is the signal line.
* `df['macdh']` is he histogram line.

The period of short and long EMA can be tuned with 
`StockDataFrame.MACD_EMA_SHORT` and `StockDataFrame.MACD_EMA_LONG`.  The default
value are 12 and 26

The period of the signal line can be tuned with 
`StockDataFrame.MACD_EMA_SIGNAL`. The default value is 9.

#### [PPO - Percentage Price Oscillator](https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:price_oscillators_ppo)

The Percentage Price Oscillator includes three lines.

* `df['ppo']` derives from the difference of 2 exponential moving average.
* `df['ppos]` is the signal line.
* `df['ppoh']` is he histogram line.

The period of short and long EMA can be tuned with 
`StockDataFrame.PPO_EMA_SHORT` and `StockDataFrame.PPO_EMA_LONG`.  The default
value are 12 and 26

The period of the signal line can be tuned with 
`StockDataFrame.PPO_EMA_SIGNAL`. The default value is 9.

#### [Simple Moving Average](https://www.investopedia.com/terms/m/mean.asp)

Follow the pattern `<columnName>_<window>_sma` to retrieve a simple moving average.

#### [Moving Standard Deviation](https://www.investopedia.com/terms/s/standarddeviation.asp)

Follow the pattern `<columnName>_<window>_mstd` to retrieve the moving STD.

#### [Moving Variance](https://www.investopedia.com/terms/v/variance.asp)

Follow the pattern `<columnName>_<window>_mvar` to retrieve the moving VAR.

#### [Volume Weighted Moving Average](https://www.investopedia.com/articles/trading/11/trading-with-vwap-mvwap.asp)

It's the moving average weighted by volume.

It has a parameter for window size.  The default window is 14.  Change it with
`StockDataFrame.VWMA`.

Examples:
* `df['vwma']` retrieves the 14 periods VWMA
* `df['vwma_6']` retrieves the 6 periods VWMA

#### [CHOP - Choppiness Index](https://www.tradingview.com/education/choppinessindex/)

The Choppiness Index determines if the market is choppy.

It has a parameter for window size.  The default window is 14.  Change it with
`StockDataFrame.CHOP`.

Examples:
* `df['chop']` retrieves the 14 periods CHOP
* `df['chop_6']` retrieves the 6 periods CHOP

#### [MFI - Money Flow Index](https://www.investopedia.com/terms/m/mfi.asp)

The Money Flow Index
identifies overbought or oversold signals in an asset.

It has a parameter for window size.  The default window is 14.  Change it with
`StockDataFrame.MFI`.

Examples:
* `df['mfi']` retrieves the 14 periods MFI
* `df['mfi_6']` retrieves the 6 periods MFI

#### [KAMA - Kaufman's Adaptive Moving Average](https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average)

Kaufman's Adaptive Moving Average is designed to account for market noise or 
volatility.

It has 2 optional parameters and 2 required parameters
* fast - optional, the parameter for fast EMA smoothing, default to 5
* slow - optional, the parameter for slow EMA smoothing, default to 34
* column - required, the column to calculate
* window - required, rolling window size

The default value for fast and slow can be configured with
`StockDataFrame.KAMA_FAST` and `StockDataFrame.KAMA_SLOW`

Examples:
* `df['close_10_kama_2_30']` retrieves 10 periods KAMA of the close price with 
  `fast = 2` and `slow = 30`
* `df['close_2_kama']` retrieves 2 periods KAMA of the close price

#### Cross Upwards and Cross Downwards

Use the pattern `<A>_xu_<B>` to check when A crosses up B.

Use the pattern `<A>_xd_<B>` to check when A crosses down B.

Use the pattern `<A>_x_<B>` to check when A crosses B.

Examples:
* `kdjk_x_kdjd` returns a series that marks the cross of KDJK and KDJD
* `kdjk_xu_kdjd` returns a series that marks where KDJK crosses up KDJD
* `kdjk_xd_kdjd` returns a series that marks where KDJD crosses down KDJD

## Issues

We use [Github Issues](https://github.com/jealous/stockstats/issues) to track
the issues or bugs.

## Others

MACDH Note:

In July 2017 the code for MACDH was changed to drop an extra 2x multiplier on
the final value to align better with calculation methods used in tools like
cryptowatch, tradingview, etc.

## Contact author:

* Cedric Zhuang <jealous@163.com>
