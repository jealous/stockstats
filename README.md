# Stock Statistics/Indicators Calculation Helper

[![build & test](https://github.com/jealous/stockstats/actions/workflows/build-test.yml/badge.svg)](https://github.com/jealous/stockstats/actions/workflows/build-test.yml)
[![codecov](https://codecov.io/gh/jealous/stockstats/branch/master/graph/badge.svg?token=IFMD1pVJ7T)](https://codecov.io/gh/jealous/stockstats)
[![pypi](https://img.shields.io/pypi/v/stockstats.svg)](https://pypi.python.org/pypi/stockstats)

VERSION: 0.6.1

## Introduction

Supply a wrapper ``StockDataFrame`` for ``pandas.DataFrame`` with inline stock
statistics/indicators support.

Supported statistics/indicators are:

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
* KER: Kaufman's efficiency ratio
* KAMA: Kaufman's Adaptive Moving Average
* PPO: Percentage Price Oscillator
* StochRSI: Stochastic RSI
* WT: LazyBear's Wave Trend
* Supertrend: with the Upper Band and Lower Band
* Aroon: Aroon Oscillator
* Z: Z-Score
* AO: Awesome Oscillator
* BOP: Balance of Power
* MAD: Mean Absolute Deviation
* ROC: Rate of Change
* Coppock: Coppock Curve
* Ichimoku: Ichimoku Cloud
* CTI: Correlation Trend Indicator
* LRMA: Linear Regression Moving Average
* ERI: Elder-Ray Index
* FTR: the Gaussian Fisher Transform Price Reversals indicator
* RVGI: Relative Vigor Index
* Inertia: Inertia Indicator
* KST: Know Sure Thing
* PGO: Pretty Good Oscillator
* PSL: Psychological Line
* PVO: Percentage Volume Oscillator
* QQE: Quantitative Qualitative Estimation

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
configurable through `set_dft_window('rsi', n)`. e.g.

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

The default window is 14. Use `set_dft_window('rsi', n)` to tune it.

Examples:

* `df['rsi']`: retrieve the RSI of 14 periods
* `df['rsi_6']`: retrieve the RSI of 6 periods

#### [Stochastic RSI](https://www.investopedia.com/terms/s/stochrsi.asp)

Stochastic RSI gives traders an idea of whether the current RSI value is 
overbought or oversold. It takes a window parameter.

The default window is 14. Use `set_dft_window('stochrsi', n)` to tune it.

Examples:

* `df['stochrsi']`: retrieve the Stochastic RSI of 14 periods
* `df['stochrsi_6']`: retrieve the Stochastic RSI of 6 periods

#### [WT - Wave Trend](https://medium.com/@samuel.mcculloch/lets-take-a-look-at-wavetrend-with-crosses-lazybear-s-indicator-2ece1737f72f)

Retrieve the LazyBear's Wave Trend with `df['wt1']` and `df['wt2']`.

Wave trend uses two parameters. You can tune them with
`set_dft_window('wt', (10, 21))`.

#### SMMA - Smoothed Moving Average

It requires column and window.

For example, use `df['close_7_smma']` to retrieve the 7 periods smoothed moving
average of the close price.

#### [ROC - Rate of Change](https://www.investopedia.com/terms/p/pricerateofchange.asp)

The Price Rate of Change (ROC) is a momentum-based technical indicator 
that measures the percentage change in price between the current price 
and the price a certain number of periods ago.

Formular:

ROC = (PriceP - PricePn) / PricePn * 100

Where:
* PriceP: the price of the current period
* PricePn: the price of the n periods ago

You need a column name and a period to calculate ROC.

Examples:
* `df['close_10_roc']`: the ROC of the close price in 10 periods
* `df['high_5_roc']`: the ROC of the high price in 5 periods

#### [MAD - Mean Absolute Deviation](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/other-measures-of-spread/a/mean-absolute-deviation-mad-review)

The mean absolute deviation of a dataset is the average
distance between each data point and the mean. It gives 
us an idea about the variability in a dataset.

Formular:
1. Calculate the mean.
2. Calculate how far away each data point is from the 
   mean using positive distances. These are called 
   absolute deviations.
3. Add those deviations together.
4. Divide the sum by the number of data points.

Example:
* `df['close_10_mad']`: the MAD of the close price in 10 periods

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

Use `set_dft_window('trix', n)` to change the default window.

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

Use `set_dft_window('tema', n)` to change the default window.

Examples:

* `df['tema']` stands for 12 periods TEMA for the close price.
* `df['middle_10_tema']` stands for the 10 periods TEMA for the typical price.

#### [VR - Volume Variation Index](https://help.eaglesmarkets.com/hc/en-us/articles/900002867026-Summary-of-volume-variation-index)

It is the strength index of the trading volume.

It has a default window of 26. Change it with `set_dft_window('vr', n)`.

Examples:
* `df['vr']` retrieves the 26 periods VR.
* `df['vr_6']` retrieves the 6 periods VR.

#### [WR - Williams Overbought/Oversold Index](https://www.investopedia.com/terms/w/williamsr.asp)

Williams Overbought/Oversold index
is a type of momentum indicator that moves between 0 and -100 and measures
overbought and oversold levels.

It takes a window parameter. The default window is 14. Use `set_dft_window('wr', n)`
to change the default window.

Examples:

* `df['wr']` retrieves the 14 periods WR.
* `df['wr_6']` retrieves the 6 periods WR.

#### [CCI - Commodity Channel Index](https://www.investopedia.com/terms/c/commoditychannelindex.asp)

CCI stands for Commodity Channel Index.

It requires a window parameter. The default window is 14. Use
`set_dft_window('cci', n)` to change it.

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

Users can modify the default window with `set_dft_window('atr', n)`.

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
* the default window size is 14.  Change it with `set_dft_window('supertrend', n)`

#### DMA - Difference of Moving Average

`df['dma']` retrieves the difference of 10 periods SMA of the close price and
the 50 periods SMA of the close price.

#### [DMI - Directional Movement Index](https://www.investopedia.com/terms/d/dmi.asp)

The directional movement index (DMI)
identifies in which direction the price of an asset is moving.

It has several lines:

* `df['pdi']` is the positive directional movement line (+DI)
* `df['ndi']` is the negative directional movement line (-DI)
* `df['dx']` is the directional index (DX)
* `df['adx']` is the average directional index (ADX)
* `df['adxr']` is an EMA for ADX

It has several parameters.

* default window for +DI is 14, change it with `set_dft_window('pdi', n)`
* default window for -DI is 14, change it with `set_dft_window('ndi', n)`
* `StockDataFrame.DX_SMMA` - window for DX, default to 14
* `StockDataFrame.ADX_EMA` - window for ADX, default to 6
* `StockDataFrame.ADXR_EMA` - window for ADXR, default to 6

#### [KDJ Indicator](https://en.wikipedia.org/wiki/Stochastic_oscillator)

The stochastic oscillator is a momenxtum indicator that uses support and 
resistance levels.

It includes three lines:
* `df['kdjk']` - K series
* `df['kdjd']` - D series
* `df['kdjj']` - J series

The default window is 9.  Use `set_dft_window('kdjk', n)` to change it.
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
* `df['cr-ma1']` - `StockDataFrame.CR_MA[0]` periods of the CR moving average,
  the default window is 5
* `df['cr-ma2']` - `StockDataFrame.CR_MA[1]` periods of the CR moving average,
  the default window is 10
* `df['cr-ma3']` - `StockDataFrame.CR_MA[2]` periods of the CR moving average,
  the default window is 20

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

The default window of boll is 20.
You can also supply your window with `df['boll_10']`.  It will also
generate the `boll_ub_10` and `boll_lb_10` column.

The default period of the Bollinger Band can be changed with
`set_dft_window('boll', n)`.  The width of the bands can be turned with
`StockDataFrame.BOLL_STD_TIMES`.  The default value is 2.

#### [MACD - Moving Average Convergence Divergence](https://en.wikipedia.org/wiki/MACD)

We use the close price to calculate the MACD lines.
* `df['macd']` is the difference between two exponential moving averages.
* `df['macds]` is the signal line.
* `df['macdh']` is he histogram line.

The period of short, long EMA and signal line can be tuned with 
`set_dft_window('macd', (short, long, signal))`.  The default
windows are 12 and 26 and 9.

#### [PPO - Percentage Price Oscillator](https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:price_oscillators_ppo)

The Percentage Price Oscillator includes three lines.

* `df['ppo']` derives from the difference of 2 exponential moving average.
* `df['ppos]` is the signal line.
* `df['ppoh']` is he histogram line.

The period of short, long EMA and signal line can be tuned with 
`set_dft_window('ppo', (short, long, signal))`.  The default
windows are 12 and 26 and 9.

#### [Simple Moving Average](https://www.investopedia.com/terms/m/mean.asp)

Follow the pattern `<columnName>_<window>_sma` to retrieve a simple moving average.

#### [Moving Standard Deviation](https://www.investopedia.com/terms/s/standarddeviation.asp)

Follow the pattern `<columnName>_<window>_mstd` to retrieve the moving STD.

#### [Moving Variance](https://www.investopedia.com/terms/v/variance.asp)

Follow the pattern `<columnName>_<window>_mvar` to retrieve the moving VAR.

#### [Volume Weighted Moving Average](https://www.investopedia.com/articles/trading/11/trading-with-vwap-mvwap.asp)

It's the moving average weighted by volume.

It has a parameter for window size.  The default window is 14.  Change it with
`set_dft_window('vwma', n)`.

Examples:
* `df['vwma']` retrieves the 14 periods VWMA
* `df['vwma_6']` retrieves the 6 periods VWMA

#### [CHOP - Choppiness Index](https://www.tradingview.com/education/choppinessindex/)

The Choppiness Index determines if the market is choppy.

It has a parameter for window size.  The default window is 14.  Change it with
`set_dft_window('chop', n)`.

Examples:
* `df['chop']` retrieves the 14 periods CHOP
* `df['chop_6']` retrieves the 6 periods CHOP

#### [MFI - Money Flow Index](https://www.investopedia.com/terms/m/mfi.asp)

The Money Flow Index
identifies overbought or oversold signals in an asset.

It has a parameter for window size.  The default window is 14.  Change it with
`set_dft_window('mfi', n)`.

Examples:
* `df['mfi']` retrieves the 14 periods MFI
* `df['mfi_6']` retrieves the 6 periods MFI

#### [ERI - Elder-Ray Index](https://admiralmarkets.com/education/articles/forex-indicators/bears-and-bulls-power-indicator)

The Elder-Ray Index contains the bull and the bear power.
Both are calculated based on the EMA of the close price.

The default window is 13.

Formular:
* Bulls Power = High - EMA
* Bears Power = Low - EMA
* EMA is exponential moving average of close of N periods

Examples:
* `df['eribull']` retrieves the 13 periods bull power
* `df['eribear']` retrieves the 13 periods bear power
* `df['eribull_5']` retrieves the 5 periods bull power
* `df['eribear_5']` retrieves the 5 periods bear power

#### [KER - Kaufman's efficiency ratio](https://strategyquant.com/codebase/kaufmans-efficiency-ratio-ker/)

The Efficiency Ratio (ER) is calculated by
dividing the price change over a period by the
absolute sum of the price movements that occurred
to achieve that change.

The resulting ratio ranges between 0 and 1 with
higher values representing a more efficient or
trending market.

The default column is close.

The default window is 10.

Formular:
* window_change = ABS(close - close[n])
* last_change = ABS(close-close[1])
* volatility = moving sum of last_change in n
* KER = window_change / volatility

Examples:
* `df['ker']` retrieves the 10 periods KER of the close price
* `df['high_5_ker']` retrieves 5 periods KER of the high price

#### [KAMA - Kaufman's Adaptive Moving Average](https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average)

Kaufman's Adaptive Moving Average is designed to account for market noise or 
volatility.

It has 2 optional parameters and 2 required parameters
* fast - optional, the parameter for fast EMA smoothing, default to 5
* slow - optional, the parameter for slow EMA smoothing, default to 34
* column - required, the column to calculate
* window - required, rolling window size

The default value for window, fast and slow can be configured with
`set_dft_window('kama', (10, 5, 34))`

Examples:
* `df['close_10,2,30_kama']` retrieves 10 periods KAMA of the close 
  price with `fast = 2` and `slow = 30`
* `df['close_2_kama']` retrieves 2 periods KAMA of the close price
  with default fast and slow

#### Cross Upwards and Cross Downwards

Use the pattern `<A>_xu_<B>` to check when A crosses up B.

Use the pattern `<A>_xd_<B>` to check when A crosses down B.

Use the pattern `<A>_x_<B>` to check when A crosses B.

Examples:
* `kdjk_x_kdjd` returns a series that marks the cross of KDJK and KDJD
* `kdjk_xu_kdjd` returns a series that marks where KDJK crosses up KDJD
* `kdjk_xd_kdjd` returns a series that marks where KDJD crosses down KDJD

#### [Aroon Oscillator](https://www.investopedia.com/terms/a/aroonoscillator.asp)

The Aroon Oscillator measures the strength of a trend and 
the likelihood that it will continue.

The default window is 25.

* Aroon Oscillator = Aroon Up - Aroon Down
* Aroon Up = 100 * (n - periods since n-period high) / n
* Aroon Down = 100 * (n - periods since n-period low) / n
* n = window size

Examples:
* `df['aroon']` returns Aroon oscillator with a window of 25
* `df['aroon_14']` returns Aroon oscillator with a window of 14

#### [Z-Score](https://www.investopedia.com/terms/z/zscore.asp)

Z-score is a statistical measurement that describes a value's relationship to 
the mean of a group of values. 

There is no default column name or window for Z-Score.

The statistical formula for a value's z-score is calculated using
the following formula:

```
z = ( x - μ ) / σ
```

Where:

* `z` = Z-score
* `x` = the value being evaluated
* `μ` = the mean
* `σ` = the standard deviation

Examples:
* `df['close_75_z']` returns the Z-Score of close price with a window of 75

#### [Awesome Oscillator](https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator)

The AO indicator is a good indicator for measuring the market dynamics,
it reflects specific changes in the driving force of the market, which
helps to identify the strength of the trend, including the points of
its formation and reversal.

Awesome Oscillator Formula

* MEDIAN PRICE = (HIGH+LOW)/2
* AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

Examples:
* `df['ao']` returns the Awesome Oscillator with default windows (5, 34)
* `df['ao_3,10']` returns the Awesome Oscillator with a window of 3 and 10

#### [Balance of Power](https://school.stockcharts.com/doku.php?id=technical_indicators:balance_of_power)

Balance of Power (BOP) measures the strength of the bulls vs. bears.

Formular:
```
BOP = (close - open) / (high - low)
```

Example:
* `df['bop']` returns the Balance of Power

#### [Chande Momentum Oscillator] (https://www.investopedia.com/terms/c/chandemomentumoscillator.asp)

The Chande Momentum Oscillator (CMO) is a technical momentum 
indicator developed by Tushar Chande.

The formula calculates the difference between the sum of recent 
gains and the sum of recent losses and then divides the result 
by the sum of all price movements over the same period.

The default window is 14.

Formular:
```
CMO = 100 * ((sH - sL) / (sH + sL))
```

where:
* sH=the sum of higher closes over N periods
* sL=the sum of lower closes of N periods

Examples:
* `df['cmo']` returns the CMO with a window of 14
* `df['cmo_5']` returns the CMO with a window of 5

#### [Coppock Curve](https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:coppock_curve)

Coppock Curve is a momentum indicator that signals
long-term trend reversals.

Formular:

Coppock Curve = 10-period WMA of (14-period RoC + 11-period RoC)
WMA = Weighted Moving Average
RoC = Rate-of-Change

Examples:
* `df['coppock']` returns the Coppock Curve with default windows
* `df['coppock_5,10,15']` returns the Coppock Curve with WMA window 5,
  fast window 10, slow window 15. 

#### [Ichimoku Cloud](https://www.investopedia.com/terms/i/ichimoku-cloud.asp)

The Ichimoku Cloud is a collection of technical indicators
that show support and resistance levels, as well as momentum
and trend direction.

In this implementation, we only calculate the delta between
lead A and lead B (which is the width of the cloud).

It contains three windows:
* window for the conversion line, default to 9
* window for the baseline and the shifts, default to 26
* window for the leading line, default to 52

Formular:
* conversion line = (PH9 + PL9) / 2
* baseline = (PH26 + PL26) / 2
* leading span A = (conversion line + baseline) / 2
* leading span B = (PH52 + PL52) / 2
* result = leading span A - leading span B

Where:
* PH = Period High
* PL = Period Low

Examples:
* `df['ichimoku']` returns the ichimoku cloud width with default windows
* `df['ichimoku_7,22,44']` returns the ichimoku cloud width with window sizes
  7, 22, 44

#### [Linear Regression Moving Average](https://www.daytrading.com/moving-linear-regression)

Linear regression works by taking various data points in a sample and
providing a “best fit” line to match the general trend in the data. 

Implementation reference:

https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/linreg.py

Examples:
* `df['close_10_lrma']` linear regression of close price with window size 10

#### [Correlation Trend Indicator](https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/C-D/CorrelationTrendIndicator)

Correlation Trend Indicator is a study that estimates
the current direction and strength of a trend.

Implementation is based on the following code:

https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/cti.py

Examples:
* `df['cti']` returns the CTI of close price with window 12
* `df['high_5_cti']` returns the CTI of high price with window 5

#### [the Gaussian Fisher Transform Price Reversals indicator](https://www.tradingview.com/script/ajZT2tZo-Gaussian-Fisher-Transform-Price-Reversals-FTR/)

The Gaussian Fisher Transform Price Reversals indicator, dubbed
FTR for short, is a stat based price reversal detection indicator
inspired by and based on the work of the electrical engineer
now private trader John F. Ehlers.

https://www.tradingview.com/script/ajZT2tZo-Gaussian-Fisher-Transform-Price-Reversals-FTR/

Implementation reference:

https://github.com/twopirllc/pandas-ta/blob/084dbe1c4b76082f383fa3029270ea9ac35e4dc7/pandas_ta/momentum/fisher.py#L9

Formular:
* Fisher Transform = 0.5 * ln((1 + X) / (1 - X))
* X is a series whose values are between -1 to 1

Examples:
* `df['ftr']` returns the FTR with window 9
* `df['ftr_20']` returns the FTR with window 20

#### [Relative Vigor Index (RVGI)](https://www.investopedia.com/terms/r/relative_vigor_index.asp)

The Relative Vigor Index (RVI) is a momentum indicator
used in technical analysis that measures the strength
of a trend by comparing a security's closing price to
its trading range while smoothing the results using a
simple moving average (SMA).

Formular

* NUMERATOR= (a+(2×b)+(2×c)+d) / 6
* DENOMINATOR= (e+(2×f)+(2×g)+h) / 6
* RVI= SMA-N of DENOMINATOR / SMA-N of NUMERATOR
* Signal Line = (RVI+(2×i)+(2×j)+k) / 6

where:

* a=Close−Open
* b=Close−Open One Bar Prior to a
* c=Close−Open One Bar Prior to b
* d=Close−Open One Bar Prior to c
* e=High−Low of Bar a
* f=High−Low of Bar b
* g=High−Low of Bar c
* h=High−Low of Bar d
* i=RVI Value One Bar Prior
* j=RVI Value One Bar Prior to i
* k=RVI Value One Bar Prior to j
* N=Minutes/Hours/Days/Weeks/Months

Examples:
* `df['rvgi']` retrieves the RVGI line of window 14
* `df['rvgis']` retrieves the RVGI signal line of window 14
* `df['rvgi_5']` retrieves the RVGI line of window 5
* `df['rvgis_5']` retrieves the RVGI signal line of window 5

#### [Inertia Indicator](https://theforexgeek.com/inertia-indicator/)

In financial markets, the concept of inertia was given by Donald Dorsey
in the 1995 issue of Technical Analysis of Stocks and Commodities
through the Inertia Indicator. The Inertia Indicator is moment-based
and is an extension of Dorsey’s Relative Volatility Index (RVI).

Formular:
* inertia = n periods linear regression of RVGI

Examples:
* `df['inertia']` retrieves the inertia of 20 periods linear regression of 14 periods RVGI
* `df['inertia_10']` retrieves the inertia of 10 periods linear regression of 14 periods RVGI

#### [Know Sure Thing (kst)](https://www.investopedia.com/terms/k/know-sure-thing-kst.asp)

The Know Sure Thing (KST) is a momentum oscillator developed by
Martin Pring to make rate-of-change readings easier for traders
to interpret.

Formular:
* KST=(RCMA1×1)+(RCMA2×2) + (RCMA3×3)+(RCMA4×4)

Where:
* RCMA1=10-period SMA of 10-period ROC
* RCMA2=10-period SMA of 15-period ROC
* RCMA3=10-period SMA of 20-period ROC
* RCMA4=15-period SMA of 30-period ROC

Example:
* `df['kst']` retrieves the KST.

#### [Pretty Good Oscillator (PGO)](https://library.tradingtechnologies.com/trade/chrt-ti-pretty-good-oscillator.html)

The Pretty Good Oscillator indicator by Mark Johnson measures the 
distance of the current close from its N-day simple moving average, 
expressed in terms of an average true range over a similar period.

Formular:
* PGO = (Close - SMA) / (EMA of TR)

Example:
* `df['pgo']` retrieves the PGO with default window 14.
* `df['pgo_10']` retrieves the PGO with window 10.

#### [Psychological Line (PSL)](https://library.tradingtechnologies.com/trade/chrt-ti-psychological-line.html)

The Psychological Line indicator is the ratio of the number of 
rising periods over the total number of periods.

Formular:
* PSL = (Number of Rising Periods) / (Total Number of Periods) * 100

Example:
* `df['psl']` retrieves the PSL with default window 12.
* `df['psl_10']` retrieves the PSL with window 10.
* `df['high_12_psl']` retrieves the PSL of high price with window 10.

#### [Percentage Volume Oscillator(PVO)](https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo)

The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume. 
The PVO measures the difference between two volume-based moving averages as
a percentage of the larger moving average.

Formular: 

* Percentage Volume Oscillator (PVO): 
  ((12-day EMA of Volume - 26-day EMA of Volume)/26-day EMA of Volume) x 100
* Signal Line: 9-day EMA of PVO
* PVO Histogram: PVO - Signal Line

Example:
* `df['pvo']` derives from the difference of 2 exponential moving average.
* `df['pvos]` is the signal line.
* `df['pvoh']` is he histogram line.

The period of short, long EMA and signal line can be tuned with 
`set_dft_window('pvo', (short, long, signal))`.  The default
windows are 12 and 26 and 9.

#### [Quantitative Qualitative Estimation(QQE)](https://www.tradingview.com/script/0vn4HZ7O-Quantitative-Qualitative-Estimation-QQE/)

The Qualitative Quantitative Estimation (QQE) indicator works like a smoother 
version of the popular Relative Strength Index (RSI) indicator. QQE expands 
on RSI by adding two volatility based trailing stop lines. These trailing 
stop lines are composed of a fast and a slow moving Average True Range (ATR). 
These ATR lines are smoothed making this indicator less susceptible to short 
term volatility.

Implementation reference:
https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/qqe.py

Example:
* `df['qqe']` retrieves the QQE with RSI window 14, MA window 5.
* `df['qqel']` retrieves the QQE long
* `df['qqes']` retrieves the QQE short
* `df['qqe_10,4']` retrieves the QQE with RSI window 10, MA window 4
* `df['qqel_10,4']` retrieves the QQE long with customized windows.
  Initialized by retrieving `df['qqe_10,4']`
* `df['qqes_10,4']` retrieves the QQE short with customized windows
  Initialized by retrieving `df['qqe_10,4']`

The period of short, long EMA and signal line can be tuned with 
`set_dft_window('qqe', (rsi, rsi_ma))`.  The default windows are 14 and 5.

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
