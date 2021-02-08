# Quantitative-Trading-Demo
This project shows how to do quant trade with abupy

## prerequisite

abupy is outdated and lack of maintainance. to run the code with the latest version of pandas, you have to update line 34 of

```shell
/path/to/site-package/abupy/CoreBu/ABuPdHelper.py
```

from 

```python
from pandas.core.window import EWM
```
to

```python
from pandas.core.window import ewm
```

## time picking demo

generate orders according to buy and sell strategies.

```python
python3 pick_time_demo.py
```

## stock picking demo

filter stocks according to given rules.

```python
python3 pick_stock_demo.py
```

## batch backtesting demo

batch the backtesting of a number of strategies.

```python
python3 batch_backtest.py
```

## parameter grid search demo

search the optimize parameter for buy and sell strategies.

```python
python3 grid_search.py
```
