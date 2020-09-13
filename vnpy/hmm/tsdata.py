#!/usr/bin/python3

from datetime import datetime;
from numpy import ndarray;
from typing import List, Optional;
import tushare as ts;
from vnpy.trader.constant import Exchange, Interval;
from vnpy.trader.object import BarData, HistoryRequest;

class TushareClient:

  def __init__(self):

    self.inited: bool = False;
    self.symbols: ndarray = None;
    self.ex = {Exchange.SHFE: 'SHF', Exchange.CZCE: 'ZCE', Exchange.CFFEX: 'CFX', Exchange.DCE: 'DCE', Exchange.INE: 'INE', Exchange.SSE: 'SH', Exchange.SZSE: 'SZ'};

  def init(self) -> bool:

    if self.inited: return True;
    # initialize tushare pro api
    self.pro = ts.pro_api(token = 'a7455de91b1ffd9ebeacf63bccec8cc3b5d7de8e3e57c6bdfdba770e');
    self.inited = True;
    return True;

  def to_ts_symbol(self, symbol: str, exchange: Exchange) -> str:

    return '.'.join([symbol, self.ex[exchange]]);

  def query_history(self, req: HistoryRequest) -> Optional[List[BarData]]:

    symbol = req.symbol;
    exchange = req.exchange;
    interval = req.interval;
    start = req.start;
    end = req.end;

    if exchange not in self.ex:
      print('不是Tushare支持的交易所');
      return None;

    if interval == Interval.DAILY:
      if exchange in [Exchange.SHFE, Exchange.CZCE, Exchange.CFFEX, Exchange.DCE, Exchange.INE]:
        df = self.pro.fut_daily(ts_code = self.to_ts_symbol(symbol, exchange), asset = 'FT', start_date = start.strftime('%Y%m%d'), end_date = end.strftime('%Y%m%d'));
      elif exchange in [Exchange.SSE, Exchange.SZSE]:
        df = self.pro.daily(ts_code = self.to_ts_symbol(symbol, exchange), start_date = start.strftime('%Y%m%d'), end_date = end.strftime('%Y%m%d'));
    else:
      df = self.pro.pro_bar(ts_code = self.to_ts_symbol(symbol, exchange), 
                            asset = 'FT' if exchange in [Exchange.SHFE, Exchange.CZCE, Exchange.CFFEX, Exchange.DCE, Exchange.INE] else ('E' if exchange in [Exchange.SSE, Exchange.SZSE] else None),
                            start_date = start.strftime('%Y%m%d'), 
                            end_date = end.strftime('%Y%m%d'), 
                            freq = '60min' if interval == Interval.HOUR else ('1min' if interval == Interval.MINUTE else None));
    df = df.sort_index();

    data: List[BarData] = [];

    if df is not None:
      for ix, row in df.iterrows():
        date = datetime.strptime(row.trade_date, '%Y%m%d');
        bar = BarData(
          symbol = symbol,
          exchange = exchange,
          interval = interval,
          datetime = date,
          open_price = row['open'],
          high_price = row['high'],
          low_price = row['low'],
          close_price = row['close'],
          volume = row['amount'],
          gateway_name = 'TS',
        );
        data.append(bar);
    return data;

tsdata_client = TushareClient();

