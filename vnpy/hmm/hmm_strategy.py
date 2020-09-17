#!/usr/bin/python3

from math import log;
import pickle;
from hmmlearn.hmm import GaussianHMM;
from datetime import datetime;
from vnpy.trader.constant import Interval, Exchange;
from vnpy.app.cta_strategy import CtaTemplate, StopOrder, TickData, BarData, TradeData, OrderData, BarGenerator, ArrayManager;

class HMMStrategy(CtaTemplate):

  def __init__(self, cta_engine, strategy_name, vt_symbol, setting):

    super(HMMStrategy, self).__init__(cta_engine, strategy_name, vt_symbol, setting);
    self.bg = BarGenerator(self.on_bar);
    self.am = ArrayManager(15);
    with open('hmm.pkl', 'rb') as f:
      self.hmm = pickle.load(f);
  
  def on_init(self):

    self.write_log('策略初始化');
    self.load_bar(len(self.cta_engine.history_data));
    
  def on_start(self):

    self.write_log('策略启动');

  def on_stop(self):

    self.write_log('策略停止');

  def on_tick(self, tick: TickData):

    self.bg.update_tick(tick);

  def on_bar(self, bar: BarData):
    
    self.cancel_all();
    self.am.update_bar(bar);
    if not self.am.inited: return;
    X = [[log(self.am.close[i]) - log(self.am.close[i-1]),
          log(self.am.close[i]) - log(self.am.close[i-5]),
          log(self.am.high[i]) - log(self.am.low[i])] for i in range(5, self.am.close().shape[0])];
    colors = self.hmm.predict(X);
    if colors[-1] in [3, 4]:
      self.buy(bar.close_price, 1, stop = True);
    elif colors[-1] in [0, 5]:
      self.short(bar.close_price, 1, stop = True);
    else:
      if self.pos > 0:
        self.sell(bar.close_price, abs(self.pos));
      elif self.pos < 0:
        self.cover(bar.close_price, abs(self.pos));
    # stop loss at a floating price
    if self.pos > 0:
      self.sell(bar.close_price * 0.99, abs(self.pos));
    elif self.pos < 0:
      self.cover(bar.close_price * 0.99, abs(self.pos));
    self.put_event();

  def on_trade(self, trade: TradeData):

    pass;

  def on_order(self, order: OrderData):

    pass;

  def on_stop_order(self, stop_order: StopOrder):

    pass;

if __name__ == "__main__":

  from vnpy.trader.database import database_manager;
  from vnpy.app.cta_strategy.backtesting import BacktestingEngine, BacktestingMode, DailyResult;
  engine = BacktestingEngine();
  engine.set_parameters(
    vt_symbol = 'IF88.CFFEX', # 设置交易品种
    interval = Interval.MINUTE, # 按分钟模拟
    start = datetime(2009,1,1), # 回测起始时间
    end = datetime(2020,8,8), # 数据库中最近的日期
    rate = 0.5 / 10000, # 手续费率
    slippage = 0.2, #  滑点大小
    size = 300, # 合约大小
    pricetick = 0.2, # 价格精度
    capital = 1000000, # 资本
    mode = BacktestingMode.BAR, # 采用k线图
    inverse = False);
  engine.add_strategy(HMMStrategy, {});
  engine.load_data();
  engine.run_backtesting();
  engine.calculate_result();
  statistics = engine.calculate_statistics(output = True);
  engine.show_chart();
