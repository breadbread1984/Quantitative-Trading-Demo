#!/usr/bin/python3

from os.path import exists;
import tensorflow as tf;
from vnpy.app.cta_strategy import CtaTemplate, BarGenerator, ArrayManager, TickData;

def PositionPredictor(days = 10):

  inputs = tf.keras.Input((days + 3,)); # inputs.shape = (batch, days + 3)
  results = tf.keras.layers.Dense(units = 1, activation = tf.math.tanh)(inputs); # results.shape = (batch, 1)
  return tf.keras.Model(inputs = inputs, outputs = results);

class MaxSharpeStrategy(CtaTemplate):

  M = 10;
  predictor = PositionPredictor(M);
  if exists('weights.h5'):
    predictor.load_weights('weights.h5');
  parameters = ["M"];

  def __init__(self, cta_engine, strategy_name, vt_symbol, setting):

    super(MaxSharpeStrategy, self).__init__(cta_engine, strategy_name, vt_symbol, setting);
    self.bg = BarGenerator(self.on_bar);
    self.am = ArrayManager(3 + self.M); # am.shape = (3 + M)
    self.F_tm1 = tf.zeros((1, self.M + 3)); # F_tm1.shape = (1, M + 3)

  def on_init(self):

    self.write_log('策略初始化');
    self.load_bar(len(self.cta_engine.history_data));
    self.mu = self.cta_engine.capital / self.cta_engine.size; # how many possible positions

  def on_start(self):

    self.write_log('策略启动');

  def on_stop(self):

    self.write_log('策略停止');

  def on_tick(self, tick: TickData):
    
    self.bg.update_tick(tick);

  def on_bar(self, bar: BarData):

    self.cancel_all();
    self.am.update_bar(bar);
    if not am.inited: return;
    xt = tf.ones((self.M + 3), dtype = tf.float32); # xt.shape = (M + 3)
    for i in range(1, self.M + 2):
      xt[i] = self.am.close[i] - self.am.close[i - 1];
    xt[self.M + 2] = self.F_tm1[0, 0];
    xt = tf.expand_dims(xt, axis = 0); # xt.shape = (batch = 1, M + 3)
    F_t = self.predictor(xt);
    pos = int(F_t * self.mu);
    self.F_tm1 = F_t;
    if self.pos > 0 and pos > 0:
      if pos > self.pos:
        self.buy(bar.close_price, pos - self.pos, stop = True);
      if pos < self.pos:
        self.sell(bar.close_price, self.pos - pos, stop = True);
    elif self.pos > 0 and pos < 0:
      self.sell(bar.close_price, abs(self.pos), stop = True);
      self.short(bar.close_price, abs(pos), stop = True);
    elif self.pos < 0 and pos < 0:
      if pos < self.pos:
        self.short(bar.close_price, self.pos - pos, stop = True);
      if pos > self.pos:
        self.cover(bar.close_price, pos - self.pos, stop = True);
    else: # self.pos < 0 and pos > 0
      self.cover(bar.close_price, abs(self.pos), stop = True);
      self.buy(bar.close_price, abs(pos), stop = True);
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
  engine.add_strategy(MaxSharpeStrategy, {'M': 10});
  engine.load_data();
  engine.strategy.on_init()
  engine.strategy.inited = True;
  engine.strategy.on_start();
  engine.strategy.trading = True;
  for data in engine.history_data[0:]:
    engine.new_bar(data);
  engine.calculate_result();
  statistics = engine.calculate_statistics(output = True);
  engine.show_chart();
