#!/usr/bin/python3

from datetime import datetime;
from typing import List, Dict;
from vnpy.app.portfolio_strategy import StrategyTemplate, StrategyEngine;
from vnpy.trader.utility import BarGenerator, ArrayManager;
from vnpy.trader.object import TickData, BarData, OrderData, TradeData;
from vnpy.trader.constant import Direction, Offset;

class TurtleStrategy(StrategyTemplate):

  # 合约->合约规模：一手>合约的美元价格
  size_dict = {'IF99.CFFEX': 300, 'I99.DCE': 100, 'CU99.SHFE': 5, 'TA99.CZCE': 5};
  entry_window = 55;
  exit_window = 20;
  atr_window = 20;
  max_product_pos = 4; # 最大仓位
  max_direction_pos = 10; # 最大仓量
  # NOTE: the following line is valid only when you use a real ctp account
  capital = 1000000; #self.strategy_engine.main_engine.get_account('breadbread1984');

  parameters = ['capital', 'entry_window', 'exit_window', 'atr_window', 'max_product_pos', 'max_direction_pos', 'size_dict'];
  variables = []

  def __init__(self, strategy_engine: StrategyEngine, strategy_name: str, vt_symbols: List[str], setting: dict):

    super(TurtleStrategy, self).__init__(strategy_engine, strategy_name, vt_symbols, setting);
    self.vt_symbols = vt_symbols; # 合约的list
    self.total_long = 0; # 所有合约的多仓位和
    self.total_short = 0; # 所有合约的空仓位和
    self.context = {
      symbol: {
        'entry_long': 0, # 开/加多仓价格阈值
        'entry_short': 0, # 开/加空仓价格阈值
        'exit_long': 0, # 平多仓价格阈值
        'exit_short': 0, # 平空仓价格阈值
        'atr_value': 0, # ATR
        'stop_loss_long': 0, # 多仓止损价格
        'stop_loss_short': 0, # 空仓止损价格
        'multiplier': 0, # 合约规模
        'unit': 0, # 仓位
        'am': ArrayManager(),
      } for symbol in vt_symbols
    };
    def on_bar(bar: BarData):
      pass;
    self.bgs = {symbol: BarGenerator(on_bar) for symbol in vt_symbols};
    self.last_tick_time: datetime = None;

  def on_init(self):

    self.write_log("策略初始化");
    self.load_bars(1);

  def on_start(self):

    self.write_log("策略启动");

  def on_stop(self):

    self.write_log("策略停止");

  def on_tick(self, tick: TickData):

    if self.last_tick_time and self.last_tick_time.minute != tick.datetime.minute:
      bars = dict();
      for vt_symbol, bg in self.bgs.items():
        bars[vt_symbol] = bg.generate();
      self.on_bars(bars);
    bg: BarGenerator = self.bgs[tick.vt_symbol];
    bg.update_tick(tick);
    self.last_tick_time = tick.datetime;

  def on_bars(self, bars: Dict[str, BarData]):

    # 取消所有没有成交的停止单
    self.cancel_all();
    for vt_symbol in self.vt_symbols:
      context = self.context[vt_symbol];
      if vt_symbol not in bars: continue;
      bar = bars[vt_symbol];
      # 检查array manager是否收集足够的数据
      context['am'].update_bar(bar);
      if not context['am'].inited: continue;
      # 1) 空仓的情况下更新价格阈值
      if not self.get_pos(vt_symbol):
        context['entry_long'], context['entry_short'] = context['am'].donchian(self.entry_window);
        context['exit_short'], context['exit_long'] = context['am'].donchian(self.exit_window);
        context['atr_value'] = context['am'].atr(self.atr_window);
        if context['atr_value'] == 0: continue;
        # 头寸规模(multiplier)=(1%*资本总数)/(ATR*合约规模)
        multiplier = self.capital * 0.01 / (context['atr_value'] * self.size_dict[vt_symbol]);
        multiplier = int(round(multiplier, 0));
        context['multiplier'] = multiplier;
        context['stop_loss_long'] = 0;
        context['stop_loss_short'] = 0;
      # 2) 加仓（开仓）/平仓
      if not self.get_pos(vt_symbol):
        # 开多/空仓二选一
        self.send_buy_orders(vt_symbol, context['entry_long']);
        self.send_short_orders(vt_symbol, context['entry_short']);
      elif self.get_pos(vt_symbol) > 0:
        # 加/平多仓二选一
        self.send_buy_orders(vt_symbol, context['entry_long']);
        sell_price = max(context['stop_loss_long'], context['exit_long']);
        self.sell(vt_symbol, sell_price, abs(self.get_pos(vt_symbol)), True);
      elif self.get_pos(vt_symbol) < 0:
        # 加/平空仓二选一
        self.send_short_orders(vt_symbol, context['entry_short']);
        cover_price = min(context['stop_loss_short'], context['exit_short']);
        self.cover(vt_symbol, cover_price, abs(self.get_pos(vt_symbol)), True);
    self.put_event();

  def update_order(self, order: OrderData):

    # 下单回调
    context = self.context[order.symbol + '.' + order.exchange.value];
    if order.direction == Direction.LONG:
      if self.total_long >= self.max_direction_pos: return;
      if context['unit'] >= self.max_product_pos: return;
    else:
      if self.total_short <= -self.max_direction_pos: return;
      if context['unit'] <= -self.max_product_pos: return;
    super(TurtleStrategy, self).update_order(order);

  def update_trade(self, trade: TradeData):

    # 成交回调
    context = self.context[trade.symbol + '.' + trade.exchange.value];
    super(TurtleStrategy, self).update_trade(trade);
    # 计算止损价格
    if trade.direction == Direction.LONG:
      context['stop_loss_long'] = trade.price - 2 * context['atr_value'];
    else:
      context['stop_loss_short'] = trade.price + 2 * context['atr_value'];
    if trade.direction == Direction.LONG:
      # 加多仓或者平空仓
      context['unit'] = 0 if self.get_pos(trade.symbol) == 0 else context['unit'] + 1;
    else:
      # 加空仓或者平多仓
      context['unit'] = 0 if self.get_pos(trade.symbol) == 0 else context['unit'] - 1;
    self.total_long = 0;
    self.total_short = 0;
    for vt_symbol, context in self.context.items():
      if self.get_pos(vt_symbol) > 0: self.total_long += context['unit'];
      elif self.get_pos(vt_symbol) < 0: self.total_short += context['unit'];

  def send_buy_orders(self, vt_symbol, price):

    context = self.context[vt_symbol];
    if context['unit'] < 1:
      self.buy(vt_symbol, price, 1 * context['multiplier'], True);
    if context['unit'] < 2:
      self.buy(vt_symbol, price + context['atr_value'] * 0.5, 1 * context['multiplier'], True);
    if context['unit'] < 3:
      self.buy(vt_symbol, price + context['atr_value'] * 1.0, 1 * context['multiplier'], True);
    if context['unit'] < 4:
      self.buy(vt_symbol, price + context['atr_value'] * 1.5, 1 * context['multiplier'], True);

  def send_short_orders(self, vt_symbol, price):

    context = self.context[vt_symbol];
    if context['unit'] > -1:
      self.short(vt_symbol, price, 1 * context['multiplier'], True);
    if context['unit'] > -2:
      self.short(vt_symbol, price - context['atr_value'] * 0.5, 1 * context['multiplier'], True);
    if context['unit'] > -3:
      self.short(vt_symbol, price - context['atr_value'] * 1.0, 1 * context['multiplier'], True);
    if context['unit'] > -4:
      self.short(vt_symbol, price - context['atr_value'] * 1.5, 1 * context['multiplier'], True);

