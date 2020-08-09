#!/usr/bin/python3

from datetime import time;
from vnpy.app.cta_strategy import CtaTemplate, StopOrder, TickData, BarData, TradeData, OrderData, BarGenerator, ArrayManager;
from vnpy.app.cta_strategy.strategies.dual_thrust_strategy import DualThrustStrategy;

class DualThrustStrategyRefined(DualThrustStrategy):

  long_stop = 0;
  short_stop = 0;
  trailing_percent = 0.8; #1 - 0.8%
  
  def __init__(self, cta_engine, strategy_name, vt_symbol, setting):

    super().__init__(cta_engine, strategy_name, vt_symbol, setting);
    
    self.bg = BarGenerator(self.onBar, 5, self.on_5min_bar);

  def onBar(self, bar):

    self.bg.updateBar(bar);

  def on_5min_bar(self, bar):

    # 取消所有之前没有成交的挂单
    self.cancel_all();

    self.bars.append(bar);
    if len(self.bars) <= 2: return;
    else: self.bars.pop(0);
    last_bar = self.bars[-2];
    
    if last_bar.datetime.date() != bar.datetime.date():
      # 如果来到新的一天，用当日开盘价更新上下界
      if self.day_high:
        self.day_range = self.day_high - self.day_low;
        self.long_entry = bar.open_price + self.kl * self.day_range;
        self.short_entry = bar.open_price - self.k2 * self.day_range;
      # 重置高/低价格
      self.day_open = bar.open_price;
      self.day_high = bar.high_price;
      self.day_low = bar.low_price;
      # 今天可以最多一次多仓，一次空仓
      self.long_entered = False;
      self.short_entered = False;
    else:
      # 更新高/低价格
      self.day_high = max(self.day_high, bar.high_price);
      self.day_low = min(self.day_low, bar.low_price);
      
    if not self.day_range: return;

    if bar.datetime.time() < self.exit_time:
      if self.pos == 0:
        if bar.close_price > self.day_open:
          # 如果当前5分钟收盘价超过当日开盘价，就挂上界停止多单
          if not self.long_entered:
            self.buy(self.long_entry, self.fixed_size, stop = True);
        else:
          # 如果当前5分钟收盘价低于当日开盘价，就挂下界停止空单
          if not self.short_entered:
            self.short(self.short_entry, self.fixed_size, stop = True);
      elif self.pos > 0:
        self.long_entered = True;
        # NOTE: 这里采用动态多单止损价格
        self.long_stop = self.day_high * (1 - self.trailing_percent / 100);
        # 如果有了多单，就设置下界为止损
        self.sell(self.long_stop, self.fixed_size, stop = True);
        
        if not self.short_entered:
          # 多单止损后，立马挂空停止单
          self.short(self.long_stop, self.fixed_size, stop = True);
      elif self.pos < 0:
        self.short_entered = True;
        # NOTE: 这里采用动态空单止损价格
        self.short_stop = self.day_low * (1 + self.trailing_percent / 100);
        # 如果有了空单，就设置上界为止损
        self.cover(self.short_stop, self.fixed_size, stop = True);
        
        if not self.long_entered:
          # 空单止损后，立马挂多停止单
          self.buy(self.short_stop, self.fixed_size, stop = True);
    else:
      if self.pos > 0:
        self.sell(bar.close_price * 0.99, abs(self.pos));
      elif self.pos < 0:
        self.cover(bar.close_price * 0.99, abs(self.pos));
        
    self.put_event();
