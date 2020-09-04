#!/usr/bin/python3

from datetime import datetime, timedelta;
import tensorflow as tf;
from tf_agents.policies import policy_saver; # policy
from tf_agents.trajectories.time_step import TimeStep, StepType, time_step_spec;
from vnpy.app.cta_strategy import CtaTemplate, BarGenerator, ArrayManager;
from vnpy.trader.object import HistoryRequest, TickData, BarData, TradeData, OrderData;
from vnpy.app.cta_strategy.base import StopOrder;

class PPOStrategy(CtaTemplate):

  policy = tf.compat.v2.saved_model.load('checkpoints/policy');

  def __init__(self, cta_engine, strategy_name, vt_symbol, setting):

    super(PPOStrategy, self).__init__(cta_engine, strategy_name, vt_symbol, setting):
    self.bg = BarGenerator(self.on_bar);
    self.am = ArrayManager(3); # the 

  def on_init(self):

    self.write_log('策略初始化');
    # FIXME: 生产环境下需要修改
    self.load_bar(len(self.cta_engine.history_data));
    self.policy_state = self.policy.get_initial_state(1);
    self.pos_history = list(); # yesterday's and today's bar
    self.last_ts = None;

  def on_start(self):

    self.write_log('策略启动');
    self.put_event();

  def on_stop(self):

    self.write_log('策略停止');
    self.put_event();

  def on_tick(self, tick: TickData):

    self.bg.update_tick(tick);

  def on_bar(self, bar: BarData):

    self.cancel_all();
    self.am.update_bar(bar);
    self.pos_history.append(self.pos);
    if len(self.pos_history) > 2: self.pos_history.pop(0);
    if not self.am.inited: return;
    # collect yesterday's trades to inference reward_{t-1}
    daily_result = DailyResult((bar.datetime - timedelta(days = 1)).date(), self.am.close[1]); # yesterday's close price
    for trade in self.cta_engine.trades.values():
      trade_date = trade.datetime.date();
      if (bar.datetime - timedelta(days = 1)).date() == trade_date:
        daily_result.add_trade(trade);
    # get yesterday's pnl
    daily_result.calculate_pnl(
      self.am.close[0], # the day before yesterday's close
      self.pos_history[0], # yesterday's start pos
      self.cta_engine.size,
      self.cta_engine.rate,
      self.cta_engine.slippage,
      self.cta_engine.inverse);
    # create time step
    status = [bar.volume, bar.open_interest, bar.open_price, bar.close_price, bar.high_price, bar.low_price, self.pos];
    last_reward = daily_result.net_pnl;
    ts = TimeStep(
      step_type = tf.constant([StepType.FIRST if self.last_ts is None else (StepType.LAST if bar.datetime.date() == self.cta_engine.end.date() else StepType.MID)], dtype = tf.int32),
      reward = tf.constant([last_reward], dtype = tf.float32),
      discount = tf.constant([1.], dtype = tf.float32),
      observation = tf.constant([status], dtype = tf.float32));
    # infer action
    action = self.agent.policy.action(ts, self.policy_state);
    self.last_ts = ts;
    if action.action[0,0] == 0:
      if self.pos >= 0: # long
        self.buy(bar.close_price, 1, True);
      else: # cover
        self.cover(bar.close_price, abs(self.pos), True);
    elif action.action[0,0] == 1:
      if self.pos <= 0: # short
        self.short(bar.close_price, 1, True);
      else: # sell
        self.sell(bar.close_price, abs(self.pos), True);
    elif action.action[0,0] == 2 and self.pos != 0:
      if self.pos > 0: # sell
        self.sell(bar.close_price, abs(self.pos), True);
      if self.pos < 0: # cover
        self.cover(bar.close_price, abs(self.pos), True);
    elif action.action[0, 0] == 3:
      pass;
    self.policy_state = action.state;
    self.put_event();

  def on_trade(self, trade: TradeData):

    pass;

  def on_order(self, order: OrderData):

    self.put_event();
    
  def on_stop_order(self, stop_order: StopOrder):

    pass;
