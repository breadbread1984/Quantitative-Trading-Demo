#!/usr/bin/python3

import tensorflow as tf;
from tf_agents.policies import policy_saver; # policy
from vnpy.app.cta_strategy import CtaTemplate, BarGenerator, ArrayManager;
from vnpy.trader.object import HistoryRequest, TickData, BarData, TradeData, OrderData;

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

