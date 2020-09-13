#!/usr/bin/python3

from os.path import exists, join;
from datetime import datetime, timedelta;
import tensorflow as tf;
from tf_agents.policies import policy_saver; # policy
from tf_agents.trajectories.time_step import TimeStep, StepType, time_step_spec;
from vnpy.app.cta_strategy import CtaTemplate, BarGenerator, ArrayManager;
from vnpy.trader.object import HistoryRequest, TickData, BarData, TradeData, OrderData;
from vnpy.app.cta_strategy.base import StopOrder;
from vnpy.app.cta_strategy.backtesting import DailyResult;

class PPOStrategy(CtaTemplate):

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 1e-3);
  # observation = (volume, open interest, open price, close price, high price, low price,)
  obs_spec = TensorSpec((7,), dtype = tf.float32, name = 'observation');
  # action = {long/cover: 0, short/sell: 1, close: 2, none: 3}
  action_spec = BoundedTensorSpec((1,), dtype = tf.int32, minimum = 0, maximum = 3, name = 'action');
  actor_net = ActorDistributionRnnNetwork(obs_spec, action_spec, lstm_size = (100,100));
  value_net = ValueRnnNetwork(obs_spec);
  agent = ppo_agent.PPOAgent(
    time_step_spec = time_step_spec(obs_spec),
    action_spec = action_spec,
    optimizer= optimizer,
    actor_net = actor_net,
    value_net = value_net,
    normalize_observations = True,
    normalize_rewards = True,
    use_gae = True,
    num_epochs = 1
  );
  agent.initialize();
  
  def __init__(self, cta_engine, strategy_name, vt_symbol, setting):

    super(PPOStrategy, self).__init__(cta_engine, strategy_name, vt_symbol, setting);
    self.bg = BarGenerator(self.on_bar);
    self.checkpointer = Checkpointer(
      ckpt_dir = 'checkpoints/policy',
      max_to_keep = 20,
      agent = self.agent,
      policy = self.agent.policy,
      global_step = tf.compat.v1.train.get_or_create_global_step()
    );
    self.checkpointer.initialize_or_restore();
    self.trading = True;

  def on_init(self):

    self.write_log('策略初始化');
    self.load_bar(len(self.cta_engine.history_data));
    self.policy_state = self.agent.policy.get_initial_state(1);
    self.last_ts = None;
    self.last_action = None;
    self.total_pnl = 0;
    self.end_of_epside = False;
    # context for calculate result
    self.pre_close = None;
    self.start_pos = 0;

  def on_start(self):

    self.write_log("策略启动");
    self.put_event();

  def on_stop(self):

    self.write_log("策略停止");
    self.put_event();
    
  def on_tick(self, tick: TickData):

    self.bg.update_tick(tick);

  def on_bar(self, bar: BarData):

    self.cancel_all();
    if self.end_of_epside: 
      self.put_event()
      return;
    # NOTE: this function is executed when everyday ends
    # 1) calculate r_{t-1}
    reward = 0;
    daily_result = DailyResult(bar.datetime, bar.close_price);
    for trade in self.cta_engine.trades.values():
      if trade.datetime.date() == bar.datetime.date():
        daily_result.add_trade(trade);
    if len(daily_result.trades):
        # only update pnl when there are new trades today
        daily_result.calculate_pnl(
          self.pre_close,
          self.start_pos,
          self.cta_engine.size,
          self.cta_engine.rate,
          self.cta_engine.slippage,
          self.cta_engine.inverse);
        self.pre_close = daily_result.close_price;
        self.start_pos = daily_result.end_pos;
        reward = daily_result.net_pnl;
    self.total_pnl += reward;
    if self.pre_close is None: self.pre_close = bar.close_price;
    # 2) rollout
    # ts = (step_type_t, reward_{t-1}, discount_t, status_t)
    ts = TimeStep(
      step_type = tf.constant([StepType.FIRST if self.last_ts is None else (StepType.LAST if bar.datetime.date() == self.cta_engine.end.date() or self.cta_engine.capital + self.total_pnl <= 0 else StepType.MID)], dtype = tf.int32), 
      reward = tf.constant([reward], dtype = tf.float32), # to reduce drawdown
      discount = tf.constant([0.8], dtype = tf.float32),
      observation = tf.constant([[bar.volume, bar.open_interest, bar.open_price, bar.close_price, bar.high_price, bar.low_price, self.pos]], dtype = tf.float32));
    if ts.step_type == StepType.LAST:
      self.end_of_epside = True;
      print('agent\'s total_pnl: %f' % (self.total_pnl));
      self.put_event();
      return;
    action = self.agent.collect_policy.action(ts, self.policy_state); # action_t
    self.last_ts = ts;
    self.last_action = action;
    if action.action[0, 0] == 0:
      if self.pos >= 0: # long
        self.buy(bar.close_price, 1, True);
      else: # cover
        self.cover(bar.close_price, abs(self.pos), True);
    elif action.action[0, 0] == 1:
      if self.pos <= 0: # short
        self.short(bar.close_price, 1, True);
      else: # sell
        self.sell(bar.close_price, abs(self.pos), True);
    elif action.action[0, 0] == 2 and self.pos != 0:
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
