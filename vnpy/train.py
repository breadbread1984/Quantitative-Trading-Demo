#!/usr/bin/python3

from os.path import exists;
from datetime import time;
import re;
import pickle;
import tensorflow as tf;
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork;
from tf_agents.networks.value_rnn_network import ValueRnnNetwork;
from tf_agents.agents.ppo import ppo_agent;
from tf_agents.trajectories.time_step import time_step_spec;
from tf_agents.specs import ArraySpec;
from tf_agents.policies import policy_saver;
import tushare as ts;
from vnpy.app.cta_strategy import CtaTemplate, BarGenerator, ArrayManager;
from vnpy.trader.constant import Interval, Exchange;
from vnpy.trader.rqdata import rqdata_client;
from vnpy.trader.object import HistoryRequest;
from vnpy.trader.database import database_manager;
from vnpy.app.cta_strategy.backtesting import BacktestingEngine;

class AgentStrategy(CtaTemplate):

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 1e-3);
  # observation = (volume, open interest, open price, close price, high price, low price,)
  obs_spec = array_spec.ArraySpec((6,), dtype = tf.float32);
  reward_spec = array_spec.ArraySpec((), dtype = tf.float32);
  # action = {long: 0, short: 1, sell: 2, cover: 3, close: 4, none: 5}
  action_spec = array_spec.BoundedArraySpec((1,), dtype = tf.float32, minimum = [0,], maximum = [5,]);
  actor_net = ActorDistributionRnnNetwork(obs_spec, action_spec, lstm_size = (100,100));
  value_net = ValueRnnNetwork(obs_spec);
  agent = ppo_agent.PPOAgent(
    time_spec(observation_spec = obs_spec, reward_spec = reward_spec),
    action_spec,
    optimizer= optimizer,
    actor_net = actor_net,
    normalize_observation = True,
    normalize_rewards = False,
    use_gea = True,
    num_epochs = 1
  );
  agent.initialize();
  
  def __init__(self, cta_engine, strategy_name, vt_symbol, setting):

    super(AgentStrategy, self).__init__(cta_engine, strategy_name, vt_symbol, setting);
    self.bg = BarGenerator(self.on_bar);
    self.am = ArrayManager();

  def on_init(self):

    self.write_log('策略初始化');
    self.load_bar(1);
    self.policy_state = self.agent.policy.get_initial_state(1);

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
    status = tf.constant([bar.volume, bar.open_interest, bar.open_price, bar.close_price, bar.high_price, bar.low_price], dtype = tf.float32);
    action = self.agent.policy.action(status,self.policy_state);
    # FIXME: 每天收盘之前平仓
    if action[0] == 0 and self.pos >= 0:
      self.buy(bar.close_price, 1, True);
    elif action[0] == 1 and self.pos <= 0:
      self.short(bar.close_price, 1, True);
    elif action[0] == 2 and self.pos > 0:
      self.sell(bar.close_price, 1, True);
    elif action[0] == 3 and self.pos < 0:
      self.cover(bar.close_price, 1, True);
    elif action[0] == 4 and self.pos != 0:
      if self.pos > 0:
        self.sell(bar.close_price, abs(self.pos), True);
      if self.pos < 0:
        self.cover(bar.close_price, abs(self.pos), True);
    elif action[0] == 5:
      pass;
    self.policy_state = action.state;
    self.put_event();

  def on_trade(self, trade: TradeData):

    pass;

  def on_order(self, order: OrderData):

    self.put_event();

  def on_stop_order(self, stop_order: StopOrder):

    pass;

def get_fut_info(futs):

  fee = {'IF': 0.23/10000, 'IC': 0.23/10000, 'IH': 0.23/10000, 'T': 3, 'TF': 3, 'TS': 3,
         'SC': 20, 'RB': 1/10000, 'HC': 1/10000, 'NI': 6, 'CU': 0.5/10000, 'ZN': 3, 'AL': 3, 'SN': 1, 'PB': 0.4/10000, 'AU': 10, 'AG': 0.5/10000, 'WR': 0.4/10000, 'RU': 0.45/10000, 'BU': 1/10000, 'FU': 0.5/10000,
         'I': 0.6/10000, 'J': 0.6/10000, 'JM': 0.6/10000, 'BB': 1/10000, 'FB': 1/10000, 'V': 2, 'C': 0.2, 'CS': 1.5, 'A': 2, 'B': 2, 'L': 2, 'M': 0.2, 'P': 2.5, 'Y': 2.5, 'PP': 0.6/10000, 'JD': 1.5/10000,
         'FG': 3, 'ZC': 4, 'MA': 2, 'SM': 3, 'SF': 3, 'OI': 2, 'SR': 3, 'TA': 3, 'WH': 2.5, 'CY': 4, 'CF':4.3, 'RM': 1.5, 'AP': 20, 'JR': 3, 'LR': 3, 'PM': 5, 'RI': 2.5, 'RS': 2};
  pro = ts.pro_api(token = 'a7455de91b1ffd9ebeacf63bccec8cc3b5d7de8e3e57c6bdfdba770e');
  p = re.compile('([A-Za-z]*)([0-9]*)');
  info = dict();
  for exchange, symbols in futs.items():
    # 合约名称，合约规模，价格单位，价格最小变动
    df = pro.fut_basic(exchange = exchange.value, fut_type = '1', fields = 'symbol, per_unit, quote_unit, quote_unit_desc');
    for symbol in symbols:
      results = p.search(symbol);
      prefix = results.group(1);
      match = df.loc[df['symbol'].str.startswith(prefix)].iloc[0];
      data = database_manager.load_bar_data(symbol = symbo, exchange = exchange, interval = Interval.MINUTE, start = datatime(2009, 1, 1), end = datetime(2020,8,15));
      if len(data) == 0:
        req = HistoryRequest(symbol = symbol, exchange = exchange, interval = Interval.MINUTE, start = datetime(2009,1,1), end = datetime(2020,8,15));
        data = rqdata_client.query_history(req);
        if data is None or len(data) == 0:
          print('symbol: ' + symbol + '没有找到数据');
          continue;
      info[symbol] = {'rate': fee[prefix], 
                      'size': float(match['per_unit']), 
                      'pricetick': float(match['quote_unit_desc'][:-len(match['quote_unit'])]), 
                      'start_date': data[0].datetime, 
                      'end_date': data[-1].datetime};
  return info;

if __name__ == "__main__":

  if not rqdata_client.inited:
    print('用账户(%s)登录米筐' % (rqdata_client.username));
    succeed = rqdata_client.init();
    if False == succeed:
      print('米筐登录失败');
      exit(1);
  futures = {
    Exchange.CFFEX: ['IF99','IC99','IH99','T99','TF99','TS99',],
    Exchange.SHFE: ['SC99','RB99','HC99','NI99','CU99','ZN99','AL99','SN99','PB99','AU99','AG99','WR99','RU99','BU99','FU99',],
    Exchange.DCE: ['I99','J99','JM99','BB99','FB99','V99','C99','CS99','A99','B99','L99','M99','P99','Y99','PP99','JD99',],
    Exchange.CZCE: ['FG99','ZC99','MA99','SM99','SF99','OI99','SR99','TA99','WH99','CY99','CF99','RM99','AP99','JR99','LR99','PM99','RI99','RS99',],
  };
  if not exists('info.pkl'):
    info = get_fut_info(futures);
    with open('info.pkl', 'wb') as f:
      f.write(pickle.dumps(info));
  else:
    with open('info.pkl', 'rb') as f:
      info = pickle.loads(f.read());
  engine = BacktestingEngine();
  for i in range(100):
    for exchange, symbols in futures.items():
      for symbol in symbols:
        engine.set_parameters(
          vt_symbol = symbol + "." + exchange.value,
          interval = Interval.MINUTE,
          start = info[symbol]['start_date'],
          end = info[symbol]['end_date'],
          rate = info[symbol]['rate'],
          slippage = info[symbol]['pricetick'],
          size = info[symbol]['size'],
          pricetick = info[symbol]['pricetick'],
          capital = 1000000,
          mode = BacktestingMode.BAR,
          inverse = False);
