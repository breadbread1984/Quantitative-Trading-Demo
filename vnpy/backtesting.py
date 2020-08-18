#!/usr/bin/python3

import json;
from datetime import datetime;
from vnpy.trader.constant import Interval, Exchange;
from vnpy.trader.object import HistoryRequest;
# data source
from vnpy.trader.rqdata import rqdata_client;
from vnpy.trader.tqdata import tqdata_client;
# data manager
from vnpy.trader.database import database_manager;
# back testing
from vnpy.app.cta_strategy.backtesting import BacktestingEngine, BacktestingMode, OptimizationSetting;
# strategies
from vnpy.app.cta_strategy.strategies.atr_rsi_strategy import AtrRsiStrategy;
from vnpy.app.cta_strategy.strategies.double_ma_strategy import DoubleMaStrategy;
from vnpy.app.cta_strategy.strategies.dual_thrust_strategy import DualThrustStrategy;
from DualThrustStrategyRefined import DualThrustStrategyRefined;

def main():

  '''
  # 0) 从天勤获取数据，免费的
  if not tqdata_client.inited:
    succeed = tqdata_client.init();
    if False == succeed:
      print('天勤连接失败');
      exit(1);
  req = HistoryRequest(symbol = 'au2012', exchange = Exchange.SHFE, interval = Interval.MINUTE, start = datetime.strptime('2020-01-01', '%Y-%m-%d'), end = datetime.strptime('2020-08-06', '%Y-%m-%d'));
  data = tqdata_client.query_history(req);
  '''

  # 0) 从米筐获取数据，收费的
  if not rqdata_client.inited:
    print('用账户(%s)登录米筐' % (rqdata_client.username));
    succeed = rqdata_client.init();
    if False == succeed:
      print('米筐登录失败');
      exit(1);
  req = HistoryRequest(symbol = 'IF88', exchange = Exchange.CFFEX, interval = Interval.MINUTE, start = datetime.strptime('2013-01-01', '%Y-%m-%d'), end = datetime.strptime('2020-08-08', '%Y-%m-%d'));
  data = rqdata_client.query_history(req);

  if data is None:
    print('获取数据失败');
    exit(1);
  database_manager.save_bar_data(data);

  # 1) 回测引擎
  engine = BacktestingEngine(); # 创建回测引擎
  engine.set_parameters(
    vt_symbol = 'IF88.CFFEX', # 设置交易品种
    interval = Interval.MINUTE, # 按分钟模拟
    start = datetime(2017,8,8), # 回测起始时间
    end = datetime(2020,8,8), # 数据库中最近的日期
    rate = 0.5 / 10000, # 手续费率
    slippage = 0.2, #  滑点大小
    size = 300, # 合约大小
    pricetick = 0.2, # 价格精度
    capital = 1000000, # 资本
    mode = BacktestingMode.BAR, # 采用k线图
    inverse = False);
  #engine.add_strategy(AtrRsiStrategy, {'atrLength': 11});
  #engine.add_strategy(DoubleMaStrategy, {'fast_window': 10, 'slow_window': 20});
  engine.add_strategy(DualThrustStrategyRefined, {'k1': 0.4, 'k2': 0.7, 'fixed_size': 1});
  # 2) 参数优化
   # target is at app/ct_strategy/backtesting.py
  setting = OptimizationSetting();
  setting.set_target('sharpe_ratio');
  setting.add_parameter('k1', 0., 1., 0.1);
  setting.add_parameter('k2', 0., 1., 0.1);
  result_values = engine.run_optimization(setting);
  params = json.loads(result_values[0][0].replace('\'','"'));
  engine.add_strategy(DualThrustStrategyRefined, {'k1': params['k1'], 'k2': params['k2'], 'fixed_size': 1});
  # 3) 回测
  engine.load_data();
  engine.run_backtesting();
  # 4) 结果
  engine.calculate_result();
  statistics = engine.calculate_statistics(output = True);
  engine.show_chart();
  daily_result = engine.get_all_daily_results();

if __name__ == "__main__":

  main();
