#!/usr/bin/python3

from datetime import datetime;
from vnpy.trader.constant import Interval, Exchange;
from vnpy.trader.object import HistoryRequest;
from vnpy.trader.rqdata import rqdata_client;
#from vnpy.trader.tqdata import tqdata_client;
from vnpy.trader.database import database_manager;
from vnpy.app.cta_strategy.backtesting import BacktestingEngine, BacktestingMode, OptimizationSetting;
from vnpy.app.cta_strategy.strategies.atr_rsi_strategy import AtrRsiStrategy;

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
  req = HistoryRequest(symbol = 'IF88', exchange = Exchange.CFFEX, interval = Interval.MINUTE, start = datetime.strptime('2017-08-08', '%Y-%m-%d'), end = datetime.strptime('2020-08-07', '%Y-%m-%d'));
  data = rqdata_client.query_history(req);

  if data is None:
    print('获取数据失败');
    exit(1);
  database_manager.save_bar_data(data);
  
  # 1) 回测引擎
  engine = BacktestingEngine(); # 创建回测引擎
  engine.set_parameters(
    vt_symbol = 'IF88.CFFEX', # 设置交易品种
    interval = '1m', # 按分钟模拟
    start = datetime(2019,1,1), # 回测起始时间
    end = datetime(2019,12,31), # 数据库中最近的日期
    rate = 0.3 / 10000, # 手续费率
    slippage = 0.2, #  滑点大小
    size = 300, # 合约大小
    pricetick = 0.2, # 价格精度
    capital = 1000000, # 资本
    mode = BacktestingMode.BAR, # 采用k线图
    inverse = False
  );
  engine.add_strategy(AtrRsiStrategy, {'atrLength': 11});
  # 3) 回测
  engine.load_data();
  engine.run_backtesting();
  # 4) 结果
  engine.calculate_result();
  statistics = engine.calculate_statistics(output = True);
  for result in engine.get_all_daily_results():
    print(result.date, "净盈亏:" + str(result.net_pnl), "总净盈亏：" + str(result.holding_pnl));

if __name__ == "__main__":

  main();
