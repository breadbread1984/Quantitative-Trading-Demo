#!/usr/bin/python3

if __name__ == "__main__":

  import numpy as np;
  import pandas as pd;
  from abupy.CoreBu import ABuEnv;
  ABuEnv.g_market_source = ABuEnv.EMarketSourceType.E_MARKET_SOURCE_sn_us;
  from abupy import AbuFactorBuyBreak; # 买入策略
  from abupy import AbuFactorSellBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop; # 卖出策略
  from abupy import AbuSlippageBuyMean; # 买入价格策略
  from abupy import AbuKellyPosition; # 买入仓位策略
  from abupy import AbuBenchmark;
  from abupy import AbuCapital;
  # n天向上突破买入策略(加入买入价格策略)
  buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak, 'slippage': AbuSlippageBuyMean},
                 {'xd': 42, 'class': AbuFactorBuyBreak, 'slippage': AbuSlippageBuyMean}];
  # n天向下突破卖出策略
  # 止盈止损卖出策略
  # 剧烈下跌平仓策略
  # 移动止盈策略
  sell_factors = [{'xd': 120, 'class': AbuFactorSellBreak},
                  {'stop_loss_n': 0.5, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop},
                  {'pre_atr_n': 1.0, 'class': AbuFactorPreAtrNStop},
                  {'close_atr_n': 1.5, 'class': AbuFactorCloseAtrNStop}];
  benchmark = AbuBenchmark();
  capital = AbuCapital(1000000, benchmark); # assign money
  '''
  # 1) tedious way
  from abupy import AbuPickTimeWorker;
  from abupy import AbuKLManager;
  kl_pd_manager = AbuKLManager(benchmark, capital);
  kl_pd = kl_pd_manager.get_pick_time_kl_pd('usTSLA'); # load k line of Tesla
  # simulate with buy factor and sell factor
  abu_worker = AbuPickTimeWorker(capital, kl_pd, benchmark, buy_factors = buy_factors, sell_factors = sell_factors);
  abu_worker.fit();
  # visualize orders
  from abupy import ABuTradeProxy;
  orders_pd, action_pd, _ = ABuTradeProxy.trade_summary(abu_worker.orders, kl_pd, draw = True, show_info = False);
  # visualize capitalism
  from abupy import ABuTradeExecute;
  ABuTradeExecute.apply_action_to_capital(capital, action_pd, kl_pd_manager);
  capital.capital_pd.capital_blance.plot();
  '''
  # 2) simple way
  from abupy import ABuPickTimeExecute;
  orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA', 'usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usWUBA', 'usVIPS'], benchmark, buy_factors, sell_factors, capital, show = False);
  print(orders_pd[:10]);
  # 3) metric the profit
  from abupy import AbuMetricsBase;
  metrics = AbuMetricsBase(orders_pd, action_pd, capital, benchmark);
  metrics.fit_metrics();
  metrics.plot_returns_cmp(only_show_returns = True);
  # 4) evaluate win_rate(为了计算胜率)
  win_rate = metrics.win_rate;
  gains_mean = metrics.gains_mean;
  losses_mean = -metrics.losses_mean;
  # 5) add position control
  # n天向上突破买入策略(加入买入价格策略,加入仓位控制策略)
  buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak, 'slippage': AbuSlippageBuyMean, 'win_rate': win_rate, 'gains_mean': gains_mean, 'losses_mean': losses_mean, 'position': AbuKellyPosition},
                 {'xd': 42, 'class': AbuFactorBuyBreak, 'slippage': AbuSlippageBuyMean, 'win_rate': win_rate, 'gains_mean': gains_mean, 'losses_mean': losses_mean, 'position': AbuKellyPosition}];
  orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA', 'usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usWUBA', 'usVIPS'], benchmark, buy_factors, sell_factors, capital, show = False);
  print(orders_pd[:10]);
  metrics = AbuMetricsBase(orders_pd, action_pd, capital, benchmark);
  metrics.fit_metrics();
  metrics.plot_returns_cmp(only_show_returns = True);
  # 6) pick time with different factors
  target_symbols = ['usSFUN', 'usNOAH'];
  buy_factors_sfun = [{'xd': 42, 'class': AbuFactorBuyBreak}];
  sell_factors_sfun = [{'xd': 60, 'class': AbuFactorSellBreak}];
  buy_factors_noah = [{'xd': 21, 'class': AbuFactorBuyBreak}];
  sell_factors_noah = [{'xd': 42, 'class': AbuFactorSellBreak}];
  factor_dict = {
    'usSFUN': {'buy_factors': buy_factors_sfun, 'sell_factors': sell_factors_sfun},
    'usNOAH': {'buy_factors': buy_factors_noah, 'sell_factors': sell_factors_noah}
  };
  orders_pd, action_pd, all_fit_symbols = ABuPickTimeExecute.do_symbols_with_diff_factors(target_symbols, benchmark, factor_dict, capital);
  # 7) cross tab
  print(pd.crosstab(orders_pd.buy_factor, orders_pd.symbol));
