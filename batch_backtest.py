#!/usr/bin/python3

if __name__ == "__main__":

  # 1) 批量选股并回测
  from abupy import AbuFactorBuyBreak; # 买入策略
  from abupy import AbuFactorSellBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop; # 卖出策略
  from abupy import AbuSlippageBuyMean; # 买入价格策略
  from abupy import AbuKellyPosition; # 买入仓位策略
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
  from abupy import AbuPickRegressAngMinMax, AbuPickStockPriceMinMax; # 选股因子
  # pose gradient restriction on stocks
  # pose price restriction on stocks
  stock_pickers = [{'threshold_ang_min': 0.0, 'threshold_ang_max': 90.0, 'reversed': False, 'class': AbuPickRegressAngMinMax},
                   {'threshold_price_min': 5.0, 'threshold_price_max': 2000.0, 'reserved': False, 'class': AbuPickStockPriceMinMax}];
  # candidates
  from abupy import abu;
  read_cash = 1000000;
  choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usTSLA', 'usWUBA', 'usVIPS'];
  abu_result_tuple, kl_pd_manager = abu.run_loop_back(read_cash, buy_factors, sell_factors, stock_pickers, choice_symbols, n_folds = 2);
  # 2) 评价回测结果
  from abupy import AbuMetricsBase;
  metrics = AbuMetricsBase(*abu_result_tuple);
  metrics.fit_metrics();
  metrics.plot_returns_cmp();
  # 3) add position control
  win_rate = metrics.win_rate;
  gains_mean = metrics.gains_mean;
  losses_mean = -metrics.losses_mean;
  # n天向上突破买入策略(加入买入价格策略,加入仓位控制策略)
  buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak, 'slippage': AbuSlippageBuyMean, 'win_rate': win_rate, 'gains_mean': gains_mean, 'losses_mean': losses_mean, 'position': AbuKellyPosition},
                 {'xd': 42, 'class': AbuFactorBuyBreak, 'slippage': AbuSlippageBuyMean, 'win_rate': win_rate, 'gains_mean': gains_mean, 'losses_mean': losses_mean, 'position': AbuKellyPosition}];  
  abu_result_tuple, kl_pd_manager = abu.run_loop_back(read_cash, buy_factors, sell_factors, stock_pickers, choice_symbols, n_folds = 2);
  metrics = AbuMetricsBase(*abu_result_tuple);
  metrics.fit_metrics();
  metrics.plot_returns_cmp();
