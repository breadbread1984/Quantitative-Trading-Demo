#!/usr/bin/python3

if __name__ == "__main__":

  import numpy as np;
  from abupy.CoreBu import ABuEnv;
  ABuEnv.g_market_source = ABuEnv.EMarketSourceType.E_MARKET_SOURCE_sn_us;
  # 1) generate parameter combinations for sell strategies
  from abupy import AbuFactorSellBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop; # 卖出策略
  sell_break_factor_grid = {
    'class': [AbuFactorSellBreak,],
    'xd': [120]#np.arange(20, 150, 10), # xd_range = [20, 150)
  };
  sell_atr_nstop_factor_grid = {
    'class': [AbuFactorAtrNStop,],
    'stop_loss_n': np.arange(0.5, 2, 0.5), # stop_loss_range = [0.5,2)
    'stop_win_n': np.arange(2.0, 4.5, 0.5), # stop_win_range = [2.0, 4.5)
  };
  sell_atr_pre_factor_grid = {
    'class': [AbuFactorPreAtrNStop,],
    'pre_atr_n': np.arange(1.0, 3.5, 0.5), # pre_atr_range = [1.0, 3.5)
  };
  sell_atr_close_factor_grid = {
    'class': [AbuFactorCloseAtrNStop],
    'close_atr_n': np.arange(1.0, 4.0, 0.5), # close_atr_range = [1.0, 4.0)
  };
  from abupy import ABuGridHelper;
  sell_factors_product = ABuGridHelper.gen_factor_grid(
    ABuGridHelper.K_GEN_FACTOR_PARAMS_SELL,
    [sell_break_factor_grid, sell_atr_nstop_factor_grid, sell_atr_pre_factor_grid, sell_atr_close_factor_grid]
  );
  print('combination number is {}'.format(len(sell_factors_product)));
  # 2) generate parameters combinations for buy strategies
  from abupy import AbuFactorBuyBreak; # 买入策略
  buy_bk_factor_grid = {
    'class': [AbuFactorBuyBreak,],
    'xd': [42, 60],
  }
  buy_factors_product = ABuGridHelper.gen_factor_grid(
    ABuGridHelper.K_GEN_FACTOR_PARAMS_BUY,
    [buy_bk_factor_grid, buy_bk_factor_grid]
  );
  print('combination number is {}'.format(len(buy_factors_product)));
  # 3) grid search
  from abupy import GridSearch;
  read_cash = 1000000;
  choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usTSLA', 'usWUBA', 'usVIPS'];
  grid_search = GridSearch(read_cash, choice_symbols, buy_factors_product = buy_factors_product, sell_factors_product = sell_factors_product);
  scores, score_tuple_array = grid_search.fit(n_jobs = -1);
  # 只考虑回报评分
  from abupy import WrsmScorer;
  scorer = WrsmScorer(score_tuple_array, weights = [0, 1, 0, 0]); # win_rate, returns, sharp, max_drawdown
  sfs = scorer.fit_score(); # 按照分数升序排序的结果
  print(sfs[::-1][:15]); # 打印分数top 15的参数
  # 四个分数等权重(并不是最优解)
  best_score_tuple_grid = grid_search.best_score_tuple_grid;
  from abupy import AbuMetricsBase;
  AbuMetricsBase.show_general(best_score_tuple_grid.orders_pd, best_score_tuple_grid.action_pd, best_score_tuple_grid.capital, best_score_tuple_grid.benchmark);
