#!/usr/bin/python3

if __name__ == "__main__":

  import numpy as np;
  from abupy.CoreBu import ABuEnv;
  ABuEnv.g_market_source = ABuEnv.EMarketSourceType.E_MARKET_SOURCE_sn_us;
  # 1) generate parameter combinations
  from abupy import AbuFactorSellBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop; # 卖出策略
  sell_break_factor_grid = {
    'class': [AbuFactorSellBreak,],
    'xd': np.arange(100, 150, 1), # xd_range = [100, 150)
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
  # 2)
