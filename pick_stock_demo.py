#!/usr/bin/python3

if __name__ == "__main__":

  from abupy.CoreBu import ABuEnv;
  ABuEnv.g_market_source = ABuEnv.EMarketSourceType.E_MARKET_SOURCE_sn_us;
  # 1) 按照选股条件筛选股票
  from abupy import AbuPickRegressAngMinMax;
  from abupy import ABuSymbol;
  from abupy import AbuBenchmark;
  from abupy import AbuCapital;
  from abupy import AbuKLManager;
  # gradient restriction posed on picked stocks
  stock_pickers = [{'threshold_ang_min': 0.0, 'threshold_ang_max': 10.0, 'reversed': False, 'class': AbuPickRegressAngMinMax}];
  symbols = ABuSymbol.search_to_symbol_dict('黄金');
  benchmark = AbuBenchmark();
  capital = AbuCapital(1000000, benchmark); # assign money
  kl_pd_manager = AbuKLManager(benchmark, capital);
  '''
  # tedious way
  from abupy import AbuPickStockWorker;
  stock_pick = AbuPickStockWorker(capital, benchmark, kl_pd_manager, choice_symbols = list(symbols.keys()), stock_pickers = stock_pickers);
  stock_pick.fit();
  print('candidates:', list(symbols.keys()));
  print('picked:', stock_pick.choice_symbols);
  '''
  # simple way
  from abupy import ABuPickStockExecute;
  picked_stocks = ABuPickStockExecute.do_pick_stock_work(list(symbols.keys()), benchmark, capital, stock_pickers);
  print('candidates:', list(symbols.keys()));
  print('picked:', picked_stocks);
