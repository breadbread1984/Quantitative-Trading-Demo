#!/usr/bin/python3

if __name__ == "__main__":

  from abupy.CoreBu import ABuEnv;
  ABuEnv.g_market_source = ABuEnv.EMarketSourceType.E_MARKET_SOURCE_sn_us;
  # 1) 按照选股条件筛选股票
  from abupy import AbuPickRegressAngMinMax;
  from abupy import AbuPickStockPriceMinMax;
  from abupy import ABuSymbol;
  from abupy import AbuBenchmark;
  from abupy import AbuCapital;
  from abupy import AbuKLManager;
  # pose gradient restriction on stocks
  # pose price restriction on stocks
  stock_pickers = [{'threshold_ang_min': 0.0, 'threshold_ang_max': 10.0, 'reversed': False, 'class': AbuPickRegressAngMinMax},
                   {'threshold_price_min': 50.0, 'threshold_price_max': 100.0, 'reserved': False, 'class': AbuPickStockPriceMinMax}];
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
  from abupy import ABuRegUtil;
  picked_stocks = ABuPickStockExecute.do_pick_stock_work(list(symbols.keys()), benchmark, capital, stock_pickers);
  print('candidates:', list(symbols.keys()));
  for stock in picked_stocks:
    kl = kl_pd_manager.get_pick_stock_kl_pd(stock);
    print('stock name: ' + stock + ' degree: {}'.format(round(ABuRegUtil.calc_regress_deg(kl.close), 3)));
  # 2) pick stocks with multiple processes
  from abupy import AbuPickStockMaster;
  picked_stocks = AbuPickStockMaster.do_pick_stock_with_process(capital, benchmark, stock_pickers, list(symbols.keys()));
  print(picked_stocks);
  # 3) pick stocks with multiple processes and threads
  picked_stocks = AbuPickStockMaster.do_pick_stock_with_process_mix_thread(capital, benchmark, stock_pickers, list(symbols.keys()), n_process = 8, n_thread = 3);
  print(picked_stocks);
