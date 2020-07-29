#!/usr/bin/python3

if __name__ == "__main__":

  import pandas as pd;
  from abupy.CoreBu import ABuEnv;
  ABuEnv.g_market_source = ABuEnv.EMarketSourceType.E_MARKET_SOURCE_sn_us;
  ABuEnv.g_enable_ml_feature = True;
  from abupy import ABuSymbolPd;
  pre_kl_pd = ABuSymbolPd.make_kl_df('usTSLA', start = '2018-01-01', end = '2018-12-31');
  kl_pd = ABuSymbolPd.make_kl_df('usTSLA', start = '2019-01-01', end = '2019-12-31');
  print(len(pre_kl_pd));
  print(len(kl_pd));
  print(kl_pd.key)
  from abupy.TradeBu import feature;
  extractor = feature.AbuMlFeature();
  f = {'buy_feature': None, 'sell_feature': None};
  index = list();
  columns = None;
  for key in kl_pd.key:
    buy_feature = extractor.make_feature_dict(kl_pd, pre_kl_pd, key, True);
    sell_feature = extractor.make_feature_dict(kl_pd, pre_kl_pd, key, False);
    if f['buy_feature'] is None: f['buy_feature'] = {key: list() for key, value in buy_feature.items()};
    if f['sell_feature'] is None: f['sell_feature'] = {key: list() for key, value in sell_feature.items()};
    [f['buy_feature'][key].append(value) for key, value in buy_feature.items()];
    [f['sell_feature'][key].append(value) for key, value in sell_feature.items()];
    index.append(key);
    if columns is None: columns = list(buy_feature.keys());
  f['buy_feature'] = pd.DataFrame(data = f['buy_feature'], index = index, columns = columns);
  f['sell_feature'] = pd.DataFrame(data = f['sell_feature'], index = index, columns = columns);
  print(f['buy_feature']);
  
