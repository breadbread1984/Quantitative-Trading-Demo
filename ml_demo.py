#!/usr/bin/python3

if __name__ == "__main__":

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
  f = {'buy_feature': list(), 'sell_feature': list()};
  for key in kl_pd.key:
    f['buy_feature'].append(extractor.make_feature_dict(kl_pd, pre_kl_pd, key, True));
    f['sell_feature'].append(extractor.make_feature_dict(kl_pd, pre_kl_pd, key, False));
  print(f['buy_feature']);
  
