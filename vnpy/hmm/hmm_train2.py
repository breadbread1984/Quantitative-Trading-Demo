#!/usr/bin/python3

from math import log;
from datetime import datetime;
import pickle;
from hmmlearn.hmm import GaussianHMM;
from vnpy.trader.database import database_manager;
from vnpy.trader.constant import Interval, Exchange;
from vnpy.trader.object import HistoryRequest;
from tsdata import tsdata_client;
import numpy as np;
import matplotlib.pyplot as plt;

def main(symbol, exchange, start, end):

  # 1) prepare observation X
  data = database_manager.load_bar_data(symbol, exchange, Interval.MINUTE, start, end);
  if len(data) == 0:
    # download data if not presented
    if not tsdata_client.inited:
      print('登录tushare');
      succeed = tsdata_client.init();
      if False == succeed:
        print('tushare登录失败');
        return;
    req = HistoryRequest(symbol = symbol, exchange = exchange, interval = Interval.MINUTE, start = start, end = end);
    data = tsdata_client.query_history(req);
    database_manager.save_bar_data(data);
    data = database_manager.load_bar_data(symbol, exchange, Interval.MINUTE, start, end);
  X = [[log(data[i].close_price) - log(data[i-1].close_price),
        log(data[i].close_price) - log(data[i-5].close_price),
        log(data[i].high_price) - log(data[i].low_price)] for i in range(5, len(data))]; # X.shape = (len(data) - 5, 3)
  # 2) learn the HMM model
  hmm = GaussianHMM(n_components = 6, covariance_type = 'diag', n_iter = 5000).fit(X);
  with open('hmm.pkl', 'wb') as f:
    pickle.dump(hmm, f);
  # 3) visualize
  latent_states_sequence = hmm.predict(X);
  plt.figure(figsize = (15,8));
  dates = [data[i].datetime.strftime('%Y-%m-%d') for i in range(5, len(data))];
  close_prices = [data[i].close_price for i in range(5, len(data))];
  print(len(close_prices));
  for i in range(hmm.n_components):
    idx = (latent_states_sequence == i); # index of day labeled with i
    plt.plot(np.array(dates)[idx], np.array(close_prices)[idx], '.', label = 'latent idx %d' % i, lw = 1);
    plt.legend();
    plt.grid(1);
  plt.axis([0, len(close_prices), min(close_prices), max(close_prices)]);
  plt.savefig('colored_k_bar.png');
  plt.show();
  for i in range(hmm.n_components):
    idx = (latent_states_sequence == i); # index of day labeled with i
    idx = np.append(False, idx[:-1]); # index of the next day of the day labeled with i, because if you trade on day with label i the reward comes one day after
    plt.plot(np.exp(np.array(X)[idx, 0].cumsum()), label = 'latent_state %d' % i);
    plt.legend();
    plt.grid(1);
  plt.savefig('return_curve.png');
  plt.show();

if __name__ == "__main__":

  main('IF88', Exchange.CFFEX, datetime(2009,1,1), datetime(2020,8,15));
