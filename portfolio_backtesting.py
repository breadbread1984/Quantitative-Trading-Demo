#!/usr/bin/python3

from datetime import datetime;
from vnpy.trader.constant import Interval, Exchange;
from vnpy.trader.database import database_manager;
from vnpy.app.portfolio_strategy import BacktestingEngine;
from turtle_strategy import TurtleStrategy;

vt_symbols = ['IF99.CFFEX', 'I99.DCE', 'CU99.SHFE', 'TA99.CZCE'];

def main():

  engine = BacktestingEngine();
  engine.set_parameters(
    vt_symbols = vt_symbols,
    interval = Interval.MINUTE,
    start = datetime(2014,1,1),
    end = datetime(2020,8,8),
    rates = {
      'IF99.CFFEX': 3/100000,
      'I99.DCE': 6/100000,
      'CU99.SHFE': 5/100000,
      'TA99.CZCE': 0,
    },
    slippages = {
      'IF99.CFFEX': 0.2,
      'I99.DCE': 0.5,
      'CU99.SHFE': 10,
      'TA99.CZCE': 2,
    },
    sizes = {
      'IF99.CFFEX': 300,
      'I99.DCE': 100,
      'CU99.SHFE': 5,
      'TA99.CZCE': 5,
    },
    priceticks = {
      'IF99.CFFEX': 0.2,
      'I99.DCE': 0.5,
      'CU99.SHFE': 10,
      'TA99.CZCE': 2,
    },
    capital = 1000000,
  );

  engine.add_strategy(TurtleStrategy, {'captial': 1000000});
  
  engine.load_data();
  engine.run_backtesting();
  df = engine.calculate_result();
  engine.calculate_statistics();
  engine.show_char();

if __name__ == "__main__":

  main();
