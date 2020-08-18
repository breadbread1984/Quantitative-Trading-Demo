#!/usr/bin/python3

from datetime import datetime;
from vnpy.trader.constant import Interval, Exchange;
from vnpy.trader.database import database_manager;
from vnpy.app.portfolio_strategy import BacktestingEngine;
from turtle_strategy import TurtleStrategy;

vt_symbols = ['IF99', 'I99', 'CU99', 'TA99'];

def main():

  engine = BacktestingEngine();
  engine.set_parameters(
    vt_symbols = vt_symbols,
    interval = Interval.DAILY,
    start = datetime(2014,1,1),
    end = datetime(2020,8,8),
    rates = {
      'IF99': 3/100000,
      'I99': 6/100000,
      'CU99': 5/100000,
      'TA99': 0,
    },
    slippages = {
      'IF99': 0.2,
      'I99': 0.5,
      'CU99': 10,
      'TA99': 2,
    },
    sizes = {
      'IF99': 300,
      'I99': 100,
      'CU99': 5,
      'TA99': 5,
    },
    priceticks = {
      'IF99': 0.2,
      'I99': 0.5,
      'CU99': 10,
      'TA99': 2,
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
