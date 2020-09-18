#!/usr/bin/python3

from os.path import join;
from datetime import datetime;
import numpy as np;
import tensorflow as tf;
from vnpy.trader.constant import Interval, Exchange, Direction, Offset;
from vnpy.trader.database import database_manager;
from max_sharpe_strategy import PositionPredictor;

def train():

  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  predictor = PositionPredictor(10);
  checkpoint = tf.train.Checkpoint(model = predictor, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  bars = database_manager.load_bar_data('IF88', Exchange.CFFEX, Interval.MINUTE, datetime(2009,1,1), datetime(2020,8,8));
  mu = 1000000 / 300; # how many possible positions
  delta = 0.5 / 10000; # rate
  F_tm1 = tf.zeros((1,1), dtype = tf.float32);
  history_length = len(bars);
  r = [bars[i].close_price - bars[i-1].close_price for i in range(1, len(bars))]; # r.shape = (history_length - 1,)
  R = list();
  for i in range(100):
    with tf.GradientTape() as tape:
      for i in range(len(r) - 11 + 1):
        xt = tf.concat([tf.ones((1,1), dtype = tf.float32), tf.expand_dims(r[i:i+11], axis = 0), F_tm1], axis = -1); # xt.shape = (1, 13)
        F_t = predictor(xt); # F_t.shape = (1, 1)
        R.append(mu * (F_tm1 * r[i] - delta * tf.math.abs(F_t - F_tm1)));
        F_tm1 = F_t;
      sharpe_rate = tf.math.reduce_mean(R) / tf.math.sqrt(tf.math.reduce_mean(tf.math.square(R)) - tf.math.square(tf.math.reduce_mean(R)));
      loss = -sharpe_rate;
    grads = tape.gradient(loss, predictor.trainable_variables);
    optimizer.apply_gradients(zip(grads, predictor.trainable_variables));
    print('#%d sharpe rate: %f' % (optimizer.iterations, sharpe_rate));
    checkpoint.save(join('checkpoints', 'ckpt'));
  predictor.save_weights('weights.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  train();
