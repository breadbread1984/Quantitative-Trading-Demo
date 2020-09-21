#!/usr/bin/python3

import tensorflow as tf
from max_sharpe_strategy import PositionPredictor;

def save_model():

  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  predictor = PositionPredictor(10);
  checkpoint = tf.train.Checkpoint(model = predictor, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  predictor.save_weights('weights.h5');  

if __name__ == "__main__":

  assert tf.executing_eagerly();
  save_model();
