#!/usr/bin/python3

from math import log;
from os.path import exists;
from datetime import datetime;
from vnpy.trader.database import database_manager;
from vnpy.trader.constant import Interval, Exchange;
from vnpy.trader.object import HistoryRequest;
from tsdata import tsdata_client;
import numpy as np;
import pickle;
import tensorflow as tf;
import tensorflow_probability as tfp;
import matplotlib.pyplot as plt;

def main(symbol, exchange, start, end):

  # 1) prepare observation X
  data = database_manager.load_bar_data(symbol, exchange, Interval.DAILY, start, end);
  if len(data) == 0:
    # download data if not presented
    if not tsdata_client.inited:
      print('登录tushare');
      succeed = tsdata_client.init();
      if False == succeed:
        print('tushare登录失败');
        return;
    req = HistoryRequest(symbol = symbol, exchange = exchange, interval = Interval.DAILY, start = start, end = end);
    data = tsdata_client.query_history(req);
    database_manager.save_bar_data(data);
    data = database_manager.load_bar_data(symbol, exchange, Interval.DAILY, start, end);
  X = [[log(data[i].close_price) - log(data[i-1].close_price),
        log(data[i].close_price) - log(data[i-5].close_price),
        log(data[i].high_price) - log(data[i].low_price)] for i in range(5, len(data))]; # X.shape = (len(data) - 5, 3)
  X = tf.expand_dims(X, axis = 0); # X.shape = (1, len(data) - 5, 3)
  # 2) sample p(theta | X)
  if exists('samples.pkl'):
    with open('samples.pkl', 'rb') as f:
      states = tf.constant(pickle.loads(f.read()));
  else:
    step_size = tf.Variable(0.5, dtype = tf.float32, trainable = False);
    initial_probs = tf.constant([1./6, 1./6, 1./6, 1./6, 1./6, 1./6], dtype = tf.float32);
    transition_probs = tf.constant([[1./6, 1./6, 1./6, 1./6, 1./6, 1./6],
                                    [1./6, 1./6, 1./6, 1./6, 1./6, 1./6],
                                    [1./6, 1./6, 1./6, 1./6, 1./6, 1./6],
                                    [1./6, 1./6, 1./6, 1./6, 1./6, 1./6],
                                    [1./6, 1./6, 1./6, 1./6, 1./6, 1./6],
                                    [1./6, 1./6, 1./6, 1./6, 1./6, 1./6]], dtype = tf.float32);
    observation_mean = tf.constant(np.random.normal(size = (6,3)), dtype = tf.float32);
    observation_std = tf.constant(np.random.normal(size = (6,3)), dtype = tf.float32);
    [states], kernel_results = tfp.mcmc.sample_chain(
      num_results = 48000,
      num_burnin_steps = 25000,
      current_state = [tf.concat([initial_probs, 
                                  tf.reshape(transition_probs, (-1,)), 
                                  tf.reshape(observation_mean, (-1,)),
                                  tf.reshape(observation_std, (-1,))], axis = 0)],
      kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn = log_prob_generator(X),
          num_leapfrog_steps = 2,
          step_size = step_size,
          step_size_update_fn = tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps = 20000),
          state_gradients_are_stopped = True
        ),
        bijector = [tfp.bijectors.RealNVP(num_masked = 2, shift_and_log_scale_fn = tfp.bijectors.real_nvp_default_template(hidden_layers = [512, 512]))]
      )
    );
    print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype = tf.float32)));
    states = states[25000:]; # states.shape = (batch = 1, sample num, state_dim = 78)
    with open('samples.pkl', 'wb') as f:
      f.write(pickle.dumps(states.numpy()));
  # 3) find the mode of p(theta | X)
  mean = tf.math.reduce_mean(states, axis = [0, 1], keepdims = True); # sample_mean.shape = (1, 1, 78)
  var = tf.math.reduce_mean(tf.math.square(states - mean), axis = [0, 1], keepdims = True); # var.shape = (1, 1, state_dim = 78)
  # reduce samples to 2-dim vectors for visualization
  s, u, v = tf.linalg.svd(tf.transpose(tf.squeeze(states - mean, axis = 0))); # u.shape = (78, 78)
  u = u[:, :2]; # u.shape = (78, 2)
  low_dim = tf.linalg.matmul(u, tf.squeeze(states - mean, axis = 0), transpose_a = True, transpose_b = True) # low_dim.shape = (2, sample_num)
  plt.figure(figsize = (12.5, 4));
  plt.title('mean shift trajectory on posterior distribution in 2D');
  plt.scatter(x = low_dim[0,:].numpy(), y = low_dim[1,:].numpy(), c = 'b');
  mode = mean;
  mode_low_dim = tf.linalg.matmul(u, tf.squeeze(mode - mean, axis = 0), transpose_a = True, transpose_b = True); # mode_low_dim.shape = (2, 1)
  plt.Circle(mode_low_dim[:, 0].numpy(), 0.2, color = 'r');
  while True:
    mahalanobis_dist = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(states - mode) / var, axis = -1)); # mahalanobis_dist.shape = (1, sample_num)
    idx = tf.argsort(mahalanobis_dist, axis = -1, direction = 'DESCENDING')[0,:states.shape[1]/1000]; # idx.shape = (sample num/1000)
    idx = tf.stack([tf.zeros_like(idx), idx], axis = -1); # idx.shape = (sample num/1000, 2)
    neighbors = tf.gather_nd(states, idx); # neighbors.shape = (sample num/1000, state dim = 78)
    new_mode = tf.reshape(tf.math.reduce_mean(neighbors, axis = 0), (1, 1, -1)); # mode.shape = (1, 1, state dim = 78)
    if tf.math.reduce_sum(tf.math.square(new_mode - mode)) < 1e-3: break;
    new_mode_low_dim = tf.linalg.matmul(u, tf.squeeze(new_mode - mean, axis = 0), transpose_a = True, transpose_b = True); # new_mode_low_dim.shape = (2,1)
    plt.Circle(new_mode_low_dim[:, 0].numpy(), 0.2, color = 'r');
    plt.arrow(mode_low_dim[0,0], mode_low_dim[1,0], new_mode_low_dim[0,0] - mode_low_dim[0,0], new_mode_low_dim[1,0] - mode_low_dim[1,0], color = 'g');
    mode = new_mode;
    mode_low_dim = new_mode_low_dim;
  plt.legend();
  plt.grid(1);
  plt.show();
  with open('hmm.pkl', 'wb') as f:
    f.write(pickle.dumps(new_mode.numpy()));

def log_prob_generator(samples):
  # samples.shape = (1, num_steps, 3)
  def func(probs):
    initial_probs = tf.slice(probs, begin = [0,], size = [6,]);
    transition_probs = tf.reshape(tf.slice(probs, begin = [6,], size = [36,]), (6,6));
    observation_mean = tf.reshape(tf.slice(probs, begin = [42,], size = [18,]), (6,3));
    observation_std = tf.reshape(tf.slice(probs, begin = [60,], size = [18,]), (6,3));
    prob_dist = tfp.distributions.HiddenMarkovModel(initial_distribution = tfp.distributions.Categorical(probs = initial_probs),
                                                    transition_distribution = tfp.distributions.Categorical(probs = transition_probs),
                                                    observation_distribution = tfp.distributions.MultivariateNormalDiag(loc = observation_mean,
                                                                                                                        scale_diag = observation_std),
                                                    num_steps = samples.shape[1]);
    return tf.math.reduce_sum(prob_dist.log_prob(samples));
  return func;

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main('000301', Exchange.SZSE, datetime(2012,6,1), datetime(2016,4,7));
