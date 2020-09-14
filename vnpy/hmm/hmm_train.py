#!/usr/bin/python3

from datetime import datetime;
from vnpy.trader.database import database_manager;
from vnpy.trader.constant import Interval, Exchange;
from vnpy.trader.object import HistoryRequest;
from tsdata import tsdata_client;
import tensorflow as tf;
import tensorflow_probability as tfp;

def main(symbol, exchange, start, end):

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
  X = tf.constant([[tf.math.log(data[i].close_price) - tf.math.log(data[i-1].close_price),
                            tf.math.log(data[i].close_price) - tf.math.log(data[i-5].close_price),
                            tf.math.log(data[i].high_price) - tf.math.log(data[i].low_price)] for i in range(5, len(data))]); # X.shape = (len(data) - 5, 3)
  X = tf.expand_dims(X, axis = 0); # X.shape = (1, len(data) - 5, 3)
  # bayesian inference parameter
  [probs], kernel_results = tfp.mcmc.sample_chain(
    num_results = 48000,
    num_burnin_steps = 25000,
    current_state = [tf.constant([1./6, 1./6, 1./6, 1./6, 1./6, 1./6]),
                     tf.constant([[1./6, 1./6, 1./6, 1./6, 1./6, 1./6],
                      [1./6, 1./6, 1./6, 1./6, 1./6, 1./6],
                      [1./6, 1./6, 1./6, 1./6, 1./6, 1./6],
                      [1./6, 1./6, 1./6, 1./6, 1./6, 1./6],
                      [1./6, 1./6, 1./6, 1./6, 1./6, 1./6],
                      [1./6, 1./6, 1./6, 1./6, 1./6, 1./6]]),
                     tf.constant(np.random.normal(size = (6,3))), tf.constant(np.random.normal(size = (6,3)))],
    kernel = tfp.mcmc.TransformedTransitionKernel(
      inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn = log_prob_generator(X),
        num_leapfrog_step = 2,
        step_size = step_size,
        step_size_update_fn = tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps = 20000),
        state_gradients_are_stopped = True
      ),
      bijector = [tfp.bijectors.Identity()]
    )
  );
  print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype = tf.float32)));

def log_prob_generator(samples):
  # samples.shape = (1, num_steps, 3)
  def func(initial_probs, transition_probs, observation_mean, observation_std):
    prob_dist = tfp.distributions.HiddenMarkovModel(initial_distribution = tfp.distributions.Categorical(probs = initial_probs),
                                                    transition_distribution = tfp.distributions.Categorical(probs = transition_probs),
                                                    observation_distribution = tfp.distributions.Normal(loc = observation_mean,
                                                                                                        scale = observation_std),
                                                    num_steps = samples.shape[1]);
    return tf.math.reduce_sum(prob_dist.log_prob(samples));
  return func;

if __name__ == "__main__":

  main('000301', Exchange.SZSE, datetime(2012,6,1), datetime(2016,4,7));
