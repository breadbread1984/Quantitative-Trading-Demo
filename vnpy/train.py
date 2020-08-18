#!/usr/bin/python3

from datetime import time;
import tensorflow as tf;
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork;
from tf_agents.networks.value_rnn_network import ValueRnnNetwork;
from tf_agents.agents.ppo import ppo_agent;
from tf_agents.trajectories.time_step import time_step_spec;
from tf_agents.specs import ArraySpec;
from tf_agents.policies import policy_saver;
from vnpy.app.cta_strategy import CtaTemplate, BarGenerator, ArrayManager;

class AgentStrategy(CtaTemplate):

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 1e-3);
  obs_spec = array_spec.ArraySpec((7 * 3,), dtype = tf.float32);
  reward_spec = array_spec.ArraySpec((), dtype = tf.float32);
  action_spec = array_spec.BoundedArraySpec((1), dtype = tf.float32, minimum = [0,], maximum = [6,]);
  actor_net = ActorDistributionRnnNetwork(obs_spec, action_spec, lstm_size = (100,100));
  value_net = ValueRnnNetwork(obs_spec);
  agent = ppo_agent.PPOAgent(
    # status = (volume, open interest, open price, close price, high price, low price, end of day) x (1min, 5min, 15min)
    time_spec(observation_spec = obs_spec, reward_spec = reward_spec),
    # action = {long: 0, short: 1, sell: 2, cover: 3, sell and short: 4, cover and long: 5, none: 6}
    action_spec,
    optimizer= optimizer,
    actor_net = actor_net,
    normalize_observation = True,
    normalize_rewards = False,
    use_gea = True,
    num_epochs = 1
  );
  agent.initialize();
  
  def __init__(self, cta_engine, strategy_name, vt_symbol, setting):

    super(AgentStrategy, self).__init__(cta_engine, strategy_name, vt_symbol, setting);
    self.5min_bg = BarGenerator(self.on_bar, 5, self.on_5min_bar);
    self.15min_bg = BarGenerator(self.on_bar, 15, self._on_15min_bar);
    self.1min_am = ArrayManager();
    self.5min_am = ArrayManager();
    self.15min_am = ArrayManager();
