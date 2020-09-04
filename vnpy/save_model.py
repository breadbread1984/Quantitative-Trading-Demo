#!/usr/bin/python3

import tensorflow as tf;
from tf_agents.specs.tensor_spec import TensorSpec, BoundedTensorSpec;
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork;
from tf_agents.networks.value_rnn_network import ValueRnnNetwork;
from tf_agents.trajectories.time_step import TimeStep, StepType, time_step_spec;
from tf_agents.policies import policy_saver;

def save_model():

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 1e-3);
  obs_spec = TensorSpec((7,), dtype = tf.float32, name = 'observation');
  action_spec = BoundedTensorSpec((1,), dtype = tf.int32, minimum = 0, maximum = 3, name = 'action');
  actor_net = ActorDistributionRnnNetwork(obs_spec, action_spec, lstm_size = (100,100));
  value_net = ValueRnnNetwork(obs_spec);
  agent = ppo_agent.PPOAgent(
    time_step_spec = time_step_spec(obs_spec),
    action_spec = action_spec,
    optimizer = optimizer,
    actor_net = actor_net,
    value_net = value_net,
    normalize_observation = True,
    normalize_rewards = False,
    use_gae = True,
    num_epochs = 1,
  );
  checkpointer = Checkpointer(
    ckpt_dir = 'checkpointer/policy',
    max_to_keep = 1,
    agent = agent,
    policy = agent.policy,
    global_step = tf.compat.v1.train.get_or_create_global_step());
  checkpointer.initialize_or_restore();
  saver = policy_saver.PolicySaver(agent.policy);
  saver.save('final_policy');

if __name__ == "__main__":

  save_model();
