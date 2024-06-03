# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import collections
import numpy as np
import os
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import pickle
# from environments.gridworld.navigation import 
import sys, os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
import environments.env_policies as env_policies
import data.tf_agents_onpolicy_dataset as tf_agents_onpolicy_dataset
import estimators.estimator as estimator_lib
import utils.common as common_utils
from data.dataset import Dataset, EnvStep, StepType
from data.tf_offpolicy_dataset import TFOffpolicyDataset


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'taxi', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 100,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 500,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 1.0,
                   'How close to target policy.')
flags.DEFINE_bool('tabular_obs', True,
                  'Whether to use tabular observations.')
flags.DEFINE_string('save_dir', None, 'Directory to save dataset to.')
flags.DEFINE_string('load_dir', None, 'Directory to load policies from.')
flags.DEFINE_bool('force', False,
                  'Whether to force overwriting any existing dataset.')
# flags.DEFINE_string('distribution', None, 'Probability distribution.')

# def get_onpolicy_dataset(load_dir, env_name, tabular_obs, max_trajectory_length,
#                          alpha, seed, distribution):
def get_onpolicy_dataset(load_dir, env_name, tabular_obs, max_trajectory_length,
                         alpha, seed):
  """Get on-policy dataset."""
  # tf_env, tf_policy = env_policies.get_env_and_policy(
  #     load_dir, env_name, alpha, distribution, env_seed=seed, tabular_obs=tabular_obs)
  tf_env, tf_policy = env_policies.get_env_and_policy(
    load_dir, env_name, alpha, env_seed=seed, tabular_obs=tabular_obs)
  dataset = tf_agents_onpolicy_dataset.TFAgentsOnpolicyDataset(
      tf_env, tf_policy,
      episode_step_limit=max_trajectory_length)
  return dataset


def add_episodes_to_dataset(episodes, valid_ids, write_dataset):
  num_episodes = 1 if tf.rank(valid_ids) == 1 else tf.shape(valid_ids)[0]
  for ep_id in range(num_episodes):
    if tf.rank(valid_ids) == 1:
      this_valid_ids = valid_ids
      this_episode = episodes
    else:
      this_valid_ids = valid_ids[ep_id, ...]
      this_episode = tf.nest.map_structure(
          lambda t: t[ep_id, ...], episodes)
    
    episode_length = tf.shape(this_valid_ids)[0]
    for step_id in range(episode_length):
      this_valid_id = this_valid_ids[step_id]
      this_step = tf.nest.map_structure(
          lambda t: t[step_id, ...], this_episode)
      if this_valid_id:
        write_dataset.add_step(this_step)
        # print("EPISODE: ", this_episode)
        # print("STEP: ", this_step)

def normalize_within_states(d):
    state_probs = {}
    state_totals = 0
    
    for state, count in d.items():
      state_totals += count
    
    for state, count in d.items():
      state_probs[state] = count / state_totals
    
    return state_probs

# def calculate_mismatch(visitation_distribution_alpha0, visitation_distribution_alpha1):
    # print("KEYS: ", sorted(set(visitation_distribution_alpha0.keys()) | set(visitation_distribution_alpha1.keys())))
def kl_divergence(p, q):
  for key in p.keys():
      if key not in q:
          q[key] = 1e-10  # Add a small value to handle missing entries in q

  kl_div = 0.0
  for k in p:
      if p[k] > 0 and q[k] > 0:
          kl_div += p[k] * np.log(p[k] / q[k])
  return kl_div

def load_visitation_distribution(alpha):
    # Load visitation distribution from file based on alpha value
    file_path = f"visitation_distribution_alpha{alpha}.npy"
    visitation_distribution = np.load(file_path, allow_pickle=True).item()
    return visitation_distribution

def load_visitation_distribution_on(alpha):
    # Load visitation distribution from file based on alpha value
    file_path = f"visitation_distribution_alpha_on{alpha}.npy"
    visitation_distribution = np.load(file_path, allow_pickle=True).item()
    return visitation_distribution

def main(argv):
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  save_dir = FLAGS.save_dir
  load_dir = FLAGS.load_dir
  force = FLAGS.force
  # distribution = FLAGS.distribution

  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  directory = os.path.join(save_dir, hparam_str)
  if tf.io.gfile.isdir(directory) and not force:
    raise ValueError('Directory %s already exists. Use --force to overwrite.' %
                     directory)
  visitation_distribution_on = {}
  np.random.seed(seed)
  tf.random.set_seed(seed)

  # dataset = get_onpolicy_dataset(load_dir, env_name, tabular_obs,
  #                                max_trajectory_length, alpha, seed, distribution)
  dataset = get_onpolicy_dataset(load_dir, env_name, tabular_obs,
                                 max_trajectory_length, alpha, seed)
  write_dataset = TFOffpolicyDataset(
      dataset.spec,
      capacity=num_trajectory * (max_trajectory_length + 1))

  batch_size = 20
  # tr = True
  for batch_num in range(1 + (num_trajectory - 1) // batch_size):
    num_trajectory_after_batch = min(num_trajectory, batch_size * (batch_num + 1))
    num_trajectory_to_get = num_trajectory_after_batch - batch_num * batch_size
    episodes, valid_steps = dataset.get_episode(
        batch_size=num_trajectory_to_get)
    batch_states = episodes.observation.numpy()
    batch_actions = episodes.action.numpy()
    # print(batch_states)
    # Mismatch between alpha=0 and alpha=1 visitation distributions: 0.736303753151139
# Mismatch between alpha=0 and alpha=1 visitation distributions on: 0.7330388872441861
    for episode in range(len(batch_actions)):
      for step in range(len(batch_actions[episode])):
        # print(batch_states[episode][step])
        # state = batch_states[episode][step]
        state = tuple(batch_states[episode][step])
        visitation_distribution_on[state] = visitation_distribution_on.get(state, 0) + 1
    # if tr:
    #   for episode in episodes:
    #     for step in episode:
    #       print("STEP: ", step)
    #   tr = False
    # print("EPISODE: ", episodes.observation)
    # state_action_pair = (episodes.observation.ref(), episodes.action.ref())
    # visitation_distribution[state_action_pair] = visitation_distribution.get(state_action_pair, 0) + 1
    # print(visitation_distribution.items())
    # print("ACTION: ", episodes.action)
    # print("VALID STEPS: ", valid_steps)
    add_episodes_to_dataset(episodes, valid_steps, write_dataset)
    # for step in episodes:
    #         state_action_pair = (episodes.observation, step.action)
    #         visitation_distribution[state_action_pair] = visitation_distribution.get(state_action_pair, 0) + 1
    print('num episodes collected: %d', write_dataset.num_total_episodes)
    print('num steps collected: %d', write_dataset.num_steps)

    estimate = estimator_lib.get_fullbatch_average(write_dataset)
    print('per step avg on offpolicy data', estimate)
    estimate = estimator_lib.get_fullbatch_average(write_dataset,
                                                   by_steps=False)
    print('per episode avg on offpolicy data', estimate)

  print('Saving dataset to %s.' % directory)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)
  write_dataset.save(directory)

  print('Loading dataset.')
  new_dataset = Dataset.load(directory)
  # with open('close_d.pkl', 'wb') as f:
  #   pickle.dump(new_dataset, f)
  all_steps = new_dataset.get_all_steps().observation.numpy()
  visitation_distribution = {}
  # print("STEPS: ", all_steps.observation.numpy())
  # Iterate over all steps and count state visitations
  for step in all_steps:  
      # state = step  
      state = tuple(step)
      visitation_distribution[state] = visitation_distribution.get(state, 0) + 1
  # print(state_visitation_counts)
  print('num loaded steps', new_dataset.num_steps)
  print('num loaded total steps', new_dataset.num_total_steps)
  print('num loaded episodes', new_dataset.num_episodes)
  print('num loaded total episodes', new_dataset.num_total_episodes)
  
  estimate = estimator_lib.get_fullbatch_average(new_dataset)
  print('per step avg on saved and loaded offpolicy data', estimate)
  estimate = estimator_lib.get_fullbatch_average(new_dataset,
                                                 by_steps=False)
  print('per episode avg on saved and loaded offpolicy data', estimate)
  np.save(f"visitation_distribution_alpha{alpha}.npy", visitation_distribution)
  np.save(f"visitation_distribution_alpha_on{alpha}.npy", visitation_distribution_on)

  if alpha == 1.0:
      behavior_policy_count = load_visitation_distribution(0.0)
      target_policy_count = load_visitation_distribution(1.0)
      behavior_policy_count_on = load_visitation_distribution_on(0.0)
      target_policy_count_on = load_visitation_distribution_on(1.0)
      # behavior_policy_distribution = dict(sorted(behavior_policy_distribution.items()))
      # target_policy_distribution = dict(sorted(target_policy_distribution.items()))
      # print("BEHAVIOR POLICY: ", behavior_policy_count)
      # print("TARGET POLICY: ", target_policy_count)
      behavior_policy_distribution = normalize_within_states(behavior_policy_count)
      target_policy_distribution = normalize_within_states(target_policy_count)
      behavior_policy_distribution_on = normalize_within_states(behavior_policy_count_on)
      target_policy_distribution_on = normalize_within_states(target_policy_count_on)
      sorted_behavior = dict(sorted(behavior_policy_distribution.items()))
      sorted_target = dict(sorted(target_policy_distribution.items()))
      sorted_behavior_on = dict(sorted(behavior_policy_distribution_on.items()))
      sorted_target_on = dict(sorted(target_policy_distribution_on.items()))
      # print("BEHAVIOR DISTRIBUTION: ", behavior_policy_distribution)
      # print("TARGET DISTRIBUTION: ", target_policy_distribution)        
      mismatch = kl_divergence(behavior_policy_distribution, target_policy_distribution)
      mismatch_on = kl_divergence(behavior_policy_distribution_on, target_policy_distribution_on)

      print("Mismatch between alpha=0 and alpha=1 visitation distributions:", mismatch)
      print("Mismatch between alpha=0 and alpha=1 visitation distributions on:", mismatch_on)

      # with open('Distribution_Close.txt', 'w') as file:
      #   for key, value in sorted_behavior.items():
      #       file.write(f"{key}: {value}\n")
      #   file.write("-----------------------------------------------------------------------------------------\n\n")
      #   for key, value in sorted_target.items():
      #       file.write(f"{key}: {value}\n")
      #   file.write("-----------------------------------------------------------------------------------------\n\n")
      #   file.write(f"Mismatch: {mismatch}\n")
      #   file.write("-----------------------------------------------------------------------------------------\n\n")
      #   for key, value in sorted_behavior_on.items():
      #       file.write(f"{key}: {value}\n")
      #   file.write("-----------------------------------------------------------------------------------------\n\n")
      #   for key, value in sorted_target_on.items():
      #       file.write(f"{key}: {value}\n")
      #   file.write("-----------------------------------------------------------------------------------------\n\n")
      #   file.write(f"Mismatch: {mismatch}\n")
      with open('distibution_test_d.pkl', 'wb') as f:
        pickle.dump({'mismatch': mismatch}, f)
  print('Done!')
  # print(write_dataset.get_all_episodes(None, 400))
  # print(write_dataset.get_step(10, 10).action)


if __name__ == '__main__':
  app.run(main)
