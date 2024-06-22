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

######################################################################

# The main changes in this file compared to the original codebase involve the 
# introduction of the "distribution" variable. This addition facilitates the 
# creation and selection of initial start distributions for the environment. 
# The "distribution" variable is a seed ranging from 1 to 7, which determines 
# the specific initial distribution. For instance, selecting distribution 1 results 
# in a uniform initial start distribution when generating the dataset for 
# the grid environment.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow.compat.v2 as tf

from gym import spaces
from gym.utils import seeding
import sys, os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.common as common_utils


class Navigation(gym.Env):
  def __init__(self, nav_map, distribution, tabular_obs=True,
               reward_fn=None, done_fn=None):
  # def __init__(self, nav_map, tabular_obs=True,
  #              reward_fn=None, done_fn=None):
    self._map = nav_map
    self._tabular_obs = tabular_obs
    self._reward_fn = reward_fn
    self._done_fn = done_fn
    # self.fixed_random_state = 0
    # self.acts = []
    if self._reward_fn is None:
      self._reward_fn = lambda x, y, tx, ty: float(x == tx and y == ty)
    if self._done_fn is None:
      self._done_fn = lambda x, y, tx, ty: False
    self.distribution = distribution
    if self.distribution is not None:
      self.fixed_random_state = np.random.RandomState(distribution)

    self._max_x = len(self._map)
    if not self._max_x:
      raise ValueError('Invalid map.')
    self._max_y = len(self._map[0])
    if not all(len(m) == self._max_y for m in self._map):
      raise ValueError('Invalid map.')

    self._start_x, self._start_y = self._find_initial_point()
    self._target_x, self._target_y = self._find_target_point()
    self._n_state = self._max_x * self._max_y
    self._n_action = 4

    if self._tabular_obs:
      self.observation_space = spaces.Discrete(self._n_state)
    else:
      self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(2,))

    self.action_space = spaces.Discrete(self._n_action)
    self.seed()
    self.reset()

  @property
  def nav_map(self):
    return self._map

  @property
  def n_state(self):
    return self._n_state

  @property
  def n_action(self):
    return self._n_action

  @property
  def target_location(self):
    return self._target_x, self._target_y

  @property
  def tabular_obs(self):
    return self._tabular_obs

  def _find_initial_point(self):
    for x in range(self._max_x):
      for y in range(self._max_y):
        if self._map[x][y] == 'S':
          break
      if self._map[x][y] == 'S':
        break
    else:
      return None, None
    return x, y

  def _find_target_point(self):
    for x in range(self._max_x):
      for y in range(self._max_y):
        if self._map[x][y] == 'T':
          break
      if self._map[x][y] == 'T':
        break
    else:
      raise ValueError('Target point not found in map.')

    return x, y

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    if self._start_x is not None and self._start_y is not None:
      self._x, self._y = self._start_x, self._start_y
    else:
      while True:  # Find empty grid cell.
        ### UNIFORM ###
        if self.distribution == 0:
          self._x = self.fixed_random_state.randint(self._max_x)
          self._y = self.fixed_random_state.randint(self._max_y)
          
        ### DISTANCE BASED DISTRIBUTION ###
        if self.distribution == 1:
          grid_size = 10
          start_distribution = self.distance_distribution(grid_size)
          sampled_position = self.sample_start_position(start_distribution)
          self._x, self._y = sampled_position

        ### MIXED MODE DISTRIBUTION ###
        if self.distribution == 2:
          grid_size = 10
          start_distribution = self.generate_mixture_distribution(grid_size)
          sampled_position = self.sample_start_position(start_distribution)
          self._x, self._y = sampled_position


        ### TARGET-CENTRIC ###
        if self.distribution == 3:
          self._x = self.fixed_random_state.randint(max(0, self._target_x - 1), min(self._max_x, self._target_x + 1))
          self._y = self.fixed_random_state.randint(max(0, self._target_y - 1), min(self._max_y, self._target_y + 1))

        ### FIXED POINT ###
        if self.distribution == 4:
          self._y = 8
          self._x = 9

        ### REMOTE ###
        if self.distribution == 5:
          self._x = self.fixed_random_state.randint(0, 2)
          self._y = self.fixed_random_state.randint(0, 2)
        
        ### EDGE BIASED ###
        if self.distribution == 6:
          grid_size = 10
          edge_biased_distribution = self.generate_edge_biased_distribution(grid_size)
          sampled_position = self.sample_start_position(edge_biased_distribution)
          self._x, self._y = sampled_position

        ### TARGET POLICY START DISTRIBUTION ###
        if self.distribution == 7:
          self._x = self.fixed_random_state.randint(self._max_x)
          self._y = self.fixed_random_state.randint(self._max_y)

        if self._map[self._x][self._y] != 'x':
          break
    return self._get_obs()

  def _get_obs(self):
    if self._tabular_obs:
      return self._x * self._max_y + self._y
    else:
      return np.array([self._x, self._y])

  def get_tabular_obs(self, status_obs):
    return self._max_y * status_obs[..., 0] + status_obs[..., 1]

  def sample_start_position(self, start_distribution):
    flat_distribution = start_distribution.flatten()
    index = self.fixed_random_state.choice(len(flat_distribution), p=flat_distribution)
    # Convert flattened index to 2D coordinates
    row = index // start_distribution.shape[1]
    col = index % start_distribution.shape[1]
    return row, col
  
  def get_xy(self, state):
    x = state / self._max_y
    y = state % self._max_y
    return x, y

  def calculate_distance(self, cell, goal):
            return np.linalg.norm(np.array(cell) - np.array(goal))

  def distance_distribution(self, grid_size):
      goal = (grid_size - 1, grid_size - 1)  # Bottom right corner
      start_distribution = np.zeros((grid_size, grid_size))
      for i in range(grid_size):
          for j in range(grid_size):
              distance = self.calculate_distance((i, j), goal)
              start_distribution[i, j] = 1 / (distance + 1)  # Adding 1 to avoid division by zero
      start_distribution /= np.sum(start_distribution)  # Normalize probabilities
      return start_distribution

  def generate_uniform_distribution(self, grid_size):
    return np.full((grid_size, grid_size), 1 / (grid_size * grid_size))
  
  def generate_mixture_distribution(self, grid_size):
    uniform_distribution = self.generate_uniform_distribution(grid_size)
    edge_distribution = self.generate_edge_biased_distribution(grid_size)
    distance_distribution = self.distance_distribution(grid_size)
    
    mixture_distribution = (uniform_distribution + edge_distribution + distance_distribution) / 3
    mixture_distribution /= np.sum(mixture_distribution)  
    return mixture_distribution
  
  def generate_edge_biased_distribution(self, grid_size):
    start_distribution = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            # Assign higher probability to cells closer to the edges
            edge_probability = min(i, j, grid_size - i - 1, grid_size - j - 1)
            start_distribution[i, j] = 1 / (edge_probability + 1)  
    start_distribution /= np.sum(start_distribution)  
    return start_distribution
  
  def step(self, action):
    #TODO(ofirnachum): Add stochasticity.
    last_x, last_y = self._x, self._y
    if action == 0:
      if self._x < self._max_x - 1:
        self._x += 1
    elif action == 1:
      if self._y < self._max_y - 1:
        self._y += 1
    elif action == 2:
      if self._x > 0:
        self._x -= 1
    elif action == 3:
      if self._y > 0:
        self._y -= 1

    if self._map[self._x][self._y] == 'x':
      self._x, self._y = last_x, last_y

    reward = self._reward_fn(self._x, self._y, self._target_x, self._target_y)
    done = self._done_fn(self._x, self._y, self._target_x, self._target_y)
    return self._get_obs(), reward, done, {}


class GridWalk(Navigation):
  """Walk on grid to target location."""
  def __init__(self, distribution, length=10, tabular_obs=True):
  # def __init__(self, length=10, tabular_obs=True):
    nav_map = [[' ' for _ in range(length)]
               for _ in range(length)]
    nav_map[-1][-1] = 'T'
    self._length = length

    def reward_fn(x, y, tx, ty):
      taxi_distance = np.abs(x - tx) + np.abs(y - ty)
      return np.exp(-2 * taxi_distance / length)

    super(GridWalk, self).__init__(nav_map, distribution=distribution, tabular_obs=tabular_obs,
                                   reward_fn=reward_fn)


class FourRooms(Navigation):
  def __init__(self, tabular_obs=True):
    nav_map = [[' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               ['x', ' ', 'x', 'x', 'x', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', 'x', 'x', ' ', 'x', 'x'],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'T', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               ]
    super(FourRooms, self).__init__(nav_map, tabular_obs=tabular_obs)


def _compute_near_optimal_actions(nav_map, target_location):
  """A rough approximation to value iteration."""
  current_points = [target_location]
  chosen_actions = {target_location: 0}
  visited_points = {target_location: True}

  while current_points:
    next_points = []
    for point_x, point_y in current_points:
      for (action, (next_point_x, next_point_y)) in [
          (0, (point_x - 1, point_y)), (1, (point_x, point_y - 1)),
          (2, (point_x + 1, point_y)), (3, (point_x, point_y + 1))]:

        if (next_point_x, next_point_y) in visited_points:
          continue

        if not (next_point_x >= 0 and next_point_y >= 0 and
                next_point_x < len(nav_map) and
                next_point_y < len(nav_map[next_point_x])):
          continue

        if nav_map[next_point_x][next_point_y] == 'x':
          continue

        next_points.append((next_point_x, next_point_y))
        visited_points[(next_point_x, next_point_y)] = True
        chosen_actions[(next_point_x, next_point_y)] = action

    current_points = next_points
  return chosen_actions

def get_navigation_policy(nav_env, epsilon_explore=0.0, py=True,
                          return_distribution=True):
  """Creates a near-optimal policy for solving the navigation environment.

  Args:
    nav_env: A navigation environment.
    epsilon_explore: Probability of sampling random action as opposed to optimal
      action.
    py: Whether to return Python policy (NumPy) or TF (Tensorflow).
    return_distribution: In the case of a TF policy, whether to return the
      full action distribution.

  Returns:
    A policy_fn that takes in an observation and returns a sampled action along
      with a dictionary containing policy information (e.g., log probability).
    A spec that determines the type of objects returned by policy_info.

  Raises:
    ValueError: If epsilon_explore is not a valid probability.
  """
  if epsilon_explore < 0 or epsilon_explore > 1:
    raise ValueError('Invalid exploration value %f' % epsilon_explore)

  near_optimal_actions = _compute_near_optimal_actions(
      nav_env.nav_map, nav_env.target_location)
  policy_distribution = (
      np.ones((nav_env.n_state, nav_env.n_action)) / nav_env.n_action)
  for location, action in near_optimal_actions.items():
    tabular_id = nav_env.get_tabular_obs(np.array(location))
    policy_distribution[tabular_id] *= epsilon_explore
    policy_distribution[tabular_id, action] += 1 - epsilon_explore
  def obs_to_index_fn(observation):
    if not nav_env.tabular_obs:
      state = nav_env.get_tabular_obs(observation)
    else:
      state = observation

    if py:
      return state.astype(np.int32)
    else:
      return tf.cast(state, tf.int32)

  if py:
    return common_utils.create_py_policy_from_table(
        policy_distribution, obs_to_index_fn)
  else:
    return common_utils.create_tf_policy_from_table(
        policy_distribution, obs_to_index_fn,
        return_distribution=return_distribution)