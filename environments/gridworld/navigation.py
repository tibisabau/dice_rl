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

import gym
import numpy as np
import tensorflow.compat.v2 as tf

from gym import spaces
from gym.utils import seeding
import sys, os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.common as common_utils


class Navigation(gym.Env):
  # def __init__(self, nav_map, distribution, tabular_obs=True,
  #              reward_fn=None, done_fn=None):
  def __init__(self, nav_map, tabular_obs=True,
               reward_fn=None, done_fn=None):
    self._map = nav_map
    self._tabular_obs = tabular_obs
    self._reward_fn = reward_fn
    self._done_fn = done_fn
    # self.acts = []
    if self._reward_fn is None:
      self._reward_fn = lambda x, y, tx, ty: float(x == tx and y == ty)
    if self._done_fn is None:
      self._done_fn = lambda x, y, tx, ty: False

    self._max_x = len(self._map)
    if not self._max_x:
      raise ValueError('Invalid map.')
    self._max_y = len(self._map[0])
    if not all(len(m) == self._max_y for m in self._map):
      raise ValueError('Invalid map.')

    self._start_x, self._start_y = self._find_initial_point()
    self._target_x, self._target_y = self._find_target_point()
    # print("START X: ", self._start_x)
    # print("START Y: ", self._start_y)
    # print("TARGET X: ", self._target_x)
    # print("TARGET Y: ", self._target_y)
    # print(self._map)
    self._n_state = self._max_x * self._max_y
    self._n_action = 4

    if self._tabular_obs:
      self.observation_space = spaces.Discrete(self._n_state)
    else:
      self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(2,))

    self.action_space = spaces.Discrete(self._n_action)
    # print(distribution)
    self.seed()
    # self.reset(distribution)
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
    # print("START NAV")
    for x in range(self._max_x):
      for y in range(self._max_y):
        if self._map[x][y] == 'S':
          break
      if self._map[x][y] == 'S':
        break
    else:
      # print((None, None))
      return None, None
    # print((x, y))
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

  def generate_random_distribution(self, k, seed=None):
    # if seed is not None:
    #     np.random.seed(seed)
    # random_numbers = np.random.rand(k)
    # lower = 0
    # upper = 100000
    # random_numbers = lower + (upper - lower) * np.random.rand(k)
    # probability_distribution = random_numbers / np.sum(random_numbers)
    mean = 5
    std = 2
    random_numbers = np.random.normal(mean, std, k)
    # Make sure all values are positive to be used as probabilities
    random_numbers = np.abs(random_numbers)
    # Normalize to create a probability distribution
    probability_distribution = random_numbers / np.sum(random_numbers)
    # print("Probability:", probability_distribution)
    # print(f"Sum of probabilities: {np.sum(probability_distribution)}")
    return probability_distribution
  
  # def reset(self, distribution):
  def reset(self):
    # print("RESTART NAV")
    # print("TARGET X: ", self._target_x)
    # print("TARGET Y: ", self._target_y)

    # print("N_STATE: ", self.n_state)
    if self._start_x is not None and self._start_y is not None:
      self._x, self._y = self._start_x, self._start_y
    else:
      while True:  # Find empty grid cell.
        ### DEFAULT ###
        self._x = self.np_random.randint(self._max_x)
        self._y = self.np_random.randint(self._max_y)

        ### State Mismatch ###
        # prob_table = np.array([
        #     [0.00013944223107569722, 0.00021912350597609563, 0.000249003984063745, 0.00047808764940239046, 0.0006673306772908366, 0.0008565737051792828, 0.0010657370517928287, 0.001444223107569721, 0.0018725099601593625, 0.002460159362549801],
        #     [0.0001693227091633466, 0.00027888446215139443, 0.0003784860557768924, 0.0005278884462151394, 0.0008565737051792828, 0.0008964143426294821, 0.0011653386454183268, 0.0015338645418326692, 0.0023705179282868527, 0.004093625498007968],
        #     [9.960159362549801e-05, 0.0002091633466135458, 0.0004581673306772908, 0.0005577689243027889, 0.0006972111553784861, 0.0007968127490039841, 0.0013545816733067729, 0.00199203187250996, 0.003247011952191235, 0.005826693227091634],
        #     [0.0001693227091633466, 0.00017928286852589642, 0.00036852589641434263, 0.0006573705179282868, 0.000747011952191235, 0.0007968127490039841, 0.0013247011952191236, 0.002430278884462151, 0.004312749003984063, 0.00852589641434263],
        #     [0.00021912350597609563, 0.00030876494023904385, 0.0003286852589641434, 0.0004681274900398406, 0.000747011952191235, 0.001145418326693227, 0.002051792828685259, 0.003894422310756972, 0.006932270916334662, 0.013725099601593625],
        #     [0.000249003984063745, 0.00036852589641434263, 0.0003286852589641434, 0.00049800796812749, 0.0008764940239043825, 0.001653386454183267, 0.002938247011952191, 0.005896414342629482, 0.011653386454183267, 0.022689243027888446],
        #     [0.00021912350597609563, 0.0003884462151394422, 0.0005677290836653387, 0.0009063745019920319, 0.0012948207171314741, 0.0020916334661354582, 0.004103585657370518, 0.008386454183266933, 0.01794820717131474, 0.03673306772908366],
        #     [0.00011952191235059761, 0.000249003984063745, 0.0005378486055776893, 0.0007669322709163346, 0.0012250996015936255, 0.002539840637450199, 0.005806772908366534, 0.012798804780876494, 0.03042828685258964, 0.06676294820717131],
        #     [0.00027888446215139443, 0.0004282868525896414, 0.0005876494023904382, 0.0008366533864541832, 0.001444223107569721, 0.0031772908366533865, 0.008087649402390438, 0.020796812749003985, 0.05408366533864542, 0.13239043824701197],
        #     [0.0002888446215139442, 0.0004681274900398406, 0.0005677290836653387, 0.0007071713147410359, 0.0014641434262948206, 0.0035856573705179283, 0.009681274900398407, 0.02993027888446215, 0.09393426294820717, 0.30994023904382473]
        # ])
        # prob_table = np.array([float(x) for x in distribution.split(',')])
        # prob_table = np.array(self.generate_random_distribution(100))
        # prob_table = prob_table.reshape((10, 10))
        # flat_probs = prob_table.flatten()
        # cdf = np.cumsum(flat_probs)
        # random_number = np.random.rand()
        # index = np.searchsorted(cdf, random_number)
        # self._x, self._y = np.unravel_index(index, prob_table.shape)

        ### DISTANCE BASED DISTRIBUTION ###
        # grid_size = 10
        # start_distribution = self.generate_exploration_decay_distribution(grid_size)
        # sampled_position = self.sample_start_position(start_distribution)
        # self._x, self._y = sampled_position

        ### MIXED MODE DISTRIBUTION ###
        # grid_size = 10
        # start_distribution = self.generate_mixture_distribution(grid_size)
        # sampled_position = self.sample_start_position(start_distribution)
        # self._x, self._y = sampled_position


        ### CLOSE TO TARGET ###
        # self._x = self.np_random.randint(max(0, self._target_x - 1), min(self._max_x, self._target_x + 1))
        # self._y = self.np_random.randint(max(0, self._target_y - 1), min(self._max_y, self._target_y + 1))
        self._y = 8
        self._x = 9

        ### FAR FROM TARGET ###
        # self._x = self.np_random.randint(0, 2)
        # self._y = self.np_random.randint(0, 2)
        
        ### EDGE ###
        # grid_size = 10
        # edge_biased_distribution = self.generate_edge_biased_distribution(grid_size)
        # sampled_position = self.sample_start_position(edge_biased_distribution)
        # self._x, self._y = sampled_position

        ### EXPLORATION BIAS ###
        # grid_size = 10
        # exploration_bias = 0.5  # Adjust the exploration bias factor
        # exploration_with_bias_distribution = self.generate_exploration_with_bias_distribution(grid_size, exploration_bias)
        # sampled_position = self.sample_start_position(exploration_with_bias_distribution)
        # self._x, self._y = sampled_position


        if self._map[self._x][self._y] != 'x':
          # print("START: ", self._x)
          # print("START: ", self._y)
          break
    # print((self._x, self._y))
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
    index = np.random.choice(len(flat_distribution), p=flat_distribution)
    # Convert flattened index to 2D coordinates
    row = index // start_distribution.shape[1]
    col = index % start_distribution.shape[1]
    return row, col
  
  def get_xy(self, state):
    # print("STATE: ", state)
    x = state / self._max_y
    y = state % self._max_y
    return x, y

  def calculate_distance(self, cell, goal):
            return np.linalg.norm(np.array(cell) - np.array(goal))

  def generate_exploration_decay_distribution(self, grid_size):
      goal = (grid_size - 1, grid_size - 1)  # Bottom right corner
      start_distribution = np.zeros((grid_size, grid_size))
      for i in range(grid_size):
          for j in range(grid_size):
              distance = self.calculate_distance((i, j), goal)
              start_distribution[i, j] = 1 / (distance + 1)  # Adding 1 to avoid division by zero
      start_distribution /= np.sum(start_distribution)  # Normalize probabilities
      return start_distribution

  def generate_grid_walk_distribution(self, grid_size):
    center = (grid_size // 2, grid_size // 2)
    start_distribution = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            distance = np.linalg.norm(np.array((i, j)) - np.array(center))
            start_distribution[i, j] = 1 / (distance + 1)  # Adding 1 to avoid division by zero
    start_distribution /= np.sum(start_distribution)  # Normalize probabilities
    return start_distribution

  def generate_uniform_distribution(self, grid_size):
    return np.full((grid_size, grid_size), 1 / (grid_size * grid_size))
  
  def generate_mixture_distribution(self, grid_size):
    uniform_distribution = self.generate_uniform_distribution(grid_size)
    grid_walk_distribution = self.generate_grid_walk_distribution(grid_size)
    exploration_decay_distribution = self.generate_exploration_decay_distribution(grid_size)
    
    mixture_distribution = (uniform_distribution + grid_walk_distribution + exploration_decay_distribution) / 3
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

  def generate_exploration_with_bias_distribution(self, grid_size, exploration_bias):
    start_distribution = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate distance to the goal
            distance_to_goal = np.linalg.norm(np.array([i, j]) - np.array([grid_size - 1, grid_size - 1]))
            # Exploration bias
            exploration_bias_factor = exploration_bias * (np.random.rand() + 1)  # Add random factor to the bias
            start_distribution[i, j] = (1 / (distance_to_goal + 1)) * exploration_bias_factor
    start_distribution /= np.sum(start_distribution)  # Normalize probabilities
    return start_distribution
  
  def step(self, action):
    # self.acts.append(action)
    # print("ACTIONS: ", len(self.acts))
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
  # def __init__(self, distribution, length=10, tabular_obs=True):
  def __init__(self, length=10, tabular_obs=True):
    nav_map = [[' ' for _ in range(length)]
               for _ in range(length)]
    nav_map[-1][-1] = 'T'
    self._length = length

    def reward_fn(x, y, tx, ty):
      taxi_distance = np.abs(x - tx) + np.abs(y - ty)
      return np.exp(-2 * taxi_distance / length)

    # super(GridWalk, self).__init__(nav_map, distribution, tabular_obs=tabular_obs,
    #                                reward_fn=reward_fn)
    super(GridWalk, self).__init__(nav_map, tabular_obs=tabular_obs,
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
  # print("Policy Distribution: ", policy_distribution)
  for location, action in near_optimal_actions.items():
    tabular_id = nav_env.get_tabular_obs(np.array(location))
    policy_distribution[tabular_id] *= epsilon_explore
    policy_distribution[tabular_id, action] += 1 - epsilon_explore
  # print("Policy Distribution: ", policy_distribution)
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
