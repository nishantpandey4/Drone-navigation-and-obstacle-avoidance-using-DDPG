import numpy as np
import airsim

import gym
from gym import spaces


class AirSimEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    # def __init__(self, image_shape):
    def __init__(self):
        # self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        # self.viewer = None
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(19,), dtype=np.float32)

    def __del__(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()
