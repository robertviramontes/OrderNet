"""
Based on stable-baselines3 example at https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/envs/identity_env.py

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from gym import Env, spaces
import numpy as np
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

class OrderNetEnv(Env):
    """
    Implements the custom OrderNet learning environment for OpenAI Gym.
    """

    def __init__(self):
        """
        Environment for OrderNet RL.
        """
        self._obs_space_shape = (1,)

        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self._obs_space_shape, dtype=np.float32
        )

        self._num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def step(self, action) -> GymStepReturn:
        # The `step()` method must return four values: obs, reward, done, info
        obs = np.random.rand(*self._obs_space_shape)
        reward = 1
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self) -> GymObs:
        self._num_resets += 1

        # reset must return an initial observation value
        return np.random.rand(*self._obs_space_shape)

    def render(self):
        pass

    def close(self):
        pass
