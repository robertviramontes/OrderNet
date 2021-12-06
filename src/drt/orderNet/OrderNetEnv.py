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

import subprocess
from typing import Tuple
from gym import Env, spaces
import numpy as np
from numpy.core.fromnumeric import std
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
import zmq
import random

class OrderNetEnv(Env):
    """
    Implements the custom OrderNet learning environment for OpenAI Gym.
    """

    def __init__(self, executable_path: str, script_path: str):
        """
        Environment for OrderNet RL.
        """
        self._init_zmq()

        self._executable_path = executable_path
        self._script_path = script_path
        self._router_process = None

        self._obs_space_shape = (1,)

        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self._obs_space_shape, dtype=np.float32
        )

        # reward metrics, will track improvement across steps
        self._violations = 0
        self._wirelength = 0

        self._num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def step(self, action) -> GymStepReturn:
        # The `step()` method must return four values: obs, reward, done, info
        # TODO we get many observations before we get a reward, how to handle that?
        obs = self._get_observation()


        (violations, wirelength) = self._receive_reward()
        violation_improvement = (
            1
            if self.violations == 0
            else (self._violations - violations) / self._violations
        )
        wirelength_improvement = (
            0
            if self._wirelength == 0
            else (self._wirelength - wirelength) / self._wirelength
        )
        # reward is the sum of the violation and wirelength improvemnets
        reward = violation_improvement + wirelength_improvement
        done = True # for now, there is one step possible
        info = {}
        return obs, reward, done, info

    def reset(self) -> GymObs:
        self._num_resets += 1

        if isinstance(self._router_process, subprocess.Popen):
            try:
                self._router_process.wait(1)
            except:
                self._router_process.terminate()
        t = self._socket.poll(500) # discard any messages from the killed process
        for i in range(t):
            self._socket.recv()
            self._socket.send_string("ack")

        # Start the router and get the information from the first run
        # which should generate the initial violations we look to fix
        self._router_process = subprocess.Popen(
            [self._executable_path, self._script_path],
            stdout=subprocess.PIPE
        )

        # get metrics after the first pass of the detailed router
        (violations, wirelength) = self._receive_reward()
        self._violations = violations
        self._wirelength = wirelength

        # reset must return an initial observation value
        return np.random.rand(*self._obs_space_shape)

    def render(self):
        pass

    def close(self):
        pass

    def _init_zmq(self):
        """
        Initializes ZMQ communication context and socket
        """
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind("tcp://*:5555")

    def _receive_reward(self) -> Tuple[int, int]:
        """
        Waits to receive the reward from tritonroute.
        Returns (numViolations, wireLength)
        """
        message = self._socket.recv_json()
        print("reward: ")
        print(message)
        if message["type"] and message["type"] == "reward":
            self._socket.send_string("ack")

        return (message["data"]["numViolations"], message["data"]["wireLength"])


    def _get_observation(self) -> GymObs:
        message = self._socket.recv_json()

        print("observation: ")
        print(message)

        if message["type"] is None or message["type"] != "inferenceData":
            raise TypeError("Expected inference data type")
        
        infData = message["data"]
        nets = infData["nets"]
        random.shuffle(nets)

        # send the net ordering back to the responder
        order = {}
        for i, net in enumerate(nets):
            order[net["name"]] = i

        self._socket.send_json(order)
        return  np.random.rand(*self._obs_space_shape)
