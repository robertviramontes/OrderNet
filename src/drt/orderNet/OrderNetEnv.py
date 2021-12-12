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
from typing import Dict, List, Optional, Tuple
from gym import Env, spaces
import numpy as np
from numpy.core.fromnumeric import std
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
import zmq
import random
import math


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


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
        self._router_process: Optional[subprocess.Popen] = None

        self._num_layers = 9
        self._obs_space_shape = (200, 200)
        self._nets_to_order = []

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
        print("stepping")

        # First, write the net ordering to TritonRoute from the action
        self._socket.recv_json()
        self._send_ordering(action)

        # Then, get the reward for the net ordering we just applied
        message = self._socket.recv_json()
        (violations, wirelength) = self._receive_reward(message)

        violation_improvement = (
            1
            if self._violations == 0
            else (self._violations - violations) / self._violations
        )
        wirelength_improvement = (
            0
            if self._wirelength == 0
            else (self._wirelength - wirelength) / self._wirelength
        )

        # reward is the sum of the violation and wirelength improvemnets
        reward = violation_improvement + wirelength_improvement

        # finally, get the observation for the next set of nets to order
        # this updated observation of the state gets returned from the step
        message = self._socket.recv_json()
        self._get_observation(message)
        self._socket.send_string("ack")

        done = violations == 0
        info = {}
        obs = np.random.rand(*self._obs_space_shape)
        return obs, reward, done, info

    def reset(self) -> GymObs:
        print("resetting")
        self._num_resets += 1

        self._cleanup_dangling_process()

        # Start the router and get the information from the first run
        # which should generate the initial violations we look to fix
        self._router_process = subprocess.Popen(
            [self._executable_path, self._script_path], stdout=subprocess.PIPE
        )

        # get metrics about the workers in this design
        message = self._socket.recv_json()
        self._get_first_observation(message)
        self._socket.send_string("ack")

        # get metrics after the first pass of the detailed router
        receivedReward = False
        while not receivedReward:
            message = self._socket.recv_json()
            if "reward" in message["type"] and message["lastInIteration"]:
                receivedReward = True
            else:
                self._get_first_observation(message)
                self._socket.send_string("ack")

        (violations, wirelength) = self._receive_reward(message)
        self._violations = violations
        self._wirelength = wirelength

        # finally get an observation of optimization iteration #1
        if self._num_resets > 0:
            message = self._socket.recv_json()
            self._get_observation(message)
            message = self._socket.send_string("ack")

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

    def _receive_reward(self, message: Dict) -> Tuple[int, int]:
        """
        Waits to receive the reward from tritonroute.
        Returns (numViolations, wireLength)
        """

        if not "reward" in message["type"]:
            raise TypeError("Expected reward type message.")

        data = message["data"]
        print("reward: ")
        print(data)

        self._socket.send_string("ack")

        return (data["numViolations"], data["wireLength"])

    def _get_observation(self, message: Dict) -> GymObs:

        if not "inferenceData" in message["type"]:
            raise TypeError("Expected inference data type message.")

        data = message["data"]

        routeBoxMin = Point(data["routeBoxes"][0]["xlo"], data["routeBoxes"][0]["ylo"])
        routeBoxMax = Point(data["routeBoxes"][0]["xhi"], data["routeBoxes"][0]["yhi"])

        routeBoxXRange = abs(routeBoxMax.x - routeBoxMin.x)
        routeBoxYRange = abs(routeBoxMax.y - routeBoxMin.y)

        with_pins = list(filter(lambda net: "pins" in net, data["nets"]))
        # TODO in this, we assume pins are the most important feature.
        # hence, we can just (randomly shuffle?) the nets without pins
        without_pins = list(filter(lambda net: "pins" not in net, data["nets"]))
        self._create_pin_maps(routeBoxXRange, routeBoxYRange, routeBoxMin, with_pins)

        self._nets_to_order = data["nets"]

        return np.random.rand(*self._obs_space_shape)

    def _create_pin_maps(
        self,
        routeBoxXRange: int,
        routeBoxYRange: int,
        routeBoxMin: Point,
        nets: List,
    ):
        pin_array = np.zeros(
            (self._num_layers, self._obs_space_shape[1], self._obs_space_shape[0]),
            dtype=np.float32,
        )
        for i, net in enumerate(nets):
            for pin in net["pins"]:
                z = pin["h"]["z"]
                xl = pin["l"]["x"]
                yl = pin["l"]["y"]
                xh = pin["h"]["x"]
                yh = pin["h"]["y"]
                # pin indices are 1 indexed, np is 1 indexed?
                pin_array[z][yl][xl] = i
                pin_array[z][yh][xh] = i

        print(pin_array)

    def _send_ordering(self, action):
        """Sends the net ordering indicated in action to the router."""
        random.shuffle(self._nets_to_order)

        # send the net ordering back to the responder
        order = {}
        for i, net in enumerate(self._nets_to_order):
            order[net["name"]] = i

        self._socket.send_json(order)

    def _get_first_observation(self, message: Dict) -> GymObs:
        """Gets the first observation after a reset."""
        if not "inferenceData" in message["type"]:
            raise TypeError("Expected inference data type message.")

        data = message["data"]

        # routeBoxMin = (data["routeBox"]["xlo"], data["routeBox"]["ylo"])
        # routeBoxMax = (data["routeBox"]["xhi"], data["routeBox"]["yhi"])

        # routeBoxXRange = abs(routeBoxMax[0] - routeBoxMin[0])
        # routeBoxYRange = abs(routeBoxMax[1] - routeBoxMin[1])

        # update the observation space based on the observed worker grid size.
        # newXRange = max(math.ceil(routeBoxXRange * 1.5), self._obs_space_shape[0])
        # newYRange = max(math.ceil(routeBoxYRange * 1.5), self._obs_space_shape[1])
        # self._obs_space_shape = (newXRange, newYRange)

        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=self._obs_space_shape, dtype=np.float32
        # )

        # for some reason, this over-reports the number of layers in the design?
        # self._num_layers = data["numLayers"]

        self._nets_to_order = data["nets"]

        return np.random.rand(*self._obs_space_shape)

    def _cleanup_dangling_process(self):
        if isinstance(self._router_process, subprocess.Popen):
            try:
                self._router_process.wait(1)
            except:
                self._router_process.terminate()
        t = self._socket.poll(500)  # discard any messages from the killed process
        for i in range(t):
            self._socket.recv()
            self._socket.send_string("ack")
