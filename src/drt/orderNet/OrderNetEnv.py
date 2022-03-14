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
from audioop import reverse
from os import path
from os import environ
import subprocess
from typing import Dict, List, Optional, Tuple
from gym import Env, spaces
import numpy as np
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
import zmq
import random
import math
import re


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class OrderNetEnv(Env):
    """
    Implements the custom OrderNet learning environment for OpenAI Gym.
    """

    def __init__(
        self,
        executable_path: str,
        script_path: str,
        ispd_def_path: path,
        zmq_port: str = "5555",
    ):
        """
        Environment for OrderNet RL.
        """
        self._zmq_port = zmq_port
        self._init_zmq()
        self._init_def_data(ispd_def_path)

        self._executable_path = executable_path
        self._script_path = script_path
        self._router_process: Optional[subprocess.Popen] = None

        self._num_layers = 9
        self._obs_space_shape = (9, 250, 250)
        self._nets_to_order: List[str] = []

        self.action_space = spaces.Box(
            low=0, high=1, shape=(self._num_nets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self._obs_space_shape, dtype=np.uint8
        )

        # reward metrics, will track improvement across steps
        self._violations = 0
        self._wirelength = 0

        self.collect_pin_maps = []

        self._num_resets = -1  # Becomes 0 after __init__ exits.
        self._num_steps = 0
        self.done = False
        # self.reset()

    def step(self, action) -> GymStepReturn:
        done = False

        # The `step()` method must return four values: obs, reward, done, info
        self._num_steps += 1
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

        # update the wirelength and drc so we can look at the improvement in the next iteration
        self._violations = violations
        self._wirelength = wirelength

        # reward is the sum of the violation and wirelength improvemnets
        violation_weight = 0.5
        wirelength_weight = 1 - violation_weight
        reward = (
            violation_weight * violation_improvement
            + wirelength_weight * wirelength_improvement
        )
        # done = violations == 0
        if reward > 0:
            print("violation improvement: " + str(violation_improvement))

        # finally, get the observation for the next set of nets to order
        # this updated observation of the state gets returned from the step
        message = self._socket.recv_json()
        self._socket.send_string("ack")

        if message["type"] == "reward" and message["lastInIteration"]:
            # handle the double reward in the last iteration.
            message = self._socket.recv_json()
            self._socket.send_string("ack")

        if message["type"] == "reward":
            # if we keep getting rewards, indicates we're skipping the searchRepair
            # process because there are no more violations.
            obs = np.zeros(self._obs_space_shape)
            while message["type"] == "reward":
                message = self._socket.recv_json()
                self._socket.send_string("ack")

                if message["type"] == "done":
                    done = True
                    break

        if message["type"] == "inferenceData":
            obs = self._get_observation(message)
        elif not done:
            raise TypeError("Should have gotten inference data at this point!")

        if done:
            print("Done at step number: " + str(self._num_steps))
            self.done = done
            try:
                outs, errs = self._router_process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                self._router_process.kill()
                outs, errs = self._router_process.communicate()

        return obs, reward, done, {}

    def reset(self) -> GymObs:
        if self.done:
            # gets around the VecEnv resetting before
            # the end of a step
            return np.random.rand(*self._obs_space_shape)

        print("resetting")
        self._num_resets += 1
        self._num_steps = 0

        self._cleanup_dangling_process()

        # Start the router and get the information from the first run
        # which should generate the initial violations we look to fix
        router_env = environ.copy()
        router_env["ZMQ_PORT"] = self._zmq_port
        self._router_process = subprocess.Popen(
            [self._executable_path, "-exit", self._script_path],
            env=router_env,  # , stdout=subprocess.PIPE
        )

        # get metrics about the workers in this design
        message = self._socket.recv_json()
        self._get_first_observation(message)
        self._socket.send_string("ack")

        # get metrics after the first pass of the detailed router
        receivedReward = False
        while not receivedReward:
            message = self._socket.recv_json()
            if "reward" in message["type"]:
                if message["lastInIteration"]:
                    receivedReward = True
                else:
                    self._socket.send_string("ack")
            else:
                self._get_first_observation(message)
                self._socket.send_string("ack")

        (violations, wirelength) = self._receive_reward(message)
        self._violations = violations
        self._wirelength = wirelength

        # finally get an observation of optimization iteration #1
        if self._num_resets == 0:
            message = self._socket.recv_json()
            observation = self._get_observation(message)
            message = self._socket.send_string("ack")

            return observation

        # reset must return an initial observation value
        return np.random.rand(*self._obs_space_shape)

    def render(self):
        pass

    def close(self):
        pass

    def _init_def_data(self, ispd_def_path: path):
        net_start_re = re.compile("NETS ([0-9]+) ;")
        net_desc_re = re.compile("-\s(.*)\s")  # matches - net0000
        in_net_section = False
        net_id = 1
        net_id_dict: Dict[str, int] = {}
        with open(ispd_def_path, "r") as f:
            for line in f:
                if not in_net_section:
                    m = re.match(net_start_re, line)
                    if m:
                        self._num_nets = int(m.group(1))
                        in_net_section = True
                else:
                    m = re.match(net_desc_re, line)
                    if m:
                        # push back the number of the net into a dictionary structure with integer ID
                        net_id_dict[m.group(1)] = net_id
                        net_id += 1
                    elif "END NETS" in line:
                        break
        self._net_id_dict = net_id_dict

    def _init_zmq(self):
        """
        Initializes ZMQ communication context and socket
        """
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind("tcp://*:" + self._zmq_port)

    def _receive_reward(self, message: Dict) -> Tuple[int, int]:
        """
        Waits to receive the reward from tritonroute.
        Returns (numViolations, wireLength)
        """

        if not "reward" in message["type"]:
            raise TypeError("Expected reward type message.")

        data = message["data"]

        self._socket.send_string("ack")

        return (data["numViolations"], data["wireLength"])

    def _get_observation(self, message: Dict) -> GymObs:

        (pin_map, nets_to_order) = get_observation(
            message, self._obs_space_shape, self._num_layers, self._net_id_dict
        )

        self.collect_pin_maps.append(pin_map)

        self._nets_to_order = nets_to_order
        return pin_map

    def _send_ordering(self, action):
        """Sends the net ordering indicated in action to the router."""

        order = parse_ordering(action, self._net_id_dict, self._nets_to_order)

        # send the net ordering back to the responder
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


def in_routebox(pin: Point, routeBoxMin: Point, routeBoxMax: Point) -> bool:
    in_routebox = True
    in_routebox &= pin.x >= routeBoxMin.x
    in_routebox &= pin.x <= routeBoxMax.x
    in_routebox &= pin.y >= routeBoxMin.y
    in_routebox &= pin.y <= routeBoxMax.y

    return in_routebox


def get_observation(
    message: Dict,
    obs_space_shape: Tuple[int, int, int],
    num_layers: int,
    net_id_dict: Dict[str, int],
) -> Tuple[GymObs, List]:
    """Takes in the entire message, the observation space shape, the number of layers and returns the pin_map, nete_numbering, and nets_to_order."""

    if not "inferenceData" in message["type"]:
        raise TypeError("Expected inference data type message.")

    data = message["data"]

    nets_to_order = data["nets"] if data["nets"] is not None else []

    if data["nets"] is None or data["routeBoxes"] is None:
        return (np.zeros(obs_space_shape, dtype=np.uint8), {}, nets_to_order)

    routeBoxMin = Point(data["routeBoxes"][0]["xlo"], data["routeBoxes"][0]["ylo"])
    routeBoxMax = Point(data["routeBoxes"][0]["xhi"], data["routeBoxes"][0]["yhi"])

    routeBoxXRange = abs(routeBoxMax.x - routeBoxMin.x) + 1
    routeBoxYRange = abs(routeBoxMax.y - routeBoxMin.y) + 1

    with_pins = list(filter(lambda net: "pins" in net, data["nets"]))
    # TODO in this, we assume pins are the most important feature.
    # hence, we can just (randomly shuffle?) the nets without pins
    without_pins = list(filter(lambda net: "pins" not in net, data["nets"]))
    with_pins_in_routebox = []

    # filter to only get pins in the range of our routebox!
    for i, net in enumerate(with_pins):
        # if i >= 15:
        #     # can only order 15 nets
        #     break
        pins = []
        for pin in net["pins"]:
            if in_routebox(
                Point(pin["l"]["x"], pin["l"]["y"]), routeBoxMin, routeBoxMax
            ) and in_routebox(
                Point(pin["h"]["x"], pin["h"]["y"]), routeBoxMin, routeBoxMax
            ):
                pins.append(pin)
        if len(pins) > 0:
            update_net = {}
            update_net["name"] = net["name"]
            update_net["pins"] = pins
            with_pins_in_routebox.append(update_net)

    pin_map = create_pin_maps(
        with_pins_in_routebox,
        routeBoxXRange,
        routeBoxYRange,
        routeBoxMin,
        obs_space_shape,
        num_layers,
        net_id_dict,
    )

    return (pin_map, nets_to_order)


def create_pin_maps(
    nets: List,
    routeBoxXRange: int,
    routeBoxYRange: int,
    routeBoxMin: Point,
    obs_space_shape: Tuple[int, int, int],
    num_layers: int,
    net_id_dict: Dict[str, int],
) -> np.array:
    pin_array = np.zeros(
        (num_layers, routeBoxYRange, routeBoxXRange),
        dtype=np.uint8,
    )

    for net in nets:
        net_id = net_id_dict[net["name"]]
        for pin in net["pins"]:
            z = pin["h"]["z"]
            low = Point(pin["l"]["x"] - routeBoxMin.x, pin["l"]["y"] - routeBoxMin.y)
            high = Point(pin["h"]["x"] - routeBoxMin.x, pin["h"]["y"] - routeBoxMin.y)

            minX = min(high.x, low.x)
            minY = min(high.y, low.y)

            # fill the entire range of the pins access box
            for tx in range(abs(high.x - low.x) + 1):
                for ty in range(abs(high.y - low.y) + 1):
                    pin_array[z][minY + ty][minX + tx] = (net_id) * 10

    # pad the pin array to the size of the observation
    x_padding_required = obs_space_shape[2] - routeBoxXRange
    y_padding_required = obs_space_shape[1] - routeBoxYRange

    # get the pin padding
    def padding_helper(padding_required) -> Tuple:
        if padding_required % 2 == 0:
            pad_a = int(padding_required / 2)
            pad_b = int(padding_required / 2)
        else:
            pad_a = math.floor(padding_required / 2)
            pad_b = math.ceil(padding_required / 2)

        return (pad_a, pad_b)

    if x_padding_required < 0 or y_padding_required < 0:
        print("WARNING: EXCEDED BOX SIZE")
        return (np.zeros(obs_space_shape, dtype=np.uint8), {})

    (x_pad_left, x_pad_right) = padding_helper(x_padding_required)
    (y_pad_top, y_pad_bottom) = padding_helper(y_padding_required)
    pin_array_padded = np.pad(
        pin_array,
        [(0, 0), (y_pad_top, y_pad_bottom), (x_pad_left, x_pad_right)],
        mode="constant",
        constant_values=[(0, 0), (0, 0), (0, 0)],
    )

    return pin_array_padded


def parse_ordering(action, net_id_dict: Dict[str, int], nets_to_order) -> Dict:
    """Sends the net ordering indicated in action to the router."""

    action_copy = action
    np.ndarray.sort(action)
    action = np.flip(action)

    net_priorities: Dict[int, str] = {}
    for net in nets_to_order:
        # net ids are 1-based, but the action space is 0-based
        net_priorities[action[net_id_dict[net["name"]] - 1]] = net["name"]

    sorted_net_priorities = sorted(list(net_priorities.keys()), reverse=True)

    net_order = 0
    order: Dict[str, int] = {}
    for n in sorted_net_priorities:
        order[net_priorities[n]] = net_order
        net_order += 1

    return order
