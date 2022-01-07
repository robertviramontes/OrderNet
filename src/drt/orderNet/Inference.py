import os

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import A2C
import subprocess
import zmq
from typing import Dict
from OrderNetEnv import get_observation, parse_ordering
import sys


def parseInfData(message: Dict, model: BaseAlgorithm):
    # create the pin maps from the data
    (pin_map, net_numbering, nets_to_order) = get_observation(message, (9, 250, 250), 9)

    # parse the pin maps through the model to create an action
    (action, next_state) = model.predict(pin_map)

    # acknowledge the request for sorting
    socket.recv_json()

    # generate the ordering object
    ordering = parse_ordering(action, net_numbering, nets_to_order)

    # send the net ordering
    socket.send_json(ordering)


build_dir = os.path.join(os.path.join("/workspaces", "OrderNet"), "build")

executable_name = str(os.path.join(build_dir, os.path.join("src", "openroad")))
script_path = "/home/share/ispd_test1.tcl"

if len(sys.argv) > 1:
    saved_model_name = str(sys.argv[1])
else:
    saved_model_name = "cnn_test1_2500steps_noSkipSr.zip"

save_path = os.path.join("/home", "share", saved_model_name)

model = A2C.load(save_path)

# start the detailed route subprocess
router_process = subprocess.Popen(
    [executable_name, script_path]  # , stdout=subprocess.PIPE
)

# start my zmq sockets
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

not_done = True
while not_done:
    message = socket.recv_json()
    if "type" in message:
        if "reward" in message["type"]:
            socket.send_string("ack")
        elif "inferenceData" in message["type"]:
            socket.send_string("ack")
            parseInfData(message, model)
        elif "done" in message["type"]:
            not_done = False
        else:
            raise TypeError("Unknown message type.")
    else:
        raise TypeError("No type for the message.")
