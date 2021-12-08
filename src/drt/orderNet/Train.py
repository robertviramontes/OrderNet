import subprocess
from typing import Dict, List
from numpy.core.fromnumeric import std
import zmq
import os
import json
from OrderNetEnv import OrderNetEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
import random

# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind("tcp://*:5555")

build_dir = os.path.join(os.path.join("/workspaces", "OrderNet"), "build")

# subprocess.run(["make", "-j8"], cwd=build_dir)

executable_name = str(os.path.join(build_dir, os.path.join("src", "openroad")))
script_path = "/home/share/ispd_sample2.tcl"

# p = subprocess.Popen([executable_name, script_path])  # , stdout=subprocess.PIPE)

# received_done = False
# while not received_done:
#     message = socket.recv()
#     message = message.decode("utf-8")
#     if "done" in message:
#         received_done = True
#         continue

#     # otherwise, we get data serialized as json
#     net_json = json.loads(message)
#     print(net_json)

#     if net_json["type"] and net_json["type"] == "reward":
#         print(net_json["data"])

#         socket.send_string("ack")
#     if net_json["type"] and net_json["type"] == "inferenceData":
#         nets = net_json["data"]["nets"]
#         random.shuffle(nets)

#         # send the net ordering back to the responder
#         order = {}
#         for i, net in enumerate(nets):
#             order[net["name"]] = i

#         socket.send_string(json.dumps(order))

env = OrderNetEnv(str(executable_name), script_path)
# check_env(env)

model = A2C('MlpPolicy', env).learn(total_timesteps=2)

# p.wait()

log_name = "train_log_rand.txt"