import os
import sys

from matplotlib.pyplot import axes
from OrderNetEnv import OrderNetEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import numpy as np
import time
from typing import Dict

num_nets: Dict[str, int] = {
    "ispd18_sample": 11,
    "ispd18_sample2": 16,
    "ispd18_sample3": 7,
    "ispd18_test1": 3153,
    "ispd18_test2": 36834,
    "ispd18_test3": 36700,
    "ispd18_test4": 72401,
    "ispd18_test5": 72394,
    "ispd18_test6": 107701,
    "ispd18_test7": 179863,
    "ispd18_test8": 179863,
    "ispd18_test9": 128857,
    "ispd18_test10": 182000,
}


class OneShotCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(OneShotCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        dones = self.locals["dones"]
        for d in dones:
            if d:
                return False
        return True


def one_shot(agent: A2C) -> A2C:
    print("Starting one-shot routine!")
    reached_end_of_routing = False

    # incurs an environment reset
    total_timesteps, callback = agent._setup_learn(
        total_timesteps=sys.maxsize, eval_env=None, callback=OneShotCallback()
    )
    while not reached_end_of_routing:
        # incurs resets
        continue_training = agent.collect_rollouts(
            agent.env, callback, agent.rollout_buffer, n_rollout_steps=5
        )

        if not continue_training:
            print("Done with one-shot pass!")
            break

        agent.train()

        # when does the router reach an end condition?
        # TEMP just test how this loop works
        # for s in agent._last_episode_starts:
        #     if s:
        #         reached_end_of_routing = True

    print("End one-shot routine.")
    return agent


parser = argparse.ArgumentParser(
    description="One-shot solution for DRT with an agent that learns as we go."
)
parser.add_argument(
    "ispd_name", default="test1", help="Name of the ispd benchmark (i.e. test1)"
)
parser.add_argument(
    "--port", dest="zmq_port", default="5555", help="Port number to connect ZMQ over."
)

args = parser.parse_args()

build_dir = os.path.join("/OrderNet", "build")

# subprocess.run(["make", "-j8"], cwd=build_dir)

executable_name = str(os.path.join(build_dir, os.path.join("src", "openroad")))

# TCL script that drives openroad
script_path = os.path.join("/ispd18", "ispd18.tcl")

# TCL script requires knowing the ISPD source directory in env var ISPD_DIR
# the name of the ISPD benchmark in env var ISPD_NAME (i.e. ispd18_test1)
# and where to save the results in env var RESULT_DIR
os.environ["ISPD_DIR"] = os.path.join("/ispd18/ispd18_" + args.ispd_name)
os.environ["ISPD_NAME"] = "ispd18_" + args.ispd_name
if "RESULT_DIR" not in os.environ:
    # often defined by an external script
    os.environ["RESULT_DIR"] = "results"

ispd_def_path = os.path.join("/ispd18",("ispd18_" + args.ispd_name), ("ispd18_" + args.ispd_name + ".input.def"))
env = OrderNetEnv(
    str(executable_name),
    script_path,
    num_nets[("ispd18_" + args.ispd_name)],
    ispd_def_path,
    args.zmq_port,
)
# check_env(env)

model = A2C("MlpPolicy", env)

start_time = time.time()
one_shot(model)
end_time = time.time()

print("One-shot process took " + str(end_time - start_time) + " ms.")

save_pin_maps = False
if save_pin_maps:
    pin_maps = env.collect_pin_maps
    for i, pin_map in enumerate(pin_maps):
        if i == 0:
            collected_pin_maps = np.expand_dims(pin_map, axis=0)
        else:
            collected_pin_maps = np.append(
                collected_pin_maps, np.expand_dims(pin_map, axis=0), axis=0
            )

    np.save("pin_maps", collected_pin_maps)

# save_path = os.path.join("/home", "share", "cnn_test1_2500steps.zip")
# model.save(save_path)
