import os
import sys
from OrderNetEnv import OrderNetEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import argparse

def one_shot(agent: A2C) -> A2C:
    print("Starting one-shot routine!")
    reached_end_of_routing = False

    # incurs an environment reset
    total_timesteps, callback = agent._setup_learn(
        total_timesteps=sys.maxsize,
        eval_env=None
    )
    while not reached_end_of_routing:
        # incurs resets
        agent.collect_rollouts(
            agent.env, 
            callback,
            agent.rollout_buffer,
            n_rollout_steps=5
        )
        agent.train()

        # when does the router reach an end condition?
        # TEMP just test how this loop works
        for s in agent._last_episode_starts:
            if s:
                reached_end_of_routing = True
        if(reached_end_of_routing):
            print("Done with one-shot pass!")
    
    print("End on-shot routine.")
    return agent

parser = argparse.ArgumentParser(description='One-shot solution for DRT with an agent that learns as we go.')
parser.add_argument("ispd_name", default="test1",
                    help="Name of the ispd benchmark (i.e. test1)")

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
os.environ["RESULT_DIR"] = "results/test/"

env = OrderNetEnv(str(executable_name), script_path)
# check_env(env)

model = A2C("CnnPolicy", env)

one_shot(model)

# save_path = os.path.join("/home", "share", "cnn_test1_2500steps.zip")
# model.save(save_path)
