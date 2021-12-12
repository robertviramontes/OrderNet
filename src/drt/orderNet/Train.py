import os
from OrderNetEnv import OrderNetEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

build_dir = os.path.join(os.path.join("/workspaces", "OrderNet"), "build")

# subprocess.run(["make", "-j8"], cwd=build_dir)

executable_name = str(os.path.join(build_dir, os.path.join("src", "openroad")))
script_path = "/home/share/ispd_sample2.tcl"

env = OrderNetEnv(str(executable_name), script_path)
# check_env(env)

model = A2C("MlpPolicy", env).learn(total_timesteps=1)

log_name = "train_log_rand.txt"
