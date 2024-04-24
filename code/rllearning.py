import time
import os

import stable_baselines3
import sb3_contrib
from stable_baselines3.common.env_checker import check_env

from userenv import UserModelEnv

def rl_training(config, user_props, task_prop_distributions, anti_cheat_settings):
    env = UserModelEnv(config, user_props, task_prop_distributions, anti_cheat_settings)
    env.seed(config.main_seed)

    monitored_env = stable_baselines3.common.monitor.Monitor(env, filename=os.path.join(config.exp_dir_path, "train"))

    check_env(env, warn=True)
    if config.rl_model == "DQN":
        model = stable_baselines3.DQN("MlpPolicy", monitored_env, seed=config.main_seed, exploration_fraction=config.exploration_fraction,
                                  exploration_initial_eps=1, exploration_final_eps=config.exploration_final_eps,
                                  learning_starts=config.learning_starts)
    elif config.rl_model == "QR-DQN":
        model = sb3_contrib.qrdqn.QRDQN("MlpPolicy", monitored_env, seed=config.main_seed,
                                      exploration_fraction=config.exploration_fraction,
                                      exploration_initial_eps=1, exploration_final_eps=config.exploration_final_eps,
                                      learning_starts=config.learning_starts)

    model.set_logger(stable_baselines3.common.logger.configure(config.exp_dir_path, ["stdout", "csv"]))

    start_time = time.time()
    model.learn(total_timesteps=config.total_timesteps, log_interval=1000)
    duration = time.time() - start_time
    print(f"Training took {duration} seconds.")

    model.save(os.path.join(config.exp_dir_path, "model.save"))



