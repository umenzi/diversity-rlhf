import time
from typing import Callable

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

"""
To multiprocess RL training, we will just have to wrap the Gym env into a SubprocVecEnv object, 
that will take care of synchronising the processes. The idea is that each process will run an 
independent instance of the Gym env.

For that, we need an additional utility function, make_env, that will instantiate the 
environments and make sure they are different (using different random seed).
"""


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed environment.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


env_id = "CartPole-v1"
num_cpu = 4  # Number of (parallel) processes to use

# Next, we create the vectorized environment

# We can do it manually:
# Because we use vectorized environment (SubprocVecEnv),
# the actions sent to the wrapped env must be an array (one action per process).
# Also, observations, rewards and dones are arrays.
env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
model = A2C("MlpPolicy", env, verbose=0)

# Or let Baselines3 do it
# By default, we use a DummyVecEnv as it is usually faster
vec_env = make_vec_env(env_id, n_envs=num_cpu)
model = A2C("MlpPolicy", vec_env, verbose=0)

# Next, we evaluate an untrained agent (hence random)

# We create a separate environment for evaluation
eval_env = gym.make(env_id)
# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

# Next, we compare the time taken using one vs 4 processes.
# In total, it should take ~30s

n_timesteps = 25000

# Multiprocessed RL Training
start_time = time.time()
model.learn(n_timesteps)
total_time_multi = time.time() - start_time

print(
    f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS"
)

# Single Process RL Training
single_process_model = A2C("MlpPolicy", env_id, verbose=0)

start_time = time.time()
single_process_model.learn(n_timesteps)
total_time_single = time.time() - start_time

print(
    f"Took {total_time_single:.2f}s for single process version - {n_timesteps / total_time_single:.2f} FPS"
)

print(
    "Multiprocessed training is {:.2f}x faster!".format(
        total_time_single / total_time_multi
    )
)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")
