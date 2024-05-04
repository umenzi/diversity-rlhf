import os
import sys

import numpy as np
import gymnasium as gym

from gymnasium.wrappers import TimeLimit

from seals.util import AutoResetWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy


# You can also download a trained expert from HuggingFace:
def download_model(env_name, env):
    print("Downloading a pretrained model from Hugging Face.")

    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name=env_name,  # e.g., "seals-CartPole-v0"
        venv=env,
    )

    return expert


def evaluate(model: SelfBaseAlgorithm, num_episodes=100, render=True, deterministic=True) -> float:
    """
    Evaluate a RLHF agent, running it in the environment the amount of episodes if render = True,
    and showing the mean reward and its standard error.

    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :param render: (bool) Whether to render the environment or not. This can slow down the evaluation
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :return: (float) Mean reward for the last num_episodes
    """

    vec_env: VecEnv = model.get_env()

    if render:
        # This function will only work for a single Environment More info about vectorized environments:
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html

        for i in range(num_episodes):
            done = False
            obs = vec_env.reset()

            while not done:
                # _states are only useful when using LSTM policies
                action, _states = model.predict(obs, deterministic=deterministic)
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                # also note that the step only returns a 4-tuple, as the env that is returned
                # by `model.get_env()` is an sb3 vecenv that wraps the >v0.26 API
                obs, reward, done, info = vec_env.step(action)

                # Render view
                vec_env.render("human")

    reward_mean, reward_std = evaluate_policy(
        model.policy,
        vec_env,
        num_episodes,
        render=render,
    )

    reward_stderr = reward_std / np.sqrt(num_episodes)

    print(f"Mean reward: {reward_mean:.2f} +/- {reward_stderr:.2f}, Num episodes: {num_episodes}")

    return reward_mean


def save_model(model):
    # Create (if necessary) directory to store models and logs
    model_name = "PPO"
    models_dir = "models/" + model_name
    logs_dir = "logs"
    rewards_dir = "rewards/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(rewards_dir):
        os.makedirs(rewards_dir)

    # We run the model some time steps.
    # This way we can stoop the training at any point, save the model, and continue training later.
    time_steps = 1000  # Note: set to a higher value (e.g. 100_000) to train a proficient expert

    for i in range(1, 2):  # increase this range too for better performance
        model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=model_name)
        model.save(f"{models_dir}/{time_steps * i}")


def get_atari_env():
    # Here we ensure that our environment has constant-length episodes by resetting
    # it when done, and running until 100 time steps have elapsed.
    # For real training, you will want a much longer time limit.
    def constant_length_asteroids(num_steps, render_mode="rgb_array"):
        atari_env = gym.make("AsteroidsNoFrameskip-v4", render_mode=render_mode)
        preprocessed_env = AtariWrapper(atari_env)
        endless_env = AutoResetWrapper(preprocessed_env)
        limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)

        return RolloutInfoWrapper(limited_env)

    # For real training, you will want a vectorized environment with 8 environments in parallel.
    # This can be done by passing in n_envs=8 as an argument to make_vec_env.
    # The seed needs to be set to 1 for reproducibility and also to avoid win32
    # np.random.randint high bound error.
    venv = make_vec_env(
        constant_length_asteroids,
        # env_kwargs={"num_steps": 100, "render_mode": "human"},
        env_kwargs={"num_steps": 100},
        seed=1
    )

    return VecFrameStack(venv, n_stack=4)


def load_model(venv) -> SelfBaseAlgorithm:
    # Create (if necessary) directory to store models and logs
    model_name = "PPO"
    model_id = "1000.zip"
    models_dir = "models/" + model_name
    model_path = f"{models_dir}/{model_id}"
    logs_dir = "logs"
    rewards_dir = "rewards/" + model_name
    reward_id = model_id
    rewards_path = f"{rewards_dir}/{reward_id}"

    if not os.path.exists(models_dir) or not os.path.exists(logs_dir):
        sys.exit("Please train a model (in train_model.py) before running run_model.py")

    print("Loading a local model.")

    return PPO.load(model_path, env=venv)
