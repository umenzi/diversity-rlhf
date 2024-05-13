import os
import re
import sys
from typing import Optional, List

import gymnasium as gym
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from gymnasium.wrappers import TimeLimit
from seals.util import AutoResetWrapper

from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy


from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def visualize_training(logdir: str, environments: List[str], separate: bool = False, selected_learners: Optional[List[str]] = None):
    """
    Visualize the training of multiple learners in the same plot.

    Args:
        logdir: (str) the directory where the tensorboard logs are stored
        environments: (list) list of environments to include in the plot
        separate: (bool) whether to plot the data of each learner separately or not
        selected_learners: (list) list of learners to include in the plot

    Returns: void
    """

    def parse_tensorboard_logs(logdir):
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()

        # Retrieve the scalars you are interested in
        rewards = event_acc.Scalars('rollout/ep_rew_mean')
        time = event_acc.Scalars('time/fps')

        # Convert to pandas DataFrame
        rewards_df = pd.DataFrame(rewards)
        time_df = pd.DataFrame(time)

        return rewards_df, time_df

    for environment in environments:
        env_logdir = os.path.join(environment, logdir)

        # Get a list of all directories in the log directory
        dirs = os.listdir(env_logdir)
        # Regular expression to match 'learner_' followed by two digits, an underscore, and one or more digits
        # and 'perfect_agent_' followed by one or more digits
        pattern_learner = re.compile(r'(learner_\d+)_(\d+)')
        pattern_perfect_agent = re.compile(r'(perfect_agent)_(\d+)')
        # Group the directories by the learner name
        learner_dirs = {}

        for dir in dirs:
            match_learner = pattern_learner.match(dir)
            match_perfect_agent = pattern_perfect_agent.match(dir)
            if match_learner:
                learner_name = match_learner.group(1)
                if learner_name not in learner_dirs:
                    learner_dirs[learner_name] = []
                learner_dirs[learner_name].append(dir)
            elif match_perfect_agent:
                learner_name = match_perfect_agent.group(1)
                if learner_name not in learner_dirs:
                    learner_dirs[learner_name] = []
                learner_dirs[learner_name].append(dir)

        # If selected_learners is specified, filter learner_dirs to only include the selected learners
        if selected_learners is not None:
            learner_dirs = {learner_name: dirs for learner_name, dirs in learner_dirs.items() if learner_name in selected_learners}

        # Remove the "/" at the end of the environment name for matplotlib title
        title_environment = environment.rstrip('/')

        # Now you can use learner_dirs to parse the tensorboard logs and plot the data
        if separate:
            for name, dirs in learner_dirs.items():
                rewards_dfs = []
                time_dfs = []
                for dir in dirs:
                    rewards_df, time_df = parse_tensorboard_logs(os.path.join(env_logdir, dir))
                    rewards_dfs.append(rewards_df)
                    time_dfs.append(time_df)

                # Calculate the average rewards and time
                avg_rewards_df = pd.concat(rewards_dfs).groupby(level=0).mean()
                avg_time_df = pd.concat(time_dfs).groupby(level=0).mean()

                # Plot the data
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.plot(avg_rewards_df['step'], avg_rewards_df['value'])
                plt.title(f'Mean reward of {name} in {title_environment} environment')

                plt.subplot(1, 2, 2)
                plt.plot(avg_time_df['step'], avg_time_df['value'])
                plt.title(f'{name} Time')

                # Adjust spacing between the subplots
                plt.subplots_adjust(wspace=0.5)

                plt.show()
        else:
            plt.figure(figsize=(10, 5))

            for name, dirs in learner_dirs.items():
                rewards_dfs = []
                time_dfs = []
                for dir in dirs:
                    rewards_df, time_df = parse_tensorboard_logs(os.path.join(env_logdir, dir))
                    rewards_dfs.append(rewards_df)
                    time_dfs.append(time_df)

                # Calculate the average rewards and time
                avg_rewards_df = pd.concat(rewards_dfs).groupby(level=0).mean()
                avg_time_df = pd.concat(time_dfs).groupby(level=0).mean()

                plt.subplot(1, 2, 1)
                plt.plot(avg_rewards_df['step'], avg_rewards_df['value'], label=name)
                plt.title(f'Mean reward in {title_environment} environment')

                plt.subplot(1, 2, 2)
                plt.plot(avg_time_df['step'], avg_time_df['value'], label=name)
                plt.title(f'{name} Time')

            # Add a legend to the plot
            plt.legend()

        # Display the plot
        plt.show()


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


def print_reward_info(reward_1, reward_2, a_1: str, a_2: str, n_episodes: int):
    print(
        f"{a_1} mean reward: {np.mean(reward_1):.2f} +/- "
        f"{np.std(reward_1) / np.sqrt(100):.2f}, Num episodes: {n_episodes}")
    print(
        f"{a_2} mean reward: {np.mean(reward_2):.2f} +/- "
        f"{np.std(reward_2) / np.sqrt(100):.2f}, Num episodes: {n_episodes}")


def evaluate(model: SelfBaseAlgorithm, env_name, n_envs=1, num_episodes=10, verbose: bool = False,
             time_steps: int = 1000, render=True, deterministic=True) -> float:
    """
    Evaluate a RLHF agent, running it in the environment the amount of episodes if render = True,
    and showing the mean reward and its standard error.

    :param model: (BaseRLModel object) the RL Agent
    :param env_name: (str) the environment name
    :param n_envs: (int) Number of environments to run the evaluation
    :param num_episodes: (int) number of episodes to evaluate it
    :param verbose: (bool) Whether to print additional information
    :param time_steps: (int) number of time steps to run the evaluation
    :param render: (bool) Whether to render the environment or not. This can slow down the evaluation
    :param deterministic: (bool) Whether to use deterministic or stochastic actions

    :return: (float) Mean reward for the last num_episodes
    """

    # vec_env: VecEnv = model.get_env()
    is_atari = False
    if env_name == "lunar":
        vec_env = get_lunar_lander_env(n_envs)
    else:
        raise Exception("Invalid environment name")

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    successes = []
    episode_start = np.ones((n_envs,), dtype=bool)

    if render:
        # This function will only work for a single Environment More info about vectorized environments:
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html

        try:
            for _ in range(time_steps):
                done = np.array([False] * vec_env.num_envs)
                obs = vec_env.reset()

                while not done.all():
                    # _states are only useful when using LSTM policies
                    action, _states = model.predict(obs, deterministic=deterministic, episode_start=episode_start)
                    # here, action, rewards and dones are arrays
                    # because we are using vectorized env
                    # also note that the step only returns a 4-tuple, as the env that is returned
                    # by `model.get_env()` is an sb3 vecenv that wraps the >v0.26 API
                    obs, reward, done, infos = vec_env.step(action)

                    episode_start = done

                    # Render view
                    vec_env.render("human")

                    episode_reward += reward[0]
                    ep_len += 1

                    if n_envs == 1:
                        # For atari, the return reward is not the atari score,
                        # so we have to get it from the infos dict
                        if is_atari and infos is not None and verbose:
                            episode_infos = infos[0].get("episode")
                            if episode_infos is not None:
                                print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                                print("Atari Episode Length", episode_infos["l"])

                        if done and not is_atari and verbose:
                            # NOTE: for env using VecNormalize, the mean reward
                            # is a normalized reward when `--norm_reward` flag is passed
                            print(f"Episode Reward: {episode_reward:.2f}")
                            print("Episode Length", ep_len)
                            episode_rewards.append(episode_reward)
                            episode_lengths.append(ep_len)
                            episode_reward = 0.0
                            ep_len = 0

                        # Reset also when the goal is achieved when using HER
                        if done and infos[0].get("is_success") is not None:
                            if verbose:
                                print("Success?", infos[0].get("is_success", False))

                            if infos[0].get("is_success") is not None:
                                successes.append(infos[0].get("is_success", False))
                                episode_reward, ep_len = 0.0, 0
        except KeyboardInterrupt:
            print("Evaluation interrupted by user.")
            pass

    if verbose and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if verbose and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(
            f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}, "
            f"Num episodes: {num_episodes}")

    if verbose and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    # TODO: I guess we should remove the following
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


def get_lunar_lander_env(n_envs: int = 1):
    # Here we ensure that our environment has constant-length episodes by resetting
    # it when done, and running until 1000 time steps have elapsed.
    # For real training, you will want a much longer time limit.
    def constant_length_asteroids(num_steps, render_mode="rgb_array"):
        lunar_env = gym.make("LunarLander-v2", render_mode=render_mode)
        endless_env = AutoResetWrapper(lunar_env)
        limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)

        return RolloutInfoWrapper(limited_env)

    # For real training, you will want a vectorized environment with 8 environments in parallel.
    # This can be done by passing in n_envs=8 as an argument to make_vec_env.
    # The seed needs to be set to 1 for reproducibility and also to avoid win32
    # np.random.randint high bound error.
    return make_vec_env(
        constant_length_asteroids,
        # env_kwargs={"num_steps": 100, "render_mode": "human"},
        env_kwargs={"num_steps": 1000},
        seed=1,
        n_envs=n_envs,
    )


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
