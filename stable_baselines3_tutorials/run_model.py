import sys
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create (if necessary) directory to store models and logs
model_name = "PPO"
models_dir = "models/" + model_name
model_path = f"{models_dir}/290000.zip"
logdir = "logs"

if not os.path.exists(models_dir) or not os.path.exists(logdir):
    sys.exit("Please train a model (in train_model.py) before running run_model.py")

# Create environment
env = gym.make("LunarLander-v2", render_mode="rgb_array")
# Reset environment is necessary before stepping
env.reset()

# Create and train RL model (using PPO)
model = PPO.load(model_path, env=env)


# Enjoy trained agent

def evaluate(model, num_episodes=100, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    # More info about vectorized environments: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    vec_env = model.get_env()
    all_episode_rewards = []

    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = vec_env.reset()

        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            # also note that the step only returns a 4-tuple, as the env that is returned
            # by model.get_env() is an sb3 vecenv that wraps the >v0.26 API
            obs, reward, done, info = vec_env.step(action)
            episode_rewards.append(reward)
            # Render view
            vec_env.render("human")

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


# We run 10 "games" with the trained agent

evaluate(model, num_episodes=10, deterministic=True)

# Alternatively, we can evaluate the agent using baselines3 built-in functions
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=True)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
