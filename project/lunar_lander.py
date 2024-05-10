import numpy as np
import torch as th
import pandas as pd

from imitation.rewards.reward_nets import (
    BasicRewardNet,
)
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.testing.reward_improvement import is_significant_reward_improvement
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

import helpers
from project.train_preference_comparisons import train_preference_comparisons


# We train agents for a total of 1_000_000 steps
TIME_STEPS = 10_000
ITERATIONS = 100


def train_agent(agent, agent_name):
    print(f"Training {agent_name}")

    for i in range(1, ITERATIONS):  # set to 1_000_000 time steps in total for better performance
        agent.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name=agent_name)
        agent.save(f"{models_dir}/{agent_name}/{TIME_STEPS * i}")


device = th.device("cuda" if th.cuda.is_available() else "cpu")

models_dir = "models/"
logdir = "./logs/"

rng = np.random.default_rng(0)

venv = helpers.get_lunar_lander_env(16)

# Hyperparameters from:
# https://huggingface.co/sb3/ppo-LunarLander-v2

perfect_agent = PPO(
    policy=MlpPolicy,
    env=venv,
    seed=0,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.999,
    gae_lambda=0.98,
    n_epochs=4,
    tensorboard_log=logdir,
)
train_agent(perfect_agent, "perfect_agent")

(reward_net_0, main_trainer_0, results_0) = train_preference_comparisons(venv, perfect_agent,
                                                                         total_timesteps=5_000,
                                                                         total_comparisons=200,
                                                                         num_iterations=5,
                                                                         # Set to 60 for better performance
                                                                         fragment_length=100,
                                                                         transition_oversampling=1,
                                                                         initial_comparison_frac=0.1,
                                                                         reward_trainer_epochs=3,
                                                                         allow_variable_horizon=False,
                                                                         initial_epoch_multiplier=4,
                                                                         rng=rng,
                                                                         exploration_frac=0.05,
                                                                         conflicting_prob=0.0,
                                                                         reward_type=BasicRewardNet)
# We train an agent that sees only the shaped, learned reward
learned_reward_venv_0 = RewardVecEnvWrapper(venv, reward_net_0.predict_processed)
learner_0 = PPO(
    policy=MlpPolicy,
    env=learned_reward_venv_0,
    seed=0,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.999,
    gae_lambda=0.98,
    n_epochs=4,
    tensorboard_log=logdir,
)
train_agent(learner_0, "learner_0")

(reward_net_25, main_trainer_25, results_25) = train_preference_comparisons(venv, perfect_agent,
                                                                            total_timesteps=5_000,
                                                                            total_comparisons=200,
                                                                            num_iterations=5,
                                                                            # Set to 60 for better performance
                                                                            fragment_length=100,
                                                                            transition_oversampling=1,
                                                                            initial_comparison_frac=0.1,
                                                                            reward_trainer_epochs=3,
                                                                            allow_variable_horizon=False,
                                                                            initial_epoch_multiplier=4,
                                                                            rng=rng,
                                                                            exploration_frac=0.05,
                                                                            conflicting_prob=0.25,
                                                                            reward_type=BasicRewardNet)
# We train an agent that sees only the shaped, learned reward
learned_reward_venv_25 = RewardVecEnvWrapper(venv, reward_net_25.predict_processed)
learner_25 = PPO(
    policy=MlpPolicy,
    env=learned_reward_venv_25,
    seed=0,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.999,
    gae_lambda=0.98,
    n_epochs=4,
    tensorboard_log=logdir,
)
train_agent(learner_25, "learner_25")

(reward_net_50, main_trainer_50, results_50) = train_preference_comparisons(venv, perfect_agent,
                                                                            total_timesteps=5_000,
                                                                            total_comparisons=200,
                                                                            num_iterations=5,
                                                                            # Set to 60 for better performance
                                                                            fragment_length=100,
                                                                            transition_oversampling=1,
                                                                            initial_comparison_frac=0.1,
                                                                            reward_trainer_epochs=3,
                                                                            allow_variable_horizon=False,
                                                                            initial_epoch_multiplier=4,
                                                                            rng=rng,
                                                                            exploration_frac=0.05,
                                                                            conflicting_prob=0.5,
                                                                            reward_type=BasicRewardNet)
# We train an agent that sees only the shaped, learned reward
learned_reward_venv_50 = RewardVecEnvWrapper(venv, reward_net_50.predict_processed)
learner_50 = PPO(
    policy=MlpPolicy,
    env=learned_reward_venv_25,
    seed=0,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.999,
    gae_lambda=0.98,
    n_epochs=4,
    tensorboard_log=logdir,
)
train_agent(learner_50, "learner_50")

(reward_net_75, main_trainer_75, results_75) = train_preference_comparisons(venv, perfect_agent,
                                                                            total_timesteps=5_000,
                                                                            total_comparisons=200,
                                                                            num_iterations=5,
                                                                            # Set to 60 for better performance
                                                                            fragment_length=100,
                                                                            transition_oversampling=1,
                                                                            initial_comparison_frac=0.1,
                                                                            reward_trainer_epochs=3,
                                                                            allow_variable_horizon=False,
                                                                            initial_epoch_multiplier=4,
                                                                            rng=rng,
                                                                            exploration_frac=0.05,
                                                                            conflicting_prob=0.75,
                                                                            reward_type=BasicRewardNet)
# We train an agent that sees only the shaped, learned reward
learned_reward_venv_75 = RewardVecEnvWrapper(venv, reward_net_75.predict_processed)
learner_75 = PPO(
    policy=MlpPolicy,
    env=learned_reward_venv_75,
    seed=0,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.999,
    gae_lambda=0.98,
    n_epochs=4,
    tensorboard_log=logdir,
)
train_agent(learner_75, "learner_75")

(reward_net_100, main_trainer_100, results_100) = train_preference_comparisons(venv, perfect_agent,
                                                                               total_timesteps=5_000,
                                                                               total_comparisons=200,
                                                                               num_iterations=5,
                                                                               # Set to 60 for better performance
                                                                               fragment_length=100,
                                                                               transition_oversampling=1,
                                                                               initial_comparison_frac=0.1,
                                                                               reward_trainer_epochs=3,
                                                                               allow_variable_horizon=False,
                                                                               initial_epoch_multiplier=4,
                                                                               rng=rng,
                                                                               exploration_frac=0.05,
                                                                               conflicting_prob=1.0,
                                                                               reward_type=BasicRewardNet)
# We train an agent that sees only the shaped, learned reward
learned_reward_venv_100 = RewardVecEnvWrapper(venv, reward_net_100.predict_processed)
learner_100 = PPO(
    policy=MlpPolicy,
    env=learned_reward_venv_100,
    seed=0,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.999,
    gae_lambda=0.98,
    n_epochs=4,
    tensorboard_log=logdir,
)
train_agent(learner_100, "learner_100")

# helpers.evaluate(learner_0, num_episodes=10, deterministic=True, render=False, env_name="lunar")

# Evaluation #

# List of learners
learners = [learner_0, learner_25, learner_50, learner_75, learner_100]

# List of learner names
learner_names = ['learner_0', 'learner_25', 'learner_50', 'learner_75', 'learner_100']

# Dictionary to hold results
results = {}

# Evaluate each policy and store the results
for learner, name in zip(learners, learner_names):
    reward, _ = evaluate_policy(learner.policy, venv, n_eval_episodes=100, return_episode_rewards=True)
    results[name] = (np.mean(reward), np.std(reward))  # mean and std reward per episode

# Convert the results to a DataFrame
df = pd.DataFrame(results, index=['Mean Reward', 'Std Reward'])

# Print the DataFrame
print(df)

# Compare each learner with all other learners
for name1, reward1 in results.items():
    for name2, reward2 in results.items():
        if name1 != name2:
            significant = is_significant_reward_improvement(reward1, reward2, 0.001)
            print(f"{name1} is {'significantly better' if significant else 'NOT significantly better'} than {name2}.")

venv.close()
