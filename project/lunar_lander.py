import wandb

import numpy as np
import torch as th
import pandas as pd

from imitation.rewards.reward_nets import (
    BasicRewardNet,
)
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.testing.reward_improvement import is_significant_reward_improvement
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.ppo import MlpPolicy
from wandb.apis.public import Run
from wandb.integration.sb3 import WandbCallback

import helpers
import project.environments
import project.graphs
from project.train_preference_comparisons import train_preference_comparisons

# We train agents for a total of 1_000_000 steps
# TIME_STEPS = 10_000
# ITERATIONS = 100
TIME_STEPS = 1000
ITERATIONS = 10

"""
TODO:
- Study what type of gpu use in Delft Blue
- Hyperparameter tuning
- Parallelize training (and then add #SBATCH --gpus=2)
- Implement evaluation
- Update architecture for several experiments
"""


def train_agent(agent: SelfBaseAlgorithm, agent_name):
    print(f"Training {agent_name}")

    for i in range(1, ITERATIONS):  # set to 1_000_000 time steps in total for better performance
        model_path = f"{environment_dir}{models_dir}{agent_name}/{TIME_STEPS * i}"

        agent.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name=agent_name,
                    callback=WandbCallback(
                        gradient_save_freq=100,
                        model_save_freq=0,  # we do not save the model while training,
                                            # only at the end after completing the training
                        model_save_path=model_path,
                        verbose=1,  # 0 = no output, 1 = info messages, "classic output", >= 2 = debug mode
                    ))
        agent.save(model_path)

        # After every 1000 training steps, we evaluate the agent
        mean_reward, std_reward = evaluate_policy(agent.policy, evaluation_env, n_eval_episodes=10)
        wandb.log({f"{agent_name}_eval_mean_reward": mean_reward, f"{agent_name}_eval_std_reward": std_reward})


device = th.device("cuda" if th.cuda.is_available() else "cpu")

environment_dir = "lunar/"
# Directory where the trained models (i.e. its weights) are stored
models_dir = "models/"
# Directory where the tensorboard logs are stored
logdir = "logs/"

venv = project.environments.get_lunar_lander_env(16)

evaluation_env = project.environments.get_lunar_lander_env(1)

# Hyperparameters to log in wandb
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 10_000,
    "env_name": "LunarLander-v2",
}

# Hyperparameters obtained with Optuna, initially inspired by:
# https://huggingface.co/sb3/ppo-LunarLander-v2

num_experiments = 3

# Parameters of RL
seed = 0
n_steps = 2048
batch_size = 64
clip_range = 0.1
ent_coef = 0.01
gamma = 0.999
gae_lambda = 0.9
n_epochs = 4
learning_rate = 0.0003
tensorboard_log = environment_dir + logdir

# Parameters for RLHF
total_timesteps = 5_000
total_comparisons = 200
num_iterations = 5
fragment_length = 100
transition_oversampling = 1
initial_comparison_frac = 0.1
reward_trainer_epochs = 3
allow_variable_horizon = False
initial_epoch_multiplier = 4
exploration_frac = 0.05

# Run the experiments
for i in range(num_experiments):
    # We change the random seed in every experiment
    rng = np.random.default_rng(i * 10)

    run = wandb.init(
        project="rlhf_lunar",
        name=f"perfect_agent_{i}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    perfect_agent = PPO(
        policy=MlpPolicy,
        env=venv,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        tensorboard_log=tensorboard_log,
    )
    train_agent(perfect_agent, "perfect_agent")

    run.finish()

    run = wandb.init(
        project="rlhf_lunar",
        name=f"learner_0_{i}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    (reward_net_0, main_trainer_0, results_0) = (
        train_preference_comparisons(venv, perfect_agent,
                                     total_timesteps=total_timesteps,
                                     total_comparisons=total_comparisons,
                                     num_iterations=num_iterations,
                                     # Set to 60 for better performance
                                     fragment_length=fragment_length,
                                     transition_oversampling=transition_oversampling,
                                     initial_comparison_frac=initial_comparison_frac,
                                     reward_trainer_epochs=reward_trainer_epochs,
                                     allow_variable_horizon=allow_variable_horizon,
                                     initial_epoch_multiplier=initial_epoch_multiplier,
                                     rng=rng,
                                     exploration_frac=exploration_frac,
                                     conflicting_prob=0.0,
                                     reward_type=BasicRewardNet))
    # We train an agent that sees only the shaped, learned reward
    learned_reward_venv_0 = RewardVecEnvWrapper(venv, reward_net_0.predict_processed)
    learner_0 = PPO(
        policy=MlpPolicy,
        env=learned_reward_venv_0,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        tensorboard_log=tensorboard_log,
    )
    train_agent(learner_0, "learner_0")

    run.finish()

    run = wandb.init(
        project="rlhf_lunar",
        name=f"learner_25_{i}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    (reward_net_25, main_trainer_25, results_25) = (
        train_preference_comparisons(venv, perfect_agent,
                                     total_timesteps=total_timesteps,
                                     total_comparisons=total_comparisons,
                                     num_iterations=num_iterations,
                                     # Set to 60 for better performance
                                     fragment_length=fragment_length,
                                     transition_oversampling=transition_oversampling,
                                     initial_comparison_frac=initial_comparison_frac,
                                     reward_trainer_epochs=reward_trainer_epochs,
                                     allow_variable_horizon=allow_variable_horizon,
                                     initial_epoch_multiplier=initial_epoch_multiplier,
                                     rng=rng,
                                     exploration_frac=exploration_frac,
                                     conflicting_prob=0.25,
                                     reward_type=BasicRewardNet))
    # We train an agent that sees only the shaped, learned reward
    learned_reward_venv_25 = RewardVecEnvWrapper(venv, reward_net_25.predict_processed)
    learner_25 = PPO(
        policy=MlpPolicy,
        env=learned_reward_venv_25,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        tensorboard_log=tensorboard_log,
    )
    train_agent(learner_25, "learner_25")

    run.finish()

    run = wandb.init(
        project="rlhf_lunar",
        name=f"learner_40_{i}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    (reward_net_40, main_trainer_40, results_40) = (
        train_preference_comparisons(venv, perfect_agent,
                                     total_timesteps=total_timesteps,
                                     total_comparisons=total_comparisons,
                                     num_iterations=num_iterations,
                                     # Set to 60 for better performance
                                     fragment_length=fragment_length,
                                     transition_oversampling=transition_oversampling,
                                     initial_comparison_frac=initial_comparison_frac,
                                     reward_trainer_epochs=reward_trainer_epochs,
                                     allow_variable_horizon=allow_variable_horizon,
                                     initial_epoch_multiplier=initial_epoch_multiplier,
                                     rng=rng,
                                     exploration_frac=exploration_frac,
                                     conflicting_prob=0.40,
                                     reward_type=BasicRewardNet))
    # We train an agent that sees only the shaped, learned reward
    learned_reward_venv_40 = RewardVecEnvWrapper(venv, reward_net_40.predict_processed)
    learner_40 = PPO(
        policy=MlpPolicy,
        env=learned_reward_venv_40,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        tensorboard_log=tensorboard_log,
    )
    train_agent(learner_40, "learner_40")

    run.finish()

    run = wandb.init(
        project="rlhf_lunar",
        name=f"learner_50_{i}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    (reward_net_50, main_trainer_50, results_50) = (
        train_preference_comparisons(venv, perfect_agent,
                                     total_timesteps=total_timesteps,
                                     total_comparisons=total_comparisons,
                                     num_iterations=num_iterations,
                                     # Set to 60 for better performance
                                     fragment_length=fragment_length,
                                     transition_oversampling=transition_oversampling,
                                     initial_comparison_frac=initial_comparison_frac,
                                     reward_trainer_epochs=reward_trainer_epochs,
                                     allow_variable_horizon=allow_variable_horizon,
                                     initial_epoch_multiplier=initial_epoch_multiplier,
                                     rng=rng,
                                     exploration_frac=exploration_frac,
                                     conflicting_prob=0.5,
                                     reward_type=BasicRewardNet))
    # We train an agent that sees only the shaped, learned reward
    learned_reward_venv_50 = RewardVecEnvWrapper(venv, reward_net_50.predict_processed)
    learner_50 = PPO(
        policy=MlpPolicy,
        env=learned_reward_venv_25,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        tensorboard_log=tensorboard_log,
    )
    train_agent(learner_50, "learner_50")

    run.finish()

    run = wandb.init(
        project="rlhf_lunar",
        name=f"learner_75_{i}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    (reward_net_75, main_trainer_75, results_75) = (
        train_preference_comparisons(venv, perfect_agent,
                                     total_timesteps=total_timesteps,
                                     total_comparisons=total_comparisons,
                                     num_iterations=num_iterations,
                                     # Set to 60 for better performance
                                     fragment_length=fragment_length,
                                     transition_oversampling=transition_oversampling,
                                     initial_comparison_frac=initial_comparison_frac,
                                     reward_trainer_epochs=reward_trainer_epochs,
                                     allow_variable_horizon=allow_variable_horizon,
                                     initial_epoch_multiplier=initial_epoch_multiplier,
                                     rng=rng,
                                     exploration_frac=exploration_frac,
                                     conflicting_prob=0.75,
                                     reward_type=BasicRewardNet))
    # We train an agent that sees only the shaped, learned reward
    learned_reward_venv_75 = RewardVecEnvWrapper(venv, reward_net_75.predict_processed)
    learner_75 = PPO(
        policy=MlpPolicy,
        env=learned_reward_venv_75,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        tensorboard_log=tensorboard_log,
    )
    train_agent(learner_75, "learner_75")

    run.finish()

    run = wandb.init(
        project="rlhf_lunar",
        name=f"learner_100_{i}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    (reward_net_100, main_trainer_100, results_100) = (
        train_preference_comparisons(venv, perfect_agent,
                                     total_timesteps=total_timesteps,
                                     total_comparisons=total_comparisons,
                                     num_iterations=num_iterations,
                                     # Set to 60 for better performance
                                     fragment_length=fragment_length,
                                     transition_oversampling=transition_oversampling,
                                     initial_comparison_frac=initial_comparison_frac,
                                     reward_trainer_epochs=reward_trainer_epochs,
                                     allow_variable_horizon=allow_variable_horizon,
                                     initial_epoch_multiplier=initial_epoch_multiplier,
                                     rng=rng,
                                     exploration_frac=exploration_frac,
                                     conflicting_prob=1.0,
                                     reward_type=BasicRewardNet))
    # We train an agent that sees only the shaped, learned reward
    learned_reward_venv_100 = RewardVecEnvWrapper(venv, reward_net_100.predict_processed)
    learner_100 = PPO(
        policy=MlpPolicy,
        env=learned_reward_venv_100,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        tensorboard_log=tensorboard_log,
    )
    train_agent(learner_100, "learner_100")

    run.finish()

# helpers.evaluate(learner_0, num_episodes=10, deterministic=True, render=False, env_name="lunar")

# Evaluation #

# List of learners
# learners = [learner_0, learner_25, learner_50, learner_75, learner_100]

# List of learner names
# learner_names = ['learner_0', 'learner_25', 'learner_50', 'learner_75', 'learner_100']

# Dictionary to hold results
# results = {}

# Evaluate each policy and store the results
# for learner, name in zip(learners, learner_names):
#     reward, _ = evaluate_policy(learner.policy, venv, n_eval_episodes=100, return_episode_rewards=True)
#     results[name] = (np.mean(reward), np.std(reward))  # mean and std reward per episode

# Convert the results to a DataFrame
# df = pd.DataFrame(results, index=['Mean Reward', 'Std Reward'])

# Print the DataFrame
# print(df)

# Compare each learner with all other learners
# for name1, reward1 in results.items():
#     for name2, reward2 in results.items():
#         if name1 != name2:
#             significant = is_significant_reward_improvement(reward1, reward2, 0.001)
#             print(f"{name1} is {'significantly better' if significant else 'NOT significantly better'} than {name2}.")


# project.graphs.visualize_training(logdir, [environment_dir], False)

venv.close()
