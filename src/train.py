import logging
import sys

import wandb
from imitation.rewards.reward_nets import (
    BasicRewardNet,
)
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

import Constants
import src.environments
import src.graphs
from Config import CONFIG
from helpers import initialize_agent
from src.train_preference_comparisons import train_preference_comparisons

# We log in to wandb using the API key
if not wandb.login(verify=True):
    wandb.login(key=Constants.API_WANDB_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)
# We add a handler, so it looks like a normal 'print' in the console
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def train_agent(agent: SelfBaseAlgorithm, agent_name, num_experiment=0):
    logging.info(f"Training {agent_name}")

    model_path = f"{environment_dir}{CONFIG.models_dir}{agent_name}/{num_experiment}/{CONFIG.ppo_train.env_steps}"

    # After every 10000 training steps, we evaluate the agent
    mean_reward, std_reward = evaluate_policy(agent.policy, eval_venv, n_eval_episodes=8)
    wandb.log({f"eval_mean_reward": mean_reward, f"eval_std_reward": std_reward})

    for j in range(1, CONFIG.ppo_train.iterations):  # set to 1_000_000 time steps in total for better performance
        logging.info(f"Training {agent_name} iteration {j}")

        agent.learn(total_timesteps=CONFIG.ppo_train.env_steps, reset_num_timesteps=False, tb_log_name=agent_name,
                    callback=WandbCallback(
                        gradient_save_freq=100,
                        model_save_freq=0,  # we do not save the model while training,
                        # only at the end after completing the training
                        model_save_path=None,
                        verbose=1,  # 0 = no output, 1 = info messages, "classic output", >= 2 = debug mode
                    ))

        logging.info(f"Evaluating {agent_name} iteration {j}")

        # After every 10000 training steps, we evaluate the agent
        mean_reward, std_reward = evaluate_policy(agent.policy, eval_venv, n_eval_episodes=8)
        wandb.log({f"eval_mean_reward": mean_reward, f"eval_std_reward": std_reward})

    agent.save(model_path)


def train_rlhf(env, env_id, agent, seed, tensorboard_log, query_strategy, conflicting_prob=0.0):
    return train_preference_comparisons(env, env_id, agent,
                                        total_timesteps=CONFIG.rlhf[env_id].total_timesteps,
                                        total_comparisons=CONFIG.rlhf[env_id].total_comparisons,
                                        num_iterations=CONFIG.rlhf[env_id].num_iterations,
                                        # Set to 60 for better performance
                                        fragment_length=CONFIG.rlhf[env_id].fragment_length,
                                        transition_oversampling=CONFIG.rlhf[env_id].transition_oversampling,
                                        initial_comparison_frac=CONFIG.rlhf[env_id].initial_comparison_frac,
                                        reward_trainer_epochs=CONFIG.rlhf[env_id].reward_trainer_epochs,
                                        allow_variable_horizon=CONFIG.rlhf[env_id].allow_variable_horizon,
                                        initial_epoch_multiplier=CONFIG.rlhf[env_id].initial_epoch_multiplier,
                                        seed=seed,
                                        exploration_frac=CONFIG.rlhf[env_id].exploration_frac,
                                        conflicting_prob=conflicting_prob,
                                        active_selection=query_strategy == "active",
                                        active_selection_oversampling=CONFIG.active_selection_oversampling,
                                        reward_type=BasicRewardNet,
                                        temperature=CONFIG.rlhf[env_id].temperature,
                                        discount_factor=CONFIG.rlhf[env_id].discount_factor,
                                        tensorboard_log=tensorboard_log)


# Run the experiments#
for query_strategy in CONFIG.QUERY_STRATEGIES:
    for env_id in CONFIG.ENVIRONMENTS:
        for i in range(CONFIG.num_experiments):
            # We change the random seed in every experiment
            seed = i * 10

            # Training environment
            venv = src.environments.get_environment(env_id, CONFIG.env.training_n_envs, seed=seed)
            # We use a separate environment for evaluation
            eval_venv = src.environments.get_environment(env_id, CONFIG.env.eval_n_envs, seed=seed)
            # We update the random number generators of PyTorch, Numpy,

            environment_dir = f"{env_id}/"

            tensorboard_log = environment_dir + CONFIG.logdir

            # Hyperparameters to log in wandb
            wandb_config = {
                "policy_type": "MlpPolicy",
                "total_timesteps": CONFIG.ppo_train.env_steps * CONFIG.ppo_train.iterations,
                "env_name": env_id,
            }

            run = wandb.init(
                project=env_id,
                name=f"perfect_agent_{i}_{query_strategy}",
                config=wandb_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=False,  # auto-upload the videos of agents playing the game
                save_code=False,  # optional
            )

            perfect_agent = initialize_agent(venv, seed, tensorboard_log, env_id)

            train_agent(perfect_agent, "perfect_agent", i)

            run.finish()

            (reward_net_0, main_trainer_0, results_0) = train_rlhf(venv, env_id, None, seed, tensorboard_log,
                                                                   query_strategy, conflicting_prob=0.0)

            run = wandb.init(
                project=env_id,
                name=f"learner_0_{i}_{query_strategy}",
                config=wandb_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=False,  # auto-upload the videos of agents playing the game
                save_code=False,  # optional
            )

            # We train an agent that sees only the shaped, learned reward
            learned_reward_venv_0 = RewardVecEnvWrapper(venv, reward_net_0.predict_processed)
            learner_0 = initialize_agent(learned_reward_venv_0, seed, tensorboard_log, env_id)
            train_agent(learner_0, "learner_0", i)

            run.finish()

            (reward_net_25, main_trainer_25, results_25) = train_rlhf(venv, env_id, None, seed, tensorboard_log,
                                                                      query_strategy, conflicting_prob=0.25)

            run = wandb.init(
                project=env_id,
                name=f"learner_25_{i}_{query_strategy}",
                config=wandb_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=False,  # auto-upload the videos of agents playing the game
                save_code=False,  # optional
            )

            # We train an agent that sees only the shaped, learned reward
            learned_reward_venv_25 = RewardVecEnvWrapper(venv, reward_net_25.predict_processed)
            learner_25 = initialize_agent(learned_reward_venv_25, seed, tensorboard_log, env_id)
            train_agent(learner_25, "learner_25", i)

            run.finish()

            (reward_net_40, main_trainer_40, results_40) = train_rlhf(venv, env_id, None, seed, tensorboard_log,
                                                                      query_strategy, conflicting_prob=0.40)

            run = wandb.init(
                project=env_id,
                name=f"learner_40_{i}_{query_strategy}",
                config=wandb_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=False,  # auto-upload the videos of agents playing the game
                save_code=False,  # optional
            )

            # We train an agent that sees only the shaped, learned reward
            learned_reward_venv_40 = RewardVecEnvWrapper(venv, reward_net_40.predict_processed)
            learner_40 = initialize_agent(learned_reward_venv_40, seed, tensorboard_log, env_id)
            train_agent(learner_40, "learner_40", i)

            run.finish()

            (reward_net_50, main_trainer_50, results_50) = train_rlhf(venv, env_id, None, seed, tensorboard_log,
                                                                      query_strategy, conflicting_prob=0.50)

            run = wandb.init(
                project=env_id,
                name=f"learner_50_{i}_{query_strategy}",
                config=wandb_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=False,  # auto-upload the videos of agents playing the game
                save_code=False,  # optional
            )

            # We train an agent that sees only the shaped, learned reward
            learned_reward_venv_50 = RewardVecEnvWrapper(venv, reward_net_50.predict_processed)
            learner_50 = initialize_agent(learned_reward_venv_50, seed, tensorboard_log, env_id)
            train_agent(learner_50, "learner_50", i)

            run.finish()

            (reward_net_75, main_trainer_75, results_75) = train_rlhf(venv, env_id, None, seed, tensorboard_log,
                                                                      query_strategy, conflicting_prob=0.75)

            run = wandb.init(
                project=env_id,
                name=f"learner_75_{i}_{query_strategy}",
                config=wandb_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=False,  # auto-upload the videos of agents playing the game
                save_code=False,  # optional
            )

            # We train an agent that sees only the shaped, learned reward
            learned_reward_venv_75 = RewardVecEnvWrapper(venv, reward_net_75.predict_processed)
            learner_75 = initialize_agent(learned_reward_venv_75, seed, tensorboard_log, env_id)
            train_agent(learner_75, "learner_75", i)

            run.finish()

            (reward_net_100, main_trainer_100, results_100) = train_rlhf(venv, env_id, None, seed, tensorboard_log,
                                                                         query_strategy, conflicting_prob=1.0)

            run = wandb.init(
                project=env_id,
                name=f"learner_100_{i}_{query_strategy}",
                config=wandb_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=False,  # auto-upload the videos of agents playing the game
                save_code=False,  # optional
            )

            # We train an agent that sees only the shaped, learned reward
            learned_reward_venv_100 = RewardVecEnvWrapper(venv, reward_net_100.predict_processed)
            learner_100 = initialize_agent(learned_reward_venv_100, seed, tensorboard_log, env_id)
            train_agent(learner_100, "learner_100", i)

            run.finish()
            venv.close()
            eval_venv.close()

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
# logging.info(df)

# Compare each learner with all other learners
# for name1, reward1 in results.items():
#     for name2, reward2 in results.items():
#         if name1 != name2:
#             significant = is_significant_reward_improvement(reward1, reward2, 0.001)
#             logging.info(f"{name1} is {'significantly better' if significant else 'NOT significantly better'} than {name2}.")


# src.graphs.visualize_training(CONFIG.logdir, [environment_dir], False)
