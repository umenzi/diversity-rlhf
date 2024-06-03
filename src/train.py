import logging
import sys

import wandb
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

import Constants
import src.environments
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

    model_path = f"{environment_dir}{CONFIG.models_dir}{agent_name}/{num_experiment}/{CONFIG.ppo[env_id].env_steps}"

    # After every 10000 training steps, we evaluate the agent
    mean_reward, std_reward = evaluate_policy(agent.policy, eval_venv, n_eval_episodes=8)
    wandb.log({f"eval_mean_reward": mean_reward, f"eval_std_reward": std_reward})

    for j in range(1, CONFIG.ppo[env_id].iterations):  # set to 1_000_000 time steps in total for better performance
        logging.info(f"Training {agent_name} iteration {j}")

        agent.learn(total_timesteps=CONFIG.ppo[env_id].env_steps, reset_num_timesteps=False, tb_log_name=agent_name,
                    callback=WandbCallback(
                        gradient_save_freq=100,
                        model_save_freq=0,  # we do not save the model while training,
                        # only at the end after completing the training
                        model_save_path=None,
                        verbose=1,  # 0 = no output, 1 = info messages, "classic output", >= 2 = debug mode
                    ))

        logging.info(f"Evaluating {agent_name} iteration {j}")

        # After every 10000 training steps, we evaluate the agent
        mean_reward, std_reward = evaluate_policy(agent.policy, eval_venv, n_eval_episodes=10)
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
                                        ensemble=query_strategy == "active",
                                        active_selection_oversampling=CONFIG.active_selection_oversampling,
                                        temperature=CONFIG.rlhf[env_id].temperature,
                                        discount_factor=CONFIG.rlhf[env_id].discount_factor,
                                        tensorboard_log=tensorboard_log)


# Run the experiments
for query_strategy in CONFIG.QUERY_STRATEGIES:
    for env_id in CONFIG.ENVIRONMENTS:
        for i in range(CONFIG.num_experiments):
            # We change the random seed in every experiment
            seed = i * 10

            # Training environment
            venv = src.environments.get_environment(env_id, CONFIG.env.training_n_envs, seed=seed)
            # We use a separate environment for evaluation
            eval_venv = src.environments.get_environment(env_id, CONFIG.env.eval_n_envs, seed=seed)

            # We set up the environment directory and the log directory
            environment_dir = f"{env_id}/"
            tensorboard_log = environment_dir + CONFIG.logdir

            # Hyperparameters to log in wandb
            wandb_config = {
                "policy_type": "MlpPolicy",
                "total_timesteps": CONFIG.ppo[env_id].env_steps * CONFIG.ppo[env_id].iterations,
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

            # RLHF agents with their corresponding diversity levels
            data = [["learner_0", 0.0], ["learner_25", 0.25], ["learner_40", 0.40],
                    ["learner_50", 0.50], ["learner_75", 0.75], ["learner_100", 1.0]]

            for learner_id, conflicting_prob in data:
                (reward_net, main_trainer, agent_trainer, results) = train_rlhf(venv, env_id, None, seed,
                                                                                tensorboard_log, query_strategy,
                                                                                conflicting_prob=conflicting_prob)

                run = wandb.init(
                    project=env_id,
                    name=f"{learner_id}_{i}_{query_strategy}",
                    config=wandb_config,
                    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                    monitor_gym=False,  # auto-upload the videos of agents playing the game
                    save_code=False,  # optional
                )

                # We train an agent that sees only the shaped, learned reward
                learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)

                learner = initialize_agent(learned_reward_venv, seed, tensorboard_log, env_id)
                train_agent(learner, learner_id, i)

                run.finish()

            venv.close()
            eval_venv.close()
