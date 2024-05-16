import numpy as np

from project.helpers import print_reward_info
from project.train_preference_comparisons import train_preference_comparisons

from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.testing.reward_improvement import is_significant_reward_improvement
from imitation.util.util import make_vec_env

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

# Hyperparameters based on:
# https://github.com/HumanCompatibleAI/imitation/blob/a8b079c469bb145d1954814f22488adff944aa0d/tests/algorithms/test_preference_comparisons.py#L42

# Look into TensorBoard:
# https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html

rng = np.random.default_rng()

venv = make_vec_env("seals/CartPole-v0", rng=rng)

print("We start by comparing two normal agents, one is an expert and the other isn't")

print("We first train a good (but not perfect) expert")
expert = PPO(
    policy=MlpPolicy,
    env=venv,
    seed=0,
    batch_size=2,
    n_epochs=1,
    n_steps=10,
)
expert.learn(20_000, progress_bar=True)  # set to 100_000 for better performance

print("We then train a not-quite-expert")
not_expert = PPO(
    policy=MlpPolicy,
    env=venv,
    seed=0,
    batch_size=2,
    n_epochs=1,
    n_steps=10,
)
not_expert.learn(1_000, progress_bar=True)  # set to 10_000 for slightly better performance

print("We will perform a permutation test using the "
      "`is_significant_reward_improvement` function. We want to be very certain—let's set the bar high "
      "and require a p-value of 0.001. We make sure enough data is collected")

expert_rewards, _ = evaluate_policy(expert, venv, 100, return_episode_rewards=True)
not_expert_rewards, _ = evaluate_policy(
    not_expert, venv, 100, return_episode_rewards=True
)

print_reward_info(expert_rewards, not_expert_rewards, "Expert", "Not expert", 100)

significant = is_significant_reward_improvement(
    not_expert_rewards, expert_rewards, 0.001
)

print(
    f"The expert is {'NOT ' if not significant else ''}significantly better than the not-expert."
)

print("We can now be 99.9% confident that the expert is better than the not-expert -- in this specific case, with these"
      "specific trained models. It might still be an extraordinary stroke of luck, or a conspiracy to make us choose "
      "the wrong algorithm, but outside of that, we can be pretty sure our data's correct.")

print("We now train two RLHF agents, based on the previous 2 agents")

print("We train a good (but not perfect) expert")

expert_reward_net, expert_pref_comparisons, expert_result = (
    train_preference_comparisons(env=venv, agent=expert,
                                 total_timesteps=5_000,
                                 total_comparisons=200,
                                 num_iterations=2,
                                 fragment_length=2,
                                 transition_oversampling=2,
                                 initial_epoch_multiplier=2,
                                 rng=rng))

# expert_result["reward_loss"] (> 0.0), expert_result["reward_accuracy"] (> 0.0, <= 1.0)

expert_learner = PPO(
    policy=MlpPolicy,
    env=RewardVecEnvWrapper(venv, expert_reward_net.predict_processed),
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
)
expert_learner.learn(10_000)  # Use something bigger, like 100_000, for better performance

print("We train a not-quite-expert")

not_expert_reward_net, not_expert_pref_comparisons, not_expert_result = (
    train_preference_comparisons(env=venv, agent=not_expert,
                                 total_timesteps=5_000,
                                 total_comparisons=200,
                                 num_iterations=2,
                                 fragment_length=2,
                                 transition_oversampling=2,
                                 initial_epoch_multiplier=2,
                                 rng=rng))

# not_expert_result["reward_loss"] (> 0.0), not_expert_result["reward_accuracy"] (> 0.0, <= 1.0)

not_expert_learner = PPO(
    policy=MlpPolicy,
    env=RewardVecEnvWrapper(venv, not_expert_reward_net.predict_processed),
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
)
not_expert_learner.learn(10_000)  # Use something bigger, like 100_000, for better performance

print("We now compare the two RLHF agents, to see if the first agent (from the expert) "
      "is better than the second one (from the not-expert)")
venv.reset()

print("We will perform a permutation test using the "
      "`is_significant_reward_improvement` function. We want to be very certain—let's set the bar high "
      "and require a p-value of 0.001. We make sure enough data is collected")

expert_learner_rewards, _ = evaluate_policy(expert_learner.policy, venv, 100, return_episode_rewards=True)
not_expert_learner_rewards, _ = evaluate_policy(
    not_expert_learner.policy, venv, 100, return_episode_rewards=True
)

print_reward_info(expert_learner_rewards, not_expert_learner_rewards, "RLHF Expert", "RLHF Not expert", 100)

significant = is_significant_reward_improvement(
    not_expert_learner_rewards, expert_learner_rewards, 0.001
)

print(
    f"The expert is {'NOT ' if not significant else ''}significantly better than the not-expert."
)

print("We can now be 99.9% confident that the expert is better than the not-expert -- in this specific case, with these"
      "specific trained models. It might still be an extraordinary stroke of luck, or a conspiracy to make us choose "
      "the wrong algorithm, but outside of that, we can be pretty sure our data's correct.")

print("How about comparing the expert RLHF agent with the expert itself?")

expert_learner_clone_rewards, _ = evaluate_policy(expert_learner.policy, venv, 100, return_episode_rewards=True)
expert_rewards, _ = evaluate_policy(expert, venv, 100, return_episode_rewards=True)

print_reward_info(expert_learner_clone_rewards, expert_rewards, "RLHF Expert", "Expert", 100)

significant = is_significant_reward_improvement(
    expert_learner_clone_rewards, expert_rewards, 0.001
)

print(
    f"Expert is {'NOT ' if not significant else ''}significantly better than the RLHF expert."
)

print("Note that this evaluation is not perfect and highly limited. "
      "We’re comparing two algorithms on one environment, with one dataset. "
      "Ideally, to compare two algorithms you would pick a more complex environment, run this experiment several "
      "times, with different random seeds and perform some hyperparameter optimization to make sure we're not just "
      "using unlucky hyperparameters. At the end, we would also need you to run the same hypothesis test across "
      "average returns of several independent runs.")
