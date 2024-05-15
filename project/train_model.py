import numpy as np
import torch as th
from imitation.rewards.reward_nets import (
    CnnRewardNet,
    ShapedRewardNet,
    cnn_transpose
)
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

import helpers
from project.environments import get_atari_env
from project.train_preference_comparisons import train_preference_comparisons

device = th.device("cuda" if th.cuda.is_available() else "cpu")

rng = np.random.default_rng()

venv = get_atari_env()

# Several hyperparameters have been fine-tuned to reach a reasonable level of performance.
# This is the agent that will generate the synthetic trajectories
agent = PPO(
    policy=CnnPolicy,  # we use a complex ANN (Convolutional Neural Networks)
    # given the env complexity and its dependency on images
    env=venv,
    seed=0,
    n_steps=16,  # To train on atari well, set this to 128
    batch_size=16,  # To train on atari well, set this to 256
    ent_coef=0.01,
    learning_rate=0.00025,
    n_epochs=4,
)

reward_net, main_trainer, results = train_preference_comparisons(venv, agent, 16, 15, 2, fragment_length=10,
                                                                 transition_oversampling=1, initial_comparison_frac=0.1,
                                                                 reward_trainer_epochs=3,
                                                                 allow_variable_horizon=False,
                                                                 initial_epoch_multiplier=1, rng=rng,
                                                                 reward_type=CnnRewardNet)

"""We can now wrap the environment with the learned reward model, shaped by the policy's learned value function. Note
that if we were training this for real, we would want to normalize the output of the reward net as well as the value
function, to ensure their values are on the same scale. To do this, use the `NormalizedRewardNet` class from
`src/imitation/rewards/reward_nets.py` on `reward_net`, and modify the potential to add a `RunningNorm` module from
`src/imitation/util/networks.py`."""


def value_potential(state):
    state_ = cnn_transpose(state)
    return agent.policy.predict_values(state_)


shaped_reward_net = ShapedRewardNet(
    base=reward_net,
    potential=value_potential,
    discount_factor=0.99,
)

# After training the reward network using the preference comparisons algorithm,
# we can wrap our environment with that learned reward.
# RewardVecEnvWrapper replaces the original reward function with the given argument.
# GOTCHA: When using the NormalizedRewardNet wrapper, you should deactivate updating
# during evaluation by passing update_stats=False to the predict_processed method.
learned_reward_venv = RewardVecEnvWrapper(venv, shaped_reward_net.predict_processed)

# We can wrap the reward function to standardize the output of the reward function, using
# its running mean and variance, which is useful for stabilizing training.
# When a reward network is saved, its wrappers are saved along with it,
# so that the normalization fit during reward learning can be used during future policy learning or evaluation.
# from imitation.rewards.reward_nets import NormalizedRewardNet
# from imitation.util.networks import RunningNorm
# train_reward_net = NormalizedRewardNet(
#     shaped_reward_net,
#     normalize_output_layer=RunningNorm,
# )

# We can easily save the reward networks, wrappers included:
# th.save(learned_reward_venv, rewards_dir + "reward_atari_model.pt")

# We train an agent that sees only the shaped, learned reward
learner = PPO(
    policy=CnnPolicy,
    env=learned_reward_venv,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
)

helpers.save_model(learner)

venv.close()
