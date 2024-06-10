from dataclasses import dataclass, field, asdict
from typing import Union, Dict

import torch as th
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.util.networks import RunningNorm
from stable_baselines3.ppo import MlpPolicy


@dataclass
class EnvConfig:
    vectorized: bool = True
    render: bool = False
    training_n_envs: int = 8
    eval_n_envs: int = 1


@dataclass
class PPOConfig:
    """
    Global PPO config.

    Hyperparameters obtained with Optuna, initially inspired by: https://huggingface.co/sb3/ppo-LunarLander-v2
    """
    # We train agents for a total of 1_000_000 steps
    env_steps: int = 10_000  # we evaluate PPO after this many steps
    iterations: int = 100

    policy_kwargs = dict(),
    batch_size = 64
    clip_range = 0.1
    ent_coef = 0.0
    learning_rate = 0.0003
    gamma = 0.999
    gae_lambda = 0.99
    n_epochs = 10
    n_steps = 2048


@dataclass
class PendulumPPOConfig(PPOConfig):
    # specific parameters for the Pendulum environment
    # https://huggingface.co/sb3/ppo-Pendulum-v1
    policy = FeedForward32Policy
    policy_kwargs = dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    ),

    env_steps = 1_000
    clip_range = 0.2
    ent_coef = 0.01
    learning_rate = 0.001
    gae_lambda = 0.95
    gamma = 0.91
    n_epochs = 10
    n_steps = 1024


@dataclass
class LunarPPOConfig(PPOConfig):
    # specific parameters for the Lunar Lander environment
    # https://huggingface.co/sb3/ppo-LunarLander-v2
    policy = MlpPolicy
    ent_coef = 0.01
    n_epochs = 4
    gae_lambda = 0.9


@dataclass
class WalkerPPOConfig(PPOConfig):
    # specific parameters for the Bipedal Walker Environment
    # https://huggingface.co/sb3/ppo-BipedalWalker-v3
    policy = MlpPolicy
    env_steps = 20_000
    batch_size = 64
    clip_range = 0.18
    learning_rate = 0.0003
    gae_lambda = 0.95
    gamma = 0.999
    n_epochs = 10
    n_steps = 2048


@dataclass
class RLHFConfig:
    total_timesteps = 500_000
    total_comparisons = 500
    num_iterations = 60

    fragment_length = 100
    transition_oversampling = 1
    initial_comparison_frac = 0.1
    reward_trainer_epochs = 4
    allow_variable_horizon = False
    initial_epoch_multiplier = 4
    exploration_frac = 0.05
    temperature = 1
    discount_factor = 0.99


@dataclass
class PendulumRLHFConfig(RLHFConfig):
    # specific parameters for the Pendulum environment
    reward_trainer_epochs = 3
    transition_oversampling = 1
    initial_comparison_frac = 0.1
    exploration_frac = 0.05
    temperature = 0.22
    discount_factor = 1


@dataclass
class LunarRLHFConfig(RLHFConfig):
    # specific parameters for the Lunar Lander environment
    fragment_length = 97
    transition_oversampling = 1.7
    initial_comparison_frac = 0.32
    exploration_frac = 0.24
    temperature = 1.7
    discount_factor = 0.95


@dataclass
class WalkerRLHFConfig(RLHFConfig):
    # specific parameters for the Bipedal Walker Environment
    total_timesteps = 2_000_000
    total_comparisons = 700

    fragment_length = 100
    transition_oversampling = 1.7
    initial_comparison_frac = 0.32
    exploration_frac = 0.25
    temperature = 1.8
    discount_factor = 0.96


@dataclass
class Config:
    ENVIRONMENTS = ["Pendulum-v1", "LunarLander-v2", "BipedalWalker-v3"]
    QUERY_STRATEGIES = ["random", "active"]

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: Dict[str, Union[PendulumPPOConfig, LunarPPOConfig, WalkerPPOConfig]] = field(
        default_factory=lambda: {"Pendulum-v1": PendulumPPOConfig(), "LunarLander-v2": LunarPPOConfig(),
                                 "BipedalWalker-v3": WalkerPPOConfig})
    rlhf: Dict[str, Union[PendulumRLHFConfig, LunarRLHFConfig, WalkerRLHFConfig]] = field(
        default_factory=lambda: {"Pendulum-v1": PendulumRLHFConfig(), "LunarLander-v2": LunarRLHFConfig(),
                                 "BipedalWalker-v3": WalkerRLHFConfig})

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    num_experiments = 3

    active_selection_oversampling = 5

    # Directory where the trained models (i.e. its weights) are stored
    models_dir = "models/"
    # Directory where the tensorboard logs are stored
    logdir = "logs/"

    def as_dict(self):
        return asdict(self)


# This is the Config object that should be used throughout the code
CONFIG: Config = Config()
