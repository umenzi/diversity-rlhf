from dataclasses import dataclass, field, asdict
from typing import Union, Dict

import torch as th
from stable_baselines3.ppo import MlpPolicy, CnnPolicy


@dataclass
class EnvConfig:
    vectorized: bool = True
    render: bool = False
    training_n_envs: int = 2
    eval_n_envs: int = 1


@dataclass
class PPOTrainConfig:
    """
    Config for training the expert PPO
    """

    # We train agents for a total of 1_000_000 steps
    env_steps: int = 10_000  # we evaluate PPO after this many steps
    iterations: int = 100


@dataclass
class PPOConfig:
    """
    Global PPO config.

    Hyperparameters obtained with Optuna, initially inspired by: https://huggingface.co/sb3/ppo-LunarLander-v2
    """

    batch_size = 64
    clip_range = 0.1
    ent_coef = 0.01
    learning_rate = 0.0003
    gamma = 0.999
    gae_lambda = 0.9
    n_epochs = 4
    n_steps = 2048


@dataclass
class LunarPPOConfig(PPOConfig):
    # specific parameters for the Lunar Lander environment
    policy = MlpPolicy


@dataclass
class SpacePPOConfig(PPOConfig):
    # specific parameters for the Space Invaders environment
    policy = CnnPolicy
    batch_size = 256
    clip_range = 0.3
    ent_coef = 0.093
    learning_rate = 1.9e-05
    gae_lambda = 0.96
    gamma = 0.93
    n_epochs = 4
    n_steps = 128


@dataclass
class AntPPOConfig(PPOConfig):
    # specific parameters for the Ant Environment
    policy = MlpPolicy
    batch_size = 256
    clip_range = 0.2
    ent_coef = 0.038
    learning_rate = 7.55e-05
    gae_lambda = 0.88
    gamma = 0.95
    n_epochs = 10
    n_steps = 128


@dataclass
class RLHFConfig:
    total_timesteps = 200_000
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
class LunarRLHFConfig(RLHFConfig):
    # specific parameters for the Lunar Lander environment
    fragment_length = 97
    transition_oversampling = 1.7
    initial_comparison_frac = 0.32
    exploration_frac = 0.24
    temperature = 1.7
    discount_factor = 0.95


@dataclass
class SpaceRLHFConfig(RLHFConfig):
    # specific parameters for the Space Invaders environment
    fragment_length = 116
    transition_oversampling = 1.69
    initial_comparison_frac = 0.29
    exploration_frac = 0.06
    temperature = 0.22
    discount_factor = 0.96


@dataclass
class AntRLHFConfig(RLHFConfig):
    total_timesteps = 2_000_000
    total_comparisons = 1_000
    num_iterations = 100
    initial_epoch_multiplier = 6

    # specific parameters for the Ant Environment
    fragment_length = 1_000
    transition_oversampling = 1.5
    initial_comparison_frac = 0.68
    exploration_frac = 0.25
    temperature = 0.84
    discount_factor = 0.99


@dataclass
class Config:
    ENVIRONMENTS = ["LunarLander-v2", "SpaceInvaders-v4", 'Ant-v1']
    QUERY_STRATEGIES = ["random", "active"]

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo_train: PPOTrainConfig = field(default_factory=PPOTrainConfig)
    ppo: Dict[str, Union[LunarPPOConfig, SpacePPOConfig, AntPPOConfig]] = field(
        default_factory=lambda: {'LunarLander-v2': LunarPPOConfig(), 'SpaceInvaders-v4': SpacePPOConfig(),
                                 'Ant-v1': AntPPOConfig})
    rlhf: Dict[str, Union[LunarRLHFConfig, SpaceRLHFConfig, AntRLHFConfig]] = field(
        default_factory=lambda: {'LunarLander-v2': LunarRLHFConfig(), 'SpaceInvaders-v4': SpaceRLHFConfig(),
                                 'Ant-v1': AntRLHFConfig})

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
