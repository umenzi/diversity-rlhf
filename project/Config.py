from dataclasses import dataclass, field, asdict
from typing import Union, Dict

import torch as th


@dataclass
class EnvConfig:
    vectorized: bool = True
    render: bool = False


@dataclass
class PPOTrainConfig:
    """
    Config for training the expert PPO
    """

    # We train agents for a total of 1_000_000 steps
    env_steps: int = 1000  # we evaluate PPO after this many steps
    iterations: int = 10


@dataclass
class PPOConfig:
    """
    Global PPO config.

    Hyperparameters obtained with Optuna, initially inspired by: https://huggingface.co/sb3/ppo-LunarLander-v2
    """

    n_steps = 2048
    batch_size = 64
    clip_range = 0.1
    ent_coef = 0.01
    gamma = 0.999
    gae_lambda = 0.9
    n_epochs = 4
    learning_rate = 0.0003


@dataclass
class LunarPPOConfig(PPOConfig):
    # specific parameters for the Lunar Lander environment
    pass


@dataclass
class SpacePPOConfig(PPOConfig):
    # specific parameters for the Space Invaders environment
    pass


@dataclass
class AntPPOConfig(PPOConfig):
    # specific parameters for the Ant Environment
    pass


@dataclass
class RLHFConfig:
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


@dataclass
class LunarRLHFConfig(RLHFConfig):
    # specific parameters for the Lunar Lander environment
    pass


@dataclass
class SpaceRLHFConfig(RLHFConfig):
    # specific parameters for the Space Invaders environment
    pass


@dataclass
class AntRLHFConfig(RLHFConfig):
    # specific parameters for the Ant Environment
    pass


@dataclass
class Config:
    ENVIRONMENTS = ["lunar", "space", "ant"]

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo_train: PPOTrainConfig = field(default_factory=PPOTrainConfig)
    ppo: Dict[str, Union[LunarPPOConfig, SpacePPOConfig, AntPPOConfig]] = field(
        default_factory=lambda: {'lunar': LunarPPOConfig(), 'space': SpacePPOConfig(), 'ant': AntPPOConfig})
    rlhf: Dict[str, Union[LunarRLHFConfig, SpaceRLHFConfig, AntRLHFConfig]] = field(
        default_factory=lambda: {'lunar': LunarRLHFConfig(), 'space': SpaceRLHFConfig(), 'ant': AntRLHFConfig})

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    num_experiments = 3

    # Directory where the trained models (i.e. its weights) are stored
    models_dir = "models/"
    # Directory where the tensorboard logs are stored
    logdir = "logs/"

    def as_dict(self):
        return asdict(self)


CONFIG: Config = Config()
