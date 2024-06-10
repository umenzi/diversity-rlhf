import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from imitation.data.wrappers import RolloutInfoWrapper
from seals.util import AutoResetWrapper
from stable_baselines3.common.env_util import make_vec_env

from Config import CONFIG


def get_environment(id: str, n_envs: int = 1, seed: int = 1):
    """
    Get the environment based on the id

    Args:
        id: of the environment
        n_envs: how many environments to run in parallel
        seed: seed for the environment

    Returns: the environment
    Raises: Exception if the environment could not be identified
    """

    if id not in CONFIG.ENVIRONMENTS:
        raise Exception("Environment could not be identified")

    # We generate the environment appropriately for imitation learning algorithms
    def constant_length(num_steps, render_mode="rgb_array"):
        env = gym.make(id, render_mode=render_mode)

        # We ensure that the environment has constant-length episodes by resetting it when done
        endless_env = AutoResetWrapper(env)
        limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)

        return RolloutInfoWrapper(limited_env)

    return make_vec_env(
        constant_length,
        env_kwargs={"num_steps": 3500},  # For real training, you will want a very high time limit (much more than 100).
        seed=seed,
        n_envs=n_envs,  # For real training, you will want several environments in parallel, like 8 or 16
    )
