import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from imitation.data.wrappers import RolloutInfoWrapper
from seals.util import AutoResetWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack


def get_atari_env():
    # Here we ensure that our environment has constant-length episodes by resetting
    # it when done, and running until 100 time steps have elapsed.
    # For real training, you will want a much longer time limit.
    def constant_length_asteroids(num_steps, render_mode="rgb_array"):
        atari_env = gym.make("AsteroidsNoFrameskip-v4", render_mode=render_mode)
        preprocessed_env = AtariWrapper(atari_env)
        endless_env = AutoResetWrapper(preprocessed_env)
        limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)

        return RolloutInfoWrapper(limited_env)

    # For real training, you will want a vectorized environment with 8 environments in parallel.
    # This can be done by passing in n_envs=8 as an argument to make_vec_env.
    # The seed needs to be set to 1 for reproducibility and also to avoid win32
    # np.random.randint high bound error.
    venv = make_vec_env(
        constant_length_asteroids,
        # env_kwargs={"num_steps": 100, "render_mode": "human"},
        env_kwargs={"num_steps": 100},
        seed=1
    )

    return VecFrameStack(venv, n_stack=4)


def get_lunar_lander_env(n_envs: int = 1):
    # Here we ensure that our environment has constant-length episodes by resetting
    # it when done, and running until 1000 time steps have elapsed.
    # For real training, you will want a much longer time limit.
    def constant_length_asteroids(num_steps, render_mode="rgb_array"):
        lunar_env = gym.make("LunarLander-v2", render_mode=render_mode)
        endless_env = AutoResetWrapper(lunar_env)
        limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)

        return RolloutInfoWrapper(limited_env)

    # For real training, you will want a vectorized environment with 8 environments in parallel.
    # This can be done by passing in n_envs=8 as an argument to make_vec_env.
    # The seed needs to be set to 1 for reproducibility and also to avoid win32
    # np.random.randint high bound error.
    return make_vec_env(
        constant_length_asteroids,
        # env_kwargs={"num_steps": 100, "render_mode": "human"},
        env_kwargs={"num_steps": 1000},
        seed=1,
        n_envs=n_envs,
    )
