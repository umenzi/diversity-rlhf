from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# We will use an atari wrapper.
# About Atari preprocessing: danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Save trained model
model.save("a2c_pong")

# Load thr agent, and then you can continue training

trained_model = A2C.load("a2c_pong", verbose=1)
env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
trained_model.set_env(env)

trained_model.learn(int(0.5e6))

trained_model.save("a2c_pong_2")