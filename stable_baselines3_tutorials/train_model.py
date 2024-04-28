import gymnasium as gym
from stable_baselines3 import PPO
import os

# Create (if necessary) directory to store models and logs
model_name = "PPO"
models_dir = "models/" + model_name
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create environment
env = gym.make("LunarLander-v2", render_mode="rgb_array")
# Reset environment is necessary before stepping
env.reset()

# Create and train RL agent (using PPO)
# We use the MlpPolicy as our input of LunarLander-v2 is a feature vector, not images
# How to choose an RL algorithm: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10_000
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

# You can visualize how the training develops using tensorboard: tensorboard --logdir=logs
# It shows a GUI in your browser where you can see how the training progresses
# (e.g. rewards, loss, ...) of the different models you are training.
# If you specify different tb_log_name in subsequent runs, you will have split graphs.
# If you want to see them all in one graph, continuous, you should use the same tb_log_name
# More info about tensorboard: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html

# It is also possible to train the model in one line:
# model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)

# We can also easily delete a trained model in-code (it will still be saved in model_dir)
del model

env.close()
