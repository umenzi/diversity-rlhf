import numpy as np

from imitation.data import rollout

import helpers

rng = np.random.default_rng()

# We load the environment
venv = helpers.get_atari_env()

# We load the reward environment
# learned_reward_venv = th.load(rewards_path)

# We can also load the reward network as a reward function for use in evaluation
# eval_rew_fn_normalized = load_reward(reward_type="RewardNet_normalized", reward_path=rewards_path, venv=venv)
# eval_rew_fn_unnormalized = load_reward(reward_type="RewardNet_unnormalized", reward_path=rewards_path, venv=venv)
# If we want to continue to update the reward networks normalization
# by default, it is frozen for evaluation and retraining
# rew_fn_normalized = load_reward(reward_type="RewardNet_normalized", reward_path=rewards_path, venv=venv,
#                                 update_stats=True)

# We load the agent previously trained
learner = helpers.load_model(venv)

# We run 10 "games" with the trained agent
helpers.evaluate(learner, num_episodes=10, deterministic=True, env_name="lunar")

"""When generating rollouts in image environments, be sure to use the agent's get_env() function rather than using
the original environment.

The learner re-arranges the observations' space to put the channel environment in the first dimension, and get_env()
will correctly provide a wrapped environment doing this."""

rollouts = rollout.rollout(
    learner,
    # Note that passing venv instead of agent.get_env()
    # here would fail.
    learner.get_env(),
    rollout.make_sample_until(min_timesteps=None, min_episodes=3),
    rng=rng,
)
# rollouts can be used, for example, to train behavioural cloning afterwards or something:
# transitions = rollout.flatten_trajectories(rollouts)

venv.close()
