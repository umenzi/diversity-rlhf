"""Tests for the conflicting synthetic gatherer implementation."""

import imitation.testing.reward_nets as testing_reward_nets
import numpy as np
import pytest
import seals  # noqa: F401
import stable_baselines3
from imitation.algorithms import preference_comparisons
from imitation.rewards import reward_nets
from imitation.util import util

from project.train_preference_comparisons import ConflictingSyntheticGatherer

UNCERTAINTY_ON = ["logit", "probability", "label"]


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def venv(rng):
    return util.make_vec_env(
        "seals/CartPole-v0",
        n_envs=1,
        rng=rng,
    )


@pytest.fixture(
    params=[
        reward_nets.BasicRewardNet,
        testing_reward_nets.make_ensemble,
        lambda *args: reward_nets.AddSTDRewardWrapper(
            testing_reward_nets.make_ensemble(*args),
        ),
    ],
)
def reward_net(request, venv):
    return request.param(venv.observation_space, venv.action_space)


@pytest.fixture
def agent(venv):
    return stable_baselines3.PPO(
        "MlpPolicy",
        venv,
        n_epochs=1,
        batch_size=2,
        n_steps=10,
    )


@pytest.fixture
def random_fragmenter(rng):
    return preference_comparisons.RandomFragmenter(
        rng=rng,
        warning_threshold=0,
    )


@pytest.fixture
def agent_trainer(agent, reward_net, venv, rng):
    return preference_comparisons.AgentTrainer(agent, reward_net, venv, rng)


def assert_info_arrays_equal(arr1, arr2):  # pragma: no cover
    def check_possibly_nested_dicts_equal(dict1, dict2):
        for key, val1 in dict1.items():
            val2 = dict2[key]
            if isinstance(val1, dict):
                check_possibly_nested_dicts_equal(val1, val2)
            else:
                assert np.array_equal(val1, val2)

    for item1, item2 in zip(arr1, arr2):
        assert isinstance(item1, dict)
        assert isinstance(item2, dict)
        check_possibly_nested_dicts_equal(item1, item2)


def test_conflicting_synthetic_gatherer(
        agent_trainer,
        random_fragmenter,
        rng,
):
    # We generate the testing trajectories
    trajectories = agent_trainer.sample(10)
    fragments = random_fragmenter(trajectories, fragment_length=2, num_pairs=2)

    # We obtain the 'normal' feedback
    gatherer_1 = ConflictingSyntheticGatherer(
        temperature=0,
        rng=rng,
    )
    preferences1 = gatherer_1(fragments)
    preferences2 = gatherer_1(fragments)

    # We obtain the 'conflicting' feedback
    gatherer_2 = ConflictingSyntheticGatherer(
        conflicting_prob=1.0,
        temperature=0,
        rng=rng,
    )
    preferences3 = gatherer_2(fragments)
    preferences4 = gatherer_2(fragments)

    # With 0.0 or 1.0, the gatherer behaves like a deterministic gatherer
    assert np.all(preferences1 == preferences2)
    assert np.all(preferences3 == preferences4)
    # Where 0.0 and 1.0 are always conflicting
    assert np.all(preferences1 == 1 - preferences3)


def test_conflicting_synthetic_gatherer_conflict_prob(
        agent_trainer,
        random_fragmenter,
        rng,
):
    # We generate the testing trajectories
    # We need a greater number of pairs to ensure the conflict_prob can approximate
    trajectories = agent_trainer.sample(5000)
    fragments = random_fragmenter(trajectories, fragment_length=2, num_pairs=100000)

    # We obtain the 'normal' feedback
    gatherer_1 = ConflictingSyntheticGatherer(
        temperature=2,
        rng=rng,
    )
    preferences1 = gatherer_1(fragments)

    # We obtain the 'conflicting' feedback
    conflicting_prob = 0.4
    gatherer_2 = ConflictingSyntheticGatherer(
        conflicting_prob=conflicting_prob,
        temperature=2,
        rng=rng,
    )
    preferences2 = gatherer_2(fragments)

    # Calculate the actual proportion of conflicting preferences
    actual_conflicting_prob = np.mean(preferences1 != preferences2)

    # Check if the actual proportion is close to the expected proportion
    assert np.isclose(actual_conflicting_prob, conflicting_prob, atol=0.15)
