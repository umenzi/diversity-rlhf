"""Train a reward model using preference comparisons."""

import pathlib
from typing import Any, Mapping, Optional, Union

import imitation.data.serialize as data_serialize
import imitation.policies.serialize as policies_serialize
import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.algorithms.preference_comparisons import PreferenceComparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.scripts.config.train_preference_comparisons import (
    train_preference_comparisons_ex,
)
from imitation.util import logger as imit_logger
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import VecEnv


def save_model(
        agent_trainer: preference_comparisons.AgentTrainer,
        save_path: pathlib.Path,
):
    """Save the model as `model.zip`."""
    policies_serialize.save_stable_model(
        output_dir=save_path / "policy",
        model=agent_trainer.algorithm,
    )


def save_checkpoint(
        trainer: preference_comparisons.PreferenceComparisons,
        save_path: pathlib.Path,
        allow_save_policy: Optional[bool],
):
    """Save reward model and optionally policy."""
    save_path.mkdir(parents=True, exist_ok=True)
    th.save(trainer.model, save_path / "reward_net.pt")
    if allow_save_policy:
        # Note: We should only save the model as model.zip if `trajectory_generator`
        # contains one. Currently, we are slightly over-conservative by requiring
        # that an AgentTrainer be used if we're saving the policy.
        assert isinstance(
            trainer.trajectory_generator,
            preference_comparisons.AgentTrainer,
        )
        save_model(trainer.trajectory_generator, save_path)
    else:
        trainer.logger.warn(
            "trainer.trajectory_generator doesn't contain a policy to save.",
        )


@train_preference_comparisons_ex.main
def train_preference_comparisons(
        env: VecEnv,
        agent,
        total_timesteps: int,
        total_comparisons: int,
        num_iterations: int,
        comparison_queue_size: Optional[int] = None,
        trajectory_path: Optional[str] = None,
        reward_trainer_epochs: int = 3,
        fragment_length: int = 100,
        transition_oversampling: float = 1,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        exploration_frac: float = 0.0,
        save_preferences: bool = False,
        active_selection: bool = False,
        active_selection_oversampling: int = 2,
        uncertainty_on: str = "logit",
        allow_variable_horizon: bool = False,
        checkpoint_interval: int = 0,
        query_schedule: Union[str, type_aliases.Schedule] = "hyperbolic",
        rng: np.random.Generator = None,
        n_episodes_eval: int = 10,
        reward_type=None,
) -> tuple[BasicRewardNet, PreferenceComparisons, dict[str, Mapping[str, float] | Any] | Mapping[str, Any]]:
    """Train a reward model using preference comparisons.

    Args:
        env: the vectorized gym environment
        agent: the algorithm to use (e.g. PPO, DQN, etc.)
        total_timesteps: number of environment interaction steps
        total_comparisons: number of preferences to gather in total
        num_iterations: number of times to train the agent against the reward model
            and then train the reward model against newly gathered preferences.
        comparison_queue_size: the maximum number of comparisons to keep in the
            queue for training the reward model. If None, the queue will grow
            without bound as new comparisons are added.
        trajectory_path: either None (default), in which case an agent will be trained
            and used to sample trajectories on the fly, or a path to a pickled
            sequence of TrajectoryWithRew to be trained on.
        reward_trainer_epochs: number of epochs to train the reward model on each batch
        fragment_length: number of timesteps per fragment that is used to elicit
            preferences
        transition_oversampling: factor by which to oversample transitions before
            creating fragments. Since fragments are sampled with replacement,
            this is usually chosen > 1 to avoid having the same transition
            in too many fragments.
        initial_comparison_frac: fraction of total_comparisons that will be
            sampled before the rest of training begins (using the randomly initialized
            agent). This can be used to pretrain the reward model before the agent
            is trained on the learned reward.
        initial_epoch_multiplier: before agent training begins, train the reward
                model for this many more epochs than usual (on fragments sampled from a
                random agent).
        exploration_frac: fraction of trajectory samples that will be created using
            partially random actions, rather than the current policy. Might be helpful
            if the learned policy explores too little and gets stuck with a wrong
            reward.
        save_preferences: if True, store the final dataset of preferences to disk.
        active_selection: use active selection fragmenter instead of random fragmenter
        active_selection_oversampling: factor by which to oversample random fragments
            from the base fragmenter of active selection.
            This is usually chosen > 1 to allow the active selection algorithm to pick
            fragment pairs with the highest uncertainty. = 1 implies no active selection.
        uncertainty_on: passed to ActiveSelectionFragmenter
        allow_variable_horizon: If False (default), algorithm will raise an
            exception if it detects trajectories of different length during
            training. If True, overrides this safety check. WARNING: variable
            horizon episodes leak information about the reward via termination
            condition, and can seriously confound evaluation. Read
            https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
            before overriding this.
        checkpoint_interval: Save the reward model and policy models (if
            trajectory_generator contains a policy) every `checkpoint_interval`
            iterations and after training is complete. If 0, then only save weights
            after training is complete. If <0, then don't save weights at all.
        query_schedule: one of ("constant", "hyperbolic", "inverse_quadratic").
            A function indicating how the total number of preference queries should
            be allocated to each iteration. "Hyperbolic" and "inverse_quadratic"
            apportion fewer queries to later iterations when the policy is assumed
            to be better and more stable.
        rng: the random number generator (if None, a random one will be created)
        n_episodes_eval: The number of episodes to average over when calculating
            the average episode reward of the imitation policy for return. Only relevant
            if trajectory_path is not None.
        reward_type: The type of reward model to use (e.g. CnnRewardNet). If None, a BasicRewardNet

    Returns:
        Rollout statistics from trained policy.

    Raises:
        ValueError: Inconsistency between config and deserialized policy normalization.
    """
    assert env is not None
    assert agent is not None

    if rng is None:
        rng = np.random.default_rng()

    logs_dir = "logs/"
    custom_logger = imit_logger.configure(logs_dir)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # A fragmenter is a class for creating pairs of trajectory fragments from a set of trajectories
    # The random fragmenter samples fragments of trajectories uniformly at random with replacement.
    fragmenter: preference_comparisons.Fragmenter = (
        preference_comparisons.RandomFragmenter(
            warning_threshold=0,
            rng=rng,
        )
    )
    # The synthetic gatherer computes synthetic preferences using ground-truth environment rewards.
    gatherer = preference_comparisons.SyntheticGatherer(rng=rng, custom_logger=custom_logger)

    if reward_type is None:
        reward_net = BasicRewardNet(
            env.observation_space, env.action_space
        ).to(device)
    else:
        reward_net = reward_type(
            env.observation_space, env.action_space
        ).to(device)
    # A preference model is a model that converts two fragments' rewards into preference probability.
    preference_model = preference_comparisons.PreferenceModel(
        model=reward_net,
    )

    if active_selection:
        fragmenter = preference_comparisons.ActiveSelectionFragmenter(
            preference_model=preference_model,
            base_fragmenter=fragmenter,
            fragment_sample_factor=active_selection_oversampling,
            uncertainty_on=uncertainty_on,
            custom_logger=custom_logger,
        )

    # reward trainer is a class that trains a basic reward model on the preferences.
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=preference_model,
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        rng=rng,
        epochs=reward_trainer_epochs,
    )

    if trajectory_path is None:
        # Setting the logger here is not necessary (PreferenceComparisons takes care
        # of it automatically), but it avoids creating unnecessary loggers.
        agent_trainer = preference_comparisons.AgentTrainer(
            algorithm=agent,
            reward_fn=reward_net,
            venv=env,
            exploration_frac=exploration_frac,
            rng=rng,
            custom_logger=custom_logger,
        )
        # Stable Baselines will automatically occupy GPU 0 if it is available.
        # Let's use the same device as the SB3 agent for the reward model.
        reward_net = reward_net.to(agent_trainer.algorithm.device)
        trajectory_generator: preference_comparisons.TrajectoryGenerator = (
            agent_trainer
        )
    else:
        if exploration_frac > 0:
            raise ValueError(
                "exploration_frac can't be set when a trajectory dataset is used",
            )
        trajectory_generator = preference_comparisons.TrajectoryDataset(
            trajectories=data_serialize.load_with_rewards(
                trajectory_path,
            ),
            rng=rng,
            custom_logger=custom_logger,
        )

    main_trainer = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,  # reward model
        num_iterations=num_iterations,  # Increase for better performance
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        comparison_queue_size=comparison_queue_size,
        fragment_length=fragment_length,
        transition_oversampling=transition_oversampling,
        initial_comparison_frac=initial_comparison_frac,
        allow_variable_horizon=allow_variable_horizon,
        query_schedule=query_schedule,
        initial_epoch_multiplier=initial_epoch_multiplier,
        custom_logger=custom_logger,
    )

    def save_callback(iteration_num):
        if checkpoint_interval > 0 and iteration_num % checkpoint_interval == 0:
            save_checkpoint(
                trainer=main_trainer,
                save_path=pathlib.Path(logs_dir + "checkpoints/" + f"{iteration_num:04d}"),
                allow_save_policy=bool(trajectory_path is None),
            )

    # We train the reward model based on the human feedback
    results = main_trainer.train(
        total_timesteps,
        total_comparisons,
        callback=save_callback,
    )

    # Storing and evaluating policy only useful if we generated trajectory data
    # This part of the code seems to not work
    #
    # if bool(trajectory_path is None):
    #     results = dict(results)
    #     results["imit_stats"] = policy_evaluation.eval_policy(agent, n_episodes_eval, env)

    if save_preferences:
        main_trainer.dataset.save(pathlib.Path(logs_dir + "preferences.pkl"))

    # Save final artifacts.
    if checkpoint_interval >= 0:
        save_checkpoint(
            trainer=main_trainer,
            save_path=pathlib.Path(logs_dir + "checkpoints/" + "final"),
            allow_save_policy=bool(trajectory_path is None),
        )

    return reward_net, main_trainer, results
