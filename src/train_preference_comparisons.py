"""Train a reward model using preference comparisons."""

import pathlib
from typing import (
    Any,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import imitation.data.serialize as data_serialize
import imitation.policies.serialize as policies_serialize
import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.algorithms.preference_comparisons import PreferenceComparisons, PreferenceGatherer
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRewPair
from imitation.rewards.reward_nets import BasicRewardNet, RewardEnsemble
from imitation.scripts.config.train_preference_comparisons import (
    train_preference_comparisons_ex,
)
from imitation.util import logger as imit_logger
from scipy import special
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import VecEnv

from helpers import initialize_agent


def save_model(
        agent_trainer: preference_comparisons.AgentTrainer,
        save_path: pathlib.Path,
):
    """
    Save the model as `model.zip`.
    """

    policies_serialize.save_stable_model(
        output_dir=save_path / "policy",
        model=agent_trainer.algorithm,
    )


def save_checkpoint(
        trainer: preference_comparisons.PreferenceComparisons,
        save_path: pathlib.Path,
        allow_save_policy: Optional[bool],
):
    """
    Save reward model and optionally policy.
    """

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


class ConflictingSyntheticGatherer(PreferenceGatherer):
    """
    Computes synthetic preferences using ground-truth environment rewards.
    Allows creating conflicting preferences.
    """

    def __init__(
            self,
            conflicting_prob: float = 0,
            temperature: float = 1,
            discount_factor: float = 1,
            sample: bool = True,
            rng: Optional[np.random.Generator] = None,
            threshold: float = 50,
            custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        """
        Initialize the synthetic preference gatherer.

        Args:
            temperature: the preferences are sampled from a softmax, this is
                the temperature used for sampling.
                `temperature=0` leads to deterministic results (for equal rewards, 0.5 will be returned).
            conflicting_prob: probability of conflicting preferences.
                Conflicting means that the preferences are flipped. If 0, no conflicts (i.e. this would
                work the exact same as SyntheticGatherer). If 1, all preferences are conflicting.
            discount_factor: discount factor that is used to compute
                how good a fragment is.
                Default is to use undiscounted sums of rewards (as in the DRLHP paper).
            sample: if True (default), the preferences are zero or one, sampled from
                a Bernoulli's distribution (or 0.5 in the case of ties with zero
                temperature).
                If False, then the underlying Bernoulli probabilities
                are returned instead.
            rng: random number generator, only used if
                ``temperature > 0`` and ``sample=True``
            threshold: preferences are sampled from a softmax of returns.
                To avoid overflows, we clip differences in returns that are
                above this threshold (after multiplying with temperature).
                This threshold is therefore in logspace.
                The default value
                of 50 means that probabilities below 2e-22 are rounded up to 2e-22.
            custom_logger: Where to log to; if None (default), create a new logger.

        Raises:
            ValueError: if `sample` is true and no random state is provided.
        """

        super().__init__(custom_logger=custom_logger)
        self.conflicting_prob = conflicting_prob
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.sample = sample
        self.rng = rng
        self.threshold = threshold

        if self.sample and self.rng is None:
            raise ValueError("If `sample` is True, then `rng` must be provided.")

    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """
        Computes probability fragment 1 is preferred over fragment 2.
        """

        returns1, returns2 = self._reward_sums(fragment_pairs)
        if self.temperature == 0:
            if self.rng.random() < self.conflicting_prob:
                return 1 - ((np.sign(returns1 - returns2) + 1) / 2)
            else:
                return (np.sign(returns1 - returns2) + 1) / 2

        returns1 /= self.temperature
        returns2 /= self.temperature

        # clip the returns to avoid overflows in the softmax below
        returns_diff = np.clip(returns2 - returns1, -self.threshold, self.threshold)
        # Instead of computing exp(rews1) / (exp(rews1) + exp(rews2)) directly,
        # we divide enumerator and denominator by exp(rews1) to prevent overflows:
        model_probs = 1 / (1 + np.exp(returns_diff))

        # Make model_probs conflicting with probability `conflicting_prob`
        if self.rng.random() < self.conflicting_prob:
            model_probs = 1 - model_probs

        # Make model_probs conflicting with probability `conflicting_prob`
        # Compute the mean binary entropy.
        # This metric helps estimate how good we can expect the performance of the learned reward
        # model to be at predicting preferences.
        entropy = -(
                special.xlogy(model_probs, model_probs)
                + special.xlogy(1 - model_probs, 1 - model_probs)
        ).mean()
        self.logger.record("entropy", entropy)

        if self.sample:
            assert self.rng is not None
            return self.rng.binomial(n=1, p=model_probs, size=model_probs.shape).astype(np.float32)
        return model_probs

    def _reward_sums(self, fragment_pairs) -> Tuple[np.ndarray, np.ndarray]:
        rews1, rews2 = zip(
            *[
                (
                    rollout.discounted_sum(f1.rews, self.discount_factor),
                    rollout.discounted_sum(f2.rews, self.discount_factor),
                )
                for f1, f2 in fragment_pairs
            ],
        )
        return np.array(rews1, dtype=np.float32), np.array(rews2, dtype=np.float32)


@train_preference_comparisons_ex.main
def train_preference_comparisons(
        env: VecEnv,
        env_id: str,
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
        reward_type=None,
        conflicting_prob: float = 0,
        temperature: float = 1,
        discount_factor: float = 1,
        ensemble: bool = False,
        seed: int | None = None,
        tensorboard_log: str | None = None,
) -> tuple[BasicRewardNet, PreferenceComparisons, preference_comparisons.AgentTrainer, dict[str, Mapping[str, float] | Any] | Mapping[str, Any]]:
    """
    Train a reward model using preference comparisons.

    Args:
        env: the vectorized gym environment
        env_id: the id of the environment
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
        reward_type: The type of reward model to use (e.g. CnnRewardNet). If None, a BasicRewardNet
        conflicting_prob: probability of conflicting preferences.
                Conflicting means that the preferences are flipped. If 0, no conflicts (i.e. this would
                work the exact same as SyntheticGatherer). If 1, all preferences are conflicting.
        temperature: the preferences are sampled from a softmax, this is the temperature used for sampling.
        discount_factor: discount factor that is used to compute how good a fragment is.
        ensemble: whether to use an ensemble of reward models
        seed: random seed. If None, an unpredictable seed will be chosen.
        tensorboard_log: the directory to save the tensorboard logs

    Returns:
        Rollout statistics from trained policy.

    Raises:
        ValueError: Inconsistency between config and deserialized policy normalization.
    """

    logs_dir = "logs/"
    reward_net, agent_trainer, main_trainer = (
        get_preference_comparisons(env=env, env_id=env_id, agent=agent,
                                   num_iterations=num_iterations,
                                   comparison_queue_size=comparison_queue_size,
                                   trajectory_path=trajectory_path,
                                   reward_trainer_epochs=reward_trainer_epochs,
                                   fragment_length=fragment_length,
                                   transition_oversampling=transition_oversampling,
                                   initial_comparison_frac=initial_comparison_frac,
                                   initial_epoch_multiplier=initial_epoch_multiplier,
                                   exploration_frac=exploration_frac,
                                   active_selection=active_selection,
                                   active_selection_oversampling=active_selection_oversampling,
                                   uncertainty_on=uncertainty_on,
                                   allow_variable_horizon=allow_variable_horizon,
                                   query_schedule=query_schedule,
                                   reward_type=reward_type,
                                   conflicting_prob=conflicting_prob,
                                   temperature=temperature,
                                   discount_factor=discount_factor,
                                   ensemble=ensemble,
                                   seed=seed,
                                   tensorboard_log=tensorboard_log))

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

    if save_preferences:
        main_trainer.dataset.save(pathlib.Path(logs_dir + "preferences.pkl"))

    # Save final artifacts.
    if checkpoint_interval >= 0:
        save_checkpoint(
            trainer=main_trainer,
            save_path=pathlib.Path(logs_dir + "checkpoints/" + "final"),
            allow_save_policy=bool(trajectory_path is None),
        )

    return reward_net, main_trainer, agent_trainer, results


@train_preference_comparisons_ex.main
def get_preference_comparisons(env: VecEnv, env_id: str, agent, num_iterations: int,
                               comparison_queue_size: Optional[int] = None,
                               trajectory_path: Optional[str] = None, reward_trainer_epochs: int = 3,
                               fragment_length: int = 100, transition_oversampling: float = 1,
                               initial_comparison_frac: float = 0.1, initial_epoch_multiplier: float = 200.0,
                               exploration_frac: float = 0.0, active_selection: bool = False,
                               active_selection_oversampling: int = 2, uncertainty_on: str = "logit",
                               allow_variable_horizon: bool = False,
                               query_schedule: Union[str, type_aliases.Schedule] = "hyperbolic",
                               reward_type=None, conflicting_prob: float = 0, temperature: float = 1,
                               discount_factor: float = 1, ensemble: bool = False,
                               seed: int | None = None, tensorboard_log: str | None = None) -> \
        Tuple[BasicRewardNet, preference_comparisons.AgentTrainer, PreferenceComparisons]:
    """
    Creates an untrained reward model using preference comparisons.

    Args:
        env: the vectorized gym environment
        env_id: the id of the environment
        agent: the algorithm to use (e.g. PPO, DQN, etc.) If None, a PPO agent with the params in CONFIG will be used.
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
        query_schedule: one of ("constant", "hyperbolic", "inverse_quadratic").
            A function indicating how the total number of preference queries should
            be allocated to each iteration. "Hyperbolic" and "inverse_quadratic"
            apportion fewer queries to later iterations when the policy is assumed
            to be better and more stable.
        reward_type: The type of reward model to use (e.g. CnnRewardNet). If None, a BasicRewardNet
        conflicting_prob: probability of conflicting preferences.
                Conflicting means that the preferences are flipped. If 0, no conflicts (i.e. this would
                work the exact same as SyntheticGatherer). If 1, all preferences are conflicting.
        temperature: the preferences are sampled from a softmax, this is the temperature used for sampling.
        discount_factor: discount factor that is used to compute how good a fragment is.
        ensemble: whether to use an ensemble of reward models
        seed: random seed. If None, an unpredictable seed will be chosen.
        tensorboard_log: the directory to save the tensorboard logs

    Returns:
        The (untrained) reward network, agent trainer, and the main RHLF trainer

    Raises:
        ValueError: Inconsistency between config and deserialized policy normalization.
    """

    assert env is not None

    rng = np.random.default_rng(seed)

    logs_dir = "logs/"
    custom_logger = imit_logger.configure(logs_dir, ["csv", "tensorboard"])
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # A fragmenter is a class for creating pairs of trajectory fragments from a set of trajectories
    # The random fragmenter samples fragments of trajectories uniformly at random with replacement.
    fragmenter: preference_comparisons.Fragmenter = (
        preference_comparisons.RandomFragmenter(
            warning_threshold=0,
            rng=rng,
            custom_logger=custom_logger,
        )
    )

    # The conflicting synthetic gatherer computes synthetic preferences using ground-truth environment rewards.
    gatherer = ConflictingSyntheticGatherer(rng=rng, temperature=temperature, custom_logger=custom_logger,
                                            conflicting_prob=conflicting_prob, discount_factor=discount_factor)

    if reward_type is None:
        if ensemble:
            reward_net = RewardEnsemble(
                env.observation_space,
                env.action_space,
                members=[
                    BasicRewardNet(env.observation_space, env.action_space)
                    for _ in range(3)  # we use 3 reward ensembles
                ]
            ).to(device)
        else:
            reward_net = BasicRewardNet(
                env.observation_space, env.action_space
            ).to(device)
    else:
        if ensemble:
            reward_net = RewardEnsemble(
                env.observation_space,
                env.action_space,
                members=[
                    reward_type(env.observation_space, env.action_space)
                    for _ in range(3)  # we use 3 reward ensembles
                ]
            ).to(device)
        else:
            reward_net = reward_type(
                env.observation_space, env.action_space
            ).to(device)

    # A preference model is a model that converts two fragments' rewards into preference probability.
    preference_model = preference_comparisons.PreferenceModel(
        model=reward_net,
    ).to(device)

    if active_selection:
        fragmenter = preference_comparisons.ActiveSelectionFragmenter(
            preference_model=preference_model,
            base_fragmenter=fragmenter,
            fragment_sample_factor=active_selection_oversampling,
            uncertainty_on=uncertainty_on,
            custom_logger=custom_logger,
        )

    # reward trainer is a class that trains a basic reward model on the preferences.
    if ensemble:
        reward_trainer = preference_comparisons.EnsembleTrainer(
            preference_model=preference_model,
            loss=preference_comparisons.CrossEntropyRewardLoss(),
            rng=rng,
            epochs=reward_trainer_epochs,
            custom_logger=custom_logger,
        )
    else:
        reward_trainer = preference_comparisons.BasicRewardTrainer(
            preference_model=preference_model,
            loss=preference_comparisons.CrossEntropyRewardLoss(),
            rng=rng,
            epochs=reward_trainer_epochs,
            custom_logger=custom_logger,
        )

    if trajectory_path is None:
        if agent is None:
            agent = initialize_agent(env, seed, tensorboard_log, env_id)

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
    else:
        if exploration_frac > 0:
            raise ValueError(
                "exploration_frac can't be set when a trajectory dataset is used",
            )
        agent_trainer = preference_comparisons.TrajectoryDataset(
            trajectories=data_serialize.load_with_rewards(
                trajectory_path,
            ),
            rng=rng,
            custom_logger=custom_logger,
        )

    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
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

    return reward_net, agent_trainer, main_trainer
