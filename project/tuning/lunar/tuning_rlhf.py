import numpy as np
import optuna
import torch as th
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

import project.environments
from project import helpers
from project.train_preference_comparisons import get_preference_comparisons

rng = np.random.default_rng(0)
venv = project.environments.get_lunar_lander_env(16)
n_epochs = 4
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Train the main agent
expert = PPO(
    policy=MlpPolicy,
    env=venv,
    seed=0,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.999,
    gae_lambda=0.9,
    n_epochs=4,
    learning_rate=0.0003,
    clip_range=0.1,
    device=device,
)
expert.learn(total_timesteps=20_000)


def objective(trial: optuna.Trial):
    reward_net, agent_trainer, main_trainer = (
        get_preference_comparisons(venv, expert,
                                   num_iterations=1,
                                   # Set to 60 for better performance
                                   fragment_length=trial.suggest_int("fragment_length", 1, 150),
                                   transition_oversampling=trial.suggest_float("transition_oversampling", 0.9, 2.0),
                                   initial_comparison_frac=trial.suggest_float("initial_comparison_frac", 0.01, 1.0),
                                   reward_trainer_epochs=trial.suggest_int("reward_trainer_epochs", 1, 11),
                                   allow_variable_horizon=False,
                                   initial_epoch_multiplier=1,  # we don't tune this yet
                                   rng=rng,
                                   exploration_frac=trial.suggest_float("exploration_frac", 0.0, 0.5),
                                   discount_factor=trial.suggest_float("discount_factor", 0.95, 1.0),
                                   conflicting_prob=0.0,
                                   temperature=trial.suggest_float("temperature", 0.0, 2.0),
                                   reward_type=BasicRewardNet))

    accuracy = 0

    for epoch in range(n_epochs):  # Assuming you have 4 epochs as defined in n_epochs
        # We train the reward model based on the human feedback
        results = main_trainer.train(
            20_000 // n_epochs,
            100,
        )

        accuracy, _ = evaluate_policy(
            agent_trainer.algorithm.policy,
            venv,
            20,
        )

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
