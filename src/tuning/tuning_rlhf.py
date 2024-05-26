import optuna
import pandas as pd
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.evaluation import evaluate_policy

from Config import CONFIG

import src.environments
from src.train_preference_comparisons import get_preference_comparisons

n_epochs = 3

def objective(trial: optuna.Trial):
    reward_net, agent_trainer, main_trainer = (
        get_preference_comparisons(venv, env_id, None, num_iterations=1, seed=0,
                                   reward_trainer_epochs=1,
                                   fragment_length=trial.suggest_int("fragment_length", 80, 150),
                                   transition_oversampling=trial.suggest_float("transition_oversampling", 0.9, 2.0),
                                   initial_comparison_frac=trial.suggest_float("initial_comparison_frac", 0.01, 1.0),
                                   initial_epoch_multiplier=1, reward_type=BasicRewardNet, allow_variable_horizon=False,
                                   exploration_frac=trial.suggest_float("exploration_frac", 0.0, 0.5),
                                   conflicting_prob=0.0, temperature=trial.suggest_float("temperature", 0.0, 2.0),
                                   discount_factor=trial.suggest_float("discount_factor", 0.95, 1.0)))

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
            10,
        )

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    for env_id in CONFIG.ENVIRONMENTS:
        venv = src.environments.get_environment(env_id, 2, 0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

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

        # Save the result to a CSV file
        pd.DataFrame([study.best_params]).to_csv(f"{env_id}_rlhf_best_params.csv", index=False)
