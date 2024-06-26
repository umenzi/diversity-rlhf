import numpy as np
import optuna
import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import src.environments
from Config import CONFIG

rng = np.random.default_rng(0)
n_epochs = 3
device = th.device("cuda" if th.cuda.is_available() else "cpu")


def objective(trial: optuna.Trial):
    agent = PPO(
        policy=CONFIG.ppo[environment].policy,
        env=venv,
        seed=0,
        n_steps=trial.suggest_categorical("n_steps", [128, 512, 1024, 2048]),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512]),
        ent_coef=trial.suggest_float("ent_coef", 0.0,  0.1),
        gamma=trial.suggest_float("gamma", 0.9, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.8, 0.98, step=0.02),
        n_epochs=n_epochs,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.3, step=0.05),
        device=device,
    )

    accuracy = None

    for epoch in range(n_epochs):  # Assuming you have 4 epochs as defined in n_epochs
        agent.learn(total_timesteps=15_000 // n_epochs)  # Divide total_timesteps by number of epochs

        accuracy, _ = evaluate_policy(
            agent.policy,
            venv,
            n_eval_episodes=10,
        )

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    for environment in CONFIG.ENVIRONMENTS:
        venv = src.environments.get_environment(environment, 2, 0)

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
        pd.DataFrame([study.best_params]).to_csv(f"{environment}_ppo_best_params.csv", index=False)
