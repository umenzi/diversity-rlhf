# The human factor: Addressing Diversity in RLHF

Code for a Bachelor thesis, where we aim to analyse the impact of diversity in RLHF.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

If you want to train the models, use `Lunar Lander`, `Ant`, or `Space Invaders`
to train the agents for the respective environments.

Relevant files:
- `train_preference_comparisons.py` include the main methods to obtain the RLHF agents and train.
  - We also implement a custom `ConflictingSyntheticGatherer` class, which is used to generate conflicting preferences.
   There are also tests in the `tests` folder.
- `train.py` includes the training loop for the agents.
- `plot_results.py` plots the results of the training, stored in the `results` folder.
- `Config.py` includes the configuration for the training process. Hyperparameters, etc.
- `helpers.py`, `environments.py`, and `graphs.py` includes many auxiliary functions to help with the training and evaluation process.