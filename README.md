# The human factor: Addressing Diversity in RLHF

Code for a Bachelor thesis, where we aim to analyse the impact of diversity in RLHF.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

If you want to train the models, use `lunar_lander`, 
`ant`, or `atari` to train the agents for the respective environments.

`train_preference_comparisons` include the main methods to obtain the RLHF agents and train.
`helpers` includes many auxiliary functions to help with the training process.

We also implement a custom `ConflictingSyntheticGatherer` class, which is used to generate conflicting preferences.
It can be found in `train_preference_comparisons.py`. There are also tests in the corresponding folder.