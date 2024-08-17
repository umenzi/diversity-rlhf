# The human factor: Addressing Diversity in RLHF

Code for Bachelor thesis, [_The Human Factor: Addressing Diversity in Reinforcement Learning from Human Feedback_](http://resolver.tudelft.nl/uuid:a7b37b44-4798-492e-822e-f1b7c347410b).

Abstract:

> Reinforcement Learning from Human Feedback (RLHF) is a promising approach to
training agents to perform complex tasks by incorporating human feedback. However,
the quality and diversity of this feedback can significantly impact the learning process.
Humans are highly diverse in their preferences, expertise, and capabilities. This paper
investigates the effects of conflicting feedback on the agent’s performance. We analyse
the impact of environmental complexity and examine various query selection strate-
gies. Our results show that RLHF performance rapidly degrades with even minimal
conflicting feedback in simple environments, and current query selection strategies are
ineffective in handling feedback diversity. We thus conclude that addressing diversity
is crucial for RLHF, suggesting alternative reward modelling approaches are needed.
Full code is available on GitHub. 

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

This repository allows training optimal RLHF agents for the `Pendulum`, `Lunar Lander`, and `Bipedal Walker` environments
given some conflicting probability. The agents can then be evaluated, plotting the average evaluating reward per episode
and conduct permutation tests to compare the agents' performance.

Relevant files:
- `train_preference_comparisons.py` include the main methods to get the RLHF agents and train them.
  - We also implement a custom `ConflictingSyntheticGatherer` class, which is used to generate conflicting preferences.
   There are also tests in the `tests` folder.
- `train.py` includes the training loop for the agents.
- `plot_results.py` plots the results of the training, stored in the `results` folder.
- `Config.py` includes the configuration for the training process. Hyperparameters, etc.
- `helpers.py`, `environments.py`, and `graphs.py` includes many auxiliary functions to help with the training and evaluation process.


### Logging into wandb

Note that the code logs the results in Weight & Biases, so you need to have an account and set up the API key.
This key should be stored in a `Constants.py` file in the root directory:

```python
API_WANDB_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
```

### Evaluating agents

The agents' performance is evaluated using csv files of their mean evaluating reward per episode.
These files are named after the environment and must be stored in the `results` folder.

The `plot_results.py` script can be used to plot the results of the agents.
The script will plot the average evaluating reward per episode and 
conduct permutation tests to compare the agents' performance.
It also includes helper methods to generate the csv files from wandb logs.

## Citation (BibTeX)

Please cite this repository if it was useful for your research:

```bibtex
@article{javi2024rlhf,
  title={The Human Factor: Addressing Diversity in Reinforcement Learning from Human Feedback},
  subtitle={How can RLHF deal with possibly conflicting feedback?},
  author={Paez Franco, Javier},
  year={2024},
  school={Delft University of Technology},
  type={Bachelor Thesis},
  url = {http://resolver.tudelft.nl/uuid:a7b37b44-4798-492e-822e-f1b7c347410b},
}
```
