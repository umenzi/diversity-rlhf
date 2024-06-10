# The human factor: Addressing Diversity in RLHF

Code for Bachelor thesis, _The Human Factor: Addressing Diversity in Reinforcement Learning from Human Feedback_.

Abstract:

> Reinforcement Learning from Human Feedback (RLHF) is a promising approach to training agents to perform complex tasks 
> by incorporating human feedback. However, the learning process can be greatly impacted by the quality and diversity of
> this feedback. Humans are highly diverse in their preferences, expertise, and capabilities. In this paper, we 
> investigate the effects of conflicting feedback on the agent's performance, comparing it to an ideal agent that receives
> optimal feedback. Furthermore, we analyse the impact of environmental complexity on agent performance and examine
> various query selection strategies. Our results show that RLHF performance rapidly degrades with even minimal
> conflicting feedback, and current query selection strategies are ineffective in handling feedback diversity.
> We thus conclude that addressing diversity is crucial for RLHF, suggesting a need for alternative reward modelling approaches. 

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
- `train_preference_comparisons.py` include the main methods to obtain the RLHF agents and train.
  - We also implement a custom `ConflictingSyntheticGatherer` class, which is used to generate conflicting preferences.
   There are also tests in the `tests` folder.
- `train.py` includes the training loop for the agents.
- `plot_results.py` plots the results of the training, stored in the `results` folder.
- `Config.py` includes the configuration for the training process. Hyperparameters, etc.
- `helpers.py`, `environments.py`, and `graphs.py` includes many auxiliary functions to help with the training and evaluation process.