import os
import re
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Series
from scipy import stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from Config import CONFIG


def visualize_training(logdir: str, environments: List[str], separate: bool = False,
                       selected_learners: Optional[List[str]] = None):
    """
    Visualize the training of multiple learners in the same plot.

    Args:
        logdir: (str) the directory where the tensorboard logs are stored
        environments: (list) list of environments to include in the plot
        separate: (bool) whether to plot the data of each learner separately or not
        selected_learners: (list) list of learners to include in the plot

    Returns: void
    """

    def parse_tensorboard_logs(logdir):
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()

        # Retrieve the scalars you are interested in
        rewards = event_acc.Scalars('rollout/ep_rew_mean')
        time = event_acc.Scalars('time/fps')

        # Convert to pandas DataFrame
        rewards_df = pd.DataFrame(rewards)
        time_df = pd.DataFrame(time)

        return rewards_df, time_df

    for environment in environments:
        env_logdir = os.path.join(environment, logdir)

        # Get a list of all directories in the log directory
        dirs = os.listdir(env_logdir)
        # Regular expression to match 'learner_' followed by two digits, an underscore, and one or more digits
        # and 'perfect_agent_' followed by one or more digits
        pattern_learner = re.compile(r'(learner_\d+)_(\d+)')
        pattern_perfect_agent = re.compile(r'(perfect_agent)_(\d+)')
        # Group the directories by the learner name
        learner_dirs = {}

        for dir in dirs:
            match_learner = pattern_learner.match(dir)
            match_perfect_agent = pattern_perfect_agent.match(dir)
            if match_learner:
                learner_name = match_learner.group(1)
                if learner_name not in learner_dirs:
                    learner_dirs[learner_name] = []
                learner_dirs[learner_name].append(dir)
            elif match_perfect_agent:
                learner_name = match_perfect_agent.group(1)
                if learner_name not in learner_dirs:
                    learner_dirs[learner_name] = []
                learner_dirs[learner_name].append(dir)

        # If selected_learners is specified, filter learner_dirs to only include the selected learners
        if selected_learners is not None:
            learner_dirs = {learner_name: dirs for learner_name, dirs in learner_dirs.items() if
                            learner_name in selected_learners}

        # Remove the "/" at the end of the environment name for matplotlib title
        title_environment = environment.rstrip('/')

        # Now you can use learner_dirs to parse the tensorboard logs and plot the data
        if separate:
            for name, dirs in learner_dirs.items():
                rewards_dfs = []
                time_dfs = []
                for dir in dirs:
                    rewards_df, time_df = parse_tensorboard_logs(os.path.join(env_logdir, dir))
                    rewards_dfs.append(rewards_df)
                    time_dfs.append(time_df)

                # Calculate the average rewards and time
                avg_rewards_df = pd.concat(rewards_dfs).groupby(level=0).mean()
                avg_time_df = pd.concat(time_dfs).groupby(level=0).mean()

                # Plot the data
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.plot(avg_rewards_df['step'], avg_rewards_df['value'])
                plt.title(f'Mean reward of {name} in {title_environment} environment')

                plt.subplot(1, 2, 2)
                plt.plot(avg_time_df['step'], avg_time_df['value'])
                plt.title(f'{name} Time')

                # Adjust spacing between the subplots
                plt.subplots_adjust(wspace=0.5)

                plt.show()
        else:
            plt.figure(figsize=(10, 5))

            for name, dirs in learner_dirs.items():
                rewards_dfs = []
                time_dfs = []
                for dir in dirs:
                    rewards_df, time_df = parse_tensorboard_logs(os.path.join(env_logdir, dir))
                    rewards_dfs.append(rewards_df)
                    time_dfs.append(time_df)

                # Calculate the average rewards and time
                avg_rewards_df = pd.concat(rewards_dfs).groupby(level=0).mean()
                avg_time_df = pd.concat(time_dfs).groupby(level=0).mean()

                plt.subplot(1, 2, 1)
                plt.plot(avg_rewards_df['step'], avg_rewards_df['value'], label=name)
                plt.title(f'Mean reward in {title_environment} environment')

                plt.subplot(1, 2, 2)
                plt.plot(avg_time_df['step'], avg_time_df['value'], label=name)
                plt.title(f'{name} Time')

            # Add a legend to the plot
            plt.legend()

        # Display the plot
        plt.show()


def plot_agent_rewards(results_folder, environment, agents, query_types):
    data = {}

    for agent in agents:
        if agent == "perfect_agent":
            rewards = []

            for run in range(3):
                column_name = f"{agent}_{run} - eval_mean_reward"
                df = pd.read_csv(f"{results_folder}/{environment}.csv")
                df = df.ffill()
                rewards.append(df[column_name])

            data[agent] = sum(rewards) / len(rewards)
        else:
            data[agent] = {}

            for query_type in query_types:
                rewards = []

                for run in range(3):
                    column_name = f"{agent}_{run}_{query_type} - eval_mean_reward"
                    df = pd.read_csv(f"{results_folder}/{environment}.csv")
                    df = df.ffill()
                    rewards.append(df[column_name])

                data[agent][query_type] = sum(rewards) / len(rewards)

    for query_type in query_types:
        plt.figure()  # Create a new figure for each query type
        for agent, query_data in data.items():
            query_data: Series  # This avoids PyCharm warning
            if agent == "perfect_agent":
                label = f"{agent}"
                query_data.plot(label=label)
            else:
                if query_type in query_data:
                    label = f"{agent}_{query_type}"
                    query_data[query_type].plot(label=label)

        plt.title(f'Rewards for {query_type} queries')  # Add title
        plt.xlabel('Episodes')  # Add x-axis label
        plt.ylabel('Rewards')  # Add y-axis label
        plt.subplots_adjust(left=0.17)
        plt.legend()
        plt.show()  # Display the current figure and start a new one


def merge_csv_files(directory, column_name, output_file_name):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter the list down to only CSV files
    csv_files = [file for file in files if file.endswith('.csv')]

    # Exclude files named after the environments
    csv_files = [file for file in csv_files if file[:-4] not in CONFIG.ENVIRONMENTS]

    # If no CSV files are found, return a message
    if not csv_files:
        return "No CSV files found in the directory."

    # Read the first CSV file into a DataFrame
    merged_df = pd.read_csv(os.path.join(directory, csv_files[0]))

    # Loop over the rest of the CSV files and merge them with the first DataFrame
    for file in csv_files[1:]:
        df = pd.read_csv(os.path.join(directory, file))
        merged_df = merged_df.merge(df, on=column_name)

    # Save the merged DataFrame to a new CSV file in the same directory
    merged_df.to_csv(path_or_buf=str(os.path.join(directory, output_file_name)), index=False)

    return merged_df


def check_and_create_merged_file(directory, column_name, output_file_name):
    # Check if the output file already exists
    output_file_path = os.path.join(directory, output_file_name)

    if os.path.exists(output_file_path):
        return f"The file {output_file_name} already exists in the directory."

    # If the output file does not exist, call the merge_csv_files function
    return merge_csv_files(directory, column_name, output_file_name)


def compare_agent_performance(agent_rewards, query_types):
    for query_type in query_types:
        print(f"\n{query_type}")

        # Initialize an empty DataFrame to store the results
        results = pd.DataFrame(columns=['Agent 1', 'Agent 2', 'P-value'])

        # Iterate over each pair of agents
        for agent1, agent2 in combinations(agent_rewards.keys(), 2):
            # Perform a permutation test to compare the performance of the two agents
            permutation_test_result = stats.permutation_test(
                (agent_rewards[agent2][query_type], agent_rewards[agent1][query_type]),
                statistic=lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
                vectorized=True,
                alternative="less",
            )

            # Store the result in the DataFrame
            results = pd.concat([results, pd.Series({
                'Agent 1': agent1,
                'Agent 2': agent2,
                'P-value': permutation_test_result.pvalue
            }, index=results.columns).to_frame().T], ignore_index=True)

        # Print the DataFrame
        print(results)


def analyse_performance(agents, query_types):
    """
    Analyse the performance of the agents.

    We conduct permutation tests based on the csv files. The csv file needs to be named after an
    environment from `CONFIG.ENVIRONMENTS` and stored in the 'results' folder. There is a "Step" column
    that represents the time steps. The other columns are the eval mean rewards, e.g.
    "learner_0_0_random - eval_mean_reward".

    Args:
        agents: List of agents to analyse
        query_types: List of query types to analyse
    """

    for environment in CONFIG.ENVIRONMENTS:
        print("Performance analysis for", environment)

        df = pd.read_csv(f"results/{environment}.csv")
        df = df.ffill()

        agent_rewards = {}

        for agent in agents:
            agent_rewards[agent] = {}

            for query_type in query_types:
                rewards = []

                for run in range(3):
                    column_name = f"{agent}_{run}_{query_type} - eval_mean_reward"
                    rewards.append(df[column_name].tail(20).values.tolist())

                average_rewards = [(x + y + z) / 3 for x, y, z in zip(rewards[0], rewards[1], rewards[2])]

                agent_rewards[agent][query_type] = average_rewards
        compare_agent_performance(agent_rewards, query_types)


def plot_average_rewards(agents, query_types):
    """
    Plot the average rewards for each agent.

    First, we check if the merged file exists. If it does not, we create it.
    Second, we plot the average rewards for each agent, based on the csv file.

    The csv file needs to be named after an environment from `CONFIG.ENVIRONMENTS` and stored in the 'results' folder.
    There is a "Step" column that represents the time steps. The other columns are the eval mean rewards, e.g.
    "learner_0_0_random - eval_mean_reward".

    Args:
        agents: List of agents to plot
        query_types: List of query types to plot
    """

    for environment in CONFIG.ENVIRONMENTS:
        # We make sure the csv exists, else we create it
        check_and_create_merged_file("results", "Step", f"{environment}.csv")

        # We plot the resulting csv file
        plot_agent_rewards("results", environment, agents, query_types)


if __name__ == "__main__":
    queries = ["random", "active"]
    agents_list = ["learner_0", "learner_25", "learner_40", "learner_50", "learner_75", "learner_100"]

    # Next, we plot the average rewards for each agent
    plot_average_rewards(agents_list, queries)

    # Finally, we compare the performance of the agents
    analyse_performance(agents_list, queries)
