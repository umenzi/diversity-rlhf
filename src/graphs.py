import os
import re
from typing import List, Optional

import pandas as pd
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


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


def process_and_plot(
        df, run_groups, group_names,
        title: str,
        smoothing_window=5,
        vertical_lines=None,
):
    """
    Process and plot the data.

    Args:
        df: the dataframe containing the data
        run_groups: list of lists, each containing the run ids of the group
        group_names: list of names for each group
        title: the title of the plot
        smoothing_window: the window size for smoothing
        vertical_lines: list of vertical lines to plot

    Returns: the processed dataframe and the processed columns
    """

    if vertical_lines is None:
        vertical_lines = []
    processed_columns = []
    means = []
    stds = []

    # Process each group
    for i, group in enumerate(run_groups):
        group_columns = []

        # Filter group columns
        for c in df.columns:
            for run_id in group:
                if str(run_id) in c:
                    group_columns.append(c)

        assert len(group_columns) == len(group), f'Could not find all runs in group {group}'

        # Smooth group columns
        for c in group_columns:
            df[c] = df[c].rolling(smoothing_window).mean()

        # Compute mean and standard deviation
        mean_col = f'group_{i}_mean'
        std_col = f'group_{i}_std'
        df[mean_col] = df[group_columns].mean(axis=1)
        df[std_col] = df[group_columns].std(axis=1)

        # Append processed columns and names
        processed_columns.append(group_columns)
        means.append(mean_col)
        stds.append(std_col)

    # Plot
    fig, ax = plt.subplots()
    fig.set_dpi(200)
    fig.set_size_inches(8, 6)

    for mean, std, name in zip(means, stds, group_names):
        ax.plot(df['Step'], df[mean], label=f'{name}')
        ax.fill_between(df['Step'], df[mean] - df[std], df[mean] + df[std], alpha=0.5)

    if len(vertical_lines) > 1:
        for i, line in enumerate(vertical_lines):
            ax.axvline(line, color=default_colors[i], linestyle='--')
    elif len(vertical_lines) > 0:
        ax.axvline(vertical_lines[0], color=default_colors[1], linestyle='--')

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Training step')
    ax.set_ylabel('True Returns')

    plt.show()

    return df, processed_columns


def diff_swap_point():
    """
    Graph of the true returns as models train.
    Dotted vertical lines of the corresponding color show when the configuration is switched.
    Running average smoothing is applied to improve graph readability.
    """
    df = pd.read_csv('../graph_data/different_swap_point.csv')
    title = 'Returns for different values of x in Equation 2'

    lines = []
    run_groups = []
    group_names = []

    # ------
    run_groups.append([214, 215, 216])
    lines.append(100000)
    group_names.append('x = 100000')

    run_groups.append([211, 212, 213])
    lines.append(150000)
    group_names.append('x = 150000')

    run_groups.append([206, 208, 209])
    lines.append(200000)
    group_names.append('x = 200000')

    run_groups.append([194, 195, 217])
    lines.append(300000)
    group_names.append('x = 300000')
    # ------

    df = df[df['Step'] <= 500000]

    return df, run_groups, group_names, title, lines


def clean_df(df):
    columns = []

    for c in df.columns:
        if "__MIN" not in c and "__MAX" not in c:
            columns.append(c)

    df = df[columns]
    df = df.fillna(method='ffill')

    return df


def main():
    plt.rcParams.update({'font.size': 15})

    smoothing_window = 1

    df, run_groups, group_names, title, lines = diff_swap_point()
    df = clean_df(df)

    process_and_plot(df, run_groups, group_names, title, smoothing_window, vertical_lines=lines)


if __name__ == '__main__':
    main()
