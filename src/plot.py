import os

import pandas as pd
import matplotlib.pyplot as plt
from Config import CONFIG


def plot_agent_rewards(results_folder, environment, agents, query_types):
    data = {}

    for agent in agents:
        if agent == "perfect_agent":
            rewards = []

            for run in range(3):
                column_name = f"{agent}_{run} - eval_mean_reward"
                df = pd.read_csv(f"{results_folder}/{environment}.csv")
                rewards.append(df[column_name])

            data[agent] = sum(rewards) / len(rewards)
        else:
            data[agent] = {}

            for query_type in query_types:
                rewards = []

                for run in range(3):
                    column_name = f"{agent}_{run}_{query_type} - eval_mean_reward"
                    df = pd.read_csv(f"{results_folder}/{environment}.csv")
                    rewards.append(df[column_name])

                data[agent][query_type] = sum(rewards) / len(rewards)

    for agent, query_data in data.items():
        if agent == "perfect_agent":
            label = f"{agent}"
            query_data.plot(label=label)
        else:
            for query_type, reward in query_data.items():
                label = f"{agent}_{query_type}"
                reward.plot(label=label)

    plt.legend()
    plt.show()


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
    merged_df.to_csv(os.path.join(directory, output_file_name), index=False)

    return merged_df


def check_and_create_merged_file(directory, column_name, output_file_name):
    # Check if the output file already exists
    output_file_path = os.path.join(directory, output_file_name)

    if os.path.exists(output_file_path):
        return f"The file {output_file_name} already exists in the directory."

    # If the output file does not exist, call the merge_csv_files function
    return merge_csv_files(directory, column_name, output_file_name)


if __name__ == "__main__":
    # query_types = ["random", "active"]
    query_types = ["random"]
    agents = ["perfect_agent", "learner_0", "learner_25", "learner_40", "learner_50", "learner_75", "learner_100"]

    for environment in CONFIG.ENVIRONMENTS:
        # We make sure the csv exists, else we create it
        check_and_create_merged_file("results", "Step", f"{environment}.csv")

        # We plot the resulting csv file
        plot_agent_rewards("results", environment, agents, query_types)
