# This is a helper file to read the latest Q-table and visualize it using a heatmap.
# This is used to understand the distribution of the learning.
# Idea: To check how may non-zero rows(actions) maybe present in the Q-table after training.

import os
import re
import numpy as np
import matplotlib.pyplot as plt


# Method to load the latest Q_table based on the last number.
# The Idea is to load the Q_table that has undergone the maximum number of episodes.
def load_latest_q_table(q_table_directory):
    try:
        files = os.listdir(q_table_directory)
        q_table_files = [file for file in files if file.startswith("Q_table-")]
        if not q_table_files:
            print("No Q-table files found in the directory.")
            return None, None
        # Extracting the numbers from the filenames.
        numbers = [int(re.search(r'\d+', file).group()) for file in q_table_files]
        latest_q_table_number = max(numbers)
        # Finding the latest Q-table file based on the maximum number in the filename.
        latest_q_table_file = f"Q_table-Table_{latest_q_table_number}.npy"
        latest_q_table_path = os.path.join(q_table_directory, latest_q_table_file)
        q_table = np.load(latest_q_table_path)
        return q_table, latest_q_table_file
    except FileNotFoundError:
        print("Q-table directory not found.")
        return None, None


# Method to visualize the latest Q_table to check non-zero actions.
def visualize_q_table(q_table):
    # Visualizing the q-table using a heatmap.
    plt.figure(figsize=(10, 6))
    plt.imshow(q_table, cmap='viridis', aspect='auto')
    plt.colorbar(label='Q-value')
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.xticks(range(q_table.shape[1]), ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])
    plt.title('Q-table Heatmap')
    plt.show()


# Method to calculate the number of zero actions.
def count_zero_rows(q_table):
    zero_rows = np.sum(np.all(q_table == 0, axis=1))
    print("Number of zero actions in Q-table:", zero_rows)


q_table_directory_path = "Q_tables"
loaded_q_table, q_table_name = load_latest_q_table(q_table_directory_path)
if loaded_q_table is not None:
    print("Loaded Q-table shape:", loaded_q_table.shape)
    print("Loaded Q-table from:", q_table_name)
    print("Loaded Q-table:")
    print(loaded_q_table)
    count_zero_rows(loaded_q_table)
    visualize_q_table(loaded_q_table)
