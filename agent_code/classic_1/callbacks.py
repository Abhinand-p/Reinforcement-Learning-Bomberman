import os
import numpy as np
import math
import itertools
import re

from collections import deque
from .features import check_bomb_presence, check_crate_presence, compute_blockage, calculate_going_to_new_tiles, \
    shortest_path_to_coin_or_crate

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_IDEAS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    q_table_folder = "Q_tables/"
    self.valid_list = Valid_States()
    self.new_state = None
    self.old_state = None
    self.old_distance = 0
    self.new_distance = 0
    if self.train:
        self.logger.info("Q-Learning algorithm.")
        self.name = "Table_1"
        self.number_of_states = len(self.valid_list)
        self.Q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))  # number_of_states * 6
        self.exploration_rate_initial = 1.0
        self.exploration_rate_end = 0.05
        self.exploration_decay_rate = set_decay_rate(self)
    else:
        self.logger.info("Loading from the latest Q_table")
        self.Q_table = load_latest_q_table(self, q_table_folder)


def act(self, game_state: dict) -> str:
    if self.new_state is None:
        self.old_state = state_to_features(self, game_state)
    else:
        self.old_state = self.new_state
    state = self.old_state
    self.logger.info(f"act: State: {state}")
    # The Idea is to perform Exploration only when training.
    if self.train and np.random.random() < self.exploration_rate:
        action = np.random.choice(ACTIONS)
        self.logger.info(f"act: Exploring: {action}")
        return action
    # The Idea is to perform only Exploitation when executing the game.
    else:
        action = ACTIONS[np.argmax(self.Q_table[state])]
        self.logger.info(f"act: Exploiting: {action}")
    return action


def state_to_features(self, game_state) -> np.array:
    features_dict = {}
    # Feature 1: Direction_coin/crate
    coin_direction = shortest_path_to_coin_or_crate(self, game_state)
    if coin_direction in ACTIONS_IDEAS:
        features_dict["Direction_coin/crate"] = coin_direction
    else:
        self.logger.info(f"!!! state_to_features: shortest_path_to_coin_or_crate: Invalid direction: {coin_direction}")
    # Feature 2: Direction_bomb
    bomb_safety_result = calculate_going_to_new_tiles(self, game_state)
    if bomb_safety_result in ["DOWN", "UP", "RIGHT", "LEFT", "SAFE"]:
        features_dict["Direction_bomb"] = bomb_safety_result
    elif bomb_safety_result == 'NO_OTHER_OPTION':
        random_choice = np.random.choice(ACTIONS_IDEAS)
        self.logger.info(f"calculate_going_to_new_tiles: No shortest path {random_choice}")
        features_dict["Direction_bomb"] = random_choice
    else:
        self.logger.info(
            f"!!! state_to_features: calculate_going_to_new_tiles: Invalid direction: {bomb_safety_result}")
    # Feature 3: Place_Bomb
    features_dict["Place_Bomb"] = check_bomb_presence(self, game_state)
    # Feature 4: Crate_Radar
    features_dict["Crate_Radar"] = check_crate_presence(game_state)
    # Feature 5: All directions actions
    (features_dict["Up"], features_dict["Right"], features_dict["Down"], features_dict["Left"]) = compute_blockage(
        game_state)
    self.logger.info(f"Feature Dictionary: {features_dict}")
    for i, state in enumerate(self.valid_list):
        if state == features_dict:
            return i


# This method returns the latest q-table based on the number of episodes trained.
def load_latest_q_table(self, q_table_directory):
    try:
        files = os.listdir(q_table_directory)
        q_table_files = [file for file in files if file.startswith("Q_table-")]
        if not q_table_files:
            self.logger.info("load_latest_q_table: No Q-table files found in the directory.")
            return None
        # Extracting the numbers from the filenames.
        numbers = [int(re.search(r'\d+', file).group()) for file in q_table_files]
        latest_q_table_number = max(numbers)
        latest_q_table_file = f"Q_table-Table_{latest_q_table_number}.npy"
        latest_q_table_path = os.path.join(q_table_directory, latest_q_table_file)
        q_table = np.load(latest_q_table_path)
        self.logger.info(f"load_latest_q_table: Q-table file loaded:{latest_q_table_path}")
        return q_table
    except FileNotFoundError:
        self.logger.info("load_latest_q_table: Q-table directory not found.")
        return None


# This method utilizes the n_rounds argument to set the decay rate.
# The Idea is to have a constant decrease in the exploration rate.
def set_decay_rate(self) -> float:
    decay_rate = -math.log((self.exploration_rate_end + 0.005) / self.exploration_rate_initial) / self.n_rounds
    self.logger.info(f" n_rounds: {self.n_rounds}")
    self.logger.info(f"Determined exploration decay rate: {decay_rate}")
    return decay_rate


# This method defines all the possible values each feature can hold in the feature dictionary
# Idea: In the Q-table each combination of these features is a row.
# Q-table shape: (1920, 6)
def Valid_States() -> np.array:
    feature_list = []
    valid_states = list(itertools.product(('UP', 'RIGHT', 'DOWN', 'LEFT'), ('UP', 'RIGHT', 'DOWN', 'LEFT', 'SAFE'),
                                          ('MOVE', 'BLOCK'), ('MOVE', 'BLOCK'), ('MOVE', 'BLOCK'), ('MOVE', 'BLOCK'),
                                          ('YES', 'NO'), ('LOW', 'MID', 'HIGH')))

    for states in valid_states:
        features = {
            "Direction_coin/crate": states[0],
            "Direction_bomb": states[1],
            "Up": states[2],
            "Right": states[3],
            "Down": states[4],
            "Left": states[5],
            "Place_Bomb": states[6],
            "Crate_Radar": states[7],
        }
        feature_list.append(features)
    return feature_list
