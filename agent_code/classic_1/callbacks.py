import os
import numpy as np
import math
from datetime import datetime
from typing import Tuple, List
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_IDEAS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    q_table_folder = "Q_tables/"
    self.history = [0, deque(maxlen=5)]  # Currently holding (number of coins collected, tiles visited)
    self.new_state = None
    self.old_distance = 0
    self.new_distance = 0

    if self.train:
        self.logger.info("Q-Learning algorithm.")
        self.timestamp = datetime.now().strftime("%dT%H:%M:%S")
        self.number_of_states = 6  # TODO: I think this should be dynamic and not a static number.
        self.Q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))  # number_of_states * 6
        self.exploration_rate_initial = 1.0
        self.exploration_rate_end = 0.05
        self.exploration_decay_rate = set_decay_rate(self)

    else:
        self.logger.info("Loading from the latest Q_table")
        q_table_directory_path = "Q_tables"
        self.Q_table = load_latest_q_table(self, q_table_directory_path)


def set_decay_rate(self) -> float:
    # This method utilizes the n_rounds to set the decay rate
    decay_rate = -math.log((self.exploration_rate_end + 0.005) / self.exploration_rate_initial) / self.n_rounds
    self.logger.info(f" n_rounds: {self.n_rounds}")
    self.logger.info(f"Determined exploration decay rate: {decay_rate}")
    return decay_rate


def act(self, game_state: dict) -> str:
    state = state_to_features(game_state, self.history)

    if self.train and np.random.random() < self.exploration_rate:
        # TODO: Check if during exploring random choice is the best option because we do not want self explosions.
        action = np.random.choice(ACTIONS)
        self.logger.info(f"Exploring: {action}")
        return action

    # TODO: Check if we have to use only exploration from q table after training
    action = ACTIONS[np.argmax(self.Q_table[state])]
    self.logger.info(f"Exploiting: {action}")
    return action


def _get_neighboring_tiles(own_coord, radius) -> List[Tuple[int]]:
    x, y = own_coord
    # Finding neighbouring tiles
    neighboring_coordinates = []
    for i in range(1, radius + 1):
        neighboring_coordinates.extend([
            (x, y + i),  # down in the matrix
            (x, y - i),  # up in the matrix
            (x + i, y),  # right in the matrix
            (x - i, y)  # left in the matrix
        ])
    return neighboring_coordinates


# Feature 1: Count the number of walls in the immediate surrounding tiles within a given radius.
def count_walls(current_position, game_state, radius):
    return sum(
        1 for coord in _get_neighboring_tiles(current_position, radius)
        if 0 <= coord[0] < game_state["field"].shape[0] and 0 <= coord[1] < game_state["field"].shape[1]
        and game_state["field"][coord] == -1
    )


# Feature 2: Check for bomb presence in the immediate surrounding tiles within a given radius.
def check_bomb_presence(current_position, game_state, radius):
    return any(
        bomb[0] in _get_neighboring_tiles(current_position, radius)
        and bomb[1] != 0
        for bomb in game_state["bombs"]
    )


# Feature 3: Check for agent presence in the immediate surrounding tiles within a given radius.
def check_agent_presence(current_position, game_state, radius):
    return any(
        agent[3] in _get_neighboring_tiles(current_position, radius)
        for agent in game_state["others"]
    )


# TODO: I think its better to return the direction of the agent as state
def state_to_features(game_state, history) -> np.array:
    if game_state is None:
        print("First game state is None")
        return np.zeros(2)

    # Get the current position
    current_position = game_state["self"][-1]

    # Calculate features
    wall_counter = count_walls(current_position, game_state, 3)
    bomb_present = check_bomb_presence(current_position, game_state, 3)
    agent_present = check_agent_presence(current_position, game_state, 3)

    # Calculate feature_id based on features
    features = np.array([int(wall_counter > 2), int(bomb_present), int(agent_present)])
    feature_id = 2 * features[0] + features[1] + 2 * features[2]

    return feature_id


def load_latest_q_table(self, q_table_directory):
    try:
        files = os.listdir(q_table_directory)
        q_table_files = [file for file in files if file.startswith("Q_table-")]

        if not q_table_files:
            self.logger.info("No Q-table files found in the directory.")
            return None

        # Finding the latest Q-table file based on the timestamp
        latest_q_table_file = max(q_table_files)
        latest_q_table_path = os.path.join(q_table_directory, latest_q_table_file)

        q_table = np.load(latest_q_table_path)

        self.logger.info(f"Q-table file loaded:{latest_q_table_path}")
        return q_table

    except FileNotFoundError:
        self.logger.info("Q-table directory not found.")
        return None
