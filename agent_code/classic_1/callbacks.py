import os
import numpy as np
from datetime import datetime
from typing import Tuple, List

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    q_table_folder = "Q_tables/"

    if self.train or not os.path.isfile(os.path.join(q_table_folder, "q_table.npy")):
        self.logger.info("Q-Learning algorithm.")
        self.timestamp = datetime.now().strftime("%dT%H:%M:%S")
        self.number_of_states = 6
        self.Q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))
        self.exploration_rate_initial = 1.0
        self.exploration_rate_end = 0.1
        self.exploration_decay_rate = 0.01

    else:
        self.logger.info("Loading from the latest Q_table")
        with open("my-saved-model.pt", "rb") as file:
            self.Q_table = np.load("Q_table.npy")
    self.history = [0, None]


def act(self, game_state: dict) -> str:
    # Current game state
    state = state_to_features(game_state, self.history)

    if self.train and np.random.random() < self.exploration_rate:
        action = np.random.choice(ACTIONS)
        self.logger.info(f"Exploring: {action}")
        return action

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
def count_walls(own_position, game_state, radius):
    return sum(
        1 for coord in _get_neighboring_tiles(own_position, radius)
        if 0 <= coord[0] < game_state["field"].shape[0] and 0 <= coord[1] < game_state["field"].shape[1]
        and game_state["field"][coord] == -1
    )


# Feature 2: Check for bomb presence in the immediate surrounding tiles within a given radius.
def check_bomb_presence(own_position, game_state, radius):
    return any(
        bomb[0] in _get_neighboring_tiles(own_position, radius)
        and bomb[1] != 0
        for bomb in game_state["bombs"]
    )


# Feature 3: Check for agent presence in the immediate surrounding tiles within a given radius.
def check_agent_presence(own_position, game_state, radius):
    return any(
        agent[3] in _get_neighboring_tiles(own_position, radius)
        for agent in game_state["others"]
    )


# TODO: I think its better to return the direction of the agent as state
def state_to_features(game_state, history) -> np.array:
    if game_state is None:
        print("First game state is None")
        return np.zeros(2)

    own_position = game_state["self"][-1]

    # Calculate features
    wall_counter = count_walls(own_position, game_state, 3)
    bomb_present = check_bomb_presence(own_position, game_state, 3)
    agent_present = check_agent_presence(own_position, game_state, 3)

    # Calculate feature_id based on features
    features = np.array([int(wall_counter > 2), int(bomb_present), int(agent_present)])
    feature_id = 2 * features[0] + features[1] + 2 * features[2]

    return feature_id
