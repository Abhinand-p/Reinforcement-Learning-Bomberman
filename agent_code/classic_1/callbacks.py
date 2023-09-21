import os
import numpy as np
import math

from datetime import datetime
from typing import Tuple, List
from collections import deque
from igraph import Graph

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_IDEAS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

# TODO: Check if this values will change. If yes then I can read from settings.py
n_rows = 17
n_cols = 17


def setup(self):
    q_table_folder = "Q_tables/"
    self.history = [0, deque(maxlen=5)]  # Currently holding (number of coins collected, tiles visited)
    self.new_state = None
    self.old_distance = 0
    self.new_distance = 0

    if self.train:
        self.logger.info("Q-Learning algorithm.")
        self.timestamp = datetime.now().strftime("%dT%H:%M:%S")
        self.number_of_states = 10  # TODO: I think this should be dynamic and not a static number.
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
    state = state_to_features(self, game_state, self.history)

    if self.train and np.random.random() < self.exploration_rate:
        # TODO: Check if during exploring random choice is the best option because we do not want self explosions.
        action = np.random.choice(ACTIONS)
        self.logger.info(f"Exploring: {action}")
        return action

    if not np.any(self.Q_table[state]):
        action = np.random.choice(ACTIONS)
        self.logger.info(f"Q-Table has no zeros, so random action chosen:{action}")
    else:
        action = ACTIONS[np.argmax(self.Q_table[state])]
        self.logger.info(f"Exploiting: {action}")

    return action


def get_neighboring_tiles(own_coord, radius) -> List[Tuple[int]]:
    x, y = own_coord
    # Finding neighbouring tiles
    neighboring_coordinates = []
    for i in range(1, radius + 1):
        neighboring_coordinates.extend([
            (x, y + i),  # down
            (x, y - i),  # up
            (x + i, y),  # right
            (x - i, y)  # left
        ])
    return neighboring_coordinates


def get_neighboring_tiles_within_distance(current_position, max_distance, game_state) -> List[Tuple[int]]:
    directions = ["top", "right_side", "bottom", "left_side"]
    current_x, current_y = current_position[0], current_position[1]
    neighboring_tiles = []

    for d, direction in enumerate(directions):
        valid_tiles = []
        for i in range(1, max_distance + 1):
            try:
                if direction == "top":
                    if game_state["field"][current_x][current_y + i] in [0, 1]:
                        valid_tiles += [(current_x, current_y + i)]
                    else:
                        break
                elif direction == "right_side":
                    if game_state["field"][current_x + i][current_y] in [0, 1]:
                        valid_tiles += [(current_x + i, current_y)]
                    else:
                        break
                elif direction == "bottom":
                    if game_state["field"][current_x][current_y - i] in [0, 1]:
                        valid_tiles += [(current_x, current_y - i)]
                    else:
                        break
                elif direction == "left_side":
                    if game_state["field"][current_x - i][current_y] in [0, 1]:
                        valid_tiles += [(current_x - i, current_y)]
                    else:
                        break
            except IndexError:
                break

        neighboring_tiles += valid_tiles

    return neighboring_tiles


# Calculates all adjacency matrix for the game grid.
def calculate_adjacency_matrix(self, game_state, consider_crates=True) -> Graph:
    if consider_crates:
        blockers = [(i, j) for i, j in np.ndindex(*game_state["field"].shape) if game_state["field"][i, j] != 0]
    else:
        blockers = [(i, j) for i, j in np.ndindex(*game_state["field"].shape) if game_state["field"][i, j] == -1]

    current_explosions = [(i, j) for i, j in np.ndindex(*game_state["explosion_map"].shape) if
                          game_state["explosion_map"][i, j] != 0]

    bombs = [
        coordinate
        for coordinate, _ in game_state["bombs"]
        if coordinate != game_state["self"][-1]
           and coordinate not in [other_agent[-1] for other_agent in game_state["others"]]
    ]

    blockers += current_explosions
    blockers += bombs

    # self.logger.info(f"Blockers matrix: {blockers}")

    graph = Graph()

    # format: vertex1--vertex2
    # --: indicates an edge
    for i in range(n_rows):
        for j in range(n_cols):
            vertex_id = i * n_cols + j
            graph.add_vertex(vertex_id)

    for i in range(n_rows):
        for j in range(n_cols):
            vertex_id = i * n_cols + j
            if j < n_cols - 1:
                graph.add_edge(vertex_id, vertex_id + 1)
            if i < n_rows - 1:
                graph.add_edge(vertex_id, vertex_id + n_cols)

    # Removing nodes that represent blockers
    graph.delete_vertices([i * n_cols + j for i, j in blockers])
    return graph


# A helper function to get the shortest path between two coordinates.
def find_shortest_path_coordinates(graph, start_coordinate, end_coordinate):
    start_vertex = graph.vs.find(name=str(start_coordinate[0] * n_cols + start_coordinate[1]))
    shortest_path_nodes = \
        graph.get_shortest_paths(start_vertex, to=str(end_coordinate[0] * n_cols + end_coordinate[1]), mode="OUT")[0]
    shortest_path_coordinates = [(int(node) // n_cols, int(node) % n_cols) for node in shortest_path_nodes]
    return shortest_path_coordinates[1:], len(shortest_path_nodes) - 1


# A helper function to select the best action given the current position
def select_best_action(current_coord, next_coord):
    x_diff = next_coord[0] - current_coord[0]
    y_diff = next_coord[1] - current_coord[1]

    # ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
    if x_diff == 1:
        return 2  # RIGHT
    elif x_diff == -1:
        return 4  # LEFT
    elif y_diff == 1:
        return 1  # UP
    elif y_diff == -1:
        return 3  # DOWN
    else:
        return 4  # WAIT


# Feature 1: Count the number of walls in the immediate surrounding tiles within a given radius.
def count_walls(current_position, game_state, radius):
    return sum(
        1 for coord in get_neighboring_tiles(current_position, radius)
        if 0 <= coord[0] < game_state["field"].shape[0] and 0 <= coord[1] < game_state["field"].shape[1]
        and game_state["field"][coord] == -1
    )


# Feature 2: Check for bomb presence in the immediate surrounding tiles within a given radius.
def check_bomb_presence(current_position, game_state, radius):
    return any(
        bomb[0] in get_neighboring_tiles(current_position, radius)
        and bomb[1] != 0
        for bomb in game_state["bombs"]
    )


# Feature 3: Check for agent presence in the immediate surrounding tiles within a given radius.
def check_agent_presence(current_position, game_state, radius):
    return any(
        agent[3] in get_neighboring_tiles(current_position, radius)
        for agent in game_state["others"]
    )


# Feature 4: Getting the number of tiles in each direction of the agent. 0: free tiles and 1:crates
def calculate_death_tile(game_state, own_position) -> int:
    all_death_tiles = []
    is_dangerous = []

    if len(game_state["bombs"]) > 0:
        for bomb in game_state["bombs"]:
            bomb_position = bomb[0]
            neighboring_death_tiles = get_neighboring_tiles_within_distance(
                bomb_position, 3, game_state=game_state
            )
            if neighboring_death_tiles:
                all_death_tiles += neighboring_death_tiles

        if len(all_death_tiles) > 0:
            for death_tile in all_death_tiles:
                in_danger = own_position == death_tile
                is_dangerous.append(in_danger)
            # 1 if the agent is on a death tile.
            # 0 if the agent is not on a death tile.
            return int(any(is_dangerous))
    else:
        return 0


# Feature 5: Checking for movable tiles
def compute_blockage(game_state, agent_position):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
    features = [0] * len(directions)

    for i, direction in enumerate(directions):
        x, y = agent_position[0] + direction[0], agent_position[1] + direction[1]

        # Check if the neighboring tile is within the game field
        if 0 <= x < len(game_state["field"]) and 0 <= y < len(game_state["field"][0]):
            content = game_state["field"][x][y]

            # Check for blockages (e.g., crates, walls, enemies)
            if content != 0:
                features[i] = 1

    return features


# Feature 6: Checking for new tile
def calculate_going_to_new_tiles(history):
    num_visited_tiles = len(history[1])
    if num_visited_tiles > 1:
        num_unique_visited_tiles = len(set(history[1]))
        # unique tiles visited to the total tiles visited ratio
        feature_value = 1 if np.floor((num_unique_visited_tiles / num_visited_tiles)) > 0.6 else 0
    else:
        feature_value = 0

    return feature_value


# TODO: I should make a feature dictionary and return the action list. Then I will have a better contol to check the
#  training process
def state_to_features(self, game_state, history) -> np.array:
    if game_state is None:
        print("First game state is None")
        return np.zeros(5)

    current_position = game_state["self"][-1]
    enemy_positions = [enemy[-1] for enemy in game_state["others"]]
    adj_graph = calculate_adjacency_matrix(self, game_state, True)

    features = np.zeros(10, dtype=np.int8)

    # Calculate features
    # wall_counter = count_walls(current_position, game_state, 3)
    # bomb_present = check_bomb_presence(current_position, game_state, 3)
    # agent_present = check_agent_presence(current_position, game_state, 3)
    death_tile = calculate_death_tile(game_state, current_position)
    features[0] = death_tile

    blockage_features = compute_blockage(game_state, current_position)
    features[1:5] = blockage_features

    visited_ratio = calculate_going_to_new_tiles(self.history)
    features[5] = visited_ratio

    # Calculate feature_id based on features
    # features = np.array([int(wall_counter > 2), int(bomb_present), int(agent_present), int(death_tile)])
    # feature_id = 2 * features[0] + features[1] + 2 * features[2] + features[3]

    self.logger.info(f"Feature vector: {features}")
    self.logger.info(f"Adjacency Graph: {adj_graph}")
    return features


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
