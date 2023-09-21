import os
import numpy as np
import math
import networkx as nx
import itertools

from datetime import datetime
from typing import Tuple, List
from collections import deque
from igraph import Graph

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_IDEAS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
n_rows = 17
n_cols = 17


def setup(self):
    q_table_folder = "Q_tables/"
    self.valid_list = Valid_States()
    self.history = [0, deque(maxlen=5)]  # Currently holding (number of coins collected, tiles visited)
    self.new_state = None
    self.old_state = None
    self.old_distance = 0
    self.new_distance = 0

    if self.train:
        self.logger.info("Q-Learning algorithm.")
        self.timestamp = datetime.now().strftime("%dT%H:%M:%S")
        self.number_of_states = len(self.valid_list)
        self.Q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))  # number_of_states * 6
        self.exploration_rate_initial = 1.0
        self.exploration_rate_end = 0.05
        self.exploration_decay_rate = set_decay_rate(self)

    else:
        self.logger.info("Loading from the latest Q_table")
        self.Q_table = load_latest_q_table(self, q_table_folder)


def set_decay_rate(self) -> float:
    # This method utilizes the n_rounds to set the decay rate
    decay_rate = -math.log((self.exploration_rate_end + 0.005) / self.exploration_rate_initial) / self.n_rounds
    self.logger.info(f" n_rounds: {self.n_rounds}")
    self.logger.info(f"Determined exploration decay rate: {decay_rate}")
    return decay_rate


def act(self, game_state: dict) -> str:
    # Testing this logic
    if self.new_state is None:
        self.old_state = state_to_features(self, game_state)
    else:
        self.old_state = self.new_state

    state = self.old_state
    self.logger.info(f"act: State: {state}")

    if self.train and np.random.random() < self.exploration_rate:
        # TODO: Check if during exploring random choice is the best option because we do not want self explosions.
        action = np.random.choice(ACTIONS)
        self.logger.info(f"Exploring: {action}")
        return action

    if not np.any(self.Q_table[state]):
        action = np.random.choice(ACTIONS)
        self.logger.info(f"Q-Table has all zeros, so random action chosen: {action}")
    else:
        action = ACTIONS[np.argmax(self.Q_table[state])]
        self.logger.info(f"Exploiting: {action}")
    return action


def Valid_States() -> np.array:
    feature_list = []
    valid_states = list(itertools.product(('UP', 'RIGHT', 'DOWN', 'LEFT'), ('UP', 'RIGHT', 'DOWN', 'LEFT', 'SAFE'),
                                          ('MOVE', 'BLOCK'), ('MOVE', 'BLOCK'), ('MOVE', 'BLOCK'), ('MOVE', 'BLOCK')))

    for states in valid_states:
        features = {
            "Direction_coin/crate": states[0],
            "Direction_bomb": states[1],
            "Up": states[2],
            "Right": states[3],
            "Down": states[4],
            "Left": states[5],
        }
        feature_list.append(features)
    return feature_list


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

    graph = nx.grid_2d_graph(m=n_cols, n=n_rows)

    # Removing nodes that represent blockers
    graph.remove_nodes_from(blockers)
    return graph


# A helper function to get the shortest path between two coordinates.
def find_shortest_path_coordinates(graph, a, b) -> Tuple[Graph, int]:
    try:
        shortest_path = nx.shortest_path(graph, source=a, target=b, weight=None, method="dijkstra")
    except nx.exception.NodeNotFound as e:
        print(graph.nodes)
        raise e

    shortest_path_length = len(shortest_path) - 1  # because path considers self as part of the path
    return shortest_path, shortest_path_length


# A helper function to select the best action given the current position
def select_best_action(self, current_coord, next_coord):
    x_diff = next_coord[0] - current_coord[0]
    y_diff = next_coord[1] - current_coord[1]

    # ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
    if np.any(x_diff == 1):
        return "RIGHT"  # RIGHT
    elif np.any(x_diff == -1):
        return "LEFT"  # LEFT
    elif np.any(y_diff == 1):
        return "UP"  # UP
    elif np.any(y_diff == -1):
        return "DOWN"  # DOWN


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


def shortest_path_to_coin_or_crate(agent, game_state):
    graph = calculate_adjacency_matrix(agent, game_state)
    graph_with_crates = calculate_adjacency_matrix(agent, game_state, consider_crates=False)

    # current coordinate Classic_1 agent
    current_position = game_state["self"][-1]
    # agent.logger.info(f"shortest_path_to_coin_or_crate: current_position: {current_position}")

    # Extract explosion area positions.
    explosion_area = [(index[0], index[1]) for index, field in np.ndenumerate(game_state["explosion_map"]) if
                      field != 0]
    # agent.logger.info(f"shortest_path_to_coin_or_crate: explosion_area: {explosion_area}")

    # Crates present that are not yet exploded.
    crates = [(index[0], index[1]) for index, field in np.ndenumerate(game_state["field"]) if field == 1]
    # agent.logger.info(f"shortest_path_to_coin_or_crate: crates: {crates}")

    # Good Coins are those that are not in the explosion_area
    good_coins = [coin for coin in game_state["coins"] if coin not in explosion_area]
    # agent.logger.info(f"shortest_path_to_coin_or_crate: good_coins: {good_coins}")

    # If no coins and crates -> Random ?
    if not any(good_coins) and not any(crates):
        random_choice = np.random.choice(ACTIONS_IDEAS)
        agent.logger.info(f"shortest_path_to_coin_or_crate: No Coins and Crates {random_choice}")
        return random_choice

    # If only there are crates but no good_coins
    elif not any(good_coins):
        next_crate_position = (None, np.inf)

        for crate_coord in crates:
            try:
                current_path, current_path_length = find_shortest_path_coordinates(graph_with_crates, current_position,
                                                                                   crate_coord)

            except nx.exception.NetworkXNoPath:
                agent.logger.info("shortest_path_to_coin_or_crate: Crate may be exploded")
                continue

            if current_path_length == 1:
                # TODO: Should I modify the code to add an action to bomb here for this case?
                agent.logger.info(f"shortest_path_to_coin_or_crate: Agent next to a crate")
                return select_best_action(agent, current_position, current_path[0])

            elif current_path_length < next_crate_position[1]:
                next_crate_position = (current_path, current_path_length)

        # If there are no crates or coins still then return a random action -> highly unlikely I guess
        if next_crate_position == (None, np.inf):
            random_choice = np.random.choice(ACTIONS_IDEAS)
            agent.logger.info(f"shortest_path_to_coin_or_crate: no crate or a good coin still: {random_choice}")
            return random_choice

        return select_best_action(agent, current_position, next_crate_position[0][0])

    # If there are only good coins
    else:
        coin_path = []

        # Finding the shortest path for all good coins
        for coin_coord in good_coins:
            try:
                current_path, current_path_length = find_shortest_path_coordinates(graph, current_position, coin_coord)
                current_reachable = True
            except nx.exception.NetworkXNoPath:
                try:
                    current_path, current_path_length = find_shortest_path_coordinates(graph_with_crates,
                                                                                       current_position, coin_coord)
                    current_reachable = False
                except nx.exception.NetworkXNoPath:
                    agent.logger.info("shortest_path_to_coin_or_crate: Corner case 1")
                    continue

            # If there are no other agents alive
            if not any(game_state["others"]):
                coin_path.append(((current_path, current_path_length, current_reachable), (None, np.inf)))
                continue

            for other_agent in game_state["others"]:
                best_other_agent = (None, np.inf)
                other_agent_coord = other_agent[3]
                try:
                    current_path_other_agent, current_path_length_other_agent = find_shortest_path_coordinates(graph,
                                                                                                               other_agent_coord,
                                                                                                               coin_coord)
                    current_other_agent_reachable = True

                except nx.exception.NetworkXNoPath:
                    try:
                        current_path_other_agent, current_path_length_other_agent = find_shortest_path_coordinates(
                            graph_with_crates, other_agent_coord, coin_coord)
                        current_other_agent_reachable = False

                    except nx.exception.NetworkXNoPath:
                        agent.logger.info("shortest_path_to_coin_or_crate: Corner case 2")
                        continue

                if not current_other_agent_reachable:
                    current_path_length_other_agent += 7

                if current_path_length_other_agent < best_other_agent[1]:
                    best_other_agent = (
                        current_path_other_agent, current_path_length_other_agent, current_other_agent_reachable)
            coin_path.append(((current_path, current_path_length, current_reachable), best_other_agent))

        # If no coins are still not reachable
        if not any(coin_path):
            random_choice = np.random.choice(ACTIONS_IDEAS)
            agent.logger.info(f"shortest_path_to_coin_or_crate: Corner case 3: {random_choice}")
            return random_choice

        # Sorting the obtained coin paths
        coin_path.sort(key=lambda x: x[0][1])
        coin_path_reachable = [path_to_coin[0][2] for path_to_coin in coin_path]

        # If no paths are reachable to coins
        if not any(coin_path_reachable):
            return select_best_action(agent, current_position, coin_path[0][0])

        # If there is only one coin path reachable
        elif coin_path_reachable.count(True) == 1:
            agent.logger.info(f"shortest_path_to_coin_or_crate: Exactly one coin path is reachable")
            index_of_reachable_path = coin_path_reachable.index(True)
            return select_best_action(agent, current_position, coin_path[index_of_reachable_path][0][0])

        for path_to_coin in coin_path:
            if path_to_coin[0][2] is True and path_to_coin[0][1] <= path_to_coin[1][1] and path_to_coin[0][1] != 0:
                return select_best_action(agent, current_position, path_to_coin[0][0])
        try:
            return select_best_action(agent, current_position, coin_path[0][0][0])

        except IndexError:
            random_choice = np.random.choice(ACTIONS_IDEAS)
            agent.logger.info(f"shortest_path_to_coin_or_crate: Corner case 4: {random_choice}")
            return random_choice


def state_to_features(self, game_state) -> np.array:
    if game_state is None:
        self.logger.info(f"state_to_features: First Game")
    features_dict = {}

    # Feature 1:
    coin_direction = shortest_path_to_coin_or_crate(self, game_state)
    if coin_direction in ["DOWN", "UP", "RIGHT", "LEFT"]:
        features_dict["Direction_coin/crate"] = coin_direction
        self.logger.info(f"state_to_features: Feature 1:{coin_direction}")
    else:
        self.logger.info(f"state_to_features: Feature 1: Invalid direction")

    # features = np.zeros(10, dtype=np.int8)
    # Calculate features
    # wall_counter = count_walls(current_position, game_state, 3)
    # bomb_present = check_bomb_presence(current_position, game_state, 3)
    # agent_present = check_agent_presence(current_position, game_state, 3)
    # death_tile = calculate_death_tile(game_state, current_position)
    # features[0] = death_tile
    # blockage_features = compute_blockage(game_state, current_position)
    # features[1:5] = blockage_features
    # visited_ratio = calculate_going_to_new_tiles(self.history)
    # features[5] = visited_ratio
    # Calculate feature_id based on features
    # features = np.array([int(wall_counter > 2), int(bomb_present), int(agent_present), int(death_tile)])
    # feature_id = 2 * features[0] + features[1] + 2 * features[2] + features[3]

    self.logger.info(f"Feature Dictionary: {features_dict}")
    for i, state in enumerate(self.valid_list):
        if state == features_dict:
            return i


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
