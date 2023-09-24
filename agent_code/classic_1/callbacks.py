import os
import numpy as np
import math
import networkx as net
import itertools
import copy as cp

from typing import List
from collections import deque
from settings import BOMB_POWER

from .support import get_neighboring_tiles, get_neighboring_tiles_within_distance, calculate_adjacency_matrix, \
    find_shortest_path_coordinates, select_best_action

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_IDEAS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

bomb_power = BOMB_POWER


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

    if self.train and np.random.random() < self.exploration_rate:
        action = np.random.choice(ACTIONS)
        self.logger.info(f"act: Exploring: {action}")
        return action

    if not np.any(self.Q_table[state]):
        action = np.random.choice(ACTIONS)
        self.logger.info(f"act: Q-Table has all zeros, so random action chosen: {action}")
    else:
        action = ACTIONS[np.argmax(self.Q_table[state])]
        self.logger.info(f"act: Exploiting: {action}")
    return action


def set_decay_rate(self) -> float:
    # This method utilizes the n_rounds to set the decay rate
    decay_rate = -math.log((self.exploration_rate_end + 0.005) / self.exploration_rate_initial) / self.n_rounds
    self.logger.info(f" n_rounds: {self.n_rounds}")
    self.logger.info(f"Determined exploration decay rate: {decay_rate}")
    return decay_rate


# This method defines all the possible values each feature can hold in the feature dictionary
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


# Feature 1: Count the number of walls in the immediate surrounding tiles within a given radius.
def count_walls(current_position, game_state, radius):
    return sum(
        1 for coord in get_neighboring_tiles(current_position, radius)
        if 0 <= coord[0] < game_state["field"].shape[0] and 0 <= coord[1] < game_state["field"].shape[1]
        and game_state["field"][coord] == -1
    )


# Feature 2: Check for bomb presence in the immediate surrounding tiles within a given radius.
def check_bomb_presence(self, game_state) -> str:
    if game_state["round"] == 1:
        return 'NO'

    if not game_state["self"][2]:
        return 'NO'

    new_game_state = cp.deepcopy(game_state)
    new_game_state["bombs"].append((game_state["self"][-1], 4))
    if calculate_going_to_new_tiles(self, new_game_state) == "NO_OTHER_OPTION":
        return 'NO'

    return 'YES'


# Feature 3: Check for crate presence in the immediate surrounding tiles within a given radius.
def check_crate_presence(game_state) -> str:
    current_position = game_state["self"][-1]
    adjacent = get_neighboring_tiles_within_distance(current_position, bomb_power, game_state)

    crate_reward = 0
    for coord in adjacent:
        if game_state["field"][coord[0]][coord[1]] == 1:
            crate_reward += 1
            # Vertical crate check
            if current_position[1] == coord[1] + 1 or current_position[1] == coord[1] - 1:
                crate_reward += 3
            # Horizontal crate check
            elif current_position[0] == coord[0] + 1 or current_position[0] == coord[0] - 1:
                crate_reward += 3

    if crate_reward == 0:
        return 'LOW'
    elif 1 <= crate_reward < 4:
        return 'MID'
    elif crate_reward >= 4:
        return 'HIGH'


# Feature 4: Getting the number of tiles in each direction of the agent. 0: free tiles and 1:crates
def calculate_death_tile(game_state, current_position) -> int:
    all_death_tiles = []
    is_dangerous = []

    if len(game_state["bombs"]) > 0:
        for bomb in game_state["bombs"]:
            bomb_position = bomb[0]
            neighboring_death_tiles = get_neighboring_tiles_within_distance(bomb_position, bomb_power, game_state)
            if neighboring_death_tiles:
                all_death_tiles += neighboring_death_tiles

        if len(all_death_tiles) > 0:
            for death_tile in all_death_tiles:
                in_danger = current_position == death_tile
                is_dangerous.append(in_danger)
            # 1 if the agent is on a death tile.
            # 0 if the agent is not on a death tile.
            return int(any(is_dangerous))
    else:
        return 0


# Feature 5: Checking for movable tiles based on other agents, bombs, explosions
def compute_blockage(game_state: dict) -> List[str]:
    # Initialize an empty set to track explosion positions
    explosion = set()
    # Initialize an empty list to store bomb positions
    bombs = []
    # Get current position
    current_position = game_state["self"][-1]
    # Get positions of other agents
    other_agent_positions = [enemy[-1] for enemy in game_state["others"]]
    # By default, let the agent move
    results = ["MOVE"] * 4
    # Calculate where the explosions can happen
    for bomb_position, bomb_timer in game_state["bombs"]:
        bombs.append(bomb_position)
        if bomb_timer == 0:
            next_explosion = get_neighboring_tiles_within_distance(bomb_position, bomb_power, game_state)
            next_explosion += bomb_position
            explosion.update(next_explosion)
    # Check if the current adjacent tile is blocked by an explosion, a bomb, a wall, or another agent,
    # and mark it as "BLOCK" in the results if it is blocked.
    for i, adjacent in enumerate(get_neighboring_tiles(current_position, 1)):
        flag_explosion = adjacent in explosion or game_state["explosion_map"][adjacent[0]][adjacent[1]] != 0
        flag_bomb = adjacent in bombs
        neighboring_content = game_state["field"][adjacent[0]][adjacent[1]]
        if neighboring_content != 0 or adjacent in other_agent_positions or flag_explosion or flag_bomb:
            results[i] = "BLOCK"
    return results


# Feature 6: Checking for new tile
def calculate_going_to_new_tiles(self, game_state) -> str:
    # Get current position
    current_position = game_state["self"][-1]

    # Get Bomb positions
    bombs_positions = [bomb[0] for bomb in game_state["bombs"]]

    # Get adjacent positions
    adjacent_positions = get_neighboring_tiles_within_distance(current_position, bomb_power, game_state)
    adjacent_positions.append(current_position)

    # Check for a clear path without bomb explosion risk
    if not any([adjacent in bombs_positions for adjacent in adjacent_positions]):
        return "SAFE"

    # Calculate tiles affected by bomb explosions and determine reach
    exploded_tiles = [current_position]

    # Power of the bomb
    effect = bomb_power
    for b in game_state["bombs"]:
        exploded_tiles += get_neighboring_tiles_within_distance(b[0], bomb_power, game_state)
        if b[1] + 1 < effect:
            effect = b[1] + 1

    graph = calculate_adjacency_matrix(self, game_state)
    adjacent_positions = get_neighboring_tiles(current_position, effect)

    shortest_path = None
    shortest_distance = 1000

    # Find the shortest safe path to a reachable tile
    for adjacent in adjacent_positions:
        if adjacent not in graph or adjacent in exploded_tiles:
            continue
        try:
            current_shortest_path, current_shortest_distance = find_shortest_path_coordinates(graph, current_position,
                                                                                              adjacent)
            if current_shortest_distance < shortest_distance:
                shortest_path = current_shortest_path
                shortest_distance = current_shortest_distance

        except net.exception.NetworkXNoPath:
            continue

    if not shortest_path:
        # random_choice = np.random.choice(ACTIONS_IDEAS)
        # self.logger.info(f"calculate_going_to_new_tiles: No shortest path {random_choice}")
        return "NO_OTHER_OPTION"

    return_action = select_best_action(self, current_position, shortest_path)
    self.logger.info(f"calculate_going_to_new_tiles: Action returned {return_action}")
    return return_action


# Feature 7: Calculating the direction to coin/crate
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

    # If no coins and crates -> Random?
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

            except net.exception.NetworkXNoPath:
                agent.logger.info("shortest_path_to_coin_or_crate: Crate may be exploded")
                continue

            if current_path_length == 1:
                # TODO: Should I modify the code to add an action to bomb here for this case?
                agent.logger.info(f"shortest_path_to_coin_or_crate: Agent next to a crate")
                return select_best_action(agent, current_position, current_path)

            elif current_path_length < next_crate_position[1]:
                next_crate_position = (current_path, current_path_length)

        # If there are no crates or coins still then return a random action -> highly unlikely I guess
        if next_crate_position == (None, np.inf):
            random_choice = np.random.choice(ACTIONS_IDEAS)
            agent.logger.info(f"shortest_path_to_coin_or_crate: no crate or a good coin still: {random_choice}")
            return random_choice

        return select_best_action(agent, current_position, next_crate_position[0])

    # If there are only good coins
    else:
        coin_path = []

        # Finding the shortest path for all good coins
        for coin_coord in good_coins:
            try:
                current_path, current_path_length = find_shortest_path_coordinates(graph, current_position, coin_coord)
                current_reachable = True
            except net.exception.NetworkXNoPath:
                try:
                    current_path, current_path_length = find_shortest_path_coordinates(graph_with_crates,
                                                                                       current_position, coin_coord)
                    current_reachable = False
                except net.exception.NetworkXNoPath:
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

                except net.exception.NetworkXNoPath:
                    try:
                        current_path_other_agent, current_path_length_other_agent = find_shortest_path_coordinates(
                            graph_with_crates, other_agent_coord, coin_coord)
                        current_other_agent_reachable = False

                    except net.exception.NetworkXNoPath:
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
            return select_best_action(agent, current_position, coin_path[0][0][0])

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
    features_dict = {}

    coin_direction = shortest_path_to_coin_or_crate(self, game_state)
    if coin_direction in ["DOWN", "UP", "RIGHT", "LEFT"]:
        features_dict["Direction_coin/crate"] = coin_direction
    else:
        self.logger.info(f"!!! state_to_features: shortest_path_to_coin_or_crate: Invalid direction: {coin_direction}")

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

    (features_dict["Up"], features_dict["Right"], features_dict["Down"], features_dict["Left"]) = compute_blockage(
        game_state)

    features_dict["Place_Bomb"] = check_bomb_presence(self, game_state)

    features_dict["Crate_Radar"] = check_crate_presence(game_state)

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
