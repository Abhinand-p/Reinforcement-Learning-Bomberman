# This is the main file containing all features implementations that are used in state_to_features().

import copy as cp
import networkx as net
import numpy as np
from typing import List
from settings import BOMB_POWER
from .support import adjacent_tiles, adjacent_tiles_within_distance, calculate_adjacency_matrix, \
    find_shortest_path_coordinates, select_best_action, get_tiles_direct

ACTIONS_IDEAS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
bomb_power = BOMB_POWER


# Feature 1: Count the number of walls in the immediate surrounding tiles within a given radius.
def count_walls(current_position, game_state, radius):
    return sum(
        1 for coord in adjacent_tiles(current_position, radius)
        if 0 <= coord[0] < game_state["field"].shape[0] and 0 <= coord[1] < game_state["field"].shape[1]
        and game_state["field"][coord] == -1
    )


# Feature 2: Check for bomb presence in the immediate surrounding tiles within a given radius.
# The Idea is to use a custom event BAD_BOMB_ACTION to reward the agent if it is safe to place a bomb.
def check_bomb_presence(self, game_state) -> str:
    if not game_state["self"][2]:
        return 'NO'
    new_game_state = cp.deepcopy(game_state)
    new_game_state["bombs"].append((game_state["self"][-1], 4))
    if calculate_going_to_new_tiles(self, new_game_state) == "NO_OTHER_OPTION":
        return 'NO'
    return 'YES'


# Feature 3: Check for crate presence in the immediate surrounding tiles within a given radius.
# The Idea is to use a custom event OK_BOMB_ACTION, GOOD_BOMB_ACTION and VERY_GOOD_BOMB_ACTION to reward the agent based
# on the number of crates.
def check_crate_presence(game_state) -> str:
    current_position = game_state["self"][-1]
    adjacent = adjacent_tiles_within_distance(current_position, bomb_power, game_state)
    crate_reward = sum(
        1 + (3 if current_position[1] == coord[1] + 1 or current_position[1] == coord[1] - 1
             else (3 if current_position[0] == coord[0] + 1 or current_position[0] == coord[0] - 1
                   else 0))
        for coord in adjacent
        if game_state["field"][coord[0]][coord[1]] == 1
    )
    if crate_reward == 0:
        return 'LOW'
    elif 1 <= crate_reward < 5:
        return 'MID'
    elif crate_reward >= 5:
        return 'HIGH'


# Feature 4: Getting the number of tiles in each direction of the agent.
# The Idea is to check if each tile is safe or unsafe.
def calculate_death_tile(game_state, current_position) -> int:
    all_death_tiles = []
    is_dangerous = []
    if len(game_state["bombs"]) > 0:
        for bomb in game_state["bombs"]:
            bomb_position = bomb[0]
            neighboring_death_tiles = adjacent_tiles_within_distance(bomb_position, bomb_power, game_state)
            if neighboring_death_tiles:
                all_death_tiles += neighboring_death_tiles
        if len(all_death_tiles) > 0:
            for death_tile in all_death_tiles:
                in_danger = current_position == death_tile
                is_dangerous.append(in_danger)
            # One if the agent is on a death tile.
            # Zero if the agent is not on a death tile.
            return int(any(is_dangerous))
    else:
        return 0


# Feature 5: Checking for movable tiles actions on other agents, bombs, explosions
# The Idea is to utilize a custom event AGENT_MOVEMENT_BLOCKED to penalize the agent if the agent performs a blocked
# movement.
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
            next_explosion = adjacent_tiles_within_distance(bomb_position, bomb_power, game_state)
            next_explosion += bomb_position
            explosion.update(next_explosion)
    # Check if the current adjacent tile is blocked by an explosion, a bomb, a wall, or another agent,
    # and mark it as "BLOCK" in the results if it is blocked.
    for i, adjacent in enumerate(adjacent_tiles(current_position, 1)):
        flag_explosion = adjacent in explosion or game_state["explosion_map"][adjacent[0]][adjacent[1]] != 0
        flag_bomb = adjacent in bombs
        neighboring_content = game_state["field"][adjacent[0]][adjacent[1]]
        if neighboring_content != 0 or adjacent in other_agent_positions or flag_explosion or flag_bomb:
            results[i] = "BLOCK"
    return results


# Feature 6: Calculating the actions to escape a bomb
# The Idea is to utilize custom events ESCAPE_BOMB_YES, ESCAPE_BOMB_NO to reward the agent based on blocked actions.
def calculate_going_to_new_tiles(self, game_state) -> str:
    # Get current position
    current_position = game_state["self"][-1]
    # Get Bomb positions
    bombs_positions = [bomb[0] for bomb in game_state["bombs"]]
    # Get adjacent positions
    adjacent_positions = adjacent_tiles_within_distance(current_position, bomb_power, game_state)
    adjacent_positions.append(current_position)
    # Check for a clear path without bomb explosion risk
    if not any([adjacent in bombs_positions for adjacent in adjacent_positions]):
        return "SAFE"
    # Calculate tiles affected by bomb explosions and determine reach
    exploded_tiles = [current_position]
    # Power of the bomb
    effect = bomb_power + 2
    for b in game_state["bombs"]:
        exploded_tiles += adjacent_tiles_within_distance(b[0], bomb_power, game_state)
        if b[1] + 1 < effect:
            effect = b[1] + 1
    graph = calculate_adjacency_matrix(self, game_state)
    adjacent_positions = get_tiles_direct(current_position, effect)
    shortest_path = None
    shortest_distance = 10000  # Initial assuming a very large number
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
    # self.logger.info(f"calculate_going_to_new_tiles: Action returned {return_action}")
    return return_action


# Feature 7: Calculating the action to the nearest coin or crate
# The Idea is to utilize custom events COIN_SEARCH_YES, COIN_SEARCH_NO to reward the agents if the action is not blocked
# to go near a crate or a coin.
def shortest_path_to_coin_or_crate(agent, game_state):
    # current coordinate Classic_1 agent
    current_position = game_state["self"][-1]
    # Extract explosion area positions.
    explosion_area = [(index[0], index[1]) for index, field in np.ndenumerate(game_state["explosion_map"]) if
                      field != 0]
    # Crates present that are not yet exploded.
    crates = [(index[0], index[1]) for index, field in np.ndenumerate(game_state["field"]) if field == 1]
    # Good Coins are those that are not in the explosion_area
    good_coins = [coin for coin in game_state["coins"] if coin not in explosion_area]
    # Adjacency matrix's
    graph = calculate_adjacency_matrix(agent, game_state)
    graph_with_crates = calculate_adjacency_matrix(agent, game_state, consider_crates=False)

    # If no coins and crates -> Random action
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
            # If no other agent alive
            if not any(game_state["others"]):
                coin_path.append(((current_path, current_path_length, current_reachable), (None, np.inf)))
                continue
            for other_agent in game_state["others"]:
                other_agent_path = (None, np.inf)
                other_agent_coord = other_agent[3]
                try:
                    current_path_other_agent, current_path_length_other_agent = find_shortest_path_coordinates(
                        graph, other_agent_coord, coin_coord)
                    agent_reachable = True
                except net.exception.NetworkXNoPath:
                    try:
                        current_path_other_agent, current_path_length_other_agent = find_shortest_path_coordinates(
                            graph_with_crates, other_agent_coord, coin_coord)
                        agent_reachable = False
                    except net.exception.NetworkXNoPath:
                        agent.logger.info("shortest_path_to_coin_or_crate: Corner case 2")
                        continue
                if not agent_reachable:
                    current_path_length_other_agent += 8
                if current_path_length_other_agent < other_agent_path[1]:
                    other_agent_path = (
                        current_path_other_agent, current_path_length_other_agent, agent_reachable)
            coin_path.append(((current_path, current_path_length, current_reachable), other_agent_path))
        # If no coins are still not reachable -> Random action
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
