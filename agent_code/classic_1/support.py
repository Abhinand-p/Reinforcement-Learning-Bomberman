# This is a support file containing all methods that support to calculate the features.
# These methods are used to provide additional information when calculating the features.

import numpy as np
import networkx as net
from typing import Tuple, List
from igraph import Graph
from settings import COLS, ROWS

n_rows = ROWS
n_cols = COLS


# Method returns the adjacent tiles for the given position and radius.
# The Idea is to return all adjacent tiles for which additional checks can be performed.
def adjacent_tiles(own_coord, radius) -> List[Tuple[int]]:
    x, y = own_coord
    # Finding adjacent tiles
    adjacent_coordinates = []
    for i in range(1, radius + 1):
        adjacent_coordinates.extend([
            (x, y - i),  # up
            (x + i, y),  # right
            (x, y + i),  # down
            (x - i, y)   # left
        ])
    return adjacent_coordinates


# Method returns the adjacent tiles for the given position and radius
# The Idea is to return the number of tiles with crates and coins around the agent
def adjacent_tiles_within_distance(current_position, max_distance, game_state) -> List[Tuple[int]]:
    directions = ["top", "right_side", "bottom", "left_side"]
    current_x, current_y = current_position[0], current_position[1]
    adjacent_tile = []
    for d, direction in enumerate(directions):
        valid_tiles = []
        for i in range(1, max_distance + 1):
            try:
                if direction == "top":
                    # If the field has crates and coins above the agent
                    if game_state["field"][current_x][current_y + i] in [0, 1]:
                        valid_tiles += [(current_x, current_y + i)]
                    else:
                        break
                elif direction == "right_side":
                    # If the field has crates and coins right to the agent
                    if game_state["field"][current_x + i][current_y] in [0, 1]:
                        valid_tiles += [(current_x + i, current_y)]
                    else:
                        break
                elif direction == "bottom":
                    # If the field has crates and coins below the agent
                    if game_state["field"][current_x][current_y - i] in [0, 1]:
                        valid_tiles += [(current_x, current_y - i)]
                    else:
                        break
                elif direction == "left_side":
                    # If the field has crates and coins left to the agent
                    if game_state["field"][current_x - i][current_y] in [0, 1]:
                        valid_tiles += [(current_x - i, current_y)]
                    else:
                        break
            except IndexError:
                break
        adjacent_tile += valid_tiles
    return adjacent_tile


# Method returns all adjacency matrix's for the game grid.
# The Idea is to return a graph that has a path for the agent without bombs or explosions on the way.
def calculate_adjacency_matrix(self, game_state, consider_crates=True) -> Graph:
    if consider_crates:
        blockers = [(i, j) for i, j in np.ndindex(*game_state["field"].shape) if game_state["field"][i, j] != 0]
    else:
        blockers = [(i, j) for i, j in np.ndindex(*game_state["field"].shape) if game_state["field"][i, j] == -1]
    # blockers are bombs and explosions.
    current_explosions = [(i, j) for i, j in np.ndindex(*game_state["explosion_map"].shape) if
                          game_state["explosion_map"][i, j] != 0]
    bombs = [bombs_coordinate for bombs_coordinate, i in game_state["bombs"]
             if
             bombs_coordinate != game_state["self"][-1] and bombs_coordinate not in [other_agent[-1] for other_agent in
                                                                                     game_state["others"]]]
    blockers += current_explosions
    blockers += bombs
    graph = net.grid_2d_graph(m=n_cols, n=n_rows)
    # Removing all nodes from graph that has blockers
    graph.remove_nodes_from(blockers)
    return graph


# Method returns the shortest path between two coordinates.
# The Idea is to use Dijkstra's algorithm to return the shortest path.
def find_shortest_path_coordinates(graph, source, target) -> Tuple[Graph, int]:
    try:
        shortest_path = net.shortest_path(graph, source=source, target=target, weight=None, method="dijkstra")
    except net.exception.NodeNotFound as e:
        print("!!! Exception raised in find_shortest_path_coordinates !!!")
        raise e

    shortest_path_length = len(shortest_path) - 1
    return shortest_path, shortest_path_length


# Method returns the best action given the current position and the next position.
# The Idea is to return the best action based on its current position and the next position.
# This could make the agent biased to move only in the horizontal direction.
def select_best_action(self, current_coord, next_coords) -> str:
    next_coord = next_coords[1]
    # Horizontal action return.
    if current_coord[1] == next_coord[1]:
        if current_coord[0] - 1 == next_coord[0]:
            return "LEFT"
        elif current_coord[0] + 1 == next_coord[0]:
            return "RIGHT"
    # Vertical action return.
    elif current_coord[0] == next_coord[0]:
        if current_coord[1] - 1 == next_coord[1]:
            return "UP"
        elif current_coord[1] + 1 == next_coord[1]:
            return "DOWN"


# Function to generate a list of adjacent coordinates within a given radius from the agent position.
# The Idea is to utilize list comprehensions to calculate adjacent coordinates.
def get_tiles_direct(own_coord, radius: int) -> List[Tuple[int]]:
    own_coord_x, own_coord_y = own_coord
    adjacent_coordinates = [(own_coord_x + x, own_coord_y + y) for x in range(radius + 1) for y in
                            range(radius + 1 - x) if
                            x <= (n_rows - 2) and y <= (n_cols - 2)]
    return list(set(adjacent_coordinates))
