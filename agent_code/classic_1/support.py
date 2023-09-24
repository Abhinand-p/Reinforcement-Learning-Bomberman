# This file contains all methods that support to calculate the features
import numpy as np
import networkx as net
from typing import Tuple, List
from igraph import Graph
from settings import COLS, ROWS

n_rows = ROWS
n_cols = COLS


# TODO: Should I limit the negative coordinates here?
# Returns the adjacent tiles for the given position and radius
def get_neighboring_tiles(own_coord, radius) -> List[Tuple[int]]:
    x, y = own_coord

    # Finding neighbouring tiles
    adjacent_coordinates = []
    for i in range(1, radius + 1):
        adjacent_coordinates.extend([
            (x, y - i),  # up
            (x + i, y),  # right
            (x, y + i),  # down
            (x - i, y)  # left
        ])
    return adjacent_coordinates


# Returns the adjacent tiles for the given position and radius
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


# Calculates all adjacency matrix's for the game grid.
# Logic has been verified
def calculate_adjacency_matrix(self, game_state, consider_crates=True) -> Graph:
    if consider_crates:
        blockers = [(i, j) for i, j in np.ndindex(*game_state["field"].shape) if game_state["field"][i, j] != 0]
    else:
        blockers = [(i, j) for i, j in np.ndindex(*game_state["field"].shape) if game_state["field"][i, j] == -1]

    current_explosions = [(i, j) for i, j in np.ndindex(*game_state["explosion_map"].shape) if
                          game_state["explosion_map"][i, j] != 0]

    bombs = [
        bombs_coordinate
        for bombs_coordinate, i in game_state["bombs"]
        if bombs_coordinate != game_state["self"][-1] and bombs_coordinate not in [other_agent[-1] for other_agent in
                                                                                   game_state["others"]]
    ]

    blockers += current_explosions
    blockers += bombs

    # self.logger.info(f"Blockers matrix: {blockers}")

    graph = net.grid_2d_graph(m=n_cols, n=n_rows)

    # Removing nodes that represent blockers
    graph.remove_nodes_from(blockers)
    return graph


# A helper function to get the shortest path between two coordinates.
def find_shortest_path_coordinates(graph, source, target) -> Tuple[Graph, int]:
    try:
        shortest_path = net.shortest_path(graph, source=source, target=target, weight=None, method="dijkstra")
    except net.exception.NodeNotFound as e:
        print("!!! Exception raised in find_shortest_path_coordinates !!!")
        raise e

    shortest_path_length = len(shortest_path) - 1
    return shortest_path, shortest_path_length


# A helper function to select the best action given the current position
def select_best_action(self, current_coord, next_coords) -> str:
    next_coord = next_coords[1]

    if current_coord[1] == next_coord[1]:
        if current_coord[0] - 1 == next_coord[0]:
            return "LEFT"
        elif current_coord[0] + 1 == next_coord[0]:
            return "RIGHT"

    elif current_coord[0] == next_coord[0]:
        if current_coord[1] - 1 == next_coord[1]:
            return "UP"
        elif current_coord[1] + 1 == next_coord[1]:
            return "DOWN"
