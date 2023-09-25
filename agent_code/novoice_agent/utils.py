import random
from queue import Queue
import numpy as np
import traceback
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from collections import deque
from .novoice_finalSettings import ACTIONS, NUMBER_OF_FEATURE

class CustomRegressor:
    def __init__(self, estimator, game_handler):
        self.reg_model = [clone(estimator) for _ in range(len(ACTIONS))]
        self.handle = game_handler

    def partial_fit(self, X, y):
        for i in range(len(ACTIONS)):
            if X[i] and y[i]:
                self._fit_regressor(i, X[i], y[i])

    def predict(self, X, action_idx=None):
        X_reshaped = X.reshape((-1, NUMBER_OF_FEATURE))
        if action_idx is None:
            return self._predict_all_regressors(X_reshaped)
        else:
            return self._predict_single_regressor(X_reshaped, action_idx)

    def _fit_regressor(self, index, X, y):
        try:
            self.reg_model[index].partial_fit(X, y)
        except Exception as e:
            self._handle_regressor_error(index, X)

    def _predict_all_regressors(self, X_reshaped):
        predictions = []
        for i in range(len(ACTIONS)):
            predictions.append(self._predict_regressor(i, X_reshaped))
        return np.vstack(predictions).T

    def _predict_single_regressor(self, X_reshaped, action_idx):
        return self._predict_regressor(action_idx, X_reshaped)

    def _predict_regressor(self, index, X_reshaped):
        try:
            return self.reg_model[index].predict(X_reshaped)
        except Exception as e:
            self._handle_regressor_error(index, X_reshaped)
            return np.zeros(len(X_reshaped))

    def _handle_regressor_error(self, index, X):
        self.handle.logger.error(f"Regressor {index} failed to predict.")
        self.handle.logger.error(traceback.format_exc())

def state_to_features(game_state: dict, coordinate_history: deque) -> np.array:
    if game_state is None:
        return None

    arena = game_state['field']
    _, score, bombs_left, (agent_x, agent_y) = game_state['self']
    bomb_positions = [position for (position, _) in game_state['bombs']]
    other_agents_positions = [position for (_, _, _, position) in game_state['others']]
    coin_positions = game_state['coins']

    danger_information = assess_danger(agent_x, agent_y, arena, bomb_positions)

    movement = find_safe_move(agent_x, agent_y, arena, bomb_positions, other_agents_positions)

    others_information, others_reached = move_towards_other_agents(agent_x, agent_y, 5, arena, bomb_positions, other_agents_positions)

    total_coins_info = move_towards_coins(agent_x, agent_y, coin_positions, arena, bomb_positions, other_agents_positions)

    crates_info, total_crates_reached = destroy_most_boxes(agent_x, agent_y, 10, arena, bomb_positions, other_agents_positions)

    target_achieved = int((others_reached or (total_crates_reached and all(others_information == (0, 0)))) and bombs_left and not danger_information)
    current_position_info = evaluate_current_position_action(agent_x, agent_y, arena, bomb_positions)
    
    coordinate_info = min(coordinate_history.count((agent_x, agent_y)), 6)

    features = np.concatenate(
        (danger_information, target_achieved, movement, others_information, total_coins_info, crates_info, current_position_info, coordinate_info),
        axis=None)
    return features.reshape(1, -1)[0]

def assess_danger(agent_x, agent_y, arena, bomb_positions):
    if not is_position_of_type(agent_x, agent_y, arena, 'wall'):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        if bomb_positions:
            for (bomb_x, bomb_y) in bomb_positions:
                if bomb_x == agent_x and bomb_y == agent_y:
                    return True
                for direction in directions:
                    new_x, new_y = bomb_x, bomb_y
                    new_x, new_y = move_forward(new_x, new_y, direction)
                    while (not is_position_of_type(new_x, new_y, arena, 'wall') and
                           abs(new_x - bomb_x) <= 3 and abs(new_y - bomb_y) <= 3):
                        if new_x == agent_x and new_y == agent_y:
                            return True
                        new_x, new_y = move_forward(new_x, new_y, direction)
        return False
    raise ValueError("Error")

def find_safe_move(agent_x, agent_y, arena, bomb_positions, other_agents_positions):
    escapable = False
    if bomb_positions:
        g = {}
        q = Queue()
        ergodic = []
        root_node = ((agent_x, agent_y), (None, None))
        ergodic.append(root_node[0])
        q.put(root_node)
        while not q.empty():
            (ix, iy), parent = q.get()
            g[(ix, iy)] = parent
            if not assess_danger(ix, iy, arena, bomb_positions):
                escapable = True
                break
            neighbours = get_surrounding_feasible_position(ix, iy, arena, bomb_positions, other_agents_positions)
            for neighbour in neighbours:
                if not neighbour in ergodic:
                    ergodic.append(neighbour)
                    q.put((neighbour, (ix, iy)))
        if escapable:
            r = []
            node = (ix, iy)
            if g[node] != (None, None) or node == (agent_x, agent_y):
                while node != (None, None):
                    r.insert(0, node)
                    node = g[node]
            if len(r) > 1:
                next_node = r[1]
                target_direction = (next_node[0] - agent_x, next_node[1] - agent_y)
                return np.array(target_direction)
    return np.zeros(2)

def evaluate_current_position_action(agent_x, agent_y, arena, bomb_positions):
    if not is_position_of_type(agent_x, agent_y, arena, 'wall'):
        crates_destroyed = count_destroyed_boxes(agent_x, agent_y, arena)
        s = 0
        if is_position_of_type(agent_x, agent_y, arena, 'free') and 0 < crates_destroyed < 3 and evade_own_bomb(agent_x, agent_y, arena):
            s = 1
        elif is_position_of_type(agent_x, agent_y, arena, 'free') and 2 < crates_destroyed < 6 and evade_own_bomb(agent_x, agent_y, arena):
            s = 2
        elif is_position_of_type(agent_x, agent_y, arena, 'free') and crates_destroyed > 5 and evade_own_bomb(agent_x, agent_y, arena):
            s = 3
        return s

def evade_own_bomb(agent_x, agent_y, arena):
    if is_position_of_type(agent_x, agent_y, arena, 'free'):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        for direction in directions:
            new_x, new_y = agent_x, agent_y
            new_x, new_y = move_forward(new_x, new_y, direction)
            while is_position_of_type(new_x, new_y, arena, 'free'):
                if abs(agent_x - new_x) > 3 or abs(agent_y - new_y) > 3:
                    return True
                jx, jy, kx, ky = positions_on_both_sides(new_x, new_y, direction)
                if (is_position_of_type(jx, jy, arena, 'free') or
                        is_position_of_type(kx, ky, arena, 'free')):
                    return True
                new_x, new_y = move_forward(new_x, new_y, direction)
        return False
    else:
        raise ValueError("error")

def count_destroyed_boxes(agent_x, agent_y, arena):
    if is_position_of_type(agent_x, agent_y, arena, 'free'):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        crates = 0
        for direction in directions:
            ix, iy = agent_x, agent_y
            ix, iy = move_forward(ix, iy, direction)
            while (not is_position_of_type(ix, iy, arena, 'wall') and
                   abs(agent_x - ix) <= 3 and abs(agent_y - iy) <= 3):
                if is_position_of_type(ix, iy, arena, 'crate'):
                    crates += 1
                ix, iy = move_forward(ix, iy, direction)
        return crates
    else:
        return -1

def destroy_most_boxes(agent_x, agent_y, n, arena, bomb_positions, other_agents_positions):
    ergodic, g = [], {}
    c = []
    q = Queue()
    root_node = ((agent_x, agent_y), 0, (None, None))
    ergodic.append(root_node[0])
    q.put(root_node)
    
    # Define a function to check if a position is too close to an existing bomb's explosion
    def is_position_safe(x, y):
        for bomb_x, bomb_y in bomb_positions:
            if abs(x - bomb_x) + abs(y - bomb_y) <= 3:
                return False
        return True

    while not q.empty():
        (ix, iy), steps, parent = q.get()
        if steps > n:
            continue
        g[(ix, iy)] = parent
        crates = count_destroyed_boxes(ix, iy, arena)
        if crates > 0 and is_position_safe(ix, iy) and evade_own_bomb(ix, iy, arena):
            c.append((crates, steps, (ix, iy)))
        neighbours = get_surrounding_feasible_position(ix, iy, arena, bomb_positions, other_agents_positions)
        for neighb in neighbours:
            if not neighb in ergodic:
                ergodic.append(neighb)
                q.put((neighb, steps + 1, (ix, iy)))
    
    if c:
        value_max = 0
        for crates, steps, (ix, iy) in c:
            value = crates / (4 + steps)
            if value > value_max:
                value_max = value
                cx, cy = ix, iy
        r = []
        node = (cx, cy)
        if g[node] != (None, None) or node == (agent_x, agent_y):
            while node != (None, None):
                r.insert(0, node)
                node = g[node]
        if len(r) > 1:
            nx, ny = r[1]
            if not assess_danger(nx, ny, arena, bomb_positions):
                target_direction = np.array([nx - agent_x, ny - agent_y])
                return target_direction, False
        elif len(r) == 1:
            return np.zeros(2), True
    
    return np.zeros(2), False


def move_towards_coins(agent_x, agent_y, coin_positions, arena, bomb_positions, other_agents_positions):
    reachable = False
    if coin_positions:
        ergodic = []
        g = {}
        q = Queue()
        root_node = ((agent_x, agent_y), 0, (None, None))
        ergodic.append(root_node[0])
        q.put(root_node)
        while not q.empty():
            (ix, iy), steps, parent = q.get()
            g[(ix, iy)] = parent
            if (ix, iy) in coin_positions:
                reachable = True
                cx, cy = ix, iy
                break
            neighbours = get_surrounding_feasible_position(ix, iy, arena, bomb_positions, other_agents_positions)
            for neighb in neighbours:
                if not neighb in ergodic:
                    ergodic.append(neighb)
                    q.put((neighb, steps + 1, (ix, iy)))
        if reachable:
            r = []
            node = (cx, cy)
            if g[node] != (None, None) or node == (agent_x, agent_y):
                while node != (None, None):
                    r.insert(0, node)
                    node = g[node]
            if len(r) > 1:
                nx, ny = r[1]
                if not assess_danger(nx, ny, arena, bomb_positions):
                    target_direction = np.array([nx - agent_x, ny - agent_y])
                    return target_direction
    return np.zeros(2)

def danger_to_other_agents(agent_x, agent_y, arena, other_agents_positions):
    if is_position_of_type(agent_x, agent_y, arena, 'free'):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        danger_to_others = False
        for direction in directions:
            ix, iy = agent_x, agent_y
            ix, iy = move_forward(ix, iy, direction)
            while (not is_position_of_type(ix, iy, arena, 'wall') and abs(agent_x - ix) <= 3 and abs(agent_y - iy) <= 3):
                if (ix, iy) in other_agents_positions:
                    danger_to_others = True
                    break
                ix, iy = move_forward(ix, iy, direction)
        return danger_to_others
    else:
        raise ValueError("error.")

def move_towards_other_agents(agent_x, agent_y, n, arena, bomb_positions, other_agents_positions):
    if other_agents_positions:
        reachable = False
        q = Queue()
        ergodic, g = [], {}
        root_node = ((agent_x, agent_y), 0, (None, None))
        ergodic.append(root_node[0])
        q.put(root_node)
        while not q.empty():
            (ix, iy), steps, parent = q.get()
            g[(ix, iy)] = parent
            if steps > n:
                continue
            if (danger_to_other_agents(ix, iy, arena, other_agents_positions) and evade_own_bomb(ix, iy, arena)):
                reachable = True
                cx, cy = ix, iy
                break
            neighbours = get_surrounding_feasible_position(ix, iy, arena, bomb_positions, other_agents_positions)
            for neighb in neighbours:
                if not neighb in ergodic:
                    ergodic.append(neighb)
                    q.put((neighb, steps + 1, (ix, iy)))

        if reachable:
            r = []
            node = (cx, cy)
            if g[node] != (None, None) or node == (agent_x, agent_y):
                while node != (None, None):
                    r.insert(0, node)
                    node = g[node]
            if len(r) > 1:
                nx, ny = r[1]
                if not assess_danger(nx, ny, arena, bomb_positions):
                    target_direction = np.array([nx - agent_x, ny - agent_y])
                    return target_direction, False
            elif len(r) == 1:
                return np.zeros(2), True
    return np.zeros(2), False

# def get_valid_actions(game_state):
#     _, _, bombs_left, (agent_x, agent_y) = game_state['self']
#     arena = game_state['field']
#     coin_positions = game_state.get('coins', [])
#     bomb_positions = [position for (position, _) in game_state.get('bombs', [])]
#     other_agents_positions = [position for (_, _, _, position) in game_state.get('others', [])]
#     bomb_map = game_state.get('explosion_map', np.zeros(arena.shape))
#     has_crates = any(arena[position] == 1 for position in np.ndindex(arena.shape))
#     directions = [(agent_x, agent_y - 1), (agent_x + 1, agent_y), (agent_x, agent_y + 1), (agent_x - 1, agent_y), (agent_x, agent_y)]

#     valid_actions = []
#     mask = np.zeros(len(ACTIONS))

#     if not has_crates:
#         for i, direction in enumerate(directions):
#             if (arena[direction] == 0 and
#                     not direction in other_agents_positions and
#                     not direction in bomb_positions):
#                 valid_actions.append(ACTIONS[i])
#                 mask[i] = 1
#     else:
#         danger_status = np.zeros(len(directions))
#         for i, (ix, iy) in enumerate(directions):
#             if not is_position_of_type(ix, iy, arena, 'wall'):
#                 danger_status[i] = int(assess_danger(ix, iy, arena, bomb_positions))
#             else:
#                 danger_status[i] = -1

#         if not any(danger_status == 0):
#             danger_status = np.zeros(len(directions))
#             danger_status[-1] = 1

#         for i, direction in enumerate(directions):
#             if (arena[direction] == 0 and
#                     bomb_map[direction] < 1 and
#                     not direction in other_agents_positions and
#                     not direction in bomb_positions and
#                     danger_status[i] == 0):
#                 valid_actions.append(ACTIONS[i])
#                 mask[i] = 1

#         if bombs_left:
#             valid_actions.append(ACTIONS[-1])
#             mask[-1] = 1

#     mask = (mask == 1)
#     valid_actions = np.array(valid_actions)

#     if len(valid_actions) == 0:
#         return np.ones(len(ACTIONS)) == 1, ACTIONS
#     else:
#         return mask, valid_actions

def get_valid_actions(game_state):
    _, _, bombs_left, (agent_x, agent_y) = game_state['self']
    arena = game_state['field']
    coin_positions = game_state.get('coins', [])
    bomb_positions = [position for (position, _) in game_state.get('bombs', [])]
    other_agents_positions = [position for (_, _, _, position) in game_state.get('others', [])]
    bomb_map = game_state.get('explosion_map', np.zeros(arena.shape))
    has_crates = any(arena[position] == 1 for position in np.ndindex(arena.shape))
    directions = [(agent_x, agent_y - 1), (agent_x + 1, agent_y), (agent_x, agent_y + 1), (agent_x - 1, agent_y), (agent_x, agent_y)]

    valid_actions = []
    mask = np.zeros(len(ACTIONS))

    # Check if the agent is in a corner
    is_in_corner = (agent_x, agent_y) in [(1, 1), (1, 15), (15, 1), (15, 15)]

    if not has_crates:
        for i, direction in enumerate(directions):
            if (arena[direction] == 0 and
                    not direction in other_agents_positions and
                    not direction in bomb_positions):
                if not is_in_corner or ACTIONS[i] != 'BOMB':
                    valid_actions.append(ACTIONS[i])
                    mask[i] = 1
    else:
        danger_status = np.zeros(len(directions))
        for i, (ix, iy) in enumerate(directions):
            if not is_position_of_type(ix, iy, arena, 'wall'):
                danger_status[i] = int(assess_danger(ix, iy, arena, bomb_positions))
            else:
                danger_status[i] = -1

        if not any(danger_status == 0):
            danger_status = np.zeros(len(directions))
            danger_status[-1] = 1

        # Check if there are nearby coins
        nearby_coins = [pos for pos in coin_positions if manhattan_distance((agent_x, agent_y), pos) <= 3]

        if nearby_coins:
            # Prioritize moving towards coins
            nearest_coin = min(nearby_coins, key=lambda pos: manhattan_distance((agent_x, agent_y), pos))
            move_direction = get_direction((agent_x, agent_y), nearest_coin)

            if move_direction and move_direction != 'WAIT':
                valid_actions.append(move_direction)
                mask[ACTIONS.index(move_direction)] = 1

        for i, direction in enumerate(directions):
            if (arena[direction] == 0 and
                    bomb_map[direction] < 1 and
                    not direction in other_agents_positions and
                    not direction in bomb_positions and
                    danger_status[i] == 0):
                # Check if the agent is not too close to an existing bomb's explosion radius
                if not is_too_close_to_explosion(agent_x, agent_y, bomb_positions, bomb_map):
                    if not (is_in_corner and ACTIONS[i] == 'BOMB'):
                        valid_actions.append(ACTIONS[i])
                        mask[i] = 1

        if bombs_left and not is_in_corner:
            valid_actions.append(ACTIONS[-1])
            mask[-1] = 1

    mask = (mask == 1)
    valid_actions = np.array(valid_actions)

    if len(valid_actions) == 0:
        return np.ones(len(ACTIONS)) == 1, ACTIONS
    else:
        return mask, valid_actions

# def get_valid_actions(game_state):
#     _, _, bombs_left, (agent_x, agent_y) = game_state['self']
#     arena = game_state['field']
#     coin_positions = game_state.get('coins', [])
#     bomb_positions = [position for (position, _) in game_state.get('bombs', [])]
#     other_agents_positions = [position for (_, _, _, position) in game_state.get('others', [])]
#     bomb_map = game_state.get('explosion_map', np.zeros(arena.shape))
#     has_crates = any(arena[position] == 1 for position in np.ndindex(arena.shape))
#     directions = [(agent_x, agent_y - 1), (agent_x + 1, agent_y), (agent_x, agent_y + 1), (agent_x - 1, agent_y), (agent_x, agent_y)]

#     valid_actions = []
#     mask = np.zeros(len(ACTIONS))

#     # Check if the agent is in a corner
#     is_in_corner = (agent_x, agent_y) in [(1, 1), (1, 15), (15, 1), (15, 15)]

#     if not has_crates:
#         for i, direction in enumerate(directions):
#             if (arena[direction] == 0 and
#                     not direction in other_agents_positions and
#                     not direction in bomb_positions):
#                 if not is_in_corner or ACTIONS[i] != 'BOMB':
#                     valid_actions.append(ACTIONS[i])
#                     mask[i] = 1
#     else:
#         danger_status = np.zeros(len(directions))
#         for i, (ix, iy) in enumerate(directions):
#             if not is_position_of_type(ix, iy, arena, 'wall'):
#                 danger_status[i] = int(assess_danger(ix, iy, arena, bomb_positions))
#             else:
#                 danger_status[i] = -1

#         if not any(danger_status == 0):
#             danger_status = np.zeros(len(directions))
#             danger_status[-1] = 1

#         # Check if there are nearby coins
#         nearby_coins = [pos for pos in coin_positions if manhattan_distance((agent_x, agent_y), pos) <= 3]

#         if nearby_coins:
#             # Prioritize moving towards coins
#             nearest_coin = min(nearby_coins, key=lambda pos: manhattan_distance((agent_x, agent_y), pos))
#             move_direction = get_direction((agent_x, agent_y), nearest_coin)

#             if move_direction and move_direction != 'WAIT':
#                 valid_actions.append(move_direction)
#                 mask[ACTIONS.index(move_direction)] = 1
#         else:
#             # Check if there are nearby crates
#             nearby_crates = [(ix, iy) for (ix, iy) in directions if arena[ix, iy] == 1]

#             if nearby_crates:
#                 # Prioritize bombing nearby crates
#                 bomb_direction = get_direction((agent_x, agent_y), nearby_crates[0])

#                 if bomb_direction and bomb_direction != 'WAIT':
#                     valid_actions.append(bomb_direction)
#                     mask[ACTIONS.index(bomb_direction)] = 1

#         for i, direction in enumerate(directions):
#             if (arena[direction] == 0 and
#                     bomb_map[direction] < 1 and
#                     not direction in other_agents_positions and
#                     not direction in bomb_positions and
#                     danger_status[i] == 0):
#                 # Check if the agent is not too close to an existing bomb's explosion radius
#                 if not is_too_close_to_explosion(agent_x, agent_y, bomb_positions, bomb_map):
#                     if not (is_in_corner and ACTIONS[i] == 'BOMB'):
#                         valid_actions.append(ACTIONS[i])
#                         mask[i] = 1

#         if bombs_left and not is_in_corner:
#             valid_actions.append(ACTIONS[-1])
#             mask[-1] = 1

#     mask = (mask == 1)
#     valid_actions = np.array(valid_actions)

#     if len(valid_actions) == 0:
#         return np.ones(len(ACTIONS)) == 1, ACTIONS
#     else:
#         return mask, valid_actions



def is_safe_to_drop_bomb(agent_x, agent_y, direction, bomb_positions, bomb_map):
    # Calculate the position after moving in the specified direction
    new_x, new_y = move_forward(agent_x, agent_y, direction)
    
    # Check if there's already a bomb or a crate in the target position
    if (new_x, new_y) in bomb_positions:
        return False
    if bomb_map[new_x, new_y] > 0:
        return False
    
    # Check if the agent will be safe after dropping the bomb
    if is_too_close_to_explosion(new_x, new_y, bomb_positions, bomb_map):
        return False
    
    return True



def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_direction(start, end):
    if start[0] < end[0]:
        return 'RIGHT'
    elif start[0] > end[0]:
        return 'LEFT'
    elif start[1] < end[1]:
        return 'DOWN'
    elif start[1] > end[1]:
        return 'UP'
    else:
        return 'WAIT'


def is_too_close_to_explosion(agent_x, agent_y, bomb_positions, bomb_map, min_distance=100):
    for bomb_x, bomb_y in bomb_positions:
        if (
            abs(agent_x - bomb_x) + abs(agent_y - bomb_y) <= min_distance
            and bomb_map[agent_x, agent_y] > 0
        ):
            return True
    return False

def get_surrounding_feasible_position(agent_x, agent_y, arena, bomb_positions, other_agents_positions):
    surround = []
    directions = [(agent_x, agent_y - 1), (agent_x + 1, agent_y), (agent_x, agent_y + 1), (agent_x - 1, agent_y)]
    random.shuffle(directions)
    for (new_x, new_y) in directions:
        if not (new_x, new_y) in bomb_positions and (is_position_of_type(new_x, new_y, arena, 'free') and not (new_x, new_y) in other_agents_positions):
            surround.append((new_x, new_y))
    return surround

def check_direction(direction, self_action):
    return ((all(direction == (0, 1)) and self_action == 'DOWN') or
            (all(direction == (1, 0)) and self_action == 'RIGHT') or
            (all(direction == (0, -1)) and self_action == 'UP') or
            (all(direction == (-1, 0)) and self_action == 'LEFT'))

def is_position_of_type(x, y, arena, object):
    if object == 'crate':
        return arena[x, y] == 1
    elif object == 'free':
        return arena[x, y] == 0
    elif object == 'wall':
        return arena[x, y] == -1

def move_forward(x, y, direction):
    if direction == 'LEFT':
        x -= 1
    elif direction == 'RIGHT':
        x += 1
    elif direction == 'UP':
        y -= 1
    elif direction == 'DOWN':
        y += 1
    return x, y

def positions_on_both_sides(x, y, direction):
    if direction == 'UP' or direction == 'DOWN':
        jx, jy, kx, ky = x + 1, y, x - 1, y
    elif direction == 'RIGHT' or direction == 'LEFT':
        jx, jy, kx, ky = x, y + 1, x, y - 1
    else:
        raise ValueError("error")
    return jx, jy, kx, ky

