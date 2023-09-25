import heapq
import os
from collections import deque
from queue import Queue
import random
import numpy as np
import pickle
from sklearn.linear_model import SGDRegressor
from .novoice_finalSettings import ACTIONS, EPSILON_END_VALUE, EPSILON_START_VALUE
from .utils import state_to_features, get_valid_actions,CustomRegressor
# Define the MOVE_ACTIONS dictionary to map action names to their corresponding (dx, dy) movements.
# Define the MOVE_ACTIONS dictionary to map action names to their corresponding (dx, dy) movements.
MOVE_ACTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "WAIT": (0, 0),  # You can include this for completeness
}


def setup(self):
    
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 7)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    #
    self.action_history = deque([], 20)
    self.actions = ACTIONS
    self.epsilon = EPSILON_START_VALUE 
    # setup models
    FINAL_MODEL_NAME = "model-1000"
    model_file = os.path.join('./models', FINAL_MODEL_NAME)
    self.logger.info(f"my_variable value: {os.path.isfile(model_file)}")
    if os.path.isfile(model_file):
        self.logger.info("Loading model from saved state.Sid_Model_from_saved_state")
        with open(model_file, "rb") as file:
            self.model = pickle.load(file)
        self.model_is_fitted = True
    if self.train:
        self.logger.info("Setting up model from scratch. Sid_Model_Frm_Scratch")
        self.model = CustomRegressor(SGDRegressor(alpha=0.0001, warm_start=True),self)
        self.model_is_fitted = False
        with open(model_file, "wb") as file:
            pickle.dump(self.model, file)
        
def reset_self(self):
    """Reset specific agent attributes to their initial values."""
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 7)
    self.ignore_others_timer = 0
    self.action_history = deque([], 20)

# def act(self, game_state):
#     """
#     Picking action according to model
#     Args:
#         self:
#         game_state:

#     Returns:
#     """
#     if game_state["round"] != self.current_round:
#         reset_self(self)
#         self.current_round = game_state["round"]
#     _, _, bombs_left, (x, y) = game_state['self']
#     self.coordinate_history.append((x, y))
#     mask, valid_actions = get_valid_actions(game_state)
#     if self.train:
#         random_prob = self.epsilon
#     else:
#         random_prob = EPSILON_END
#     if random.random() < random_prob or not self.model_is_fitted:
#         #execute_action = np.random.choice(valid_actions) if len(valid_actions) > 0 else "WAIT"
#         execute_action = np.random.choice(valid_actions)
#         #self.logger.debug(f"Choose action uniformly at random. valid action {valid_actions}, execute action {execute_action}")
#     else:
#         q_values = self.model.predict(state_to_features(game_state, self.coordinate_history))[0]
#         execute_action = valid_actions[np.argmax(q_values[mask])]
#         #self.logger.info(f'Choose action according to model: {self.actions[np.argmax(q_values)]}, valid execution action: {execute_action}')
#         #self.logger.info(f'q_values: {q_values}, mask: {mask}')
#     self.action_history.append((execute_action, x, y))
#     return execute_action

# def act(self, game_state):
#     """
#     Picking action according to model
#     Args:
#         self:
#         game_state:

#     Returns:
#     """
#     if game_state["round"] != self.current_round:
#         reset_self(self)
#         self.current_round = game_state["round"]
#     _, _, bombs_left, (x, y) = game_state['self']
#     self.coordinate_history.append((x, y))
#     mask, valid_actions = get_valid_actions(game_state)

#     # Check if it's safe to drop a bomb
#     if "BOMB" in valid_actions:
#         # Calculate the blast range of the bomb
#         blast_range = game_state['self'][2]  # Assuming you have the bomb range in game_state
#         # Check if any explosion will hit the agent's current position
#         if any((x, y) in game_state['explosion_map'][i] for i in range(1, blast_range + 1)):
#             # If the agent's current position is in danger, don't drop a bomb
#             valid_actions.remove("BOMB")

#     if self.train:
#         random_prob = self.epsilon
#     else:
#         random_prob = EPSILON_END
#     if random.random() < random_prob or not self.model_is_fitted:
#         execute_action = np.random.choice(valid_actions)
#     else:
#         q_values = self.model.predict(state_to_features(game_state, self.coordinate_history))[0]
#         execute_action = valid_actions[np.argmax(q_values[mask])]
#     self.action_history.append((execute_action, x, y))
#     return execute_action
# def act(self, game_state):
#     """
#     Picking action according to model
#     Args:
#         self:
#         game_state:

#     Returns:
#     """
#     if game_state["round"] != self.current_round:
#         reset_self(self)
#         self.current_round = game_state["round"]
#     _, _, bombs_left, (x, y) = game_state['self']
#     self.coordinate_history.append((x, y))
#     mask, valid_actions = get_valid_actions(game_state)

#     # Check if it's safe to drop a bomb
#     if "BOMB" in valid_actions:
#         # Calculate the blast range of the bomb
#         blast_range = game_state['self'][2]  # Assuming you have the bomb range in game_state
#         # Check if any explosion will hit the agent's current position
#         if any((x, y) in game_state['explosion_map'][i] for i in range(1, blast_range + 1)):
#             # If the agent's current position is in danger, try to find a safe place

#             # First, check if there is a safe direction to move
#             safe_direction = None
#             for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
#                 new_x, new_y = get_new_position(x, y, direction)
#                 if is_safe_location(game_state, new_x, new_y):
#                     safe_direction = direction
#                     break
            
#             if safe_direction is not None:
#                 return safe_direction  # Move to a safe location

#             # If there's no safe direction, go in the opposite direction of the explosion
#             opposite_direction = get_opposite_direction(game_state, x, y)
#             if opposite_direction is not None:
#                 return opposite_direction

#             # If no safe place and no opposite direction, don't drop a bomb
#             valid_actions.remove("BOMB")

#     # Check if it's safe to collect a coin
#     if np.any(np.array(game_state['field'], dtype=str) == "COIN"):
#         # There is at least one "COIN" in the game field
#         # Find the nearest coin and move towards it
#         nearest_coin_direction = get_nearest_coin_direction(game_state, x, y)
#         if nearest_coin_direction:
#             return nearest_coin_direction

#     # If no coins are nearby or it's not safe to collect them, continue with the previous logic
#     if self.train:
#         random_prob = self.epsilon
#     else:
#         random_prob = EPSILON_END
#     if random.random() < random_prob or not self.model_is_fitted:
#         execute_action = np.random.choice(valid_actions)
#     else:
#         q_values = self.model.predict(state_to_features(game_state, self.coordinate_history))[0]
#         execute_action = valid_actions[np.argmax(q_values[mask])]
#     self.action_history.append((execute_action, x, y))
#     return execute_action
def act(self, game_state):
    """
    Picking action according to model
    Args:
        self:
        game_state:

    Returns:
    """
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    _, _, bombs_left, (x, y) = game_state['self']
    self.coordinate_history.append((x, y))
    mask, valid_actions = get_valid_actions(game_state)

    # Check if it's safe to collect a coin
    if np.any(np.array(game_state['field'], dtype=str) == "COIN"):
        # There is at least one "COIN" in the game field
        # Find the nearest coin and move towards it
        nearest_coin_direction = get_nearest_coin_direction(game_state, x, y)
        if nearest_coin_direction:
            return nearest_coin_direction

    # Check if it's safe to drop a bomb
    if "BOMB" in valid_actions:
        # Calculate the blast range of the bomb
        blast_range = game_state['self'][2]  # Assuming you have the bomb range in game_state
        # Check if any explosion will hit the agent's current position
        if any((x, y) in game_state['explosion_map'][i] for i in range(1, blast_range + 1)):
            # If the agent's current position is in danger, try to find a safe place

            # First, check if there is a safe direction to move
            safe_direction = None
            for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                new_x, new_y = get_new_position(x, y, direction)
                if is_safe_location(game_state, new_x, new_y):
                    safe_direction = direction
                    break

            if safe_direction is not None:
                return safe_direction  # Move to a safe location

            # If there's no safe direction, go in the opposite direction of the explosion
            opposite_direction = get_opposite_direction(game_state, x, y)
            if opposite_direction is not None:
                return opposite_direction

            # If no safe place and no opposite direction, don't drop a bomb
            valid_actions.remove("BOMB")

    # Check if the agent is cornered by crates and walls
    if is_cornered(game_state, x, y):
        # If cornered and no safe escape, don't drop a bomb
        valid_actions.remove("BOMB")

    # If no coins are nearby or it's not safe to collect them, continue with the previous logic
    if self.train:
        random_prob = self.epsilon
    else:
        random_prob = EPSILON_END_VALUE
    if random.random() < random_prob or not self.model_is_fitted:
        execute_action = np.random.choice(valid_actions)
    else:
        q_values = self.model.predict(state_to_features(game_state, self.coordinate_history))[0]
        execute_action = valid_actions[np.argmax(q_values[mask])]

    # Check if the agent would drop a bomb when in a corner
    if execute_action == "BOMB" and is_cornered(game_state, x, y):
        valid_actions.remove("BOMB")  # Remove bomb action to avoid self-harm

    self.action_history.append((execute_action, x, y))
    return execute_action

def is_cornered(game_state, x, y):
    # Get the game field and its dimensions
    field = game_state['field']
    field_width = len(field[0])
    field_height = len(field)

    # Check if there are crates or walls on all four sides of the agent
    up = field[y - 1][x] == "WALL" or field[y - 1][x] == "CRATE" if y > 0 else True
    down = field[y + 1][x] == "WALL" or field[y + 1][x] == "CRATE" if y < field_height - 1 else True
    left = field[y][x - 1] == "WALL" or field[y][x - 1] == "CRATE" if x > 0 else True
    right = field[y][x + 1] == "WALL" or field[y][x + 1] == "CRATE" if x < field_width - 1 else True

    # If there are crates or walls on all four sides, the agent is cornered
    if up and down and left and right:
        # Check if there's a coin nearby
        coin_direction = get_nearest_coin_direction(game_state, x, y)
        if coin_direction:
            return False  # There's a coin nearby, don't drop a bomb
        return True  # No coin nearby, consider dropping a bomb
    return False  # Not completely cornered, don't drop a bomb


def drop_bomb_to_collect_coin(game_state, x, y):
    """
    Drop a bomb to collect a nearby coin and create an escape route if possible.

    Args:
        game_state: The current game state.
        x: X-coordinate of the agent's position.
        y: Y-coordinate of the agent's position.

    Returns:
        True if a bomb was dropped, False otherwise.
    """
    # Check if there's a coin nearby and if the agent has bombs left
    if (
        np.any(np.array(game_state['field'], dtype=str) == "COIN")
        and game_state['self'][2] > 0  # Bombs left
    ):
        # Iterate through possible directions
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            new_x, new_y = get_new_position(x, y, direction)
            # Check if the new position contains a coin
            if (
                is_valid_position(game_state, new_x, new_y)
                and game_state['field'][new_y][new_x] == "COIN"
            ):
                return "BOMB"  # Drop a bomb to collect the nearby coin and create an escape route
    return False  # No bomb dropped

def is_valid_position(game_state, x, y):
    """
    Check if a position (x, y) is valid within the game boundaries.

    Args:
        game_state: The current game state.
        x: X-coordinate of the position to check.
        y: Y-coordinate of the position to check.

    Returns:
        True if the position is valid, False otherwise.
    """
    field_width = len(game_state['field'][0])
    field_height = len(game_state['field'])
    return 0 <= x < field_width and 0 <= y < field_height

def get_new_position(x, y, direction):
    if direction == "UP":
        return x, y - 1
    elif direction == "DOWN":
        return x, y + 1
    elif direction == "LEFT":
        return x - 1, y
    elif direction == "RIGHT":
        return x + 1, y
    else:
        return x, y


def get_new_position(x, y, direction):
    if direction == "UP":
        return x, y - 1
    elif direction == "DOWN":
        return x, y + 1
    elif direction == "LEFT":
        return x - 1, y
    elif direction == "RIGHT":
        return x + 1, y
    else:
        return x, y


def is_safe_location(game_state, x, y):
    """
    Check if a location (x, y) is safe in the game state.

    Args:
        game_state: The current game state.
        x: X-coordinate of the location to check.
        y: Y-coordinate of the location to check.

    Returns:
        True if the location is safe, False otherwise.
    """
    # Check if the location is within the game board boundaries
    if x < 0 or x >= game_state['field'].shape[1] or y < 0 or y >= game_state['field'].shape[0]:
        return False

    # Check if there are no bombs at the location
    for bomb in game_state['bombs']:
        if bomb[0] == (x, y):
            return False

    # Check if there are no flames (explosions) at the location
    for flame in game_state['explosion_map']:
        if (x, y) in flame:
            return False

    # Check if there are no other agents at the location
    for agent in game_state['others']:
        if agent[3] == (x, y):
            return False

    # If none of the above conditions are met, the location is safe
    return True

def get_opposite_direction(game_state, x, y):
    """
    Get the opposite direction of the nearest explosion relative to the agent's position.

    Args:
        game_state: The current game state.
        x: X-coordinate of the agent's position.
        y: Y-coordinate of the agent's position.

    Returns:
        The opposite direction (e.g., "UP" if the nearest explosion is "DOWN").
        Returns None if no explosion is nearby.
    """
    nearest_explosion_direction = None
    nearest_explosion_distance = float('inf')

    # Iterate through explosion locations and find the nearest one
    for explosion in game_state['explosion_map']:
        for ex_x, ex_y in explosion:
            distance = abs(x - ex_x) + abs(y - ex_y)
            if distance < nearest_explosion_distance:
                nearest_explosion_distance = distance
                # Determine the direction of the nearest explosion
                if ex_x > x:
                    nearest_explosion_direction = "RIGHT"
                elif ex_x < x:
                    nearest_explosion_direction = "LEFT"
                elif ex_y > y:
                    nearest_explosion_direction = "DOWN"
                elif ex_y < y:
                    nearest_explosion_direction = "UP"

    # Return the opposite direction if a nearest explosion was found
    if nearest_explosion_direction:
        opposite_directions = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        return opposite_directions[nearest_explosion_direction]

    # Return None if no explosion is nearby
    return None

def get_nearest_coin_direction(game_state, x, y):
    """
    Get the direction to move towards the nearest coin.

    Args:
        game_state: The current game state.
        x: X-coordinate of the agent's position.
        y: Y-coordinate of the agent's position.

    Returns:
        The direction to move towards the nearest coin or None if no coins are nearby.
    """
    coin_positions = [(i, j) for i in range(game_state['field'].shape[0]) for j in range(game_state['field'].shape[1]) if game_state['field'][i][j] == 'COIN']
    if not coin_positions:
        return None

    # Calculate distances to all coins and find the nearest one
    nearest_coin = min(coin_positions, key=lambda pos: abs(x - pos[0]) + abs(y - pos[1]))

    # Determine the direction to move towards the nearest coin
    if nearest_coin[0] > x:
        return "DOWN"
    elif nearest_coin[0] < x:
        return "UP"
    elif nearest_coin[1] > y:
        return "RIGHT"
    elif nearest_coin[1] < y:
        return "LEFT"
    else:
        return None  # Agent is already at the nearest coin

