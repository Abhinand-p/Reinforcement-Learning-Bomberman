from collections import namedtuple, deque, Counter
import copy
import pickle
from typing import List
import random
import numpy as np
import events as e
import settings as s
from .utils import state_to_features, check_direction
from .novoice_finalSettings import (SAVE_MODEL, TRAINING_ROUNDS,
                           BATCH_SIZE, TRANSITION_BUFFER_SIZE_MAX,
                           DECAY_GAMMA_VALUE, EPSILON_END_VALUE, EPSILON_DECAY_VALUE,
                           N_STEP_LEARNING_RATE, N_STEP, PRIORITY_LEARNING, PRIORITY_RATIO)

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
loop_old = 0
# Events
BOMBED_1_TO_2_CRATES = "BOMBED_1_TO_2_CRATES"
BOMBED_3_TO_5_CRATES = "BOMBED_3_TO_5_CRATES"
BOMBED_5_PLUS_CRATES = "BOMBED_5_PLUS_CRATES"
GET_IN_LOOP = "GET_IN_LOOP"
PLACEHOLDER_EVENT = "PLACEHOLDER"
ESCAPE = "ESCAPE"
NOT_ESCAPE = "NOT_ESCAPE"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
AWAY_FROM_CRATE = "AWAY_FROM_CRATE"
SURVIVED_STEP = "SURVIVED_STEP"
DESTROY_TARGET = "DESTROY_TARGET"
MISSED_TARGET = "MISSED_TARGET"
WAITED_NECESSARILY = "WAITED_NECESSARILY"
WAITED_UNNECESSARILY = "WAITED_UNNECESSARILY"
CLOSER_TO_PLAYERS = "CLOSER_TO_PLAYERS"
AWAY_FROM_PLAYERS = "AWAY_FROM_PLAYERS"



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_BUFFER_SIZE_MAX)
    self.n_step_buffer = deque(maxlen=N_STEP)
    # evaluation initialization
    self.game_nr = 0
    self.historic_data = {
        'rewards': [],
        'coins': [],
        'crates': [],
        'enemies': [],
        'exploration': [],
        'games': [],
        'offline_acc': [],
        'offline_rewards': []
    }
    # Initialization
    self.rewards    = 0
    self.coins   = 0
    self.crates  = 0
    self.enemies    = 0
    self.exploration = 0
    self.model_action = []
    self.online_action = []
    self.model_rewards = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    new_events = events
    self_coor = None
    if self_action is not None:
        self_coor = self.coordinate_history.pop()
    old_feature = state_to_features(old_game_state, self.coordinate_history)
    if self_coor is not None:
        self.coordinate_history.append(self_coor)
    if old_game_state is not None and self_action is not None:
        next_feature = state_to_features(new_game_state, self.coordinate_history)
        new_events = get_bomberman_events(old_feature, self_action, events)
        if N_STEP_LEARNING_RATE:
            self.n_step_buffer.append(Transition(old_feature, self_action, next_feature, reward_from_events(self, new_events)))
            if len(self.n_step_buffer) >= N_STEP:
                reward_arr = np.array([self.n_step_buffer[i][-1] for i in range(N_STEP)])
                # Sum with the discount factor to get the accumulated rewards over N_STEP transitions.
                reward = ((DECAY_GAMMA_VALUE) ** np.arange(N_STEP)).dot(reward_arr)
                self.transitions.append(
                    Transition(self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[-1][2], reward))
        else:
            self.transitions.append(Transition(old_feature, self_action, next_feature, reward_from_events(self, new_events)))

    # Check if placing a bomb would lead to self-destruction when not in the four corners
    if self_action == 'BOMB' and 'KILLED_SELF' in new_events and not is_in_four_corners(new_game_state):
        self.logger.debug('Avoiding self-bombing')
        self_action = 'WAIT'  # Replace self-bomb action with a wait action

    # collect evaluation data
    update_bomberman_stats(self, new_events, False)
    

def is_in_four_corners(game_state: dict) -> bool:
    """
    Check if the agent's position is in one of the four corners of the game board.
    """
    pos = game_state['self'][3]
    corners = [(1, 1), (1, 15), (15, 1), (15, 15)]  # Coordinates of the four corners
    return pos in corners



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.logger.debug('Encountered event(s) AAAAAAAA')
    #self.logger.debug(f'end of round {last_game_state is None}, {last_action is None} in final step')
    self.game_nr += 1

    new_events = events
    self_coor = None
    if last_action is not None:
        self_coor = self.coordinate_history.pop()
    last_feature = state_to_features(last_game_state, self.coordinate_history)
    if self_coor is not None:
        self.coordinate_history.append(self_coor)
    #last_feature = state_to_features(last_game_state)
    if last_game_state is not None and last_action is not None:
        new_events = get_bomberman_events(last_feature, last_action, events)
        if N_STEP_LEARNING_RATE:
            self.n_step_buffer.append(Transition(last_feature, last_action, None, reward_from_events(self, new_events)))
            if len(self.n_step_buffer) >= N_STEP:
                reward_arr = np.array([self.n_step_buffer[i][-1] for i in range(N_STEP)])
                # Sum with the discount factor to get the accumulated rewards over N_STEP transitions.
                reward = ((DECAY_GAMMA_VALUE) ** np.arange(N_STEP)).dot(reward_arr)
                self.transitions.append(
                    Transition(self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[-1][2], reward))
        else:
            self.transitions.append(Transition(last_feature, last_action, None, reward_from_events(self, new_events)))

    # train and update the model
    if len(self.transitions) > BATCH_SIZE:
        # mini_batch sampling
        batch = random.sample(self.transitions, BATCH_SIZE)
        # Initialization.
        X = [[] for i in range(len(self.actions))]  # Feature matrix for each action
        y = [[] for i in range(len(self.actions))]  # Target vector for each action
        residuals = [[] for i in range(len(self.actions))]
        for old_state, action, next_state, reward in batch:
            if old_state is not None:
                # Index of action taken in 'state'.
                action_idx = self.actions.index(action)
                # Q-value for the given state and action.
                if self.model_is_fitted and next_state is not None:
                    # Non-terminal next state and pre-existing model.
                    maximal_response = np.max(self.model.predict(next_state))
                    q_update = (reward + DECAY_GAMMA_VALUE * maximal_response)
                else:
                    # Either next state is terminal or a model is not yet fitted.
                    q_update = reward # Equivalent to a Q-value of zero for the next state.

                # Append feature data and targets for the regression,
                # corresponding to the current action.
                X[action_idx].append(old_state)
                y[action_idx].append(q_update)

                # Prioritized experience replay.
                if PRIORITY_LEARNING and self.model_is_fitted:
                    # Calculate the residuals for the training instance.
                    X_tmp = X[action_idx][-1].reshape(1, -1)
                    target = y[action_idx][-1]
                    q_estimate = self.model.predict(X_tmp, action_idx=action_idx)[0]
                    res = (target - q_estimate) ** 2
                    residuals[action_idx].append(res)
        # update model
        if PRIORITY_LEARNING and self.model_is_fitted:
            # Initialization
            X_new = [[] for i in range(len(self.actions))]
            y_new = [[] for i in range(len(self.actions))]
            # For the training set of every action:
            for i in range(len(self.actions)):
                # Keep the specifed fraction of samples with the largest squared residuals.
                prio_size = int(len(residuals[i]) * PRIORITY_RATIO)
                idx = np.argpartition(residuals[i], -prio_size)[-prio_size:]
                X_new[i] = [X[i][j] for j in list(idx)]
                y_new[i] = [y[i][j] for j in list(idx)]
            # Update the training set.
            #self.logger.info(f'priority learning data: original: {[np.array(X[i]).shape for i in range(len(self.actions))]}, priority :{[np.array(X_new[i]).shape for i in range(len(self.actions))]}')
            X = X_new
            y = y_new

        self.logger.info(f'current round training data info, priority learning: {PRIORITY_LEARNING}, {[np.array(X[i]).shape for i in range(len(self.actions))]}')
        #self.logger.info(f'{np.array(X[0])}')
        self.model.partial_fit(X, y)
        self.model_is_fitted = True
        #if self.game_nr % 99 == 0:
        #    self.logger.info(f'training data print: {X}')
        #    self.logger.info(f'training data print: {y}')
        # update greedy epsilon
        if self.epsilon > EPSILON_END_VALUE:
            self.epsilon *= EPSILON_DECAY_VALUE
        self.logger.info(f"Training policy e-greedy:{self.epsilon}, n_step: {N_STEP_LEARNING_RATE}")

    update_bomberman_stats(self, new_events, True)
   
    if not last_game_state["round"] % SAVE_MODEL:
        with open("./models/model-"+ str(last_game_state["round"]), "wb") as file:
            pickle.dump(self.model, file)

    if last_game_state["round"] == TRAINING_ROUNDS:
        with open("./models/model-"+ str(last_game_state["round"]), "wb") as file:
            pickle.dump(self.model, file)

    # evaluate from json.



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent gets to encourage
    certain behavior like killing other agents, collecting coins, and avoiding self-destruction.
    """
    global loop_old
    
    # Base rewards:
    aggressive_action = 0.3
    coin_action = 2  # Increase the reward for coin-related actions
    crate_action = 0.4  # Increase the reward for crate-related actions
    escape = 10
    bombing = 0.5
    waiting = 0.5
    passive = 0

    game_rewards = {
        # SPECIAL EVENTS
        ESCAPE: escape,
        NOT_ESCAPE: -escape,
        DESTROY_TARGET: bombing,
        MISSED_TARGET: -4 * escape,  # Prevent self-bomb
        BOMBED_1_TO_2_CRATES: 0.1,
        BOMBED_3_TO_5_CRATES: 0.8,
        BOMBED_5_PLUS_CRATES: 1,
        WAITED_NECESSARILY: waiting,
        WAITED_UNNECESSARILY: -waiting,
        CLOSER_TO_PLAYERS: aggressive_action,
        AWAY_FROM_PLAYERS: -aggressive_action,
        CLOSER_TO_COIN: coin_action,
        AWAY_FROM_COIN: -coin_action,
        CLOSER_TO_CRATE: crate_action,
        AWAY_FROM_CRATE: -crate_action,
        GET_IN_LOOP : -0.025 * loop_old,
        SURVIVED_STEP: passive,

        # DEFAULT EVENTS
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.INVALID_ACTION: -5,

        # bombing
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        # crates, coins
        e.CRATE_DESTROYED: 8,  # Increase the reward for destroying crates
        e.COIN_FOUND: 100,  # Increase the reward for finding a coin
        e.COIN_COLLECTED: 600,  # Increase the reward for collecting a coin

        # kills
        e.KILLED_OPPONENT: 200,  # Increase the reward for killing an opponent
        e.KILLED_SELF: -3000,  # Penalize for killing oneself
        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 0,

        # passive
        e.SURVIVED_ROUND: 0,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    return reward_sum


def get_bomberman_events(old_feature, self_action, events_src) -> list:
    """
    Get Bomberman game events.

    This method calculates and returns a list of game events based on the agent's current state,
    action taken, and the events that occurred in the game.

    :param old_feature: The agent's feature representation of the old game state.
    :param self_action: The action that the agent intends to take.
    :param events_src: The events that occurred when transitioning from the old game state to the new game state.
    :return: A list of game events.
    """
    global loop_old
    events = events_src.copy()  # Avoid deepcopy for efficiency
    danger = old_feature[0] == 1

    if danger:
        # When in mortal danger, prioritize escape.
        escape_direction = old_feature[2:4]
        events.append(ESCAPE if check_direction(escape_direction, self_action) else NOT_ESCAPE)
    else:
        # NOT DANGER
        target_location = old_feature[1]
        players_location = old_feature[4:6]
        coins_location = old_feature[6:8]
        crates_location = old_feature[8:10]
        current_position = old_feature[10]
        loop_old = old_feature[11]

        if loop_old > 2:
            events.append(GET_IN_LOOP)

        if self_action == 'BOMB':
            if current_position == 1:
                events.append(BOMBED_1_TO_2_CRATES)
            elif current_position == 2:
                events.append(BOMBED_3_TO_5_CRATES)
            elif current_position == 3:
                events.append(BOMBED_5_PLUS_CRATES)

            events.append(DESTROY_TARGET if target_location == 1 else MISSED_TARGET)
        elif self_action == 'WAIT':
            if (
                all(p == 0 for p in players_location)
                and all(c == 0 for c in coins_location)
                and all(cr == 0 for cr in crates_location)
                and target_location == 0
            ):
                events.append(WAITED_NECESSARILY)
            else:
                events.append(WAITED_UNNECESSARILY)
        else:
            if target_location == 1:
                events.append(MISSED_TARGET)

            if not all(p == 0 for p in players_location):
                events.append(CLOSER_TO_PLAYERS if check_direction(players_location, self_action) else AWAY_FROM_PLAYERS)
            if not all(c == 0 for c in coins_location):
                events.append(CLOSER_TO_COIN if check_direction(coins_location, self_action) else AWAY_FROM_COIN)

            if not all(cr == 0 for cr in crates_location):
                events.append(CLOSER_TO_CRATE if check_direction(crates_location, self_action) else AWAY_FROM_CRATE)

    if 'GOT_KILLED' not in events:
        events.append(SURVIVED_STEP)

    return events



def update_bomberman_stats(self, events: List[str], data_reset=False):

    if 'COIN_COLLECTED' in events:
        self.coins += 1
    if 'CRATE_DESTROYED' in events:
        self.crates += events.count('CRATE_DESTROYED')
    if 'KILLED_OPPONENT' in events:
        self.enemies += events.count('KILLED_OPPONENT')
    self.rewards += reward_from_events(self, events)


