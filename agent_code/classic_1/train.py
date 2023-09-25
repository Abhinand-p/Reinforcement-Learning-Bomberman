import os
import wandb
import events as e
import numpy as np

from collections import namedtuple, deque
from typing import List
from .callbacks import state_to_features

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}

# Hyper parameters
TRANSITION_HISTORY_SIZE = 1

# Custom Event: 1 -> Bomb location event
BOMB_DISTANCE_NEAR = "BOMB_DISTANCE_NEAR"
BOMB_DISTANCE_FAR = "BOMB_DISTANCE_FAR"

# Custom Event: 2 -> Blocking the movement
AGENT_MOVEMENT_BLOCKED = "AGENT_MOVEMENT_BLOCKED"

# Custom Event: 3 -> Bad Bomb action
BAD_BOMB_ACTION = "BAD_BOMB_ACTION"
GOOD_BOMB_ACTION = "GOOD_BOMB_ACTION"

# Custom Event: 4 -> Escape Bomb
ESCAPE_BOMB_YES = "ESCAPE_BOMB_YES"
ESCAPE_BOMB_NO = "ESCAPE_BOMB_NO"

# Custom Event: 5 -> Coin search
COIN_SEARCH_YES = "COIN_SEARCH_YES"
COIN_SEARCH_NO = "COIN_SEARCH_NO"


def setup_training(self):
    # Initial exploration rate for training
    self.exploration_rate = self.exploration_rate_initial
    # Alpha = Learning Rate.
    self.learning_rate = 0.5
    # Gamma = Discount Rate.
    self.discount_rate = 0.2
    # episode number
    self.episodes = 0.0
    # Gathered return of rewards per episode
    self.episode_gathered_rewards = 0.0
    # Transactions
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    # Logging in wandb
    wandb.init(project="bomberman_rl", entity="abhinand-po")


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    old_state = self.old_state
    self.new_state = state_to_features(self, new_game_state)
    new_state = self.new_state
    previous_feature_dict = self.valid_list[old_state]

    # Feature 1: Escape Bomb
    if previous_feature_dict["Direction_bomb"] != "SAFE":
        if self_action == previous_feature_dict["Direction_bomb"]:
            events.append(BOMB_DISTANCE_FAR)
        else:
            events.append(BOMB_DISTANCE_NEAR)
    # --------------------------------

    # Feature 2: Agent block feature
    if previous_feature_dict["Up"] == "BLOCK" and self_action == "UP":

        events.append(AGENT_MOVEMENT_BLOCKED)
    elif previous_feature_dict["Down"] == "BLOCK" and self_action == "DOWN":
        events.append(AGENT_MOVEMENT_BLOCKED)
    elif previous_feature_dict["Right"] == "BLOCK" and self_action == "RIGHT":
        events.append(AGENT_MOVEMENT_BLOCKED)
    elif previous_feature_dict["Left"] == "BLOCK" and self_action == "LEFT":
        events.append(AGENT_MOVEMENT_BLOCKED)
    # --------------------------------

    # Feature 3: Correct Bomb check
    if previous_feature_dict["Place_Bomb"] == 'NO' and self_action == "BOMB":
        events.append(BAD_BOMB_ACTION)

    if previous_feature_dict["Place_Bomb"] == 'YES' and previous_feature_dict["Direction_bomb"] == 'SAFE':
        if self_action == "BOMB":
            if previous_feature_dict["Crate_Radar"] == 'HIGH':
                events.append(GOOD_BOMB_ACTION)
            elif previous_feature_dict["Crate_Radar"] == 'LOW':
                events.append(GOOD_BOMB_ACTION)
            else:
                events.append(BAD_BOMB_ACTION)
    # --------------------------------

    # Feature 4: Escape bomb
    if previous_feature_dict["Direction_bomb"] != 'SAFE':
        escape_bomb = True
        if previous_feature_dict["Direction_bomb"] == 'UP' and previous_feature_dict["Up"] == "BLOCK":
            escape_bomb = False
        if previous_feature_dict["Direction_bomb"] == 'RIGHT' and previous_feature_dict["Right"] == "BLOCK":
            escape_bomb = False
        if previous_feature_dict["Direction_bomb"] == 'DOWN' and previous_feature_dict["Down"] == "BLOCK":
            escape_bomb = False
        if previous_feature_dict["Direction_bomb"] == 'LEFT' and previous_feature_dict["Left"] == "BLOCK":
            escape_bomb = False

        if escape_bomb:
            if previous_feature_dict["Direction_bomb"] == self_action:
                events.append(ESCAPE_BOMB_YES)
            else:
                events.append(ESCAPE_BOMB_NO)
    # --------------------------------

    # Feature 5: Search Coin
        coin_collect = True
        if previous_feature_dict["Direction_coin/crate"] == 'UP' and previous_feature_dict["Up"] == "BLOCK":
            coin_collect = False
        if previous_feature_dict["Direction_coin/crate"] == 'RIGHT' and previous_feature_dict["Right"] == "BLOCK":
            coin_collect = False
        if previous_feature_dict["Direction_coin/crate"] == 'DOWN' and previous_feature_dict["Down"] == "BLOCK":
            coin_collect = False
        if previous_feature_dict["Direction_coin/crate"] == 'LEFT' and previous_feature_dict["Left"] == "BLOCK":
            coin_collect = False

        if coin_collect:
            if previous_feature_dict["Direction_coin/crate"] == self_action:
                events.append(COIN_SEARCH_YES)
            else:
                events.append(COIN_SEARCH_NO)

    # --------------------------------

    reward = reward_from_events(self, events)
    self.transitions.append(Transition(old_state, self_action, new_state, reward))

    action_idx = ACTION_INDEX[self_action]
    self.logger.debug(f"game_events_occurred: game_events_occurred: Action: {self_action}, Action Index: {action_idx}")
    self.logger.debug(f"game_events_occurred: Old Q-value for state {old_state}: {self.Q_table[old_state]}")

    self.episode_gathered_rewards += reward

    new_q_value = self.Q_table[old_state, action_idx] + self.learning_rate * (
            reward + self.discount_rate * np.max(self.Q_table[new_state]) - self.Q_table[old_state, action_idx])

    self.Q_table[old_state, action_idx] = new_q_value

    # self.logger.debug(f"Updated Q-value for state {state} and action {action}: {new_q_value}")
    # self.logger.debug(f"Classic_1 Updated Q-table: {self.Q_table}")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None,
                                       reward_from_events(self, events)))

    self.episode_gathered_rewards += self.transitions[-1][3]

    self.logger.debug(f"end_of_round: Total rewards in episode {self.episodes}: {self.episode_gathered_rewards}")

    # Performing exploration rate decay
    self.exploration_rate = self.exploration_rate_end + (
            self.exploration_rate_initial - self.exploration_rate_end) * np.exp(
        -self.exploration_decay_rate * self.episodes)

    # Update the existing Q-table
    q_table_folder = "Q_tables/"
    q_table_file = os.path.join(q_table_folder, f"Q_table-{self.name}.npy")
    if os.path.exists(q_table_file):
        existing_q_table = np.load(q_table_file)
        existing_q_table[self.Q_table != 0] = self.Q_table[self.Q_table != 0]
        self.Q_table = existing_q_table

    # Save the updated Q-table to the file
    np.save(q_table_file, self.Q_table)

    # Log exploration rate
    wandb.log({"Exploration_rate": self.exploration_rate}, step=int(self.episodes))

    # Log total rewards for the episode
    wandb.log({"Total_Rewards_Per_Episode": int(self.episode_gathered_rewards)}, step=int(self.episodes))

    # Log additional metrics: BOMB_DISTANCE_FAR
    if BOMB_DISTANCE_FAR in events:
        wandb.log({"BOMB_DISTANCE_FAR": 1}, step=int(self.episodes))
    else:
        wandb.log({"BOMB_DISTANCE_FAR": 0}, step=int(self.episodes))

    # Log additional metrics: AGENT_MOVEMENT_BLOCKED
    if AGENT_MOVEMENT_BLOCKED in events:
        wandb.log({"AGENT_MOVEMENT_BLOCKED": 1}, step=int(self.episodes))
    else:
        wandb.log({"AGENT_MOVEMENT_BLOCKED": 0}, step=int(self.episodes))

    # Log additional metrics: GOOD_BOMB_ACTION
    if GOOD_BOMB_ACTION in events:
        wandb.log({"GOOD_BOMB_ACTION": 1}, step=int(self.episodes))
    else:
        wandb.log({"GOOD_BOMB_ACTION": 0}, step=int(self.episodes))

    # Log additional metrics: ESCAPE_BOMB_YES
    if ESCAPE_BOMB_YES in events:
        wandb.log({"ESCAPE_BOMB_YES": 1}, step=int(self.episodes))
    else:
        wandb.log({"ESCAPE_BOMB_YES": 0}, step=int(self.episodes))

    # Log additional metrics: COIN_SEARCH_YES
    if COIN_SEARCH_YES in events:
        wandb.log({"COIN_SEARCH_YES": 1}, step=int(self.episodes))
    else:
        wandb.log({"COIN_SEARCH_YES": 0}, step=int(self.episodes))

    self.episode_gathered_rewards = 0
    self.episodes += 1
    # self.logger.debug(f"Exploration_rate{self.episodes}: {self.exploration_rate}")


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        BOMB_DISTANCE_NEAR: -10,
        BOMB_DISTANCE_FAR: 20,

        AGENT_MOVEMENT_BLOCKED: -100,

        BAD_BOMB_ACTION: --100,
        GOOD_BOMB_ACTION: 75,

        ESCAPE_BOMB_YES: 75,
        ESCAPE_BOMB_NO: -100,

        COIN_SEARCH_YES: 75,
        COIN_SEARCH_NO: -100,
    }
    self.logger.debug(f"reward_from_events: {events}")
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
