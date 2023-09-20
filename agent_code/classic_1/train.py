import os
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from typing import List

import events as e
import numpy as np

from .callbacks import state_to_features

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}

# Hyper parameters
TRANSITION_HISTORY_SIZE = 3
RECORD_ENEMY_TRANSITIONS = 1.0

# Events
# Custom Event: 1 -> Coin Collection event
COIN_DISTANCE_NEAR = "COIN_DISTANCE_NEAR"
COIN_DISTANCE_FAR = "COIN_DISTANCE_FAR"

# Custom Event: 2 -> Crate explosion event
CRATE_DISTANCE_NEAR = "CRATE_DISTANCE_NEAR"
CRATE_DISTANCE_FAR = "CRATE_DISTANCE_FAR"

# TODO: Add more events like this to handle bomb state(to avoid killing itself), Enemies distance(to play safe) ...

PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    # Initial exploration rate for training
    self.exploration_rate = self.exploration_rate_initial
    # Alpha = Learning Rate.
    self.learning_rate = 0.1
    # Gamma = Discount Rate.
    self.discount_rate = 0.99
    # episode number
    self.episodes = 0.0
    # Gathered return of rewards per episode
    self.episode_gathered_rewards = 0.0

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(
        f'Classic_1 Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Skip the first game state
    if old_game_state is None:
        print("First game state is None")
        return

    self.transitions.append(Transition(state_to_features(old_game_state, self.history), self_action,
                                       state_to_features(new_game_state, self.history),
                                       reward_from_events(self, events), ))

    state, action, next_state, reward = (
        self.transitions[-1][0], self.transitions[-1][1], self.transitions[-1][2], self.transitions[-1][3],)

    action_idx = ACTION_INDEX[action]
    self.logger.debug(f"Action: {action}, Action Index: {action_idx}")
    self.logger.debug(f"Old Q-value for state {state}: {self.Q_table[state, action_idx]}")

    self.episode_gathered_rewards += reward

    new_q_value = self.Q_table[state, action_idx] + self.learning_rate * (
            reward + self.discount_rate * np.max(self.Q_table[next_state]) - self.Q_table[state, action_idx])

    self.Q_table[state, action_idx] = new_q_value

    self.logger.debug(f"Updated Q-value for state {state} and action {action}: {new_q_value}")

    self.logger.debug(f"Classic_1 Updated Q-table: {self.Q_table}")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.transitions.append(Transition(state_to_features(last_game_state, self.history), last_action, None,
                                       reward_from_events(self, events), ))

    self.episode_gathered_rewards += self.transitions[-1][3]

    self.logger.debug(f"Total rewards in episode {self.episodes}: {self.episode_gathered_rewards}")

    # Reset the gathered rewards to 0 after each episode
    self.episode_gathered_rewards = 0

    # TODO: Should I update the same Q-table instead of new one everytime?
    q_table_folder = "Q_tables/"
    if self.episodes % 100 == 0:
        q_table_file = os.path.join(q_table_folder, f"Q_table-{self.timestamp}")
        np.save(q_table_file, self.Q_table)

    self.episodes += 1

    # Performing exploration rate decay
    self.exploration_rate = self.exploration_rate_end + (
            self.exploration_rate_initial - self.exploration_rate_end) * np.exp(
        -self.exploration_decay_rate * self.episodes)
    # self.logger.debug(f"Exploration_rate{self.episodes}: {self.exploration_rate}")


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.BOMB_DROPPED: 4,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 7,
        e.COIN_FOUND: 0,
        e.KILLED_SELF: -5,
        e.GOT_KILLED: 0,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0,
        PLACEHOLDER_EVENT: -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def plot_rewards(rewards):
    episodes = range(1, len(rewards) + 1)
    plt.plot(episodes, rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.title('Rewards per Episode')
    plt.grid(True)
    plt.show()
