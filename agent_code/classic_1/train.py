"""
Reinforcement Learning: How an agent act in an environment in order to maximise some given reward.

MDP: Markov Decision Process: Formalize sequential decision-making.
Decision Maker -> Interact with the environment -> Select an action -> environment transits to a new state
-> Agent is given a reward based on its previous action.

Agent, Environment, States(S), Actions(A) and Rewards(R).
f(S(t),A(t)) = R(t+1)

Trajectory: sequence of states, actions and rewards.
S(0), A(0), R(1), S(1), A(1), R(2), S(2). A(2), R(3), ...
Since the sets S and R are finite, the random variables R(t) and S(t) have well-defined probability distributions.

Return -> Driving the agent to make decisions
expected return: rewards at a given time step.
G(t) = R(t+1) + R(t+2) + R(t+3) + ... R(T)
Expected Return is the agents objective of maximising the rewards

Episodes: The Agent Environment Interaction naturally breaks up into sub-sequences called episodes.
Similar to new rounds in a game, where its current state does not depend on its previous episode.

Types of tasks
I> Episodic tasks: Tasks within the episodes.
T is known.
Agent objective will be to maximise the total return.
G(t) = R(t+1) + R(t+2) + R(t+3) + ... R(T)
II> Continuing tasks: Agent environment interactions don't break up naturally into episodes.
  T ~ infinity.
  Agent objective will be to maximise the discounted return.
  Gamma = discount rate = [0,1]
  Based on the idea of Time value of money.
  G(t) = R(t+1) + (Gamma)*R(t+2) + (Gamma)^2*R(t+3) + ...
  G(t) = R(t+1) + (Gamma)G(t+1)

Policies and value functions
I> Policies(pi): What's the probability that an agent will select a specific action from a specific state?
A function that maps a given state to probabilities of actions to select from that given state.
If an agent follows policy pi at time t, then pi(a|s) is the probability that A(t) = a if S(t) = s.
II> Value Functions: How good is a specific action or a specific state for the agent?
Value functions -> expected return -> the way the agent acts -> policy.
Two types:
  State-value function:
      How good any given state is for an agent following policy pi.
      Value of a state under pi.
  Action-value function:
      How good it is for the agent to take any given action from a given state while following policy pi.
      Value of an action under pi.
      "Q-function" q(pi)(s,a) = E[G(t)"Q-value" | S(t) = s, A(t) = a]
          The value of action(a) in state(s) under policy(pi) is the expected return from starting from state(s) at
          time(t) taking action(a) and following policy(pi) thereafter
      Q = "Quality"

Optimal Policy: A policy that is better than or at least the same as all other policies is called the optimal policy.
pi >= pi' iff v(pi)(s) >= v(pi')(s) for all s belonging to S

Optimal state-value function v(*):
  Largest expected return achievable by any policy pi for each state.
Optimal action-value function q(*):
  Largest expected return achievable by any policy pi for each possible state-action pair.
  Satisfies the Bellman optimality equation for q(*)
  q(*)(s,a) = E[R(t+1) + (Gamma) * max q(*)(s',a')]

Using Q-Learning:
Value interation process.
Solve for the optimal policy in an MDP.
  The algorithm iteratively updates the q values for each state-action pair using the Bellman equation until the
  q function converges to the optimal q(*)

Q-Table:(Number of states X Number of actions)
  Table storing q values for each state-action pair.
  Horizontal = Actions
  Vertical = States

Tradeoff between Exploration and Exploitation.
  Epsilon Greedy strategy:
      Exploration rate(Epsilon) Probability that the agent will explore the environment rather than exploitation.
      Initially set to 1 and then is chosen randomly.
      A random value is generated to decide if the agent will explore or exploit. If it performs exploitation then
          it would choose the greatest q value action from the q-table. If it performs exploration then it will
          randomly choose an action to explore the environment.

The Bellman equation is used to update the q value in the Q-table of the given state.
  Objective: Make the Q-value for the given state-action pair as close as we can to the right-hand side of the
  Bellman equation so that the Q-value will eventually converge to the optimal Q-value q(*).
      Reduce the loss = q(*)(s,a) - q(s,a)
  We use the learning rate to determine how much of information has to be retained by the previously encountered state
  Learning rate = alpha. Higher the learning rate, the faster the agent will adapt to the new Q-value.
  new Q-value(s,a) = (1-alpha)*(old value) + (alpha)*(learned value)
      learned value is derived from the Bellman equation.


Learning Material used to understand the concepts:
1. Reinforcement Learning - Developing Intelligent Agents https://deeplizard.com/learn/video/nyjbcRQ-uQ8

"""
import os
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
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    # Initial exploration rate is set to 1.0
    self.exploration_rate = 1.0
    # Alpha = Learning Rate.
    self.learning_rate = 0.7
    # Gamma = Discount Rate.
    self.discount_rate = 0.2
    # episode number
    self.episodes = 0.0
    # Gathered return of rewards per episode
    self.episode_gathered_rewards = 0.0

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Classic_1 Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Skip the first game state
    if old_game_state is None:
        print("First game state is None")
        return

    self.transitions.append( Transition(state_to_features(old_game_state, self.history), self_action, state_to_features(new_game_state, self.history), reward_from_events(self, events), ))
    state, action, next_state, reward = ( self.transitions[-1][0], self.transitions[-1][1], self.transitions[-1][2], self.transitions[-1][3],)

    action_idx = ACTION_INDEX[action]
    self.logger.debug(f"Action-ID: {action_idx}")

    self.episode_gathered_rewards += reward
    self.Q_table[state, action_idx] = self.Q_table[state, action_idx] + self.learning_rate * (reward + self.discount_rate * np.max(self.Q_table[next_state]) - self.Q_table[state, action_idx])
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
    self.exploration_rate = self.exploration_rate_end + (
            self.exploration_rate_initial - self.exploration_rate_end) * np.exp(
        -self.exploration_decay_rate * self.episodes)
    self.logger.debug(f"Exploration_rate{self.episodes}: {self.exploration_rate}")


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 1,
        e.BOMB_DROPPED: -1,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 0,
        e.COIN_FOUND: 0,
        e.KILLED_SELF: 0,
        e.GOT_KILLED: 0,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0,
        PLACEHOLDER_EVENT: -.1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
