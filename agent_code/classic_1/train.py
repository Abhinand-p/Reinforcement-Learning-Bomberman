# Reinforcement Learning: How an agent act in an environment in order to maximise some given reward.

# MDP: Markov Decision Process: Formalize sequential decision-making.
# Decision Maker -> Interact with the environment -> Select an action -> environment transits to a new state
# -> Agent is given a reward based on its previous action.

# Agent, Environment, States(S), Actions(A) and Rewards(R).
# f(S(t),A(t)) = R(t+1)

# Trajectory: sequence of states, actions and rewards.
# S(0), A(0), R(1), S(1), A(1), R(2), S(2). A(2), R(3), ...
# Since the sets S and R are finite, the random variables R(t) and S(t) have well-defined probability distributions.

# Return -> Driving the agent to make decisions
# expected return: rewards at a given time step.
# G(t) = R(t+1) + R(t+2) + R(t+3) + ... R(T)
# Expected Return is the agents objective of maximising the rewards

# Episodes: The Agent Environment Interaction naturally breaks up into sub-sequences called episodes.
# Similar to new rounds in a game, where its current state does not depend on its previous episode.

# Types of tasks
# I> Episodic tasks: Tasks within the episodes.
# T is known.
# Agent objective will be to maximise the total return.
# G(t) = R(t+1) + R(t+2) + R(t+3) + ... R(T)
# II> Continuing tasks: Agent environment interactions don't break up naturally into episodes.
#   T ~ infinity.
#   Agent objective will be to maximise the discounted return.
#   Gamma = discount rate = [0,1]
#   Based on the idea of Time value of money.
#   G(t) = R(t+1) + (Gamma)*R(t+2) + (Gamma)^2*R(t+3) + ...
#   G(t) = R(t+1) + (Gamma)G(t+1)

# Policies and value functions
# I> Policies(pi): What's the probability that an agent will select a specific action from a specific state?
# A function that maps a given state to probabilities of actions to select from that given state.
# If an agent follows policy pi at time t, then pi(a|s) is the probability that A(t) = a if S(t) = s.
# II> Value Functions: How good is a specific action or a specific state for the agent?
# Value functions -> expected return -> the way the agent acts -> policy.
# Two types:
#   State-value function:
#       How good any given state is for an agent following policy pi.
#       Value of a state under pi.
#   Action-value function:
#       How good it is for the agent to take any given action from a given state while following policy pi.
#       Value of an action under pi.
#       "Q-function" q(pi)(s,a) = E[G(t)"Q-value" | S(t) = s, A(t) = a]
#           The value of action(a) in state(s) under policy(pi) is the expected return from starting from state(s) at
#           time(t) taking action(a) and following policy(pi) thereafter
#       Q = "Quality"

# Optimal Policy: A policy that is better than or at least the same as all other policies is called the optimal policy.
# pi >= pi' iff v(pi)(s) >= v(pi')(s) for all s belonging to S

# Optimal state-value function v(*):
#   Largest expected return achievable by any policy pi for each state.
# Optimal action-value function q(*):
#   Largest expected return achievable by any policy pi for each possible state-action pair.
#   Satisfies the Bellman optimality equation for q(*)
#   q(*)(s,a) = E[R(t+1) + (Gamma) * max q(*)(s',a')]

# Using Q-Learning:
# Value interation process.
# Solve for the optimal policy in an MDP.
#   The algorithm iteratively updates the q values for each state-action pair using the Bellman equation until the
#   q function converges to the optimal q(*)

# Q-Table:(Number of states X Number of actions)
#   Table storing q values for each state-action pair.
#   Horizontal = Actions
#   Vertical = States

# Tradeoff between Exploration and Exploitation.
#   Epsilon Greedy strategy:
#       Exploration rate(Epsilon) Probability that the agent will explore the environment rather than exploitation.
#       Initially set to 1 and then is chosen randomly.
#       A random value is generated to decide if the agent will explore or exploit. If it performs exploitation then
#           it would choose the greatest q value action from the q-table. If it performs exploration then it will
#           randomly choose an action to explore the environment.

# The Bellman equation is used to update the q value in the Q-table of the given state.
#   Objective: Make the Q-value for the given state-action pair as close as we can to the right-hand side of the
#   Bellman equation so that the Q-value will eventually converge to the optimal Q-value q(*).
#       Reduce the loss = q(*)(s,a) - q(s,a)
#   We use the learning rate to determine how much of information has to be retained by the previously encountered state
#   Learning rate = alpha. Higher the learning rate, the faster the agent will adapt to the new Q-value.
#   new Q-value(s,a) = (1-alpha)*(old value) + (alpha)*(learned value)
#       learned value is derived from the Bellman equation.


# Learning Material used to understand the concepts:
# 1. Reinforcement Learning - Developing Intelligent Agents https://deeplizard.com/learn/video/nyjbcRQ-uQ8


from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3
RECORD_ENEMY_TRANSITIONS = 1.0

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    # This is called after `setup` in callbacks.py.
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    # Called once per step to allow intermediate rewards based on game events.
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # Called at the end of each game or when the agent died to hand out final rewards.
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    # TODO: add all the events from the events.py
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum