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
