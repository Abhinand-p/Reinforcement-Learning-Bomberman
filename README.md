
# Reinforcement-Learning-Bomberman

Welcome to the Reinforcement-Learning-Bomberman project! This project leverages reinforcement learning, specifically Q-learning, to develop an intelligent agent capable of playing the classic Bomberman game. The aim is to create an agent that can strategically place bombs, navigate through a maze-like environment, avoid obstacles, and outmaneuver opponents to achieve high scores.

## Introduction

Bomberman is a classic arcade game where players navigate a maze, strategically placing bombs to clear obstacles and eliminate opponents. In this project, we use Q-learning, a popular reinforcement learning algorithm, to train an agent that can make optimal decisions based on the game state, maximizing its cumulative reward over time. 

The agent, named `classic_1`, is designed to navigate the maze, place bombs judiciously, and avoid hazards like explosions and enemy attacks.

## Features

The following features were implemented to enable the agent's decision-making process:

1. **Wall Counts in Surrounding Tiles**: Calculates the number of walls around the agent to determine safe movement paths and potential obstacles.
2. **Bomb Presence**: Analyzes the presence of bombs and the agent's ability to place a bomb without risking self-damage.
3. **Crate Presence**: Evaluates the potential rewards of destroying crates within the agent's proximity.
4. **Blockages**: Detects potential blockages, such as bombs, walls, or opponents, that could hinder the agent's movement.
5. **New Tile Action**: Determines the safest movement options based on the current game state.
6. **Shortest Path to Coin or Crate**: Finds the shortest path to valuable targets like coins or crates while avoiding hazards.

## Methods

### Reinforcement Learning Approach

The agent utilizes Q-learning to learn from the environment. The Q-learning algorithm is based on the concept of updating a Q-table, which stores the potential rewards for each action taken from each state. The agent uses this Q-table to make informed decisions, aiming to maximize its cumulative reward.

### Feature Engineering

Key features were carefully selected and computed to accurately capture the state of the game. These features form the basis of the agent's decision-making process, guiding its actions in various scenarios.

## Training

Training involved defining custom events and utilizing tricks like exploration decay rates to shape the agent's behavior. Custom events were crafted to align with the game features, helping the agent learn from its experiences.

Key custom events defined include:

- **Escape Bomb Direction**: Encourages the agent to move away from nearby bombs.
- **Movement Blocked**: Penalizes the agent for attempting to move into a blocked path.
- **Bomb Placement Check**: Assesses the appropriateness of bomb placement actions.
- **Coin Search**: Rewards the agent for successfully moving towards coins or crates.

## Experiments and Results

The training results showed significant improvement in the agent's ability to play the game effectively. The Q-table was visualized to monitor the learning progress, with 1920 potential states being explored during training. The agent demonstrated proficiency in strategic movement, bomb placement, and avoiding hazards.

### Performance

The agent `classic_1` was compared with rule-based agents and showed a competitive edge in most scenarios. It prioritized reaching the nearest coin while evading dangers and demonstrated effective bomb placement strategies.

## Challenges

Several challenges were encountered during the development and training of the agent:

1. **Q-table Dimension**: Initially, distinct states for each feature were treated separately, which did not enhance action selection. The approach was revised to consider all possible state combinations, leading to better performance.
2. **Feature Selection**: Identifying relevant features that significantly impact the agent's decision-making was crucial for success.
3. **Event Definitions**: Custom events were introduced to better guide the agent's learning process, improving its overall gameplay strategy.

## Conclusion

The `classic_1` agent effectively plays Bomberman by leveraging Q-learning and custom-defined features. Future enhancements could involve expanding the feature set to include interactions with multiple agents, thereby broadening the scope beyond the current agent-focused policies.

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: `numpy`, `pandas`, `gym`, `wandb`, and `matplotlib`
