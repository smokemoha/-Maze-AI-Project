# Maze AI Project

A reinforcement learning project that trains an AI agent to escape a maze using Q-learning algorithm.

## Overview

This project demonstrates how an AI agent can learn to navigate through a maze environment using reinforcement learning. The agent starts with no knowledge of the maze and gradually learns the optimal path to reach the goal through trial and error.

## Features

- **Maze Environment**: A customizable grid-based maze with walls, start position, and goal
- **Q-Learning Agent**: Implements the Q-learning algorithm with epsilon-greedy exploration
- **Training Visualization**: Plots showing reward progression and epsilon decay over episodes
- **Path Visualization**: Visual representation of the agent's learned path through the maze
- **Interactive Demo**: Web interface to interact with the trained agent

## Project Structure

```
maze_ai_project/
├── src/
│   ├── maze_env.py          # Maze environment implementation
│   ├── q_learning_agent.py  # Q-learning agent implementation
│   └── train.py             # Training script and visualization
├── web_demo/                # Web interface for demonstration
├── README.md                # Project documentation
└── todo.md                  # Project progress tracking
```

## How It Works

### 1. Maze Environment
The maze is represented as a 2D grid where:
- `S` represents the start position
- `G` represents the goal position
- `#` represents walls (impassable)
- `.` represents open spaces (passable)

### 2. Q-Learning Algorithm
The agent uses Q-learning to learn the optimal policy:
- **Q-Table**: Stores action values for each state-action pair
- **Epsilon-Greedy**: Balances exploration vs exploitation
- **Reward System**: -1 for each step, +100 for reaching goal

### 3. Training Process
The agent trains over multiple episodes:
- Starts with high exploration (epsilon = 1.0)
- Gradually reduces exploration as it learns
- Updates Q-values based on rewards received

## Usage

### Training the Agent
```bash
cd src
python train.py
```

This will:
1. Create a maze environment
2. Initialize a Q-learning agent
3. Train the agent for 500 episodes
4. Generate training progress plots
5. Demonstrate the trained agent's performance
6. Create a path visualization

### Running Individual Components

#### Test the Maze Environment
```bash
cd src
python maze_env.py
```

#### Interactive Maze Navigation
The maze environment can be run interactively where you control the agent manually.

## Results

The training typically shows:
- **Reward Improvement**: Total reward per episode increases as the agent learns
- **Epsilon Decay**: Exploration rate decreases over time
- **Optimal Path**: The trained agent finds an efficient path to the goal

## Customization

### Creating Custom Mazes
You can create custom mazes by modifying the `maze_map` in `train.py`:

```python
maze_map = [
    "S.##",
    "#.#.",
    "#.#G",
    "...."
]
```

### Adjusting Hyperparameters
Modify the Q-learning parameters in `train.py`:
- `alpha`: Learning rate (default: 0.1)
- `gamma`: Discount factor (default: 0.99)
- `epsilon`: Initial exploration rate (default: 1.0)
- `epsilon_decay_rate`: How fast exploration decreases (default: 0.001)

## Requirements

- Python 3.11+
- NumPy
- Matplotlib

## Future Enhancements

- Support for larger and more complex mazes
- Different RL algorithms (SARSA, Deep Q-Networks)
- Real-time training visualization
- Multiple agents in the same environment
- Maze generation algorithms


