import numpy as np

# Defines a Q-learning agent for reinforcement learning
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay_rate=0.001, min_epsilon=0.01):
        # Store the environment (assumed to have state_space_size, action_space_size, and actions)
        self.env = env
        # Learning rate: controls how much the agent updates Q-values per step
        self.alpha = alpha
        # Discount factor: balances immediate vs. future rewards
        self.gamma = gamma
        # Exploration rate: determines likelihood of random action (starts high for exploration)
        self.epsilon = epsilon
        # Rate at which exploration decreases over time
        self.epsilon_decay_rate = epsilon_decay_rate
        # Minimum exploration rate to ensure some randomness
        self.min_epsilon = min_epsilon

        # Create a Q-table initialized with zeros (rows: states, columns: actions)
        self.q_table = np.zeros((env.state_space_size, env.action_space_size))
        # Map actions (e.g., 'up', 'down') to indices for Q-table
        self.action_map = {action: i for i, action in enumerate(env.actions.keys())}
        # Reverse map to convert Q-table indices back to actions
        self.reverse_action_map = {i: action for action, i in self.action_map.items()}

    # Selects an action for a given state
    def choose_action(self, state_index):
        # Generate a random number to decide exploration vs. exploitation
        if np.random.uniform(0, 1) < self.epsilon:
            # Exploration: randomly choose an action from available actions
            return np.random.choice(list(self.env.actions.keys()))
        else:
            # Exploitation: choose the action with the highest Q-value for the current state
            return self.reverse_action_map[np.argmax(self.q_table[state_index, :])]

    # Updates the Q-table based on experience
    def learn(self, state_index, action, reward, next_state_index, done):
        # Convert action to its index in the Q-table
        action_index = self.action_map[action]
        
        # Get the current Q-value for the state-action pair
        old_value = self.q_table[state_index, action_index]
        # Find the maximum Q-value for the next state (best future reward estimate)
        next_max = np.max(self.q_table[next_state_index, :])
        
        # Apply Q-learning formula: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        # Update the Q-table with the new Q-value
        self.q_table[state_index, action_index] = new_value

        # If episode is done, reduce exploration rate (epsilon) to favor exploitation
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    # Reduces exploration rate (epsilon) manually
    def decay_epsilon(self):
        # Ensure epsilon doesn't go below the minimum threshold
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    # Returns the current Q-table
    def get_q_table(self):
        return self.q_table

    # Returns the current exploration rate (epsilon)
    def get_epsilon(self):
        return self.epsilon