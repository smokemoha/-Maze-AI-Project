
import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay_rate=0.001, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.state_space_size, env.action_space_size))
        self.action_map = {action: i for i, action in enumerate(env.actions.keys())}
        self.reverse_action_map = {i: action for action, i in self.action_map.items()}

    def choose_action(self, state_index):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(list(self.env.actions.keys()))  # Explore
        else:
            return self.reverse_action_map[np.argmax(self.q_table[state_index, :])]  # Exploit

    def learn(self, state_index, action, reward, next_state_index, done):
        action_index = self.action_map[action]
        
        # Q-learning update rule
        old_value = self.q_table[state_index, action_index]
        next_max = np.max(self.q_table[next_state_index, :])
        
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state_index, action_index] = new_value

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    def get_q_table(self):
        return self.q_table

    def get_epsilon(self):
        return self.epsilon


