import numpy as np

# Defines a maze environment for a reinforcement learning agent
class MazeEnv:
    def __init__(self, maze_map):
        # Convert the input maze map (list of strings) into a NumPy array for easier manipulation
        self.maze = np.array([list(row) for row in maze_map])
        # Find the starting position ('S') in the maze (returns first occurrence as a tuple)
        self.start_state = tuple(np.argwhere(self.maze == 'S')[0])
        # Find the goal position ('G') in the maze (returns first occurrence as a tuple)
        self.goal_state = tuple(np.argwhere(self.maze == 'G')[0])
        # Set the agent's initial position to the start state
        self.current_state = self.start_state
        # Get the dimensions of the maze (rows and columns)
        self.rows, self.cols = self.maze.shape
        # Define possible actions and their effect on coordinates (row, col)
        self.actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        # Store the number of possible actions (4 in this case)
        self.action_space_size = len(self.actions)
        # Calculate total number of states (grid cells) in the maze
        self.state_space_size = self.rows * self.cols

    # Resets the environment to the initial state
    def reset(self):
        # Move the agent back to the start position
        self.current_state = self.start_state
        # Return the starting state
        return self.current_state

    # Takes an action and updates the environment
    def step(self, action):
        # Check if the provided action is valid
        if action not in self.actions:
            raise ValueError("Invalid action")

        # Get the change in row and column for the chosen action
        dr, dc = self.actions[action]
        # Calculate the new position after taking the action
        new_row, new_col = self.current_state[0] + dr, self.current_state[1] + dc

        # Check if the new position is within bounds and not a wall ('#')
        if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.maze[new_row, new_col] != '#':
            # Update the agent's position to the new valid position
            self.current_state = (new_row, new_col)

        # Default reward for each step (encourages faster paths)
        reward = -1
        # Flag to indicate if the episode is complete
        done = False

        # Check if the agent has reached the goal
        if self.current_state == self.goal_state:
            # Large positive reward for reaching the goal
            reward = 100
            # Mark episode as complete
            done = True
        # Check if the agent hit a wall (included for robustness, though logic prevents this)
        elif self.maze[self.current_state] == '#':
            # Large negative reward for hitting a wall
            reward = -100

        # Return the new state, reward, and done flag
        return self.current_state, reward, done

    # Displays the current state of the maze
    def render(self):
        # Create a copy of the maze to avoid modifying the original
        display_maze = np.copy(self.maze).astype(str)
        # Mark the agent's current position with 'A'
        display_maze[self.current_state] = 'A'
        # Print the maze row by row, joining characters into strings
        print("\n".join(["".join(row) for row in display_maze]))

    # Converts a state (row, col) to a single index for Q-table
    def get_state_index(self, state):
        # Calculate index: row * number_of_columns + column
        return state[0] * self.cols + state[1]

    # Converts a Q-table index back to a state (row, col)
    def get_state_from_index(self, index):
        # Calculate row: integer division of index by number of columns
        row = index // self.cols
        # Calculate column: remainder of index divided by number of columns
        col = index % self.cols
        # Return the state as a tuple
        return (row, col)


# Example usage of the MazeEnv class
if __name__ == '__main__':
    # Define a sample maze where:
    # 'S' = start, 'G' = goal, '#' = wall, '.' = open path
    maze_map = [
        "S.##",
        "#.#.",
        "#.#G",
        "...."
    ]
    # Create a MazeEnv instance with the sample maze
    env = MazeEnv(maze_map)

    # Reset the environment to start a new episode
    state = env.reset()
    # Display the initial maze state
    env.render()

    # Initialize variables for tracking episode progress
    done = False
    total_reward = 0

    # Run the episode until completion
    while not done:
        # Prompt user for an action (for manual testing)
        action = input("Enter action (up, down, left, right): ")
        try:
            # Take a step in the environment with the chosen action
            next_state, reward, done = env.step(action)
            # Accumulate the reward
            total_reward += reward
            # Display the updated maze
            env.render()
            # Print the step's results
            print(f"Reward: {reward}, Total Reward: {total_reward}, Done: {done}")
        except ValueError as e:
            # Handle invalid action input
            print(e)

    # Print message when the episode is complete
    print("Episode finished.")