
import numpy as np

class MazeEnv:
    def __init__(self, maze_map):
        self.maze = np.array([list(row) for row in maze_map])
        self.start_state = tuple(np.argwhere(self.maze == 'S')[0])
        self.goal_state = tuple(np.argwhere(self.maze == 'G')[0])
        self.current_state = self.start_state
        self.rows, self.cols = self.maze.shape
        self.actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        self.action_space_size = len(self.actions)
        self.state_space_size = self.rows * self.cols

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        if action not in self.actions:
            raise ValueError("Invalid action")

        dr, dc = self.actions[action]
        new_row, new_col = self.current_state[0] + dr, self.current_state[1] + dc

        if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.maze[new_row, new_col] != '#':
            self.current_state = (new_row, new_col)

        reward = -1  # Default reward for each step
        done = False

        if self.current_state == self.goal_state:
            reward = 100  # Reward for reaching the goal
            done = True
        elif self.maze[self.current_state] == '#': # Should not happen with current logic, but good for robustness
            reward = -100 # Penalty for hitting a wall

        return self.current_state, reward, done

    def render(self):
        display_maze = np.copy(self.maze).astype(str)
        display_maze[self.current_state] = 'A'  # Agent's current position
        print("\n".join(["".join(row) for row in display_maze]))

    def get_state_index(self, state):
        return state[0] * self.cols + state[1]

    def get_state_from_index(self, index):
        row = index // self.cols
        col = index % self.cols
        return (row, col)


if __name__ == '__main__':
    # Example Usage:
    maze_map = [
        "S.##",
        "#.#.",
        "#.#G",
        "...."
    ]
    env = MazeEnv(maze_map)

    state = env.reset()
    env.render()

    done = False
    total_reward = 0

    while not done:
        action = input("Enter action (up, down, left, right): ")
        try:
            next_state, reward, done = env.step(action)
            total_reward += reward
            env.render()
            print(f"Reward: {reward}, Total Reward: {total_reward}, Done: {done}")
        except ValueError as e:
            print(e)

    print("Episode finished.")


