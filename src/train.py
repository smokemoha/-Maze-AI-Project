# Import required libraries
import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For plotting training results and maze visualization
from maze_env import MazeEnv  # Custom environment class for the maze
from q_learning_agent import QLearningAgent  # Custom Q-learning agent class

def train_agent(env, agent, episodes=1000):
    """
    Train the Q-learning agent in the maze environment.

    Args:
        env (MazeEnv): The maze environment instance.
        agent (QLearningAgent): The Q-learning agent instance.
        episodes (int): Number of training episodes (default: 1000).

    Returns:
        tuple: Lists of rewards per episode and epsilon values over time.
    """
    # Initialize lists to track rewards and epsilon values for analysis
    rewards_per_episode = []
    epsilon_history = []

    # Run training for the specified number of episodes
    for episode in range(episodes):
        # Reset the environment to start a new episode and get initial state
        state = env.reset()
        # Convert state (e.g., (row, col)) to a single index for Q-table
        state_index = env.get_state_index(state)
        done = False  # Flag to indicate if the episode is complete
        total_reward = 0  # Track cumulative reward for this episode

        # Continue until the episode ends (agent reaches goal or max steps)
        while not done:
            # Choose an action (index) based on the current state and epsilon-greedy policy
            action = agent.choose_action(state_index)
            # Execute action in the environment, get next state, reward, and done flag
            next_state, reward, done = env.step(action)
            # Convert next state to index for Q-table
            next_state_index = env.get_state_index(next_state)

            # Update Q-table using Q-learning update rule
            agent.learn(state_index, action, reward, next_state_index, done)

            # Move to the next state
            state = next_state
            state_index = next_state_index
            total_reward += reward  # Accumulate reward for this episode
        
        # Reduce epsilon (exploration rate) after each episode
        agent.decay_epsilon()
        # Store metrics for visualization
        rewards_per_episode.append(total_reward)
        epsilon_history.append(agent.get_epsilon())

        # Print progress every 100 episodes for monitoring
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.get_epsilon():.4f}")

    print("\nTraining finished.")
    return rewards_per_episode, epsilon_history

def visualize_training(rewards_per_episode, epsilon_history):
    """
    Plot training metrics: rewards per episode and epsilon decay.

    Args:
        rewards_per_episode (list): Total rewards for each episode.
        epsilon_history (list): Epsilon values for each episode.
    """
    # Create a figure with two subplots, sized 12x5 inches
    plt.figure(figsize=(12, 5))

    # Plot 1: Rewards per episode
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.plot(rewards_per_episode)  # Plot rewards over episodes
    plt.title("Reward per Episode")  # Set title
    plt.xlabel("Episode")  # Label x-axis
    plt.ylabel("Total Reward")  # Label y-axis
    plt.grid(True)  # Add grid for readability

    # Plot 2: Epsilon decay
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.plot(epsilon_history)  # Plot epsilon values over episodes
    plt.title("Epsilon Decay")  # Set title
    plt.xlabel("Episode")  # Label x-axis
    plt.ylabel("Epsilon")  # Label y-axis
    plt.grid(True)  # Add grid for readability

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save the plot as an image file
    plt.savefig("training_progress.png")
    # Display the plot
    plt.show()

def run_trained_agent(env, agent, max_steps=100):
    """
    Run the trained agent in the maze to demonstrate its learned policy.

    Args:
        env (MazeEnv): The maze environment instance.
        agent (QLearningAgent): The trained Q-learning agent.
        max_steps (int): Maximum steps to prevent infinite loops (default: 100).

    Returns:
        list: List of states visited (path taken by the agent).
    """
    print("\nRunning trained agent...")
    # Reset the environment and get initial state
    state = env.reset()
    # Render the initial maze state
    env.render()
    done = False  # Flag to indicate if the goal is reached
    steps = 0  # Track number of steps taken
    path = [state]  # Store the agent's path starting with initial state

    # Run until the goal is reached or max steps are exceeded
    while not done and steps < max_steps:
        # Convert current state to index for Q-table
        state_index = env.get_state_index(state)
        # Choose the best action (highest Q-value) for the current state
        action_index = np.argmax(agent.q_table[state_index, :])
        # Convert action index to action (e.g., 'up', 'down') using reverse mapping
        action = agent.reverse_action_map[action_index]
        
        # Execute action, get next state, reward, and done flag
        next_state, reward, done = env.step(action)
        state = next_state  # Update current state
        path.append(state)  # Add state to path
        env.render()  # Render the updated maze
        # Print action details for debugging/monitoring
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        steps += 1

    # Print outcome based on whether the goal was reached
    if done:
        print("Agent reached the goal!")
    else:
        print("Agent could not reach the goal within max steps.")
    return path

def visualize_path(maze_map, path, filename="maze_path.png"):
    """
    Visualize the maze and the agent's path.

    Args:
        maze_map (list): List of strings representing the maze layout.
        path (list): List of (row, col) coordinates of the agent's path.
        filename (str): Output file name for the visualization (default: 'maze_path.png').
    """
    # Convert maze_map to a NumPy array for easier manipulation
    maze = np.array([list(row) for row in maze_map])
    rows, cols = maze.shape  # Get maze dimensions

    # Create a figure and axis for plotting, sized to match maze dimensions
    fig, ax = plt.subplots(figsize=(cols, rows))
    # Create a blank image (black background) for the maze
    ax.imshow(np.zeros((rows, cols)), cmap="Greys", origin="upper")

    # Draw maze elements
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == '#':
                # Draw walls as black rectangles
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
            elif maze[r, c] == 'S':
                # Mark start position with 'S'
                ax.text(c, r, 'S', ha='center', va='center', color='green', fontsize=20)
            elif maze[r, c] == 'G':
                # Mark goal position with 'G'
                ax.text(c, r, 'G', ha='center', va='center', color='red', fontsize=20)

    # Extract row and column coordinates from path for plotting
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    # Plot the agent's path as a blue line with markers
    ax.plot(path_cols, path_rows, color='blue', linewidth=2, marker='o', markersize=5)

    # Set ticks for grid alignment
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    # Hide tick labels for a cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Add a grid to align with maze cells
    ax.grid(True, which='both', color='gray', linestyle='-', linewidth=1)
    # Ensure equal aspect ratio for square cells
    ax.set_aspect('equal', adjustable='box')
    # Set plot title
    plt.title("Agent's Path in Maze")
    # Save the visualization as an image
    plt.savefig(filename)
    # Display the plot
    plt.show()

# Standard Python idiom to run the script when executed directly
if __name__ == '__main__':
    # Define a simple 4x4 maze layout
    # 'S': Start, '.': Open path, '#': Wall, 'G': Goal
    maze_map = [
        "S.##",
        "#.#.",
        "#.#G",
        "...."
    ]
    # Initialize the maze environment
    env = MazeEnv(maze_map)
    # Initialize the Q-learning agent with the environment
    agent = QLearningAgent(env)

    # Train the agent for 500 episodes and collect metrics
    rewards, epsilons = train_agent(env, agent, episodes=500)
    # Visualize training progress (rewards and epsilon decay)
    visualize_training(rewards, epsilons)
    
    # Run the trained agent to demonstrate its learned policy
    path = run_trained_agent(env, agent)
    # Visualize the agent's path through the maze
    visualize_path(maze_map, path)