
import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from q_learning_agent import QLearningAgent

def train_agent(env, agent, episodes=1000):
    rewards_per_episode = []
    epsilon_history = []

    for episode in range(episodes):
        state = env.reset()
        state_index = env.get_state_index(state)
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state_index)
            next_state, reward, done = env.step(action)
            next_state_index = env.get_state_index(next_state)

            agent.learn(state_index, action, reward, next_state_index, done)

            state = next_state
            state_index = next_state_index
            total_reward += reward
        
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        epsilon_history.append(agent.get_epsilon())

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.get_epsilon():.4f}")

    print("\nTraining finished.")
    return rewards_per_episode, epsilon_history

def visualize_training(rewards_per_episode, epsilon_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epsilon_history)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.show()

def run_trained_agent(env, agent, max_steps=100):
    print("\nRunning trained agent...")
    state = env.reset()
    env.render()
    done = False
    steps = 0
    path = [state]

    while not done and steps < max_steps:
        state_index = env.get_state_index(state)
        action_index = np.argmax(agent.q_table[state_index, :])
        action = agent.reverse_action_map[action_index]
        
        next_state, reward, done = env.step(action)
        state = next_state
        path.append(state)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        steps += 1

    if done:
        print("Agent reached the goal!")
    else:
        print("Agent could not reach the goal within max steps.")
    return path

def visualize_path(maze_map, path, filename="maze_path.png"):
    maze = np.array([list(row) for row in maze_map])
    rows, cols = maze.shape

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.imshow(np.zeros((rows, cols)), cmap="Greys", origin="upper")

    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == '#':
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
            elif maze[r, c] == 'S':
                ax.text(c, r, 'S', ha='center', va='center', color='green', fontsize=20)
            elif maze[r, c] == 'G':
                ax.text(c, r, 'G', ha='center', va='center', color='red', fontsize=20)

    # Draw path
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    ax.plot(path_cols, path_rows, color='blue', linewidth=2, marker='o', markersize=5)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='gray', linestyle='-', linewidth=1)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Agent's Path in Maze")
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    maze_map = [
        "S.##",
        "#.#.",
        "#.#G",
        "...."
    ]
    env = MazeEnv(maze_map)
    agent = QLearningAgent(env)

    rewards, epsilons = train_agent(env, agent, episodes=500)
    visualize_training(rewards, epsilons)
    
    path = run_trained_agent(env, agent)
    visualize_path(maze_map, path)


