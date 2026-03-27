import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Create environment (training)
# -----------------------------
env = gym.make("FrozenLake-v1", is_slippery=True)

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

# -----------------------------
# Q-table
# -----------------------------
num_states = env.observation_space.n
num_actions = env.action_space.n

Q_table = np.zeros((num_states, num_actions))

# -----------------------------
# Hyperparameters
# -----------------------------
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 10000
max_steps = 100

rewards_per_episode = []

# -----------------------------
# Training loop
# -----------------------------
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    step = 0
    while not done and step < max_steps:
        # ε-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state, :])

        # step environment
        new_state, reward, terminated, truncated, _ = env.step(action)

        # Q-learning update
        best_future_q = np.max(Q_table[new_state, :])

        Q_table[state, action] += alpha * (
            reward + gamma * best_future_q - Q_table[state, action]
        )

        state = new_state
        total_reward += reward

        done = terminated or truncated
        step += 1

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode+1}, Avg Reward: {np.mean(rewards_per_episode[-1000:]):.3f}")

# -----------------------------
# Final Q-table
# -----------------------------
print("\nFinal Q-table:\n")
print(Q_table)

# -----------------------------
# Policy visualization
# -----------------------------
def print_policy(Q_table):
    symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    size = int(np.sqrt(Q_table.shape[0]))
    policy = []

    for state in range(Q_table.shape[0]):
        best_action = np.argmax(Q_table[state])
        policy.append(symbols[best_action])

    policy = np.array(policy).reshape((size, size))

    print("\nLearned Policy:\n")
    for row in policy:
        print(" ".join(row))

print_policy(Q_table)

# -----------------------------
# Evaluation (100 episodes)
# -----------------------------
test_episodes = 100
successes = 0

for _ in range(test_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(Q_table[state, :])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done and reward == 1:
            successes += 1

success_rate = (successes / test_episodes) * 100
print(f"\nSuccess rate: {success_rate:.2f}%")

if success_rate > 70:
    print("✅ Good learning")
elif success_rate < 30:
    print("⚠️ Needs more training")
else:
    print("➖ Moderate performance")

# -----------------------------
# Mac-safe visualization (IMPORTANT FIX)
# -----------------------------
print("\nVisualizing agent (Mac-safe)...")

render_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")

state, _ = render_env.reset()
done = False

plt.ion()
fig, ax = plt.subplots()

img = None

while not done:
    frame = render_env.render()

    if img is None:
        img = ax.imshow(frame)
        ax.axis("off")
    else:
        img.set_data(frame)

    plt.pause(0.5)

    action = np.argmax(Q_table[state, :])
    state, reward, terminated, truncated, _ = render_env.step(action)

    done = terminated or truncated

plt.ioff()
plt.show()

render_env.close()

if reward == 1:
    print("✅ Agent reached the goal!")
else:
    print("❌ Agent fell into a hole.")