import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import subprocess

# -----------------------------
# CONFIG
# -----------------------------
num_episodes = 10
video_dir = os.path.abspath("cartpole_videos")
final_video = os.path.join(video_dir, "cartpole_full_run.mp4")

# -----------------------------
# CLEAN OUTPUT DIRECTORY
# -----------------------------
os.makedirs(video_dir, exist_ok=True)

for f in os.listdir(video_dir):
    if f.endswith(".mp4") or f.endswith(".txt"):
        os.remove(os.path.join(video_dir, f))

# -----------------------------
# CREATE ENV + VIDEO WRAPPER
# -----------------------------
env = gym.make("CartPole-v1", render_mode="rgb_array")

env = RecordVideo(
    env,
    video_folder=video_dir,
    episode_trigger=lambda episode_id: True  # record every episode
)

# -----------------------------
# RUN EPISODES
# -----------------------------
episode_stats = []

for episode in range(num_episodes):

    obs, info = env.reset()
    done = False

    total_reward = 0
    steps = 0

    while not done:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        done = terminated or truncated

    episode_stats.append((steps, total_reward))
    print(f"Episode {episode+1}: Steps={steps}, Reward={total_reward}")

env.close()

# -----------------------------
# PRINT SUMMARY
# -----------------------------
print("\n=== RUN SUMMARY ===")
for i, (steps, reward) in enumerate(episode_stats):
    print(f"Episode {i+1}: Steps={steps}, Reward={reward}")

# -----------------------------
# STITCH VIDEOS USING FFMPEG
# -----------------------------
videos = sorted([
    f for f in os.listdir(video_dir)
    if f.endswith(".mp4")
])

if not videos:
    print("No videos found to stitch.")
    exit()

# Create ffmpeg input file list
list_file = os.path.join(video_dir, "file_list.txt")

with open(list_file, "w") as f:
    for v in videos:
        f.write(f"file '{os.path.join(video_dir, v)}'\n")

print("\nStitching videos with ffmpeg...")

subprocess.run([
    "ffmpeg",
    "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", list_file,
    "-c", "copy",
    final_video
])

print("Final video created:", final_video)

# -----------------------------
# OPEN FINAL VIDEO (macOS)
# -----------------------------
subprocess.run(["open", final_video])