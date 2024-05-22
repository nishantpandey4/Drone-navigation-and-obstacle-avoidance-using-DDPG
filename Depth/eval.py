from stable_baselines3 import DDPG
from eval_ddpg_drone_env import AirSimDroneEnv
import numpy as np
import time
import matplotlib.pyplot as plt
import tqdm

env = AirSimDroneEnv()


model_path = r"Depth\model\depth_model.zip"
model = DDPG.load(model_path, env=env)

# Prepare for evaluation
num_episodes = 50
episode_rewards = []
collisions_per_episode = []
successes = []

# Evaluate the model

for episode in tqdm.tqdm(range(num_episodes)):
    obs = env.reset()
    done = False
    total_reward = 0
    collisions = 0
    success = True  # Assume success unless a failure (collision) occurs
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if 'is_crash' in info and info['is_crash']:
            collisions += 1
            success = False  # Episode is unsuccessful due to collision
    episode_rewards.append(total_reward)
    collisions_per_episode.append(collisions)
    successes.append(success)

# Calculate average reward and collision rate
average_reward = np.mean(episode_rewards)
collision_rate = [c / num_episodes for c in collisions_per_episode]
success_rate = [1 if s else 0 for s in successes]
average_success_rate = np.mean(success_rate) * 100  # Convert to percentage

# Plotting
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o', color='b', label='Reward per Episode')
plt.axhline(y=average_reward, color='r', linestyle='--', label=f'Average Reward = {average_reward:.2f}')
plt.title('Average Reward Per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(1, num_episodes + 1), collision_rate, marker='x', linestyle='-', color='red', label='Collision Rate per Episode')
plt.title('Collision Rate Per Episode')
plt.xlabel('Episode')
plt.ylabel('Collision Rate')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(1, num_episodes + 1), success_rate, marker='^', linestyle='-', color='green', label='Success Rate per Episode')
plt.axhline(y=average_success_rate, color='purple', linestyle='--', label=f'Average Success Rate = {average_success_rate:.2f}%')
plt.title('Success Rate Per Episode')
plt.xlabel('Episode')
plt.ylabel('Success Rate (%)')
plt.legend()
plt.grid(True)

# Save the plot
plot_filename = f"{time.strftime('%Y%m%d-%H%M%S')}_ddpg_airsim_drone_evaluation_plot.png"
plt.savefig(plot_filename)
plt.show()
print(f"Combined plot saved to {plot_filename}")