from stable_baselines3 import DDPG
from eval_ddpg_lidar import AirSimDroneEnv
import numpy as np
import time
import matplotlib.pyplot as plt
import tqdm
# Initialize the environment
env = AirSimDroneEnv()


model_path = r"Lidar\Model\lidar_model.zip"
model = DDPG.load(model_path, env=env)


num_episodes = 10
episode_rewards = []
collisions_per_episode = []
successes = []


for episode in tqdm.tqdm(range(num_episodes)):
    obs = env.reset()
    done = False
    total_reward = 0
    collisions = 0
    success = True
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if 'is_crash' in info and info['is_crash']:
            collisions += 1
            success = False
    episode_rewards.append(total_reward)
    collisions_per_episode.append(collisions)
    successes.append(success)

average_reward = np.mean(episode_rewards)
collision_rate = [c / num_episodes for c in collisions_per_episode]
success_rate = [1 if s else 0 for s in successes]
average_success_rate = np.mean(success_rate) * 100

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


plot_filename = f"{time.strftime('%Y%m%d-%H%M%S')}_ddpg_airsim_drone_evaluation_plot.png"
plt.savefig(plot_filename)
plt.show()
print(f"Combined plot saved to {plot_filename}")