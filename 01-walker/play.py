from TD3 import TD3
import numpy as np
import os
import torch
import gym
import pybullet_envs
from gym import wrappers

monitor_path = './monitor/'
if not os.path.exists(monitor_path):
    os.makedirs(monitor_path)


env_name = "Walker2DBulletEnv-v0"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
max_episode_steps = env._max_episode_steps


seed = 0
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


eval_episodes = 10
max_episode_steps = env._max_episode_steps
env = wrappers.Monitor(env, monitor_path, force=True)
obs = env.reset()

policy = TD3(state_dim, action_dim, max_action)
policy.load_checkpoint('models/actor.pt', 'models/critic.pt')
policy.actor.eval()

def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            
            # if np.random.rand() < 0.0:
            #     action = env.action_space.sample()
            # else:
            action = policy.choose_action(obs)
            
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("\n------------------------------------------")
    print(f"Average Reward over the Evaluation Step: {avg_reward}")
    print("------------------------------------------\n")
    return avg_reward


evaluate_policy(policy, eval_episodes=eval_episodes)
