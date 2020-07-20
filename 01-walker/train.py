from TD3 import TD3
import numpy as np
import os
import torch
import gym
import pybullet_envs
from gym import wrappers


env_name = "Walker2DBulletEnv-v0"
env = gym.make(env_name)
start_timesteps = 10_000
eval_freq = 5_000
max_timesteps = 500_000
expl_noise = 0.1
batch_size = 100
discount = 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_delay = 2
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
max_episode_steps = env._max_episode_steps

if not os.path.exists("./models"):
    os.makedirs("./models")


def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.choose_action(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("\n------------------------------------------")
    print(f"Average Reward over the Evaluation Step: {avg_reward}")
    print("------------------------------------------\n")
    return avg_reward


seed = 0
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


total_timesteps = 0
episode_reward = 0
episode_timesteps = 0
episode_num = 0
done = False
obs = env.reset()
policy = TD3(state_dim, action_dim, max_action)
import ipdb
ipdb.set_trace()

while total_timesteps < max_timesteps:

    if total_timesteps < start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.choose_action(obs)
        if expl_noise != 0:
            action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)

    new_obs, reward, done, _ = env.step(action)
    episode_reward += reward

    policy.replay_buffer.add(obs, action, reward, new_obs, bool(done))
    obs = new_obs

    episode_timesteps += 1
    total_timesteps += 1

    if done:
        print("Total Timesteps: {} Episode Timesteps {} Episode Num: {} Reward: {}".format(total_timesteps, episode_timesteps, episode_num, episode_reward))
        policy.train(episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_delay)
        obs = env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if total_timesteps % eval_freq == 0:
        evaluate_policy(policy)
        policy.save_checkpoint('models/actor.pt', 'models/critic.pt')
