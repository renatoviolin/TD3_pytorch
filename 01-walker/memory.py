import numpy as np


class ReplayBuffer():
    def __init__(self, obs_shape, action_shape, max_size=1000000):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state = np.zeros((self.mem_size, obs_shape), dtype=np.float32)
        self.action = np.zeros((self.mem_size, action_shape), dtype=np.float32)
        self.reward = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state = np.zeros((self.mem_size, obs_shape), dtype=np.float32)
        self.done = np.zeros(self.mem_size, dtype=np.bool)

    def add(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.new_state[index] = state_
        self.done[index] = done
        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        idx = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state[idx]
        actions = self.action[idx]
        rewards = self.reward[idx]
        states_ = self.new_state[idx]
        done = self.done[idx]

        return states, actions, rewards, states_, done
