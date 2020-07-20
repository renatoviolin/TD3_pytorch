# %load_ext autoreload
# %autoreload 2

import torch
import numpy as np
import TD3

state_dim = 5
action_dim = 2
max_action = 1
batch_size = 1

state_in = np.random.randint(0, 2, size=(batch_size, state_dim))
action_in = np.random.randint(0, 2, size=(batch_size, action_dim))
action_new = np.random.randint(0, 2, size=(batch_size, state_dim))
reward = np.random.randint(0, 2, size=2)
done = False


td = TD3.TD3(state_dim, action_dim, max_action)

for _ in range(32):
    done = np.random.choice([True, False])
    td.replay_buffer.add(state_in, action_in, 0.0, action_new, done)

td.train(1, batch_size=5)
