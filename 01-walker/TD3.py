import torch
import numpy as np
from memory import ReplayBuffer
from models import Actor, Critic
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TD3():
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
            action = self.actor(state)
            return action.detach().cpu().numpy().flatten()

    def save_checkpoint(self, actor_checkpoint, critic_checkpoint):
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)

    def load_checkpoint(self, actor_checkpoint, critic_checkpoint):
        self.actor.load_state_dict(torch.load(actor_checkpoint, map_location=device))
        self.actor_target.load_state_dict(torch.load(actor_checkpoint, map_location=device))
        self.critic.load_state_dict(torch.load(critic_checkpoint, map_location=device))
        self.critic_target.load_state_dict(torch.load(critic_checkpoint, map_location=device))

    def train(self, iterations, batch_size=100, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        if self.replay_buffer.mem_cntr < batch_size:
            return

        for i in range(iterations):
            # 4. Sample from memory
            batch_state, batch_action, batch_reward, batch_state_, batch_done = self.replay_buffer.sample(batch_size)
            state = torch.tensor(batch_state, dtype=torch.float, device=device)
            action = torch.tensor(batch_action, dtype=torch.float, device=device)
            reward = torch.tensor(batch_reward, dtype=torch.float, device=device).view(-1, 1)
            state_ = torch.tensor(batch_state_, dtype=torch.float, device=device)
            done = torch.tensor(batch_done, dtype=torch.bool, device=device).view(-1, 1)

            # 5. Actor target play state_
            action_ = self.actor_target(state_)

            # 6. Add gaussian noise the same shape as action (bs, n_actions)
            noise = torch.Tensor(batch_action).data.normal_(0, policy_noise).clamp(-noise_clip, noise_clip).to(device)
            action_ = (action_ + noise).clamp(-self.max_action, self.max_action)

            # 7. Critic target play with state_, action_
            q1_target, q2_target = self.critic_target(state_, action_)

            # 8. Min of q1_target and q2_target
            min_q = torch.min(q1_target, q2_target)  # remove from backpropagation

            # 9. Calculate bellman equation
            q_target = reward + (gamma * min_q).detach()
            q_target[done] = 0.0

            # 10. Critic play with state, action
            q1, q2 = self.critic(state, action)

            # 11. calculate critic loss of each Critic and the q_target
            critic_loss_1 = torch.nn.functional.mse_loss(q1, q_target)
            critic_loss_2 = torch.nn.functional.mse_loss(q2, q_target)
            critic_loss = critic_loss_1 + critic_loss_2

            # 12. backpropagation
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 13. Delayed update of Actor model by perform gradient ascent trying to
            if i % policy_freq == 0:
                actor_loss = - self.critic.q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 14. Update actor_target using Polyak Average
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # 15. Update critic_target using Polyak Average
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
