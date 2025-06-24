# pcac_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class PCACActor(nn.Module):
    def __init__(self, state_dim, pref_dim, action_dim):
        super(PCACActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + pref_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, state, preference):
        x = torch.cat([state, preference], dim=-1)
        return self.net(x)


class PCACCritic(nn.Module):
    def __init__(self, state_dim, pref_dim, action_dim, reward_dim):
        super(PCACCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + pref_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, reward_dim)
        )

    def forward(self, state, preference, action):
        x = torch.cat([state, preference, action], dim=-1)
        return self.net(x)


class PCACAgent:
    def __init__(self, state_dim, action_dim, reward_dim, device='cpu', lr=1e-3, gamma=0.99, logger=None):
        self.device = device
        self.gamma = gamma
        self.reward_dim = reward_dim
        self.logger = logger

        self.actor = PCACActor(state_dim, reward_dim, action_dim).to(device)
        self.critic = PCACCritic(state_dim, reward_dim, action_dim, reward_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = []  # Simple replay buffer
        self.batch_size = 32

    def select_action(self, state_tensor, preference_vector, noise_std=0.05):
        state_tensor = state_tensor.unsqueeze(0).to(self.device)  # [1, state_dim]
        preference_vector = preference_vector.unsqueeze(0).to(self.device)  # [1, reward_dim]
        with torch.no_grad():
            action = self.actor(state_tensor, preference_vector).squeeze(0)
        # Add exploration noise
        action += torch.randn_like(action) * noise_std
        return torch.clamp(action, -1.0, 1.0)

    def store_transition(self, s, a, r, s_next, preference_vector):
        self.memory.append((s, a, r, s_next, preference_vector))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_next, p = zip(*batch)

        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.stack([torch.tensor(ai, dtype=torch.float32) for ai in a]).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        s_next = torch.tensor(s_next, dtype=torch.float32).to(self.device)
        p = torch.stack([pi.to(self.device) for pi in p])

        # --- Critic update ---
        with torch.no_grad():
            next_action = self.actor(s_next, p)
            q_next = self.critic(s_next, p, next_action)  # [B, reward_dim]
            target_q = r + self.gamma * q_next

        q_pred = self.critic(s, p, a)
        critic_loss = nn.MSELoss()(q_pred, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions = self.actor(s, p)
        actor_loss = -torch.mean(torch.sum(self.critic(s, p, new_actions) * p, dim=1))  # scalarized reward

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.logger:
            self.logger.info(f"[PCAC] Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")
