# pcac_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F

class PCACActor(nn.Module):
    def __init__(self, state_dim, pref_dim, action_dim):
        super(PCACActor, self).__init__()
        assert action_dim == 2, "This design assumes 2 actions: [L2, cosine]."
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim + pref_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Separate output layers for each action
        self.out_l2 = nn.Linear(64, 1)     # Action[0]
        self.out_cos = nn.Linear(64, 1)    # Action[1]

    def forward(self, state, preference):
        x = torch.cat([state, preference], dim=-1)  # shape: [batch_size, state_dim + pref_dim]
        features = self.shared(x)  # shape: [batch_size, 64]

        # Apply separate activations
        a0 = torch.tanh(self.out_l2(features))      # [-1, 1]
        # a1 = torch.sigmoid(self.out_cos(features))  # [0, 1]
        a1 = torch.tanh(self.out_cos(features))      # [-1, 1]

        action = torch.cat([a0, a1], dim=-1)        # shape: [batch_size, 2]
        return action



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
        
        self.reward_ema = 0.0
        self.noise_std = 0.2  # starts high
        self.exploration_mode = 'noise'     # or 'epsilon'
        self.alpha = 0.1      # smoothing factor for EMA
        self.epsilon = 1.0          # Start with full exploration
        self.epsilon_min = 0.05     # Minimum ε (explore at least a little)
        self.epsilon_decay = 0.995  # Decay rate per training step (or episode)


    def update_noise_std(self, reward_scalar):
        if self.exploration_mode == 'noise':
            """
            Update exploration noise based on reward feedback.
            If reward improves → reduce noise (exploit more)
            If reward worsens → increase noise (explore more)
            """
            # Normalize reward to [0, 1] for stability (optional)
            reward_scalar = float(reward_scalar)
            reward_scalar = np.clip(reward_scalar, 0.0, 1.0)

            # Update EMA of recent rewards
            self.reward_ema = (1 - self.alpha) * self.reward_ema + self.alpha * reward_scalar

            # Compute delta
            delta = reward_scalar - self.reward_ema

            # Adjust noise_std based on delta
            if delta > 0:
                # Reward better than expected → decrease noise
                self.noise_std = max(0.05, self.noise_std * 0.9)
            else:
                # Reward worse than expected → increase noise
                self.noise_std = min(0.4, self.noise_std * 1.1)




    def select_action(self, state_tensor, preference_vector):
        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        preference_vector = preference_vector.unsqueeze(0).to(self.device)
        
        def wrap_action(x, low=-1.0, high=1.0):
            """
            Wrap a tensor x into the range [low, high] using modular arithmetic.
            """
            range_size = high - low
            return ((x - low) % range_size) + low
        
        with torch.no_grad():
            actor_output = self.actor(state_tensor, preference_vector).squeeze(0)

        if self.exploration_mode == 'noise':
            # Add Gaussian noise to actor output
            noisy_action = actor_output + torch.randn_like(actor_output) * self.noise_std

            # Wrap values instead of clamping
            noisy_action[0] = wrap_action(noisy_action[0], -1.0, 1.0)
            noisy_action[1] = wrap_action(noisy_action[1], -1.0, 1.0)

            print(f"[PCAC] actor_output: {actor_output.cpu().numpy()}, noise_std: {self.noise_std:.4f}")
            print(f"[PCAC] noisy_action: {noisy_action.cpu().numpy()}")

            return noisy_action

        elif self.exploration_mode == 'epsilon':
            if random.random() < self.epsilon:
                # Fully random action
                random_action = torch.zeros(2).to(self.device)
                random_action[0] = torch.FloatTensor(1).uniform_(-1.0, 1.0).to(self.device)
                random_action[1] = torch.FloatTensor(1).uniform_(-1.0, 1.0).to(self.device)

                print(f"[PCAC] random_action (ε={self.epsilon:.4f}): {random_action.cpu().numpy()}")
                return random_action
            else:
                # Use actor output
                actor_output[0] = torch.clamp(actor_output[0], -1.0, 1.0)
                actor_output[1] = torch.clamp(actor_output[1], -1.0, 1.0)

                print(f"[PCAC] actor_output (ε={self.epsilon:.4f}): {actor_output.cpu().numpy()}")
                return actor_output

        else:
            raise ValueError(f"Unknown exploration_mode: {self.exploration_mode}")





    def update_epsilon(self):
        if self.exploration_mode == 'epsilon':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
