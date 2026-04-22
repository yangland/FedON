# pcac_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from copy import deepcopy

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
    def __init__(self, state_dim, action_dim, reward_dim, device='cpu', lr=1e-3, gamma=0.99, logger=None,
                 noise_std_init=0.2, noise_std_min=0.05, noise_std_max=0.4,
                 noise_decay=0.9, noise_growth=1.1, alpha=0.1,
                 exploration_mode='noise', epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):

        # Existing...
        self.device = device
        self.gamma = gamma
        self.reward_dim = reward_dim
        self.logger = logger

        self.actor = PCACActor(state_dim, reward_dim, action_dim).to(device)
        self.critic = PCACCritic(state_dim, reward_dim, action_dim, reward_dim).to(device)
        self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = []  # Simple replay buffer
        self.batch_size = 8 # 32

        # Noise exploration parameters
        self.reward_ema = 0.0
        self.noise_std = noise_std_init
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.noise_decay = noise_decay
        self.noise_growth = noise_growth
        self.alpha = alpha

        # Exploration mode: 'noise' or 'epsilon'
        self.exploration_mode = exploration_mode
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.prev_action = torch.zeros(action_dim).to(self.device)  # initialize to 0
        self.no_improvement_counter = 0


    # def update_noise_std_old(self, reward_scalar, reward_threshold=0.01):
    #     if self.exploration_mode != 'noise':
    #         return

    #     # Clip reward to [0, 1] for stability
    #     reward_scalar = float(np.clip(reward_scalar, 0.0, 1.0))

    #     # Exponential Moving Average of Reward
    #     self.reward_ema = (1 - self.alpha) * self.reward_ema + self.alpha * reward_scalar

    #     # Compute reward delta
    #     delta = reward_scalar - self.reward_ema

    #     # Only update noise if delta passes threshold
    #     if delta > reward_threshold:
    #         # Reward increasing → reduce noise (exploit)
    #         self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay)
    #     elif delta < -reward_threshold:
    #         # Reward decreasing → increase noise (explore)
    #         self.noise_std = min(self.noise_std_max, self.noise_std * self.noise_growth)
    #     # else: delta is too small → keep current noise_std

    #     # Optional: log the noise adjustment
    #     if self.logger:
    #         self.logger.info(
    #             f"[PCAC][Noise] reward: {reward_scalar:.4f}, EMA: {self.reward_ema:.4f}, "
    #             f"delta: {delta:.4f}, noise_std: {self.noise_std:.4f}"
    #         )


    # def update_noise_std(self, reward_scalar, reward_threshold=0.01):
    #     if self.exploration_mode != 'noise':
    #         return

    #     # Clip reward to [0, 1] for stability
    #     reward_scalar = float(np.clip(reward_scalar, 0.0, 1.0))

    #     # Update EMA of reward
    #     self.reward_ema = (1 - self.alpha) * self.reward_ema + self.alpha * reward_scalar

    #     # Compute delta
    #     delta = reward_scalar - self.reward_ema

    #     # Only update if change is significant
    #     if delta > reward_threshold:
    #         # Reward increasing → reduce noise (exploit)
    #         self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay)

    #     elif delta < -reward_threshold:
    #         # Reward decreasing → adaptively increase noise (explore)
    #         # Scale growth factor based on the magnitude of delta
    #         delta_magnitude = abs(delta)
    #         # Clamp the multiplier between 1.0 and some multiple of noise_growth
    #         dynamic_growth = min(self.noise_growth ** (1 + delta_magnitude * 5), 2.0)
    #         self.noise_std = min(self.noise_std_max, self.noise_std * dynamic_growth)

    #     # else: delta within threshold → no change

    #     if self.logger:
    #         self.logger.info(
    #             f"[PCAC][Noise] reward: {reward_scalar:.4f}, EMA: {self.reward_ema:.4f}, "
    #             f"delta: {delta:.4f}, dynamic_growth: {dynamic_growth if delta < -reward_threshold else 'N/A'}, "
    #             f"noise_std: {self.noise_std:.4f}"
    #         )
    
    def update_noise_std(
        self,
        reward_scalar: float,
        reward_threshold: float = 0.01,
        reward_valid: bool = True,
        max_plateau_rounds: int = 10,
    ):
        """Adaptively update exploration noise based on reward signal."""
        if self.exploration_mode != 'noise':
            return

        # Normalize reward to [0, 1] softly for smoother dynamics
        reward_scalar = float(np.clip(reward_scalar, 0.0, 1.0))

        if not reward_valid:
            # Skip update if reward is not valid (e.g., no attack was applied)
            return

        # Update EMA of reward
        self.reward_ema = (1 - self.alpha) * self.reward_ema + self.alpha * reward_scalar

        # Compute delta between current reward and EMA
        delta = reward_scalar - self.reward_ema

        # Track flat reward plateaus
        if abs(delta) < reward_threshold:
            self.no_improvement_counter += 1
        else:
            self.no_improvement_counter = 0

        # --- Adaptive Noise Update ---
        if delta > reward_threshold:
            # Reward increasing → reduce noise (exploit)
            self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay)

        elif delta < -reward_threshold:
            # Reward decreasing → increase noise (explore)
            delta_magnitude = abs(delta)
            dynamic_growth = min(self.noise_growth ** (1 + delta_magnitude * 5), 2.0)
            self.noise_std = min(self.noise_std_max, self.noise_std * dynamic_growth)

        # --- Plateau Handling ---
        if self.no_improvement_counter >= max_plateau_rounds:
            self.noise_std = min(self.noise_std_max, self.noise_std * self.noise_growth)
            if self.logger:
                self.logger.info(
                    f"[PCAC][Noise] Plateau detected for {max_plateau_rounds} rounds → forcing noise increase"
                )
            self.no_improvement_counter = 0

        # Clamp to range
        self.noise_std = np.clip(self.noise_std, self.noise_std_min, self.noise_std_max)

        # --- Logging ---
        if self.logger:
            self.logger.info(
                f"[PCAC][Noise] reward: {reward_scalar:.4f}, EMA: {self.reward_ema:.4f}, "
                f"delta: {delta:.4f}, noise_std: {self.noise_std:.4f}, "
                f"no_improvement_counter: {self.no_improvement_counter}"
            )


    def select_action(self, state_tensor, preference_vector):
        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        preference_vector = preference_vector.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            actor_output = self.actor(state_tensor, preference_vector).squeeze(0)

        if self.exploration_mode == 'noise':
            # Add Gaussian noise to actor output
            noisy_action = actor_output + torch.randn_like(actor_output) * self.noise_std

            # Wrap values instead of clamping
            noisy_action[0] = torch.clamp(noisy_action[0], -1.0, 1.0)
            noisy_action[1] = torch.clamp(noisy_action[1], -1.0, 1.0)

            # print(f"[PCAC] actor_output: {actor_output.cpu().numpy()}, noise_std: {self.noise_std:.4f}")
            # print(f"[PCAC] noisy_action: {noisy_action.cpu().numpy()}")
            self.prev_action = noisy_action.detach()
            return noisy_action

        elif self.exploration_mode == 'epsilon':
            if random.random() < self.epsilon:
                # Fully random action
                random_action = torch.zeros(2).to(self.device)
                random_action[0] = torch.FloatTensor(1).uniform_(-1.0, 1.0).to(self.device)
                random_action[1] = torch.FloatTensor(1).uniform_(-1.0, 1.0).to(self.device)

                print(f"[PCAC] random_action (ε={self.epsilon:.4f}): {random_action.cpu().numpy()}")
                self.prev_action = random_action.detach()
                return random_action
            else:
                # Use actor output
                actor_output[0] = torch.clamp(actor_output[0], -1.0, 1.0)
                actor_output[1] = torch.clamp(actor_output[1], -1.0, 1.0)

                print(f"[PCAC] actor_output (ε={self.epsilon:.4f}): {actor_output.cpu().numpy()}")
                self.prev_action = actor_output.detach()
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

    def update_target_critic(self, tau=0.005):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

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

        # Normalize rewards (optional but helps with critic stability)
        r_mean = r.mean(dim=0, keepdim=True)
        r_std = r.std(dim=0, keepdim=True) + 1e-6
        r_normalized = (r - r_mean) / r_std

        # --- Critic update ---
        with torch.no_grad():
            next_action = self.actor(s_next, p)
            # q_next = self.critic(s_next, p, next_action)  # [B, reward_dim]
            q_next = self.target_critic(s_next, p, next_action)
            # target_q = r + self.gamma * q_next
            target_q = r_normalized + self.gamma * q_next

        q_pred = self.critic(s, p, a)
        critic_loss = nn.MSELoss()(q_pred, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # ✅ Add gradient clipping here
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions = self.actor(s, p)
        actor_loss = -torch.mean(torch.sum(self.critic(s, p, new_actions) * p, dim=1))  # scalarized reward

        self.actor_optimizer.zero_grad()
        
        actor_loss.backward()

        # ✅ Optional: clip actor gradients too
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)        
        
        self.actor_optimizer.step()
        self.update_target_critic()
        
        if self.logger:
            self.logger.info(f"[PCAC] Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")
