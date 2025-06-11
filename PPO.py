import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class RewardNormalizer:
    def __init__(self, clip_range=(-1.0, 1.0)):
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.clip_range = clip_range

    def update(self, value: float):
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def normalize(self, value: float) -> float:
        self.update(value)
        if self.max_val == self.min_val:
            return 0.0  # No variation yet
        normalized = 2 * (value - self.min_val) / (self.max_val - self.min_val) - 1
        return max(min(normalized, self.clip_range[1]), self.clip_range[0])



class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable std

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        mean = self.actor_mean(x)
        std = torch.exp(self.log_std)
        value = self.critic(x)
        return mean, std, value



class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2,
                 use_mo=False, device="cuda:0", reward_weights=None):
        self.device = device
        self.policy = PPOActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.use_mo = use_mo  # Multi-objective or scalar
        self.buffer = []
        self.r2_normalizer = RewardNormalizer()
        self.r3_normalizer = RewardNormalizer()
        self.last_action = None

        
        # Use provided weights or default to uniform (if multi-objective is used)
        if reward_weights is not None:
            self.reward_weights = torch.tensor(reward_weights, dtype=torch.float32, device=self.device)
        else:
            self.reward_weights = None  # Will be handled during training

        self.replay_buffer = deque(maxlen=100)  # Store up to 100 past experiences

        
    def select_action(self, state, round_number=None):
        state = state.unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, std, _ = self.policy(state)
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)

            # Correct log_prob for tanh squashing
            log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1)

        # Smooth current action with previous one
        if hasattr(self, 'last_action') and self.last_action is not None:
            alpha = 0.7  # Smoothing factor: higher = more stable
            smoothed_action = alpha * action + (1 - alpha) * self.last_action.to(self.device)
            action = smoothed_action

        # Save current action as last action
        self.last_action = action.detach().cpu()

        return action.squeeze(0), log_prob.detach()



    def compute_returns(self, rewards):
        """Compute discounted returns (scalar or vector)."""
        if self.use_mo:
            # Vector-valued rewards
            returns = []
            G = torch.zeros_like(rewards[0])
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            return returns
        else:
            # Scalar rewards
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            return returns


    def combine_reward(self, round_number, reward_vector):
            """
            Combine a reward vector into a single scalar reward using weighted average.

            Args:
                round_number (int): Current round (optional, can be used for dynamic weighting later)
                reward_vector (list or tensor): A list or tensor of reward components

            Returns:
                float: Scalar reward
            """
            r1, r2, r3 = reward_vector

            # Normalize each reward
            r1_n = r1
            r2_n = self.r2_normalizer.normalize(r2)
            r3_n = self.r3_normalizer.normalize(r3)

            reward_vector = torch.tensor([r1_n, r2_n, r3_n], dtype=torch.float32, device=self.device)

            # Normalize weights to sum to 1
            weights = (self.reward_weights / self.reward_weights.sum()).to(self.device)

            # Weighted average
            scalar_reward = (weights * reward_vector).sum().item()

            return scalar_reward
        
        
    def store_experience(self, state, action, reward):
        self.replay_buffer.append((state.detach().cpu(), action.detach().cpu(), reward.detach().cpu()))


    def train(self, memory=None, batch_size=64, round_number=0, total_rounds=300):
        # Store new memory into replay buffer
        if memory:
            for state, action, reward in memory:
                self.store_experience(state, action, reward)

        # Sample from replay buffer
        if len(self.replay_buffer) < batch_size:
            return  # Not enough experience yet

        sampled = list(torch.utils.data.RandomSampler(self.replay_buffer, replacement=False, num_samples=batch_size))
        sampled_batch = [self.replay_buffer[idx] for idx in sampled]

        states, actions, rewards = zip(*sampled_batch)

        states = torch.stack([s.to(self.device) for s in states])
        actions = torch.stack([a.to(self.device) for a in actions])
        rewards = torch.stack([r.to(self.device) for r in rewards])

        # Compute returns
        self.use_mo = True
        returns = self.compute_returns(rewards)
        returns = torch.stack(returns)
        if self.reward_weights is not None:
            returns = torch.matmul(returns, self.reward_weights.unsqueeze(1))  # shape: (batch, 1)
        else:
            returns = returns.mean(dim=1, keepdim=True)


        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Get old action predictions using frozen policy
        with torch.no_grad():
            mean_old, std_old, _ = self.policy(states)
            dist_old = torch.distributions.Normal(mean_old, std_old)
            log_probs_old = dist_old.log_prob(actions).sum(dim=1, keepdim=True)

        for _ in range(5):  # PPO epochs
            mean_new, std_new, state_values = self.policy(states)
            dist_new = torch.distributions.Normal(mean_new, std_new)
            log_probs_new = dist_new.log_prob(actions).sum(dim=1, keepdim=True)
            entropy = dist_new.entropy().sum(dim=1, keepdim=True)

            # PPO clipped objective
            ratios = torch.exp(log_probs_new - log_probs_old)
            advantages = returns - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(state_values, returns)

            # Anneal entropy bonus based on training round
            # You need to pass round_number into train() for this to work
            entropy_weight = max(0.01 * (1 - round_number / total_rounds), 0.001)
            loss = actor_loss + 0.5 * critic_loss - entropy_weight * entropy.mean()


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
