import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2, use_mo=False):
        self.policy = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.use_mo = use_mo  # Multi-objective or scalar

    def select_action(self, state):
        """Sample action from policy."""
        state = state.unsqueeze(0)  # Add batch dim
        with torch.no_grad():
            action, _ = self.policy(state)
        return action.squeeze(0)

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

    def train(self, memory, mode="Scalar"):
        states, actions, rewards = zip(*memory)
        states = torch.stack(states)
        actions = torch.stack(actions)

        if mode == "MO":
            self.use_mo = True
            returns = self.compute_returns(rewards)
            returns = torch.stack(returns)
            # Reduce to scalar using uniform weights (or apply Pareto loss)
            returns = returns.mean(dim=1, keepdim=True)
        else:
            self.use_mo = False
            returns = self.compute_returns(rewards)
            returns = torch.tensor(returns).unsqueeze(1)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Get old action predictions
        old_actions, old_values = self.policy(states)
        dist = torch.distributions.Normal(old_actions, 0.1)
        log_probs_old = dist.log_prob(actions).sum(dim=1, keepdim=True)

        for _ in range(5):  # PPO epochs
            new_actions, state_values = self.policy(states)
            dist_new = torch.distributions.Normal(new_actions, 0.1)
            log_probs_new = dist_new.log_prob(actions).sum(dim=1, keepdim=True)

            ratios = torch.exp(log_probs_new - log_probs_old)
            advantages = returns - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
