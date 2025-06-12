# moq_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class MOQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim):
        super(MOQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # output_dim = number of objectives
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class MOQAgent:
    def __init__(self, state_dim, action_dim, reward_dim, lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.99, device='cpu'):
        self.device = device
        self.q_net = MOQNetwork(state_dim, action_dim, reward_dim).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim

        self.last_state = None
        self.last_action = None

    def select_action(self, state_tensor, preference_vector, action_space):
        if random.random() < self.epsilon:
            action = random.choice(action_space)
        else:
            state_rep = state_tensor.repeat(len(action_space), 1)
            action_tensor = torch.tensor(action_space, dtype=torch.float32).to(self.device)
            q_values = self.q_net(state_rep, action_tensor)  # shape [N, reward_dim]
            scores = torch.matmul(q_values, preference_vector.to(self.device))  # [N]
            action = action_space[torch.argmax(scores).item()]
        return action

    def update(self, s, a, r, s_next, action_space):
        # Convert to tensors
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
        a = torch.tensor(a, dtype=torch.float32).unsqueeze(0).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        s_next = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Compute target Q
        with torch.no_grad():
            next_actions = torch.tensor(action_space, dtype=torch.float32).to(self.device)
            next_states = s_next.repeat(len(action_space), 1)
            q_next = self.q_net(next_states, next_actions)  # [N, reward_dim]
            q_next_max = q_next.max(dim=0).values  # [reward_dim]

            target = r + self.gamma * q_next_max

        # Current Q
        q_pred = self.q_net(s, a).squeeze(0)

        loss = nn.MSELoss()(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
