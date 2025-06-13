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
                 epsilon_min=0.1, epsilon_decay=0.99, epsilon_grow=1.01, device='cpu', action_grid_size=9, logger=None):
        self.device = device
        self.q_net = MOQNetwork(state_dim, action_dim, reward_dim).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_grow = epsilon_grow
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.logger = logger
        
        # Set centralized action space
        self.action_space = [
            [x, y]
            for x in np.linspace(-1, 1, action_grid_size)
            for y in np.linspace(-1, 1, action_grid_size)
        ]

        self.last_state = None
        self.last_action = None

    def select_action(self, state_tensor, preference_vector):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
            self.logger.info(f"[MOQAgent] Selecting random action: {[float(x) for x in action]} with epsilon: {self.epsilon:.4f}")
        else:
            state_rep = state_tensor.repeat(len(self.action_space), 1)
            action_tensor = torch.tensor(self.action_space, dtype=torch.float32).to(self.device)
            q_values = self.q_net(state_rep, action_tensor)  # [N, reward_dim]
            scores = torch.matmul(q_values, preference_vector.to(self.device))  # [N]
            action = self.action_space[torch.argmax(scores).item()]
        return action

    def update(self, s, a, r, s_next, preference_vector):
        # Convert to tensors
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)            # [1, state_dim]
        a = torch.tensor(a, dtype=torch.float32).unsqueeze(0).to(self.device)            # [1, action_dim]
        r = torch.tensor(r, dtype=torch.float32).to(self.device)                         # [reward_dim]
        s_next = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, state_dim]
        preference_vector = preference_vector.to(self.device)                            # [reward_dim]

        # Compute scalar target Q-value
        with torch.no_grad():
            next_actions = torch.tensor(self.action_space, dtype=torch.float32).to(self.device)       # [N, action_dim]
            next_states = s_next.repeat(len(self.action_space), 1)                                    # [N, state_dim]
            q_next = self.q_net(next_states, next_actions)                                       # [N, reward_dim]
            q_next_scalar = torch.matmul(q_next, preference_vector)                              # [N]
            q_next_max_scalar = q_next_scalar.max()                                              # scalar

            r_scalar = torch.dot(r, preference_vector)                                           # scalar
            target = r_scalar + self.gamma * q_next_max_scalar                                   # scalar

        # Compute predicted scalar Q-value
        q_pred = self.q_net(s, a).squeeze(0)             # [reward_dim]
        q_pred_scalar = torch.dot(q_pred, preference_vector)  # scalar

        # Loss and optimization
        loss = nn.MSELoss()(q_pred_scalar, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Optional: Logging for debugging
        self.logger.info(f"[MOQAgent] Loss: {loss.item():.4f}, Q_pred_scalar: {q_pred_scalar.item():.4f}, "
              f"Target: {target.item():.4f}, Epsilon: {self.epsilon:.4f}")


    def adjust_epsilon(self, r1, r2, r1_threshold=0.5, r2_threshold=0.01, epsilon_cap=0.5):
        """
        Adjust epsilon based on r1 and r2:
        - r1 = 1 - acc_curr (how bad the current performance is)
        - r2 = (prev_acc - acc_curr) * 5 (how much worse the model got)

        Logic:
        - If performance is good and improving ⇒ exploit more (lower epsilon)
        - If performance is bad and getting worse ⇒ explore more (higher epsilon)
        """

        old_epsilon = self.epsilon

        # Case 1: Good performance and improving → exploit more
        if r1 < r1_threshold and r2 < r2_threshold:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

        # Case 2: Bad performance and worsening → explore more
        elif r1 > r1_threshold and r2 > r2_threshold:
            self.epsilon *= self.epsilon_grow
            self.epsilon = min(epsilon_cap, self.epsilon)

        # Case 3: Mixed signals → do nothing or mild decay
        else:
            self.epsilon *= 0.99  # mild decay to encourage gradual exploitation
            self.epsilon = max(self.epsilon_min, self.epsilon)

        return old_epsilon, self.epsilon
