# es_agent.py
import numpy as np

class ESAgent:
    def __init__(self, init_action, sigma, alpha, bounds=None):
        self.mean = np.array(init_action)
        self.sigma = sigma
        self.alpha = alpha
        self.dim = len(init_action)
        self.bounds = bounds if bounds is not None else [(-np.inf, np.inf)] * self.dim
        self.last_noise = None

    def sample_action(self):
        noise = np.random.randn(self.dim)
        action = self.mean + self.sigma * noise

        # Clip to bounds
        for i in range(self.dim):
            low, high = self.bounds[i]
            action[i] = np.clip(action[i], low, high)

        self.last_noise = noise
        return action

    def update(self, reward):
        # Simple gradient ascent update
        if self.last_noise is None:
            raise ValueError("No noise to update from. Call sample_action() first.")
        
        # Update mean based on reward signal
        self.mean += self.alpha * reward * self.last_noise
