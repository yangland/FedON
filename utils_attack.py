import torch
import torch.nn.functional as F
import math
from typing import List, Optional
from skopt import Optimizer
import numpy as np
import random

def flatten_model(state_dict):
    return torch.cat([p.view(-1).float() for p in state_dict.values() if isinstance(p, torch.Tensor)])


def vector_to_state_dict(vector, model):
    ref_sd = model.state_dict()
    new_sd = {}
    idx = 0
    for k, v in ref_sd.items():
        if isinstance(v, torch.Tensor):
            numel = v.numel()
            new_sd[k] = vector[idx: idx + numel].view_as(v).clone().detach()
            idx += numel
        else:
            new_sd[k] = v
    return new_sd


def log2_map(norm_value: float, min_val: float, max_val: float) -> float:
    assert 0.0 <= norm_value <= 1.0
    log_min = math.log2(min_val)
    log_max = math.log2(max_val)
    log_val = log_min + norm_value * (log_max - log_min)
    return 2 ** log_val


def log2_map_from_tanh(norm_value, min_val, max_val):
    norm_01 = (norm_value + 1.0) / 2.0  # Map [-1,1] to [0,1]
    
    if min_val == 0:
        # Linear mapping when min_val is 0 to avoid log(0)
        return norm_01 * max_val
    else:
        log_min = math.log2(min_val)
        log_max = math.log2(max_val)
        log_val = log_min + norm_01 * (log_max - log_min)
        return 2 ** log_val

def log_interp_scaling_cos(norm_value, min_val, max_val):
    """
    Logarithmic interpolation between max_val (cos_dist_mali_benign)
    and min_val (avg_cos_distance), based on norm_value ∈ [-1, 1].

    Returns scaling_cos ∈ [max_val, min_val], log-spaced.
    """
    norm_01 = (norm_value + 1.0) / 2.0  # Map [-1, 1] → [0, 1]
    log_min = np.log2(min_val)
    log_max = np.log2(max_val)
    log_val = log_max - norm_01 * (log_max - log_min)  # Reverse direction
    return 2 ** log_val


def linear_map_from_tanh(norm_value, min_val, max_val):
    """Linearly map [-1, 1] to [min_val, max_val]."""
    return min_val + (norm_value + 1.0) / 2.0 * (max_val - min_val)

def calculate_bo_reward(
    acc_history: List[float],
    current_round: int,
    gamma: float = 0.95,
    warmup_rounds: int = 3,
    logger=None
) -> Optional[float]:
    if len(acc_history) < 2 or current_round < warmup_rounds:
        return None

    current_acc = acc_history[-1]
    prev_acc = acc_history[-2]
    delta_acc = prev_acc - current_acc
    discount = gamma ** current_round
    acc_scale = max(0.0, min(1.0, 1.0 - (current_acc / 100)))
    reward = delta_acc * acc_scale * discount

    if logger:
        logger.info(f"[BO Reward] Δacc: {delta_acc:.4f}, scale: {acc_scale:.4f}, reward: {reward:.6f}")

    return reward


def compute_attack_influence_r3(fedoda):
    if (fedoda.prev_server_model is None or
        fedoda.prev_benign_update is None or
        fedoda.prev_malicious_update is None):
        fedoda.logger.warning("[r3] Missing history. Skipping.")
        return None

    delta_s = flatten_model(fedoda.server_model.state_dict()) - flatten_model(fedoda.prev_server_model.state_dict())
    delta_b = fedoda.prev_benign_update
    delta_m = fedoda.prev_malicious_update

    cos_b_s = F.cosine_similarity(delta_b.unsqueeze(0), delta_s.unsqueeze(0), dim=1).item()
    cos_m_s = F.cosine_similarity(delta_m.unsqueeze(0), delta_s.unsqueeze(0), dim=1).item()

    r3 = cos_b_s / (cos_m_s + 1e-8)
    fedoda.logger.info(f"[r3] cos_b_s={cos_b_s:.4f}, cos_m_s={cos_m_s:.4f}, r3={r3:.4f}")
    return r3


class BayesOptimizer:
    def __init__(self, bounds, base_estimator="GP", acq_func="EI", acq_func_kwargs=None, random_state=None):
        self.optimizer = Optimizer(
            dimensions=bounds,
            base_estimator=base_estimator,
            acq_func=acq_func,
            acq_func_kwargs=acq_func_kwargs,
            random_state=random_state
        )
        self.xs = []
        self.ys = []
        self.best_x = None
        self.best_y = -float("inf")


    def suggest_next_point(self):
        if self.best_x is not None and random.random() < 0.2:
            # 20% of the time, exploit best action with small noise
            x_best = self.best_x
            noise = np.random.normal(scale=0.05, size=len(x_best))
            x_next = (np.array(x_best) + noise).clip(-1.0, 1.0).tolist()
        else:
            x_next = self.optimizer.ask()
        self.xs.append(x_next)
        return x_next


    def observe(self, x, reward):
        # Optional: overwrite the last point if it was just suggested
        if len(self.xs) > 0 and self.xs[-1] == x:
            self.ys.append(reward)
            self.optimizer.tell(x, reward)
        else:
            # for consistency in external use
            self.xs.append(x)
            self.ys.append(reward)
            self.optimizer.tell(x, reward)
        
        if reward > self.best_y:
            self.best_y = reward
            self.best_x = x
