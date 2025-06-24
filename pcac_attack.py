
from utils_attack import linear_map_from_tanh, log2_map_from_tanh, log_interp_scaling_cos, get_model_l2_norm
import torch
import torch.nn.functional as F
import numpy as np
from utils_attack import vector_to_state_dict

def pcac_attack(fedoda, mali_direction, benign_update, avg_cos_distance, avg_l2_distance,
                avg_update_norm, prev_acc, curr_acc):
    agent = fedoda.pcac_agent
    preference = fedoda.pcac_preference_vector

    acc_delta = curr_acc - prev_acc
    if fedoda.prev_server_model is None:
        model_norm_delta = 0
    else:
        prev_model_norm = get_model_l2_norm(fedoda.prev_server_model)
        curr_model_norm = get_model_l2_norm(fedoda.server_model)
        model_norm_delta = float(np.tanh((curr_model_norm - prev_model_norm) / prev_model_norm))

    # ----> Construct State
    state = [curr_acc, acc_delta, model_norm_delta]
    state_tensor = torch.tensor(state, dtype=torch.float32).to(agent.device)

    # ----> Compute r1 for dynamic noise
    noise_std = max(0.01, 0.5 * curr_acc)  # More noise when accuracy is high

    # ----> Action selection with adaptive noise
    action = agent.select_action(state_tensor, preference, noise_std=noise_std)
    norm_scaling_l2, norm_scaling_cos = action

    # === L2 Scaling ===
    scaling_l2 = linear_map_from_tanh(norm_scaling_l2.item(), 0.5, 20.0)

    # === Cosine Distance High Bound ===
    cos_dist_mali_benign = 1 - F.cosine_similarity(
        mali_direction.unsqueeze(0), benign_update.unsqueeze(0), dim=1
    ).item()
    cos_dist_mali_benign = max(0.0, min(2.0, cos_dist_mali_benign))
    cos_dist_high_bound = max(0.0, min(2.0, cos_dist_mali_benign / avg_cos_distance))

    # === Corrected Cosine Distance Mapping (start from 0) ===
    def map_action_to_cos_dist(action_val, max_val=1.0, log_base=2.0):
        """
        Maps action ∈ [-1, 1] to cosine distance ∈ [0, max_val]
        - action ≤ 0  → 0.0
        - action > 0 → grows log-scale up to max_val
        """
        action_tensor = torch.tensor(action_val)
        action_tensor = torch.clamp(action_tensor, -1.0, 1.0)

        if action_tensor.item() <= 0.0:
            return 0.0
        else:
            # log1p(x): log(1 + x)
            scaled = torch.log1p(action_tensor * (log_base - 1)) / torch.log(torch.tensor(log_base))
            return (max_val * scaled).item()


    scaling_cos = map_action_to_cos_dist(norm_scaling_cos.item(), max_val=cos_dist_high_bound)

    # === Store scalings ===
    fedoda.scaling_l2 = scaling_l2
    fedoda.scaling_cos = scaling_cos
    fedoda.epsilon = scaling_l2 * avg_update_norm

    # === Craft malicious update ===
    malicious_update = fedoda.mali_update_craft_l2_cos(
        model=fedoda.server_model,
        benign_update=benign_update,
        target_direction=mali_direction,
        scaling_cos_dist=scaling_cos,
        epsilon=fedoda.epsilon,
        harm_threshold=0.0,
        retry_limit=5,
        epsilon_growth=1.0,
        cos_growth=0.9,
        mode="PROJ",              # Could also be "PGD", "PROJ"
        pgd_steps=20,
        pgd_lr=0.05
    )

    # fedoda.compare_proj_vs_pgd_attack(...)  # Optional debugging

    malicious_update_dict = vector_to_state_dict(malicious_update, fedoda.server_model)

    return malicious_update_dict, malicious_update, action.tolist(), state, noise_std

