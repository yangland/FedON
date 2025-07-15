
from utils_attack import linear_map_from_tanh, log2_map_from_tanh, log_interp_scaling_cos, get_model_l2_norm
import torch
import torch.nn.functional as F
import numpy as np
from utils_attack import vector_to_state_dict

def pcac_attack(fedoda, mali_direction, benign_update, avg_cos_distance, avg_l2_distance,
                avg_update_norm, prev_acc, curr_acc, iteration):

    agent = fedoda.pcac_agent
    preference = fedoda.pcac_preference_vector
    cos_dist_mali_benign = 1 - F.cosine_similarity(
                            mali_direction.unsqueeze(0), benign_update.unsqueeze(0), dim=1
                            ).item()
    
    acc_delta = curr_acc - prev_acc
    model_norm_delta = 0.0
    cos_dist_rate = 1.0
    if fedoda.prev_server_model is not None:
        prev_model_norm = get_model_l2_norm(fedoda.prev_server_model)
        curr_model_norm = get_model_l2_norm(fedoda.server_model)
        model_norm_delta = float(np.tanh((curr_model_norm - prev_model_norm) / prev_model_norm))

        cos_dist_rate = cos_dist_mali_benign / (avg_cos_distance + 1e-8)
 
    state = [curr_acc, acc_delta,  cos_dist_rate] # model_norm_delta,
    state_tensor = torch.tensor(state, dtype=torch.float32).to(agent.device)


    # ----> Action selection
    action = agent.select_action(state_tensor, preference)
    norm_scaling_l2, norm_scaling_cos = action  # cosine ∈ [0,1] now
    
        # === Apply Ablation Configurations ===
    if fedoda.ablation_mode == "fix_degree":
        # Fix cosine distance to baseline (no directional constraint)
        norm_scaling_cos = torch.tensor(0.0).to(agent.device)  # action[1] = 0

    elif fedoda.ablation_mode == "fix_l2":
        # Fix L2 scaling so scaling_l2 == 3.0
        # Reverse of: scaling_l2 = linear_map_from_tanh(action_val, 0.5, 20.0)
        def inverse_linear_map(val, low=0.5, high=20.0):
            val = np.clip(val, low, high)
            norm = 2 * (val - low) / (high - low) - 1
            return float(np.clip(norm, -1.0, 1.0))

        fixed_action0 = inverse_linear_map(3.0, 0.5, 20.0)  # L2 scaling == 3
        norm_scaling_l2 = torch.tensor(fixed_action0).to(agent.device)  # action[0] = value

    if fedoda.ablation_mode:
        fedoda.logger.info(f"[FedOda][Ablation] Mode: {fedoda.ablation_mode}, "
                        f"Action Override → L2: {norm_scaling_l2.item():.4f}, Cos: {norm_scaling_cos.item():.4f}")

    
    
    # ----> L2 scaling
    scaling_l2 = linear_map_from_tanh(norm_scaling_l2.item(), 0.5, 20.0)

    # ----> Cosine upper bound
    # cos_dist_high_bound = max(0.0, min(2.0, cos_dist_mali_benign - avg_cos_distance/2)) # TODO: change to min(2.0, cos_dist_mali_benign)
    cos_dist_high_bound = max(0.0, min(2.0, cos_dist_mali_benign))

    # # ----> Updated 0-1 cosine mapping
    # scaling_cos = cosine_mapping_from_0_to_1(norm_scaling_cos.item(), max_val=cos_dist_high_bound)
    
    # -- Use the new symmetric log mapping
    scaling_cos = cosine_mapping_symmetric_log(
        action_val=norm_scaling_cos.item(),
        max_pos_val=cos_dist_high_bound,
        cos_dist_mali_benign=cos_dist_mali_benign,  # used to compute 2 - cos_dist
        log_base=4.0,
        start_at=0.01
    )

    # ----> Apply attack
    fedoda.scaling_l2 = scaling_l2
    fedoda.scaling_cos = scaling_cos
    fedoda.epsilon = scaling_l2 * avg_update_norm

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
        mode="PROJ",
        pgd_steps=20,
        pgd_lr=0.05
    )

    malicious_update_dict = vector_to_state_dict(malicious_update, fedoda.server_model)
    return malicious_update_dict, malicious_update, action.tolist(), state, agent.noise_std


def cosine_mapping_from_0_to_1(action_val, max_val=1.0, log_base=4.0, start_at=0.01):
    """
    Maps action ∈ [0, 1] to cosine distance ∈ [start_at, max_val]
    - action = 0 → start_at (very small constraint)
    - action = 1 → max_val
    - log-scaled growth
    """
    action = float(torch.clamp(torch.tensor(action_val), 0.0, 1.0))
    scaled = np.log1p(action * (log_base - 1)) / np.log(log_base)  # ∈ [0, 1]
    return start_at + (max_val - start_at) * scaled


def cosine_mapping_symmetric_log(action_val, 
                                 max_pos_val,  # toward benign
                                 cos_dist_mali_benign,
                                 log_base=4.0, 
                                 start_at=0.01):
    """
    Maps action_val ∈ [-1, 1] to cosine distance from mali_direction using log scale:
    - action = 0 → cos_dist = 0 (aligned with mali_direction)
    - action = +1 → cos_dist = max_pos_val (toward benign)
    - action = -1 → cos_dist = max_neg_val = 2 - cos_dist_mali_benign (opposite of benign)
    """
    action_val = float(torch.clamp(torch.tensor(action_val), -1.0, 1.0))

    if action_val > 0:
        # Log mapping toward benign direction
        scaled = np.log1p(action_val * (log_base - 1)) / np.log(log_base)
        return start_at + (max_pos_val - start_at) * scaled

    elif action_val < 0:
        # Log mapping toward opposite of benign
        max_neg_val = 2.0 - cos_dist_mali_benign
        scaled = np.log1p(-action_val * (log_base - 1)) / np.log(log_base)
        return start_at + (max_neg_val - start_at) * scaled

    else:
        return 0.0


def compute_noise_std(iteration, curr_acc, prev_acc, explore_window=50):
    if iteration < explore_window:
        return 0.5  # aggressively explore in early rounds

    acc_delta = curr_acc - prev_acc

    if curr_acc < 0.5:
        # Attack likely effective, stabilize
        return 0.1
    else:
        # Attack ineffective, encourage exploration
        return min(0.5, 0.1 + 0.4 * acc_delta)
