# moq_attack.py
from utils_attack import linear_map_from_tanh, log2_map_from_tanh, log_interp_scaling_cos
import torch
import torch.nn.functional as F
import numpy as np
from utils_attack import vector_to_state_dict

def moq_attack(fedoda, mali_direction, benign_update, avg_cos_distance, avg_l2_distance,
               avg_update_norm, prev_acc, curr_acc):
    agent = fedoda.moq_agent
    preference = fedoda.moq_preference_vector

    # === Step 1: Build state vector
    # round_number = len(fedoda.val_acc_pooled_history)
    acc_delta = curr_acc - prev_acc
    if fedoda.prev_server_model is None:
        model_norm_delta = 0
    else:
        prev_model_norm = get_model_l2_norm(fedoda.prev_server_model)
        curr_model_norm = get_model_l2_norm(fedoda.server_model)
        model_norm_delta = float(np.tanh((curr_model_norm - prev_model_norm)/prev_model_norm))

    # state = [curr_acc, acc_delta, model_norm_delta, avg_cos_distance]
    state = [curr_acc, acc_delta, model_norm_delta]
    # state = [curr_acc, acc_delta, avg_cos_distance]
    state_tensor = torch.tensor(state, dtype=torch.float32).to(agent.device)

    # === Step 2: Sample action (normalized space)
    action = agent.select_action(state_tensor, preference)
    norm_scaling_l2, norm_scaling_cos = action

    # === Step 3: Map normalized action to real values
    scaling_l2 = linear_map_from_tanh(norm_scaling_l2, 0.5, 20.0)

    cos_dist_mali_benign = 1 - F.cosine_similarity(
            mali_direction.unsqueeze(0), benign_update.unsqueeze(0), dim=1).item()
    cos_dist_mali_benign = max(0.0, min(2.0, cos_dist_mali_benign))
    cos_dist_mali_max =  cos_dist_mali_benign - avg_cos_distance
    scaling_cos_ = cos_dist_mali_max/avg_cos_distance
    fedoda.logger.info(f"Cos Attack Space: {scaling_cos_:1f} X times of avg_cos_distance")
    
    raw_high_bound = avg_cos_distance * scaling_cos_
    cos_dist_low_bound = 0
    cos_dist_high_bound = max(cos_dist_low_bound, min(2.0, raw_high_bound))

    fedoda.logger.info(f"[MOQ Attack] cos_dist_high_bound: {cos_dist_high_bound:.4f}")

    scaling_cos = log2_map_from_tanh(
        norm_scaling_cos,
        min_val=cos_dist_low_bound,
        max_val=cos_dist_high_bound
    )

    fedoda.scaling_l2 = scaling_l2
    fedoda.scaling_cos = scaling_cos
    fedoda.epsilon = scaling_l2 * avg_update_norm

    malicious_update = fedoda.mali_update_craft_l2_cos(
        model=fedoda.server_model,
        benign_update=benign_update,
        target_direction=mali_direction,
        scaling_cos_dist=fedoda.scaling_cos,
        epsilon=fedoda.epsilon,
        harm_threshold=0.0,
        retry_limit=10,
        epsilon_growth=1,
        cos_growth=0.75,
    )
    
    # === Step 7: Return malicious update
    malicious_update_dict = vector_to_state_dict(malicious_update, fedoda.server_model)

    return malicious_update_dict, malicious_update, [norm_scaling_l2, norm_scaling_cos], state


def get_model_l2_norm(model):
    total_norm = 0.0
    with torch.no_grad():
        for param in model.state_dict().values():
            total_norm += torch.norm(param.float(), p=2).item() ** 2
    return total_norm ** 0.5