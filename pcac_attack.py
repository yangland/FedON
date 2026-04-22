
from utils_attack import linear_map_from_tanh
import torch
import torch.nn.functional as F
import numpy as np
from utils_attack import vector_to_state_dict

def pcac_attack(fedon, mali_direction, benign_update, avg_cos_distance, avg_l2_distance,
                avg_update_norm, prev_acc, curr_acc, iteration):

    agent = fedon.pcac_agent
    attack_cfg = fedon.attack_config
    preference = fedon.pcac_preference_vector
    cos_dist_mali_benign = 1 - F.cosine_similarity(
                            mali_direction.unsqueeze(0), benign_update.unsqueeze(0), dim=1
                            ).item()
    
    acc_delta = curr_acc - prev_acc

    if fedon.prev_server_model is not None:
        # prev_model_norm = get_model_l2_norm(fedon.prev_server_model)
        # curr_model_norm = get_model_l2_norm(fedon.server_model)
        # model_norm_delta = float(np.tanh((curr_model_norm - prev_model_norm) / prev_model_norm))
        # cos_dist_rate = cos_dist_mali_benign / (avg_cos_distance + 1e-8)
        cos_dist_mb = cos_dist_mali_benign 
    else:
        cos_dist_mb = 2.0
        
    malicious_ratio_prev = fedon.prev_s_m / fedon.total_malicious if fedon.total_malicious > 0 else 0.0
    state = [curr_acc, acc_delta,  cos_dist_mb, avg_cos_distance, malicious_ratio_prev]
    prev_action_np = fedon.pcac_agent.prev_action.detach().cpu().numpy()
    state += prev_action_np.tolist()

    state_tensor = torch.tensor(state, dtype=torch.float32).to(agent.device)


    # ----> Action selection
    action = agent.select_action(state_tensor, preference)
    norm_scaling_l2, norm_scaling_cos = action  
    
    def inverse_linear_map(val, low=0.5, high=20.0):
        val = np.clip(val, low, high)
        norm = 2 * (val - low) / (high - low) - 1
        return float(np.clip(norm, -1.0, 1.0))
    
        # === Apply Ablation Configurations ===
    if fedon.ablation_mode == "fix_degree":
        # Fix cosine distance to baseline, opposite of benign
        norm_scaling_cos = torch.tensor(1.0).to(agent.device) 

    elif fedon.ablation_mode == "fix_l2":
        # Fix L2 scaling so scaling_l2 == 3.0
        # Reverse of: scaling_l2 = linear_map_from_tanh(action_val, 0.5, 20.0)
        fixed_action0 = inverse_linear_map(3.0, 0.5, 20.0)  # L2 scaling == 3
        norm_scaling_l2 = torch.tensor(fixed_action0).to(agent.device)  # action[0] = value

    elif fedon.ablation_mode == "both":
        # Fix both cosine distance and norm scaling
        norm_scaling_cos = torch.tensor(2.0).to(agent.device)  # Same as fix_degree

        fixed_action0 = inverse_linear_map(3.0, 0.5, 20.0)  # Same as fix_l2
        norm_scaling_l2 = torch.tensor(fixed_action0).to(agent.device)

    if fedon.ablation_mode:
        fedon.logger.info(f"[FedON][Ablation] Mode: {fedon.ablation_mode}, "
                        f"Action Override → L2: {norm_scaling_l2.item():.4f}, Cos: {norm_scaling_cos.item():.4f}")

    
    
    # ----> L2 scaling
    scaling_l2 = linear_map_from_tanh(norm_scaling_l2.item(), 0.5, 20.0)
    scaling_cos = linear_map_from_tanh(norm_scaling_cos.item(), 0.0, 2.0)
    
    # ----> Cosine upper bound
    # cos_dist_high_bound = max(0.0, min(2.0, cos_dist_mali_benign - avg_cos_distance/2)) # TODO: change to min(2.0, cos_dist_mali_benign)

    # ----> Apply attack
    fedon.scaling_l2 = scaling_l2
    fedon.scaling_cos = scaling_cos
    fedon.epsilon = scaling_l2 * avg_update_norm

    fedon.logger.info(
        f"[PCAC][Attack] RL Action → Norm(L2): {norm_scaling_l2.item():.4f}, Norm(Cos): {norm_scaling_cos.item():.4f} | "
        f"Scaled L2: {fedon.scaling_l2:.4f}, Scaled CosDist: {fedon.scaling_cos:.4f}, Epsilon: {fedon.epsilon:.4f}"
    )

    malicious_update = fedon.mali_update_craft_l2_cos(
        model=fedon.server_model,
        benign_update=benign_update,
        mali_update=mali_direction,
        scaling_cos_dist=fedon.scaling_cos,
        epsilon=fedon.epsilon,
        harm_threshold=attack_cfg.get('harm_threshold', 0.0),
        retry_limit=attack_cfg.get('retry_limit', 5),
        epsilon_growth=attack_cfg.get('epsilon_growth', 1.0),
        cos_growth=attack_cfg.get('cos_growth', 0.9),
        mode=attack_cfg.get('mode', "PROJ"),
        pgd_steps=attack_cfg.get('pgd_steps', 20),
        pgd_lr=attack_cfg.get('pgd_lr', 0.01),
        alpha=attack_cfg.get('alpha', 0.1),
        lambda_cos=attack_cfg.get('lambda_cos', 0.1)
    )

    log_cos_distance_alignment(
        mali_direction=mali_direction,               # Δ_g
        benign_direction=benign_update,              # Δ_b
        target_cos_dist=fedon.scaling_cos,
        crafted_update=malicious_update              # Δₘ, after PGD
    )

    malicious_update_dict = vector_to_state_dict(malicious_update, fedon.server_model)
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
    Maps action_val ∈ [-1, 1] to a signed cosine distance from mali_direction using log scale:
    - action = 0 → cos_dist = 0 (aligned with mali_direction)
    - action = +1 → cos_dist = +max_pos_val (toward benign)
    - action = -1 → cos_dist = -max_neg_val = -(2 - cos_dist_mali_benign) (opposite of benign)
    """
    action_val = float(torch.clamp(torch.tensor(action_val), -1.0, 1.0))

    if action_val == 0.0:
        return 0.0

    sign = np.sign(action_val)

    if sign > 0:
        # Toward benign
        scaled = np.log1p(action_val * (log_base - 1)) / np.log(log_base)
        return start_at + (max_pos_val - start_at) * scaled

    else:
        # Toward opposite of benign (negative direction)
        magnitude = np.log1p(-action_val * (log_base - 1)) / np.log(log_base)
        max_neg_val = 2.0 - cos_dist_mali_benign
        return -1.0 * (start_at + (max_neg_val - start_at) * magnitude)


def cosine_mapping_symmetric_linear(action_val, 
                                    max_pos_val, 
                                    cos_dist_mali_benign, 
                                    start_at=0.01):
    """
    Maps action ∈ [-1, 1] to signed cosine distance from mali_direction.
    Positive = toward benign, Negative = toward opposite of benign.
    """
    action_val = float(torch.clamp(torch.tensor(action_val), -1.0, 1.0))

    if action_val == 0.0:
        return 0.0

    if action_val > 0:
        # Positive action → toward benign
        return start_at + (max_pos_val - start_at) * action_val

    else:
        # Negative action → toward opposite of benign
        max_neg_val = 2.0 - cos_dist_mali_benign
        return -1.0 * (start_at + (max_neg_val - start_at) * (-action_val))



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


def log_cos_distance_alignment(mali_direction, benign_direction, target_cos_dist, crafted_update):
    """
    Logs the cosine distance alignment between Δₘ and Δ_b.

    Parameters:
    - mali_direction: torch.Tensor (Δ_g - intended malicious direction)
    - benign_direction: torch.Tensor (Δ_b - average benign update)
    - target_cos_dist: float (target cosine distance from Δₘ to Δ_b, from RL)
    - crafted_update: torch.Tensor (Δₘ - actual update after PGD)
    """

    # Normalize vectors
    g_m = F.normalize(mali_direction, dim=0)        # Δ_g
    g_b = F.normalize(benign_direction, dim=0)      # Δ_b
    delta_m = F.normalize(crafted_update, dim=0)    # Δₘ

    # 1. Cosine distance between mali_direction and benign
    cos_sim_gm_gb = F.cosine_similarity(g_m.unsqueeze(0), g_b.unsqueeze(0)).item()
    cos_dist_gm_gb = 1.0 - cos_sim_gm_gb

    # 2. Actual cosine distance between crafted update and benign
    cos_sim_real_mb = F.cosine_similarity(delta_m.unsqueeze(0), g_b.unsqueeze(0)).item()
    cos_dist_real_mb = 1.0 - cos_sim_real_mb

    # 3. Desired cosine distance is now an absolute target w.r.t. Δ_b
    print(f"[FedON] CosDist Δ_g~Δ_b: {cos_dist_gm_gb:.4f}, CosDist Δₘ~Δ_b: target = {target_cos_dist:.4f} | actual = {cos_dist_real_mb:.4f}")
