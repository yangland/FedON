import torch
import torch.nn.functional as F
from utils_attack import vector_to_state_dict, flatten_model, log2_map_from_tanh
import numpy as np

def ppo_attack(
    fedoda,
    round_number,
    prev_acc,
    curr_acc,
    avg_cos_distance,
    avg_l2_distance,
    avg_update_norm,
    avg_full_model_cos,
    malicious_participation_rate,
    mali_direction,
    benign_update,
    action,
    state,
):
    """
    Perform PPO-guided attack.
    """
    iter = round_number + 1

    fedoda.logger.info(f"[FedOda][PPO] Server model pooled dataset validation (last 5): {[round(x, 4) for x in fedoda.val_acc_pooled_history[-5:]]}")

    norm_scaling_l2 = action[0].item()
    norm_scaling_cos = action[1].item()
    fedoda.logger.info(f"[FedOda][PPO] Selected action: normalized L2 scaling={norm_scaling_l2:.4f}, "
                       f"normalized Cosine scaling={norm_scaling_cos:.4f}")

    fedoda.scaling_l2 = log2_map_from_tanh(norm_scaling_l2, 0.5, 20.0)

    cos_dist_low_bound = 1e-4

    # Cosine distance between mali_direction and benign_update
    cos_dist_mali_benign = 1 - F.cosine_similarity(
        mali_direction.unsqueeze(0),
        benign_update.unsqueeze(0),
        dim=1
    ).item()

    # Sanity check: cosine distance must be in [0, 2]
    cos_dist_mali_benign = max(0.0, min(2.0, cos_dist_mali_benign))

    # Compute high bound with clamping
    raw_high_bound = cos_dist_mali_benign - avg_cos_distance
    cos_dist_high_bound = max(cos_dist_low_bound, min(2.0, raw_high_bound))

    # Optional: log for debugging
    # print(f"[DEBUG] cos_dist_mali_benign = {cos_dist_mali_benign:.4f}")
    # print(f"[DEBUG] avg_cos_distance = {avg_cos_distance:.4f}")
    # print(f"[DEBUG] raw_high_bound = {raw_high_bound:.4f}")
    # print(f"[DEBUG] cos_dist_high_bound (clamped) = {cos_dist_high_bound:.4f}")
    fedoda.scaling_cos = log2_map_from_tanh(norm_scaling_cos, cos_dist_low_bound, cos_dist_high_bound)

    fedoda.epsilon = fedoda.scaling_l2 * avg_update_norm
    fedoda.logger.info(f"[FedOda][PPO] Cosine Dist between g_m and benign={cos_dist_mali_benign:.4f}, "
                       f"Avg Cosine Dist={avg_cos_distance:.4f}, "
                       f"Cosine Dist between g_m and benign - avg cos={cos_dist_mali_benign - avg_cos_distance:.4f} ")
    
    fedoda.logger.info(f"[FedOda][PPO] Scaling factors: L2={fedoda.scaling_l2:.4f}, Cosine={fedoda.scaling_cos:.4f} ")

    # === Craft update ===
    malicious_update = fedoda.mali_update_craft_l2_cos(
        model=fedoda.server_model,
        benign_update=benign_update,
        target_direction=mali_direction,
        scaling_cos_dist=fedoda.scaling_cos,
        epsilon=fedoda.epsilon,
        harm_threshold=0.01,
        retry_limit=5
    )

    # === Rewards ===
    # Long-term accuracy reward
    # beta = 0.05
    # r1 = (1 - curr_acc) * (1 - np.exp(-beta * iter))
    r1 = fedoda.compute_attack_influence_r1(curr_acc, target_acc=0.2, tolerance=0.05)
    # Short term accuracy drop reward
    r2 = prev_acc - curr_acc
    # Cos distance reward
    r3 = fedoda.compute_attack_influence_r3() or 0.0
    fedoda.logger.info(f"[FedOda][PPO] Rewards: r1={r1:.4f}, r2={r2:.4f}, r3={r3:.4f}")
    
    reward_vector = torch.tensor([r1, r2, r3], dtype=torch.float32)

    
    fedoda.ppo_episode_memory.append((
        state.detach(),                # Detach and move to CPU (optional for safety)
        action.detach(),
        reward_vector.detach()
    ))



    # === PPO Training ===
    # Train every 5 rounds if enough data is collected
    if iter % 5 == 0 and len(fedoda.ppo_episode_memory) >= 5:
        fedoda.logger.info(f"[FedOda][PPO] Training PPO agent with {len(fedoda.ppo_episode_memory)} new steps.")
        weighted_reward = fedoda.ppo_agent.train(memory=fedoda.ppo_episode_memory)
        
        # Only clear the episode buffer (not the replay buffer)
        fedoda.ppo_episode_memory.clear()


    malicious_update_dict = vector_to_state_dict(malicious_update, fedoda.server_model)
    return malicious_update_dict, malicious_update, reward_vector
