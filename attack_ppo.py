import torch
import torch.nn.functional as F
from utils_attack import vector_to_state_dict, flatten_model, log2_map_from_tanh


def ppo_attack(
    fedoda,
    round_number,
    prev_acc,
    avg_cos_distance,
    avg_l2_distance,
    avg_update_norm,
    avg_full_model_cos,
    mali_direction,
    benign_update
):
    """
    Perform PPO-guided attack.
    """

    fedoda.logger.info(f"[FedOda][PPO] Starting PPO attack — Mode: {fedoda.ppo_mode}")
    t = round_number + 1

    # === State Vector ===
    state = torch.tensor([
        prev_acc,
        avg_cos_distance,
        avg_l2_distance,
        round_number / 100.0  # normalized round
    ], dtype=torch.float32)

    # === Action ===
    with torch.no_grad():
        action = fedoda.ppo_agent.select_action(state)

    norm_scaling_l2 = action[0].item()
    norm_scaling_cos = action[1].item()

    fedoda.scaling_l2 = log2_map_from_tanh(norm_scaling_l2, 0.5, 20.0)
    cos_dist_low_bound = avg_cos_distance * 0.9
    fedoda.scaling_cos = log2_map_from_tanh(norm_scaling_cos, cos_dist_low_bound, avg_cos_distance)

    fedoda.epsilon = fedoda.scaling_l2 * avg_update_norm

    # === Craft update ===
    malicious_update = fedoda.mali_update_craft_l2_cos(
        model=fedoda.server_model,
        base_update=benign_update,
        target_direction=mali_direction,
        scaling_cos=fedoda.scaling_cos,
        epsilon=fedoda.epsilon,
        max_iters=10,
        alpha=1e-2
    )
    malicious_update = malicious_update / (malicious_update.norm() + 1e-10) * fedoda.epsilon

    # === Validate ===
    attacked_model = fedoda.apply_update_to_model(fedoda.server_model, malicious_update)
    val_accuracy = fedoda.validate(attacked_model, fedoda.pooled_balanced_loader)
    fedoda.val_acc_pooled_history.append(val_accuracy)

    # === Rewards ===
    beta = 0.05
    r1 = (1 - val_accuracy) * (1 - torch.exp(-beta * t))
    r2 = prev_acc - val_accuracy
    r3 = fedoda.compute_attack_influence_r3() or 0.0

    if fedoda.ppo_mode == "MO":
        reward = torch.tensor([r1, r2, r3], dtype=torch.float32)
    else:
        reward = 0.6 * r1 + 0.3 * r2 + 0.1 * r3

    fedoda.ppo_episode_memory.append((state, action, reward))

    malicious_update_dict = vector_to_state_dict(malicious_update, fedoda.server_model)
    return malicious_update_dict, malicious_update, reward
