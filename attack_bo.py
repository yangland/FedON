import torch.nn.functional as F
from utils_attack import vector_to_state_dict, log2_map


def bo_attack(fedoda, mali_direction, benign_update, avg_cos_distance, avg_l2_norm, avg_update_norm, val_acc_history):
    """
    Bayesian Optimization guided attack.
    """
    cos_sim_dirs = F.cosine_similarity(
        benign_update.unsqueeze(0),
        mali_direction.unsqueeze(0)
    ).item()
    cos_dist_dirs = 1 - cos_sim_dirs

    x_next = fedoda.bayes_optimizer.suggest_next_point()
    norm_scaling_l2, norm_scaling_cos = x_next[0], x_next[1]

    fedoda.scaling_l2 = log2_map(norm_scaling_l2, 0.5, 10.0)
    fedoda.scaling_cos = log2_map(norm_scaling_cos, avg_cos_distance * 0.9, cos_dist_dirs)
    fedoda.epsilon = fedoda.scaling_l2 * avg_l2_norm

    malicious_update = fedoda.craft_malicious_update_cos_budget(
        mali_direction, benign_update, fedoda.scaling_cos
    )
    malicious_update = malicious_update / (malicious_update.norm() + 1e-10) * fedoda.epsilon

    from utils_attack import calculate_bo_reward
    reward = calculate_bo_reward(val_acc_history, len(val_acc_history) - 1, logger=fedoda.logger)

    if reward is not None:
        fedoda.bayes_optimizer.observe(x_next, reward)

    malicious_update_dict = vector_to_state_dict(malicious_update, fedoda.server_model)
    return malicious_update_dict, malicious_update
