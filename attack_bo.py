import torch.nn.functional as F
from utils_attack import vector_to_state_dict, log2_map, linear_map_from_tanh
import torch
import numpy as np


def bo_attack(
    fedoda,
    mali_direction,
    benign_update,
    avg_cos_distance,
    avg_l2_distance,
    avg_update_norm,
    prev_acc,
    curr_acc
):
    """
    Bayesian Optimization guided attack with reward based on validation accuracy drop (r2).
    """

    # === Step 1: Cosine distance between benign and mali direction
    cos_sim_dirs = F.cosine_similarity(
        benign_update.unsqueeze(0),
        mali_direction.unsqueeze(0)
    ).item()
    cos_dist_dirs = 1 - cos_sim_dirs

    # === Step 2: Sample next candidate action from BO
    update_kappa(fedoda, curr_acc)
    x_next = fedoda.bayes_optimizer.suggest_next_point()
    print(f"[FedOda][BO] Suggested action: {x_next}")
    norm_scaling_l2, norm_scaling_cos = x_next[0], x_next[1]

    # === Step 3: Map normalized values to real scale
    scaling_l2 = linear_map_from_tanh(norm_scaling_l2, 0.5, 20.0)
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
    scaling_cos = linear_map_from_tanh(norm_scaling_cos, cos_dist_low_bound, cos_dist_high_bound)

    fedoda.scaling_l2 = scaling_l2
    fedoda.scaling_cos = scaling_cos
    fedoda.epsilon = scaling_l2 * avg_update_norm

    # === Step 4: Craft malicious update
    malicious_update = fedoda.mali_update_craft_l2_cos(
        model=fedoda.server_model,
        benign_update=benign_update,
        target_direction=mali_direction,
        scaling_cos_dist=fedoda.scaling_cos,
        epsilon=fedoda.epsilon,
        harm_threshold=0.01,
        retry_limit=15
    )

    # === Step 5: Compute reward based on accuracy drop
    r1 = 0
    # r2 = fedoda.compute_attack_influence_r2(
    #     prev_acc,
    #     curr_acc,
    #     target_acc=0.1,
    #     tolerance=0.03,
    #     max_penalty=20.0,
    #     scale=50.0
    # )
    print(f"[BO DEBUG] prev_acc: {prev_acc}, curr_acc: {curr_acc}")
    r2 = fedoda.compute_attack_influence_r2(
        prev_acc,
        curr_acc,
        scale=50.0
    )
    print(f"[BO DEBUG] r2 reward: {r2}, expected: {(prev_acc - curr_acc) * 50 / 100}")

    r3= 0

    reward_scalar = r2
    reward_vector = torch.tensor([r1, r2, r3], dtype=torch.float32)

    # Optional logs
    fedoda.logger.info(
        f"[FedOda][BO] Sampled action: norm_l2={norm_scaling_l2:.4f}, norm_cos={norm_scaling_cos:.4f} | " +
        f"scaling_l2={scaling_l2:.4f}, scaling_cos={scaling_cos:.4f} | reward={reward_scalar:.4f}"
    )

    # === Step 6: Observe reward into BO model
    if reward_scalar is not None:
        fedoda.bayes_optimizer.observe(x_next, reward_scalar)

    # === Step 7: Return malicious update
    malicious_update_dict = vector_to_state_dict(malicious_update, fedoda.server_model)
    return malicious_update_dict, malicious_update, reward_vector, torch.tensor(x_next)


def update_kappa(fedoda, curr_acc):
    acc_percent = curr_acc * 100.0
    kappa = np.interp(acc_percent, [10, 70], [0.5, 5.0])
    kappa = float(np.clip(kappa, 0.5, 6.0))
    fedoda.bayes_optimizer.optimizer.acq_func_kwargs["kappa"] = kappa
    fedoda.logger.info(f"[FedOda][BO] Updated kappa={kappa:.2f} based on curr_acc={curr_acc:.4f}")
