from utils_attack import vector_to_state_dict


def bi_attack(fedoda, mali_direction, benign_update, avg_cos_distance, avg_l2_norm):
    """
    Baseline BI attack with cosine constraint.
    """
    fedoda.epsilon = fedoda.scaling_factor * avg_l2_norm

    malicious_update = fedoda.craft_malicious_update_cos_budget(
        mali_direction, benign_update, avg_cos_distance
    )
    malicious_update = malicious_update / (malicious_update.norm() + 1e-10) * fedoda.epsilon

    malicious_update_dict = vector_to_state_dict(malicious_update, fedoda.server_model)
    return malicious_update_dict, malicious_update
