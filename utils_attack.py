import torch


def flatten_model(state_dict):
    return torch.cat([
        p.view(-1).float()
        for p in state_dict.values()
        if isinstance(p, torch.Tensor)
    ])


def vector_to_state_dict(vector, model):
    ref_sd = model.state_dict()
    new_sd = {}
    idx = 0
    for k, v in ref_sd.items():
        if isinstance(v, torch.Tensor):
            numel = v.numel()
            new_sd[k] = vector[idx:idx + numel].view_as(v).clone().detach()
            idx += numel
        else:
            new_sd[k] = v
    return new_sd


def linear_map_from_tanh(norm_value, min_val, max_val):
    """Linearly map [-1, 1] to [min_val, max_val]."""
    return min_val + (norm_value + 1.0) / 2.0 * (max_val - min_val)
