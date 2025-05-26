import numpy as np
import torch
from copy import deepcopy

def coordinate_wise_median(model_dict):
    new_weights = dict()
    client_ids = list(model_dict.keys())

    with torch.no_grad():
        for name, param in model_dict[client_ids[0]].items():
            layer_weights = []
            for client_id in client_ids:
                layer_weights.append(model_dict[client_id][name])

            t = torch.stack(layer_weights)
            med = t.float().median(0)[0]
            new_weights[name] = med
    
    return new_weights


def fedavg(models):
    """
    Federated Averaging for both dict (client_id -> model) and list of model state_dicts.

    This version correctly handles BatchNorm buffers: running_mean and running_var are averaged;
    num_batches_tracked is skipped.

    Args:
        models (dict or list): A dictionary of client_id -> model_state_dict
                               or a list of model_state_dicts.

    Returns:
        Averaged model_state_dict
    """
    if isinstance(models, dict):
        model_list = list(models.values())
    elif isinstance(models, list):
        model_list = models
    else:
        raise TypeError("Input should be a dict or list of model state_dicts.")

    avg_model = deepcopy(model_list[0])

    # Initialize to zero
    for key in avg_model:
        if "num_batches_tracked" in key:
            continue
        avg_model[key] = torch.zeros_like(avg_model[key])

    # Sum all models
    for model in model_list:
        for key in avg_model:
            if "num_batches_tracked" in key:
                continue
            avg_model[key] += model[key]

    # Average
    num_models = len(model_list)
    for key in avg_model:
        if "num_batches_tracked" in key:
            continue
        avg_model[key] /= num_models

    # Copy over num_batches_tracked from the first model (safe default)
    for key in avg_model:
        if "num_batches_tracked" in key:
            avg_model[key] = model_list[0][key]

    return avg_model



def trimmed_mean(model_dict, m):
    new_weights = dict()
    client_ids = list(model_dict.keys())

    with torch.no_grad():
        for layer_name, param in model_dict[client_ids[0]].items():
            size = param.size()
            layer_weights = []
            for client_id in client_ids:
                layer_weight = model_dict[client_id][layer_name]
                layer_weights.append(torch.flatten(layer_weight))

            t = torch.stack(layer_weights)
            meds = torch.median(t, 0).values
            dist = -(t - meds.repeat(t.size()[0], 1))**2
            m_closet_idx = torch.topk(dist, m, dim=0).indices
            # print("m_closet_idx", m_closet_idx.size())
            y_axis = torch.arange(0, t.size()[1]).unsqueeze(0).repeat(m,1)
            # print("y_axis", y_axis.size())
            m_closet = t[m_closet_idx.flatten(start_dim=0), y_axis.flatten(start_dim=0).long(),...].view(m,-1)
            mean_m_closet = torch.mean(m_closet.float(), 0)
            new_weights[layer_name] = mean_m_closet.view(size)
    # print("new_weights.keys", new_weights.keys())
    return new_weights


def get_layer_weight(model, name_list):
    splitted = name_list.split('.')
    result = model
    for p in splitted:
        result = getattr(result, p)
    return result


