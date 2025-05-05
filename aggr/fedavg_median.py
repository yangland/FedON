import numpy as np
import torch
import heapq
import itertools


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


def fedavg(model_dict):
    new_weights = dict()
    client_ids = list(model_dict.keys())
    
    with torch.no_grad():
        for name, param in model_dict[client_ids[0]].items():
            layer_weights = []
            for client_id in client_ids:
                layer_weights.append(model_dict[client_id][name])

            t = torch.stack(layer_weights)
            mean = t.float().mean(0)
            new_weights[name] = mean
    return new_weights



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


