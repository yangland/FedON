import copy
import torch
from torch import nn

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    # print("w_avg", w_avg)
    # print("len(w)", len(w))
    # print("w[0]", w[0])
    for k in w_avg.keys():
        # print("w_avg[k] before", w_avg[k])
        # print("k", k)
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
        # print("w_avg[k] after", w_avg[k])
    return w_avg