import torch

from typing import Dict

def traverse_state_dict(d:Dict):
    for k,v in d.items():
        if isinstance(v,Dict):
            traverse_state_dict(v)
            d[k] = v
        elif isinstance(v, torch.Tensor) and len(v.shape) == 5:
            d[k] = v.permute(4, 0, 1, 2, 3)
            print_paras(k, v)
        else:
            print_paras(k, v)
    return d

def print_paras(k, v):
    if isinstance(v, torch.Tensor):
        if len(v.shape) > 0:
            print(k, v.shape)
        else:
            print(k, v)