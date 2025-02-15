#!/usr/bin/env python3

import torch

def euclidean_distance(x1, x2):
    return torch.cdist(x1, x2, p=2).float()

def euclidean_distance_2():
    return lambda x1, x2: torch.cdist(x1, x2, p=2).float()

def manhattan_distance():
    return lambda x1, x2: torch.cdist(x1, x2, p=1).float()

def mst_distance():
    raise NotImplementedError
