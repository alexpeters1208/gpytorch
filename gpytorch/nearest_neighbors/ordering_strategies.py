#!/usr/bin/env python3

import torch

def coordinate_ordering(coordinate: int):
    return lambda data: torch.argsort(data[:, coordinate]).long()

def norm_ordering(p: float, dim: int):
    return lambda data: torch.argsort(torch.linalg.norm(data, ord=p, dim=dim)).long()

def mst_ordering():
    raise NotImplementedError

def minmax_ordering():
    raise NotImplementedError
