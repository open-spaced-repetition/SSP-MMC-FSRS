import numpy as np
import torch


def power_forgetting_curve(t, s, decay):
    factor = 0.9 ** (1.0 / decay) - 1.0
    return (1 + factor * t / s) ** decay


def power_forgetting_curve_torch(t, s, decay):
    factor = 0.9 ** (1.0 / decay) - 1.0
    return (1 + factor * t / s) ** decay


def next_interval(s, r, decay):
    factor = 0.9 ** (1.0 / decay) - 1.0
    ivl = s / factor * (r ** (1.0 / decay) - 1.0)
    return np.maximum(1, np.floor(ivl))


def next_interval_torch(s, r, decay):
    factor = 0.9 ** (1.0 / decay) - 1.0
    ivl = s / factor * (r ** (1.0 / decay) - 1.0)
    return torch.maximum(torch.ones_like(ivl), torch.floor(ivl))
