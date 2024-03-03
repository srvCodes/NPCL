import torch

def compute_shannon_entropy(logits, return_mean=True):
    uncertainty = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(-1)
    if return_mean:
        uncertainty = uncertainty.mean(0)
    return uncertainty

