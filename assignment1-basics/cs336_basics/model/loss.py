import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import math
def cross_entropy_loss(inputs: Tensor, targets: Tensor):
    return F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))

def learning_rate_schedule(curr_iter, max_lr, min_lr, warm_iters, cos_iters):
    if curr_iter < warm_iters:
        return max_lr * (curr_iter + 1) / warm_iters
    if curr_iter >= cos_iters:
        return min_lr
    progress = (curr_iter - warm_iters) / (cos_iters - warm_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)

def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6):
    grads = [
        p.grad for p in parameters
        if p.grad is not None
    ]
    if not grads:
        return 0.0

    total_norm_sq = 0.0
    for g in grads:
        total_norm_sq += g.norm(2).item() ** 2

    total_norm = total_norm_sq ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)

    return total_norm
