import torch.nn as nn
import torch
from .util import Linear
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    def silu(self, input):
        return input * torch.sigmoid(input)
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        gate = self.silu(self.w1(x))
        content = gate * (self.w3(x))
        return self.w2(content)
