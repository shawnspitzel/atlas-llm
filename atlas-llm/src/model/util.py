import torch.nn as nn
import torch
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=self.dtype, device=self.device))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


