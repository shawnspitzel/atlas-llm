import torch.nn as nn
import torch
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model # hidden dimension
        self.eps = eps # epsilon value for numeric stability
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.weight = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype)) # learned parameter initialized to ones
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       # process x tensor with shape (batch_size, seq_length, d_model), return same shape
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.mean(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(rms+self.eps)
        rms_norm = (x / rms) * self.weight
        return rms_norm.to(in_dtype)