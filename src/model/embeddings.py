import torch.nn as nn
import torch
from jaxtyping import Int
from torch import Tensor
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings # vocab_size
        self.embedding_dim = embedding_dim # dimension of embedding vector
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=self.dtype, device=self.device))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)
    def forward(self, token_ids: Int[Tensor, " ..."]) -> torch.Tensor:
       # Given a list of token ids, we want to find the vector for a 
       # corresponding token by performing a lookup into a tensor of 
       # token ids with shape (num_embeddings, embedding_dim)
       return self.weight[token_ids]
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        pair_idx = torch.arange(d_k//2, device=device)
        freq = theta ** (2 * pair_idx / d_k)
        positions = torch.arange(max_seq_len, device=device)
        angles = positions[:, None] / freq[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        *batch_dims, seq_len, d_k = x.shape
        x = x.reshape(*batch_dims, seq_len, d_k // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        rotated = torch.stack((rotated_x1, rotated_x2), dim=-1)
        rotated = rotated.reshape(*batch_dims, seq_len, d_k)
        return rotated