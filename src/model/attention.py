from .embeddings import RotaryPositionalEmbedding
from .util import Linear
from .util import softmax
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

def SCPAttention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        assert q.shape[-1] == k.shape[-1]
        assert k.shape[-2] == v.shape[-2]

        d_k = q.shape[-1]
        attn_obj = torch.matmul(q, k.transpose(-2, -1))
        attn_obj = attn_obj / math.sqrt(d_k)
        if mask is not None:
            attn_obj = attn_obj.masked_fill(mask==0, -float("inf"))
        attn_obj = softmax(attn_obj, dim=-1)
        result = torch.matmul(attn_obj, v)
        return result

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model//num_heads
        d_k = self.d_head
        self.q_proj = Linear(d_model, num_heads*d_k)
        self.k_proj = Linear(d_model, num_heads*d_k)
        self.v_proj = Linear(d_model, num_heads*d_k)
        self.output_proj = Linear(num_heads*d_k, d_model)
        self.rope = rope

    
    def forward(self, x: Tensor, token_positions: Tensor = None):
        B, T, _ = x.shape
        h = self.num_heads
        d = self.d_head

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, h, d).transpose(1, 2)
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        token_positions = torch.arange(T, device=x.device)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(T, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        q = q.reshape(B * h, T, d)
        k = k.reshape(B * h, T, d)
        v = v.reshape(B * h, T, d)

        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )
        causal_mask = causal_mask.unsqueeze(0)
        
        # TODO: Benchmark naive attention with FlashAttention
        
        # Naive attention
        # out = SCPAttention(q, k, v, mask=causal_mask) 
            
        # Flash Attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)

        out = out.view(B, h, T, d)
        out = out.transpose(1, 2).contiguous()  
        out = out.view(B, T, h * d)

        out = self.output_proj(out)

        return out


