from src.model.embeddings import Embedding
from src.model.util import Linear
from src.model.mlp import FeedForward
from src.model.norms import RMSNorm
from src.model.embeddings import RotaryPositionalEmbedding
from src.model.attention import MultiHeadAttention
import torch.nn as nn
from torch import Tensor
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        d_k = d_model//num_heads

        rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len
        )

        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads,rope=rope)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length)
            for _ in range(num_layers)
        ])

    def forward(self, indices: Tensor) -> Tensor:
        x = self.token_embeddings(indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
    