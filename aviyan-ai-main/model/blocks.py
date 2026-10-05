import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_key_value_heads, intermediate_size, dropout=0.0, rope_theta=10000.0, rms_norm_eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, rms_norm_eps)
        self.attention = CausalSelfAttention(hidden_size, num_heads, num_key_value_heads, dropout, rope_theta)
        self.norm2 = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
