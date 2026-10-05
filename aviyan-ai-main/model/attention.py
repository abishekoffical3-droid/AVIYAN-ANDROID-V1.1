import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rotary import apply_rotary, build_rope_cache


class CausalSelfAttention(nn.Module):
    """GQA causal attention using PyTorch SDPA/Flash Attention when available."""

    def __init__(self, hidden_size, num_heads, num_key_value_heads, dropout=0.0, rope_theta=10000.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = num_key_value_heads * self.head_dim
        self.num_kv_groups = num_heads // num_key_value_heads
        self.dropout_p = dropout
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(seq_len, self.head_dim, x.device, x.dtype, self.rope_theta)
        q, k = apply_rotary(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        dropout_p = self.dropout_p if self.training else 0.0
        # SDPA selects Flash Attention / memory-efficient kernels when the
        # installed PyTorch + GPU supports them, otherwise it falls back safely.
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.out_proj(output)
