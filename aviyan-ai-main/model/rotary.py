import torch


def build_rope_cache(seq_len, head_dim, device, dtype, theta=10000.0):
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype)[None, None, :, :], emb.sin().to(dtype=dtype)[None, None, :, :]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)
