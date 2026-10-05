import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.blocks import RMSNorm, TransformerBlock


class AviyanTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                dropout=config.hidden_dropout,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x)
