import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.transformer import AviyanTransformer


class AviyanModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.transformer = AviyanTransformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.embedding.weight

    def enable_gradient_checkpointing(self):
        self.transformer.gradient_checkpointing = True

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return self.lm_head(x)
