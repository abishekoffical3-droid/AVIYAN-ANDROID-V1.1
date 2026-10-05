import torch
import torch.nn.functional as F


batch = 2
seq_len = 8
vocab_size = 1000

logits = torch.randn(
    batch,
    seq_len,
    vocab_size,
)

labels = torch.randint(
    0,
    vocab_size,
    (batch, seq_len),
)

loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    labels.reshape(-1),
)

print("Causal LM Loss Test")
print("===================")
print("Logits:", logits.shape)
print("Labels:", labels.shape)
print("Loss:", loss.item())
