import torch
import torch.nn.functional as F

from model.aviyan_model import AviyanModel
from model.config import load_config
from training.dataloader import create_dataloader


config = load_config("configs/aviyan_test.yaml")

model = AviyanModel(config)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
)

loader = create_dataloader(
    seq_len=32,
    batch_size=2,
)

batch = next(iter(loader))

input_ids = batch["input_ids"]
labels = batch["labels"]

model.train()

logits = model(input_ids)

loss = F.cross_entropy(
    logits.reshape(-1, config.vocab_size),
    labels.reshape(-1),
)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Training Step Test")
print("==================")
print("Input:", input_ids.shape)
print("Logits:", logits.shape)
print(f"Loss: {loss.item():.6f}")
print("Backward: OK")
print("Optimizer step: OK")
