import torch

from model.aviyan_model import AviyanModel
from model.config import load_config


config = load_config("configs/aviyan_test.yaml")

model = AviyanModel(config)

checkpoint = torch.load(
    "checkpoints/aviyan_test.pt",
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

input_ids = torch.randint(
    0,
    config.vocab_size,
    (1, 8),
)

with torch.no_grad():
    logits = model(input_ids)

print("Checkpoint Test")
print("================")
print("Checkpoint loaded: OK")
print("Input:", input_ids.shape)
print("Output:", logits.shape)
print("Model restored: OK")
