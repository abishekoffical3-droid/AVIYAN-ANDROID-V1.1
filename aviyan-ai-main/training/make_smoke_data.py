from pathlib import Path
import torch

out = Path("data/cleaned")
out.mkdir(parents=True, exist_ok=True)
g = torch.Generator().manual_seed(42)
torch.save(torch.randint(0, 512, (8192,), generator=g, dtype=torch.long), out / "smoke_train_tokens.pt")
torch.save(torch.randint(0, 512, (2048,), generator=g, dtype=torch.long), out / "smoke_val_tokens.pt")
print("smoke data ready")
