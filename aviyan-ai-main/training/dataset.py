import torch
from torch.utils.data import Dataset


class CausalLanguageModelingDataset(Dataset):
    """Packed causal-LM sequences without token-level shuffling.

    The input tensor is treated as one continuous token stream. Samples advance
    by `seq_len`, avoiding the enormous overlap/repetition caused by stride=1.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.token_ids = token_ids.to(dtype=torch.long, device="cpu").contiguous()
        self.seq_len = int(seq_len)
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")

    def __len__(self):
        return max(0, (self.token_ids.numel() - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.token_ids[start:start + self.seq_len + 1]
        if chunk.numel() != self.seq_len + 1:
            raise IndexError(idx)
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }
