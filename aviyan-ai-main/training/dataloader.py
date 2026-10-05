import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from training.dataset import CausalLanguageModelingDataset


def create_dataloader(
    token_file,
    seq_len,
    batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    distributed=False,
    rank=0,
    world_size=1,
):
    tokens = torch.load(token_file, map_location="cpu", weights_only=True)
    dataset = CausalLanguageModelingDataset(tokens, seq_len)
    if len(dataset) == 0:
        raise RuntimeError(
            f"Not enough tokens in {token_file} for sequence_length={seq_len}. "
            f"Need at least {seq_len + 1:,} tokens."
        )

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=42,
            drop_last=True,
        )

    kwargs = dict(
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    if num_workers > 0:
        kwargs.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(dataset, **kwargs), sampler
