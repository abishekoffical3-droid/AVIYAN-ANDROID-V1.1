"""Distributed checkpoint helpers.

Uses torch.distributed.checkpoint when distributed training is active. For a
single process it falls back to a normal torch checkpoint.
"""
from pathlib import Path
import torch


def save_sharded(path, model, optimizer, scheduler, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        import torch.distributed.checkpoint as dcp
        state = {"model": model, "optimizer": optimizer}
        if scheduler is not None:
            state["scheduler"] = scheduler
        dcp.save(state, checkpoint_id=str(path))
        if torch.distributed.get_rank() == 0:
            torch.save(metadata, path / "metadata.pt")
    else:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            **metadata,
        }, path / "single.pt")


def load_sharded(path, model, optimizer, scheduler=None):
    path = Path(path)
    if torch.distributed.is_available() and torch.distributed.is_initialized() and (path / "metadata.pt").exists():
        import torch.distributed.checkpoint as dcp
        state = {"model": model, "optimizer": optimizer}
        if scheduler is not None:
            state["scheduler"] = scheduler
        dcp.load(state, checkpoint_id=str(path))
        metadata = torch.load(path / "metadata.pt", map_location="cpu", weights_only=False)
        return metadata
    state = torch.load(path / "single.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state
