"""AVIYAN 5B training engine.

Includes BF16 mixed precision, gradient accumulation/checkpointing, SDPA/Flash
Attention, DDP/FSDP, distributed sampling, sharded checkpoints, resume,
validation, clipping, cosine decay, and efficient loading.
"""
import argparse
import contextlib
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from model.aviyan_model import AviyanModel
from model.blocks import TransformerBlock
from model.config import load_config, load_raw_config
from training.dataloader import create_dataloader
from training.sharded_checkpoint import save_sharded, load_sharded


def setup_distributed():
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return True, rank, world, local_rank, device


def seed_everything(seed, rank=0):
    seed += rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def wrap_fsdp(model, device):
    from functools import partial
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=device,
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
    )


def reduce_loss(total, count, device, distributed):
    stats = torch.tensor([total, count], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1].clamp_min(1)).item()


def evaluate(model, loader, device, distributed, max_batches=20):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
            total += float(loss.item())
            count += 1
            if count >= max_batches:
                break
    avg = reduce_loss(total, count, device, distributed)
    model.train()
    return avg


def train(config_path):
    distributed, rank, world, local_rank, device = setup_distributed()
    raw = load_raw_config(config_path)
    cfg = load_config(config_path)
    tcfg = raw["training"]
    seed_everything(tcfg["seed"], rank)

    strategy = tcfg.get("strategy", "ddp").lower()
    if cfg.name.lower().endswith("5b") and device.type != "cuda":
        raise RuntimeError(
            "AVIYAN-5B full training requires CUDA GPUs. Use configs/aviyan_smoke.yaml "
            "for a CPU correctness test."
        )

    # Keep the unsharded 5B model off GPU before FSDP wrapping. This avoids
    # materializing ~20 GB of FP32 parameters on every GPU prior to sharding.
    model = AviyanModel(cfg)
    if tcfg.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()

    if distributed and strategy == "fsdp":
        model = wrap_fsdp(model, device)
    else:
        model = model.to(device)
        if distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        betas=(0.9, 0.95),
        fused=(device.type == "cuda" and hasattr(torch.optim.AdamW, "__call__")),
    )
    scheduler = build_scheduler(
        optimizer,
        tcfg["warmup_steps"],
        tcfg["max_steps"],
        tcfg["min_learning_rate"] / tcfg["learning_rate"],
    )

    train_loader, train_sampler = create_dataloader(
        tcfg["train_data"], tcfg["sequence_length"], tcfg["micro_batch_size"], True,
        tcfg["num_workers"], tcfg["pin_memory"], tcfg["persistent_workers"],
        tcfg["prefetch_factor"], distributed, rank, world,
    )
    val_loader = None
    if Path(tcfg["validation_data"]).exists():
        val_loader, _ = create_dataloader(
            tcfg["validation_data"], tcfg["sequence_length"], tcfg["micro_batch_size"], False,
            tcfg["num_workers"], tcfg["pin_memory"], tcfg["persistent_workers"],
            tcfg["prefetch_factor"], distributed, rank, world,
        )

    out = Path(tcfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest-sharded"
    step = 0
    if tcfg.get("resume") and latest.exists():
        meta = load_sharded(latest, model, optimizer, scheduler)
        step = int(meta.get("step", 0))
        if rank == 0:
            print(f"Resumed from step {step}", flush=True)
    if distributed:
        dist.barrier()

    accumulation = int(tcfg["gradient_accumulation_steps"])
    micro_step = 0
    epoch = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while step < tcfg["max_steps"]:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            sync = ((micro_step + 1) % accumulation == 0)
            no_sync = getattr(model, "no_sync", None)
            sync_ctx = no_sync() if no_sync is not None and not sync else contextlib.nullcontext()

            with sync_ctx:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                    ) / accumulation
                loss.backward()
            micro_step += 1

            if sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if rank == 0 and step % tcfg["log_interval"] == 0:
                    print(
                        f"step={step} loss={loss.item()*accumulation:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )

                if step % tcfg["validation_interval"] == 0 and val_loader is not None:
                    val_loss = evaluate(model, val_loader, device, distributed)
                    if rank == 0:
                        print(
                            f"validation step={step} loss={val_loss:.4f} "
                            f"ppl={math.exp(min(val_loss, 20)):.2f}",
                            flush=True,
                        )

                if step % tcfg["save_interval"] == 0:
                    save_sharded(
                        latest,
                        model,
                        optimizer,
                        scheduler,
                        {"step": step, "epoch": epoch, "world_size": world},
                    )
                    if distributed:
                        dist.barrier()

                if step >= tcfg["max_steps"]:
                    break
        epoch += 1

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(f"AVIYAN training finished at step {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aviyan_5b.yaml")
    train(parser.parse_args().config)
