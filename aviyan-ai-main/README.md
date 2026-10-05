# AVIYAN

**AVIYAN — Created, Founded & Developed by Abishek Bhusal**  
**AB DEV STUDIO · Nepal 🇳🇵**

AVIYAN is being built as a modular AI ecosystem: a ~5B decoder-only language model, training stack, RAG/knowledge layer, tools, agents, developer API and eventually AVIYAN Deploy hosting.

## Current foundation

- AVIYAN 5B architecture configuration
- GQA attention
- RoPE
- RMSNorm
- SwiGLU
- tied embeddings
- PyTorch SDPA with Flash-Attention-compatible kernels
- BF16 autocast
- gradient accumulation
- gradient checkpointing
- distributed/DDP-ready training entry point
- checkpoint + resume support
- validation loop
- efficient DataLoader settings
- 65,536-token tokenizer target

## Stack

- Python + PyTorch: AI/model/training
- CUDA/C++: performance-critical GPU kernels when needed
- TypeScript + React/Next.js: UI/platform
- Go: deployment infrastructure
- Rust: security-critical sandbox components
- PostgreSQL + Redis: platform data/cache
- Docker/OCI: isolated runtime

## Run

```bash
python -m model.parameter_count
python -m tokenizer.train_tokenizer
python -m training.prepare_data
python -m training.train --config configs/aviyan_5b.yaml
```

For multi-GPU DDP:

```bash
torchrun --nproc_per_node=4 training/train.py --config configs/aviyan_5b.yaml
```

> The 5B configuration is a training target. Real pretraining requires a sufficiently large, high-quality corpus and appropriate GPU infrastructure.
