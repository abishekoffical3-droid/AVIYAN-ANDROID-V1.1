# AVIYAN 5B Training

## 1. Verify the code on CPU

```bash
python training/make_smoke_data.py
PYTHONPATH=. python training/train.py --config configs/aviyan_smoke.yaml
```

## 2. Prepare real data

Place licensed/owned/public-domain UTF-8 `.txt` documents under `data/raw/`, then train the final tokenizer and prepare ordered token streams. Do not shuffle individual tokens.

```bash
python tokenizer/train_tokenizer.py
PYTHONPATH=. python training/prepare_data.py
```

For a real 5B corpus, replace the compact `.pt` token stream with memory-mapped or streaming shards before very large runs. The trainer interface is isolated so this can be upgraded without changing the model.

## 3. Launch AVIYAN-5B with FSDP

Example 8-GPU single-node launch:

```bash
torchrun --standalone --nproc_per_node=8 training/train.py --config configs/aviyan_5b.yaml
```

Multi-node launch uses normal `torchrun` rendezvous options (`--nnodes`, `--node_rank`, `--master_addr`, `--master_port`).

## 4. Resume

`training.resume: true` causes the trainer to load `checkpoints/aviyan-5b/latest-sharded` when present.

## Important

A 5.231B model cannot be meaningfully pretrained from the sample files shipped with this repository. A final trained checkpoint requires a genuinely large, legally usable, high-quality corpus and CUDA GPU infrastructure. Never label an untrained/random checkpoint as a trained AVIYAN model.
