"""Prepare ordered train/validation token streams for causal-LM training.

Important: documents are shuffled, not individual tokens. Each document keeps
its original token order so the model can learn language and code structure.
"""
from pathlib import Path
import random
import torch
from tokenizer.load_tokenizer import load_tokenizer

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = OUTPUT_DIR / "train_tokens.pt"
VAL_FILE = OUTPUT_DIR / "val_tokens.pt"


def clean_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def encode_documents(tokenizer, paths):
    eos_id = tokenizer.token_to_id("<eos>")
    streams = []
    for path in paths:
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            continue
        ids = tokenizer.encode(text).ids
        if eos_id is not None:
            ids.append(eos_id)
        streams.extend(ids)
        print(f"Processed: {path} | tokens={len(ids):,}")
    return torch.tensor(streams, dtype=torch.long)


def main():
    tokenizer = load_tokenizer()
    files = sorted(DATA_DIR.rglob("*.txt"))
    if not files:
        raise RuntimeError("No .txt files found in data/raw")

    rng = random.Random(42)
    rng.shuffle(files)

    if len(files) == 1:
        train_files, val_files = files, files
    else:
        val_count = max(1, round(len(files) * 0.02))
        val_files = files[:val_count]
        train_files = files[val_count:]

    train = encode_documents(tokenizer, train_files)
    val = encode_documents(tokenizer, val_files)

    if train.numel() < 100:
        raise RuntimeError("Dataset is too small for even a smoke training run")

    torch.save(train, TRAIN_FILE)
    torch.save(val, VAL_FILE)
    print(f"Train documents: {len(train_files):,} | tokens: {train.numel():,}")
    print(f"Validation documents: {len(val_files):,} | tokens: {val.numel():,}")
    print(f"Saved: {TRAIN_FILE}")
    print(f"Saved: {VAL_FILE}")


if __name__ == "__main__":
    main()
