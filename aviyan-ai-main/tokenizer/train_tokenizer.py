from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

DATA_DIR = Path("data/raw")
CORPUS_FILE = Path("data/processed/corpus.txt")
OUTPUT_DIR = Path("tokenizer")
TOKENIZER_PATH = OUTPUT_DIR / "tokenizer.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    files = [str(p) for p in sorted(DATA_DIR.rglob("*.txt"))]
    if not files and CORPUS_FILE.exists():
        files = [str(CORPUS_FILE)]
    if not files:
        raise RuntimeError("No tokenizer corpus files found")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=65536,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tokenizer.train(files, trainer)
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Saved to: {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
