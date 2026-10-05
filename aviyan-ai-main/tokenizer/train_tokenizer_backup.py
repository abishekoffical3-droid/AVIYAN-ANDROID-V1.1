from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("tokenizer")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_PATH = OUTPUT_DIR / "tokenizer.json"


def get_training_files():
    return [str(path) for path in DATA_DIR.rglob("*.txt")]


def main():
    files = get_training_files()

    if not files:
        raise RuntimeError("No .txt training files found in data/raw")

    print("AVIYAN AI Tokenizer Training")
    print("============================")
    print(f"Training files: {len(files)}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=1000,
        min_frequency=1,
        special_tokens=[
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
        ],
    )

    tokenizer.train(files, trainer)

    tokenizer.save(str(TOKENIZER_PATH))

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Saved to: {TOKENIZER_PATH}")

    test_text = "नमस्ते, म AVIYAN AI हुँ। Hello! नमस्ते!"
    encoded = tokenizer.encode(test_text)

    print("\nTest:")
    print(test_text)
    print("Tokens:", encoded.tokens)
    print("IDs:", encoded.ids)


if __name__ == "__main__":
    main()