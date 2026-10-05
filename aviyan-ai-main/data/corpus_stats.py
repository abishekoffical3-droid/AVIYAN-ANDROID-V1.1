from pathlib import Path

corpus = Path("data/processed/corpus.txt")
text = corpus.read_text(encoding="utf-8")

print("AVIYAN Corpus Statistics")
print("========================")
print(f"Characters: {len(text):,}")
print(f"Bytes:      {corpus.stat().st_size:,}")
print(f"Words:      {len(text.split()):,}")
print(f"Lines:      {len(text.splitlines()):,}")
