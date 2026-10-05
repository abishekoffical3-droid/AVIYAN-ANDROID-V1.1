from pathlib import Path

INPUT_DIR = Path("data/processed")
OUTPUT_FILE = Path("data/processed/corpus.txt")

files = sorted(INPUT_DIR.rglob("*.txt"))

with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    for path in files:
        if path.name == "corpus.txt":
            continue

        text = path.read_text(encoding="utf-8").strip()

        if text:
            out.write(text)
            out.write("\n\n")

print("AVIYAN Corpus Merge")
print("===================")
print(f"Files merged: {len(files)}")
print(f"Output: {OUTPUT_FILE}")
print(f"Characters: {len(OUTPUT_FILE.read_text(encoding='utf-8')):,}")
