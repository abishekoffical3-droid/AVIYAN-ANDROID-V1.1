from pathlib import Path
import re

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    total_files = 0
    total_chars = 0

    for path in RAW_DIR.rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        cleaned = clean_text(text)

        if not cleaned:
            continue

        relative = path.relative_to(RAW_DIR)
        output_path = OUTPUT_DIR / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned + "\n", encoding="utf-8")

        total_files += 1
        total_chars += len(cleaned)

        print(f"Cleaned: {path} -> {output_path}")

    print("\nAVIYAN Dataset Cleaning")
    print("=======================")
    print(f"Files: {total_files}")
    print(f"Characters: {total_chars:,}")


if __name__ == "__main__":
    main()
