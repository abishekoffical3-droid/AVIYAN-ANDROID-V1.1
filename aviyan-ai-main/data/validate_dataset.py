from pathlib import Path

ROOT = Path("data/raw")

print("AVIYAN Dataset Validation")
print("=========================")

total_files = 0
total_chars = 0

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    files = list(folder.rglob("*.txt"))
    chars = 0

    for file in files:
        text = file.read_text(encoding="utf-8", errors="replace")
        chars += len(text)

    total_files += len(files)
    total_chars += chars

    print(f"{folder.name:15} files={len(files):4} chars={chars:,}")

print("-------------------------")
print(f"Total files:    {total_files}")
print(f"Total chars:    {total_chars:,}")
