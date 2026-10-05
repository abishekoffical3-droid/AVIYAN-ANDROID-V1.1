from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []

for path in ROOT.rglob("*.py"):
    if any(part in {"__pycache__", ".venv", "venv"} for part in path.parts):
        continue
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as exc:
        errors.append(f"{path}: {exc}")

required = [
    ROOT / "api" / "main.py",
    ROOT / "aviyan_core" / "runtime.py",
    ROOT / "mobile_app" / "main.py",
    ROOT / "mobile_app" / "buildozer.spec",
    ROOT / ".github" / "workflows" / "android-apk.yml",
]
for path in required:
    if not path.exists():
        errors.append(f"Missing required file: {path}")

if errors:
    print("AVIYAN CHECK FAILED")
    print("\n".join(errors))
    raise SystemExit(1)

print("AVIYAN PROJECT CHECK: PASS")
print("Python syntax: PASS")
print("Android Buildozer files: PASS")
print("GitHub Actions APK workflow: PASS")
