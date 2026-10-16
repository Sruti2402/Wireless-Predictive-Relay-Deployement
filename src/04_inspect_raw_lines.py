
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

for filename in ["Dataset1.datalog", "Dataset6.datalog"]:
    path = RAW_DIR / filename

    print(f"\n{'=' * 70}")
    print(f"FILE: {filename}")

    with path.open("r", encoding="utf-8", errors="replace") as file:
        shown = 0

        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            fields = stripped.split()

            print(f"\nLine {line_number}")
            print("Raw:", repr(stripped[:500]))
            print("Field count:", len(fields))
            print("First 25 fields:", fields[:25])

            shown += 1
            if shown == 5:
                break
