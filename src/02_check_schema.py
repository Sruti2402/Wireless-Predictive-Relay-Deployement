
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

for path in sorted(RAW_DIR.glob("*.datalog")):
    print(f"\n{'=' * 70}")
    print(f"FILE: {path.name}")
    print("=" * 70)

    try:
        # Use whitespace as the delimiter, as suggested by the output.
        df = pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            comment="#"
        )

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))
        print("\nMissing values:")
        print(df.isna().sum().to_string())

    except Exception as exc:
        print("Error:", exc)
