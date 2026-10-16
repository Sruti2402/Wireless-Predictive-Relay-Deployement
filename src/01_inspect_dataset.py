
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SUPPORTED_EXTENSIONS = {
    ".csv", ".tsv", ".txt", ".data", ".datalog"
}


def main():
    if not RAW_DIR.exists():
        print(f"Dataset folder not found: {RAW_DIR}")
        print("Create data/raw and copy your dataset into it.")
        return

    files = [
        path for path in RAW_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print("No tabular files found.")
        print("Inspect the original folder for ROS bags,")
        print("archives, or other file formats.")
        return

    for path in files:
        print("\n" + "=" * 70)
        print(f"FILE: {path.relative_to(RAW_DIR)}")
        print("=" * 70)

        try:
            df = None

            # Try common delimiters.
            for separator in [None, ",", "\t", r"\s+"]:
                try:
                    candidate = pd.read_csv(
                        path,
                        sep=separator,
                        engine="python",
                        comment="#",
                        nrows=5
                    )

                    if candidate.shape[1] > 1:
                        df = candidate
                        break
                except (ValueError, pd.errors.ParserError):
                    continue

            if df is None:
                print("Could not identify the table format.")
                continue

            print("\nColumns:")
            print(df.columns.tolist())

            print("\nFirst five rows:")
            print(df.to_string(index=False))

            print("\nShape of sample:", df.shape)

        except Exception as exc:
            print("Inspection error:", exc)


if __name__ == "__main__":
    main()
