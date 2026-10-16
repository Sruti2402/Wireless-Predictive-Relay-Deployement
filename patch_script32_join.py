from pathlib import Path

path = Path("src/32_simulate_relay_deployment.py")
source = path.read_text(encoding="utf-8")

old = '''    # One coordinate record per recording/time.
    # This avoids multiplying prediction rows during the join.
    positions = positions[position_cols].copy()
    positions = positions.dropna(
        subset=["recording_id", "elapsed_sec",
                "robot_pos_x", "robot_pos_y"]
    )
    positions = positions.drop_duplicates(
        ["recording_id", "elapsed_sec"], keep="first"
    )

    pred = pred.merge(
        positions,
        on=["recording_id", "elapsed_sec"],
        how="left",
        validate="many_to_one",
    )
'''

new = '''    # One coordinate record per recording/time.
    # Preserve original timestamps; round only a temporary join key.
    positions = positions[position_cols].copy()
    positions = positions.dropna(
        subset=["recording_id", "elapsed_sec",
                "robot_pos_x", "robot_pos_y"]
    )
    positions = positions.drop_duplicates(
        ["recording_id", "elapsed_sec"], keep="first"
    )

    pred["_join_time"] = pred["elapsed_sec"].round(9)
    positions["_join_time"] = positions["elapsed_sec"].round(9)
    join_keys = ["recording_id", "_join_time"]

    if positions.duplicated(join_keys).any():
        raise ValueError(
            "Rounded timestamp collision in coordinates; "
            "cannot safely join prediction and position data."
        )

    # Keep prediction elapsed_sec as the authoritative simulation time.
    positions = positions.drop(columns=["elapsed_sec"])

    pred = pred.merge(
        positions,
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    pred = pred.drop(columns=["_join_time"])
'''

if source.count(old) != 1:
    raise SystemExit(
        "STOP: Expected join block was not found exactly once. "
        "No changes were made."
    )

backup = path.with_suffix(path.suffix + ".bak")
if not backup.exists():
    backup.write_text(source, encoding="utf-8")

path.write_text(source.replace(old, new), encoding="utf-8")
print("Patched Script 32 successfully.")
print(f"Backup: {backup}")
