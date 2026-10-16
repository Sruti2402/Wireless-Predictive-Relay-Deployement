import pandas as pd

keys = ["recording_id", "elapsed_sec"]

pred = pd.read_csv(
    "data/processed/script29/two_stage_oof_predictions.csv",
    usecols=keys
).drop_duplicates()

coords = pd.read_csv(
    "data/processed/script24/simulation_inputs.csv",
    usecols=keys + ["robot_pos_x", "robot_pos_y"]
)

valid_coords = coords.dropna(
    subset=keys + ["robot_pos_x", "robot_pos_y"]
).drop_duplicates(keys)

missing = pred.merge(
    valid_coords[keys],
    on=keys,
    how="left",
    indicator=True
)
missing = missing.loc[missing["_merge"] == "left_only", keys]

print("Unmatched unique timestamps:", len(missing))
print("\nrecording | time | nearest time difference (sec) | exact row has missing coordinates")

for row in missing.itertuples(index=False):
    recording = row.recording_id
    timestamp = row.elapsed_sec
    group = coords[coords["recording_id"] == recording]

    if group.empty:
        print(recording, timestamp, "NO RECORDING COORDINATES", sep=" | ")
        continue

    nearest = (group["elapsed_sec"] - timestamp).abs().min()
    exact = group[group["elapsed_sec"] == timestamp]
    coords_missing = (
        not exact.empty
        and exact[["robot_pos_x", "robot_pos_y"]].isna().any(axis=1).any()
    )

    print(
        recording,
        timestamp,
        round(nearest, 6),
        coords_missing,
        sep=" | "
    )
