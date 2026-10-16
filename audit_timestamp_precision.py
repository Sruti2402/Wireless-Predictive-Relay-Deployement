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
valid = coords.dropna(
    subset=keys + ["robot_pos_x", "robot_pos_y"]
).drop_duplicates(keys)

check = pred.merge(valid[keys], on=keys, how="left", indicator=True)
bad = check.loc[check["_merge"] == "left_only", keys].head(8)

for row in bad.itertuples(index=False):
    g = coords[coords["recording_id"] == row.recording_id]
    nearest_idx = (g["elapsed_sec"] - row.elapsed_sec).abs().idxmin()
    nearest = g.loc[nearest_idx, "elapsed_sec"]
    exact = g[g["elapsed_sec"] == row.elapsed_sec]

    print(
        f"{row.recording_id}: prediction={row.elapsed_sec:.15f}, "
        f"nearest_coordinate={nearest:.15f}, "
        f"absolute_difference={abs(nearest-row.elapsed_sec):.15g}, "
        f"exact_rows={len(exact)}"
    )
