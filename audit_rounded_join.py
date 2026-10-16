import pandas as pd

keys = ["recording_id", "elapsed_sec"]
pred = pd.read_csv(
    "data/processed/script29/two_stage_oof_predictions.csv",
    usecols=keys
).drop_duplicates()

coords = pd.read_csv(
    "data/processed/script24/simulation_inputs.csv",
    usecols=keys + ["robot_pos_x", "robot_pos_y"]
).dropna(subset=keys + ["robot_pos_x", "robot_pos_y"])

coords = coords.drop_duplicates(keys).copy()
pred = pred.copy()

pred["join_time"] = pred["elapsed_sec"].round(9)
coords["join_time"] = coords["elapsed_sec"].round(9)

join_keys = ["recording_id", "join_time"]
collisions = coords[coords.duplicated(join_keys, keep=False)]

print("Unique prediction timestamps:", len(pred))
print("Unique coordinate timestamps:", len(coords))
print("Coordinate rows in rounded-key collisions:", len(collisions))
print("Distinct rounded coordinate keys:", coords[join_keys].drop_duplicates().shape[0])

if not collisions.empty:
    print("\nExamples of rounded-key collisions:")
    print(
        collisions.sort_values(join_keys)[
            ["recording_id", "elapsed_sec", "join_time"]
        ].head(20).to_string(index=False)
    )
else:
    matched = pred.merge(
        coords[join_keys + ["elapsed_sec"]].rename(
            columns={"elapsed_sec": "coordinate_time"}
        ),
        on=join_keys,
        how="left",
        validate="many_to_one",
        indicator=True
    )
    ok = matched["_merge"].eq("both")
    print("\nMatched prediction timestamps:", int(ok.sum()))
    print("Unmatched prediction timestamps:", int((~ok).sum()))
    if ok.any():
        diff = (
            matched.loc[ok, "elapsed_sec"]
            - matched.loc[ok, "coordinate_time"]
        ).abs()
        print("Maximum matched timestamp difference (sec):", f"{diff.max():.15g}")
