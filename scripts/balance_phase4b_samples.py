# scripts/balance_phase4b_samples.py

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SEED = 42

INPUT_FILE = Path("data/MOIL_SIH_Phase4B_Enhanced_Features.csv")
OUTPUT_FILE = Path("data/MOIL_SIH_Phase4B_Balanced_Features.csv")

BLOCK_SIZE = 0.05

MAX_POSITIVE_PER_BLOCK = 8

NEGATIVE_RATIO = 2

MIN_NEGATIVE_DISTANCE_KM = 3.0
PREFERRED_NEGATIVE_DISTANCE_KM = 30.0


# ============================================================
# HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance in kilometres.
    """

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(a))


def nearest_positive_distance_km(
    negative_df,
    positive_df
):
    """
    Compute nearest known-positive distance for every negative point.
    """

    neg_lat = negative_df["latitude"].to_numpy(dtype=float)
    neg_lon = negative_df["longitude"].to_numpy(dtype=float)

    pos_lat = positive_df["latitude"].to_numpy(dtype=float)
    pos_lon = positive_df["longitude"].to_numpy(dtype=float)

    result = np.empty(len(negative_df), dtype=float)

    # Small dataset, so vectorized chunks are more than sufficient.
    for i in range(len(negative_df)):

        distances = haversine_km(
            neg_lat[i],
            neg_lon[i],
            pos_lat,
            pos_lon
        )

        result[i] = np.min(distances)

    return result


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PHASE 4B.4 - SPATIAL SAMPLE BALANCING")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Input records: {len(df)}")

required = {
    "label",
    "latitude",
    "longitude"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================
# BASIC CLEANUP
# ============================================================

df = df.copy()

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)

df = df.dropna(
    subset=["label", "latitude", "longitude"]
)

df["label"] = df["label"].astype(int)


# ============================================================
# REMOVE DUPLICATE COORDINATES
# ============================================================

before_duplicates = len(df)

duplicate_mask = df.duplicated(
    subset=["latitude", "longitude"],
    keep=False
)

duplicate_count = int(duplicate_mask.sum())

print()
print(f"Duplicate coordinate rows before cleanup: {duplicate_count}")

if duplicate_count > 0:

    duplicate_groups = (
        df.loc[
            duplicate_mask,
            ["latitude", "longitude", "label"]
        ]
        .sort_values(["latitude", "longitude"])
    )

    print("Duplicate coordinate groups:")
    print(duplicate_groups.to_string(index=False))

    # Keep the first occurrence deterministically.
    df = df.drop_duplicates(
        subset=["latitude", "longitude"],
        keep="first"
    ).copy()

after_duplicates = len(df)

print(
    f"Removed duplicate-coordinate rows: "
    f"{before_duplicates - after_duplicates}"
)


# ============================================================
# SPLIT POSITIVES / NEGATIVES
# ============================================================

positive = df[df["label"] == 1].copy()
negative = df[df["label"] == 0].copy()

print()
print(f"Positives: {len(positive)}")
print(f"Negatives: {len(negative)}")


# ============================================================
# SPATIAL BLOCKS FOR POSITIVE THINNING
# ============================================================

positive["block_lat"] = np.floor(
    positive["latitude"] / BLOCK_SIZE
).astype(int)

positive["block_lon"] = np.floor(
    positive["longitude"] / BLOCK_SIZE
).astype(int)

positive["block"] = (
    positive["block_lat"].astype(str)
    + "_"
    + positive["block_lon"].astype(str)
)


# ============================================================
# THIN POSITIVES
# ============================================================

rng = np.random.default_rng(SEED)

selected_positive_indices = []

for block_name, block_df in positive.groupby("block"):

    block_indices = block_df.index.to_numpy()

    if len(block_indices) > MAX_POSITIVE_PER_BLOCK:

        chosen = rng.choice(
            block_indices,
            size=MAX_POSITIVE_PER_BLOCK,
            replace=False
        )

    else:

        chosen = block_indices

    selected_positive_indices.extend(
        chosen.tolist()
    )


balanced_positive = positive.loc[
    selected_positive_indices
].copy()

print()
print(
    f"Positive thinning: "
    f"{len(balanced_positive)} positives, "
    f"{balanced_positive['block'].nunique()} spatial blocks"
)


# ============================================================
# DISTANCE OF NEGATIVES FROM POSITIVES
# ============================================================

negative = negative.copy()

negative["nearest_positive_km"] = (
    nearest_positive_distance_km(
        negative,
        balanced_positive
    )
)

print()
print(
    "Nearest-positive negative distances:"
)

print(
    f"  mean   = "
    f"{negative['nearest_positive_km'].mean():.4f} km"
)

print(
    f"  min    = "
    f"{negative['nearest_positive_km'].min():.4f} km"
)

print(
    f"  median = "
    f"{negative['nearest_positive_km'].median():.4f} km"
)

print(
    f"  max    = "
    f"{negative['nearest_positive_km'].max():.4f} km"
)


# ============================================================
# REMOVE TOO-CLOSE NEGATIVES
# ============================================================

safe_negative = negative[
    negative["nearest_positive_km"]
    >= MIN_NEGATIVE_DISTANCE_KM
].copy()

removed_close = (
    len(negative) - len(safe_negative)
)

print()
print(
    f"<{MIN_NEGATIVE_DISTANCE_KM:g} km negatives removed: "
    f"{removed_close}"
)

print(
    f"Safe negatives: {len(safe_negative)}"
)


# ============================================================
# PREFERRED NEGATIVE POOL
# ============================================================

preferred_negative = safe_negative[
    safe_negative["nearest_positive_km"]
    <= PREFERRED_NEGATIVE_DISTANCE_KM
].copy()

print(
    f"Preferred <= {PREFERRED_NEGATIVE_DISTANCE_KM:g} km: "
    f"{len(preferred_negative)}"
)


# ============================================================
# TARGET NEGATIVE COUNT
# ============================================================

target_negative_count = (
    len(balanced_positive) * NEGATIVE_RATIO
)

print()
print(
    f"Target negatives: {target_negative_count}"
)


# ============================================================
# FIRST PASS:
# ONE NEGATIVE FROM EACH POSITIVE-RELEVANT BLOCK
# ============================================================

positive_blocks = set(
    balanced_positive["block"]
)

safe_negative["block_lat"] = np.floor(
    safe_negative["latitude"] / BLOCK_SIZE
).astype(int)

safe_negative["block_lon"] = np.floor(
    safe_negative["longitude"] / BLOCK_SIZE
).astype(int)

safe_negative["block"] = (
    safe_negative["block_lat"].astype(str)
    + "_"
    + safe_negative["block_lon"].astype(str)
)

first_pass_indices = []

for block_name in sorted(positive_blocks):

    candidates = safe_negative[
        safe_negative["block"] == block_name
    ]

    if len(candidates) == 0:
        continue

    chosen = rng.choice(
        candidates.index.to_numpy(),
        size=1,
        replace=False
    )[0]

    first_pass_indices.append(chosen)

first_pass_indices = list(
    dict.fromkeys(first_pass_indices)
)

first_pass_negative = safe_negative.loc[
    first_pass_indices
].copy()

print()
print(
    f"First-pass negative selection: "
    f"{len(first_pass_negative)}"
)


# ============================================================
# SECOND PASS:
# FILL FROM PREFERRED POOL
# ============================================================

selected_indices = set(
    first_pass_negative.index
)

remaining_needed = (
    target_negative_count
    - len(first_pass_negative)
)

preferred_remaining = preferred_negative[
    ~preferred_negative.index.isin(
        selected_indices
    )
].copy()

if remaining_needed > 0:

    take_count = min(
        remaining_needed,
        len(preferred_remaining)
    )

    if take_count > 0:

        chosen = rng.choice(
            preferred_remaining.index.to_numpy(),
            size=take_count,
            replace=False
        )

        second_pass_negative = (
            preferred_remaining.loc[chosen]
            .copy()
        )

    else:

        second_pass_negative = (
            preferred_remaining.iloc[0:0]
            .copy()
        )

else:

    second_pass_negative = (
        preferred_remaining.iloc[0:0]
        .copy()
    )


# ============================================================
# THIRD PASS:
# FALLBACK IF NECESSARY
# ============================================================

selected_indices.update(
    second_pass_negative.index
)

remaining_needed = (
    target_negative_count
    - len(first_pass_negative)
    - len(second_pass_negative)
)

if remaining_needed > 0:

    fallback_pool = safe_negative[
        ~safe_negative.index.isin(
            selected_indices
        )
    ].copy()

    if len(fallback_pool) < remaining_needed:

        raise RuntimeError(
            "Not enough safe negative samples "
            "to achieve requested 2:1 ratio."
        )

    chosen = rng.choice(
        fallback_pool.index.to_numpy(),
        size=remaining_needed,
        replace=False
    )

    third_pass_negative = (
        fallback_pool.loc[chosen]
        .copy()
    )

else:

    third_pass_negative = (
        safe_negative.iloc[0:0]
        .copy()
    )


# ============================================================
# COMBINE
# ============================================================

balanced_negative = pd.concat(
    [
        first_pass_negative,
        second_pass_negative,
        third_pass_negative
    ],
    axis=0
)

balanced_negative = balanced_negative[
    ~balanced_negative.index.duplicated(
        keep="first"
    )
].copy()


# ============================================================
# FINAL DATASET
# ============================================================

balanced = pd.concat(
    [
        balanced_positive,
        balanced_negative
    ],
    axis=0
)

# Shuffle deterministically.
balanced = balanced.sample(
    frac=1.0,
    random_state=SEED
).reset_index(drop=True)


# ============================================================
# REMOVE TEMPORARY COLUMNS
# ============================================================

temporary_columns = [
    "block_lat",
    "block_lon",
    "block",
    "nearest_positive_km"
]

balanced = balanced.drop(
    columns=[
        c for c in temporary_columns
        if c in balanced.columns
    ],
    errors="ignore"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

label_counts = (
    balanced["label"]
    .value_counts()
    .sort_index()
)

print()
print("=" * 70)
print("FINAL SAMPLE AUDIT")
print("=" * 70)

print(
    f"Final records: {len(balanced)}"
)

print(
    f"Label 0: {label_counts.get(0, 0)}"
)

print(
    f"Label 1: {label_counts.get(1, 0)}"
)

if label_counts.get(1, 0) > 0:

    ratio = (
        label_counts.get(0, 0)
        / label_counts.get(1, 0)
    )

    print(
        f"Negative:positive ratio: {ratio:.2f}:1"
    )

duplicate_coordinates = balanced.duplicated(
    subset=["latitude", "longitude"]
).sum()

duplicate_rows = balanced.duplicated().sum()

print(
    f"Duplicate coordinates: "
    f"{duplicate_coordinates}"
)

print(
    f"Exact duplicate rows: "
    f"{duplicate_rows}"
)

if duplicate_coordinates != 0:
    raise RuntimeError(
        "Duplicate coordinates remain after balancing."
    )

if duplicate_rows != 0:
    raise RuntimeError(
        "Exact duplicate rows remain after balancing."
    )

if label_counts.get(0, 0) != (
    label_counts.get(1, 0) * NEGATIVE_RATIO
):
    raise RuntimeError(
        "Final class ratio is not the requested ratio."
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

balanced.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print("PHASE 4B.4 COMPLETE")