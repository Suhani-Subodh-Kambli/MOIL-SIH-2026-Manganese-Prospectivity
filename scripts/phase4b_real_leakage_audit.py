import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

DATA = Path("data/MOIL_SIH_Phase4B_Balanced_Features.csv")
OUT = Path("outputs/phase4b_real_leakage_audit.csv")

df = pd.read_csv(DATA)

print("=" * 70)
print("REAL PHASE 4B LEAKAGE AUDIT")
print("=" * 70)

print("\nDataset:", DATA)
print("Shape:", df.shape)

# ---------------------------------------------------------
# 1. LABEL DISTRIBUTION
# ---------------------------------------------------------
print("\n[1] LABEL DISTRIBUTION")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))

# ---------------------------------------------------------
# 2. FORBIDDEN FEATURES
# ---------------------------------------------------------
print("\n[2] FORBIDDEN / DANGEROUS FEATURES")

for c in ["Distance_Mn", "latitude", "longitude", "lat", "lon"]:
    if c in df.columns:
        print(f"\n{c}:")
        print(df.groupby("label")[c].agg(["count", "min", "mean", "median", "max"]))

# ---------------------------------------------------------
# 3. AUC OF EVERY NUMERIC FEATURE
# ---------------------------------------------------------
print("\n[3] INDIVIDUAL NUMERIC FEATURE AUC")

results = []

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for c in numeric_cols:

    if c == "label":
        continue

    x = df[c].replace([np.inf, -np.inf], np.nan)

    if x.nunique(dropna=True) < 2:
        continue

    valid = x.notna() & df["label"].notna()

    try:
        auc = roc_auc_score(df.loc[valid, "label"], x[valid])
        inv_auc = max(auc, 1 - auc)

        results.append({
            "feature": c,
            "auc": auc,
            "inverse_auc": inv_auc,
            "unique_values": x.nunique()
        })

    except Exception:
        pass

auc_df = pd.DataFrame(results).sort_values(
    "inverse_auc",
    ascending=False
)

print(
    auc_df[
        ["feature", "auc", "inverse_auc", "unique_values"]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# 4. VERY SUSPICIOUS FEATURES
# ---------------------------------------------------------
print("\n[4] FEATURES WITH VERY HIGH LABEL SEPARATION")

suspicious = auc_df[auc_df["inverse_auc"] >= 0.80]

if len(suspicious) == 0:
    print("NONE >= 0.80")
else:
    print(suspicious.to_string(index=False))

# ---------------------------------------------------------
# 5. GEOLOGY FEATURES
# ---------------------------------------------------------
print("\n[5] GEOLOGY CATEGORICAL FEATURES")

geo_cols = [
    c for c in df.columns
    if c.startswith("geo_")
]

for c in geo_cols:

    if df[c].dtype == "object":

        print(f"\n--- {c} ---")

        tab = pd.crosstab(
            df[c].fillna("UNKNOWN"),
            df["label"],
            normalize="index"
        )

        print(tab.sort_values(
            1,
            ascending=False
        ).head(20).to_string())

# ---------------------------------------------------------
# 6. NUMERIC GEOLOGY / ENGINEERED FEATURES
# ---------------------------------------------------------
print("\n[6] GEOLOGY / ENGINEERED NUMERIC FEATURES")

engineered = [
    "Sausar_Group_Proxy",
    "Metamorphic_Host_Proxy",
    "Geology_Unknown",
    "Geo_Boundary_Distance_km",
    "Geo_Boundary_Density_1km",
    "Geo_Boundary_Density_3km",
    "Lithology_Diversity_3km",
    "Formation_Diversity_3km",
    "Local_Elevation_STD",
    "TPI_Local",
    "Local_Slope_STD",
    "Local_Aspect_Dispersion"
]

for c in engineered:

    if c not in df.columns:
        continue

    print(f"\n--- {c} ---")

    print(
        df.groupby("label")[c].agg(
            ["count", "min", "mean", "median", "max"]
        )
    )

# ---------------------------------------------------------
# 7. DUPLICATE COORDINATES
# ---------------------------------------------------------
print("\n[7] DUPLICATE COORDINATES")

if "latitude" in df.columns and "longitude" in df.columns:

    coord_counts = (
        df.groupby(["latitude", "longitude"])
        .size()
        .sort_values(ascending=False)
    )

    duplicates = coord_counts[coord_counts > 1]

    print("Unique coordinates:", len(coord_counts))
    print("Duplicate coordinate locations:", len(duplicates))
    print("Rows involved:", duplicates.sum())

    if len(duplicates):
        print("\nTop duplicate locations:")
        print(duplicates.head(20))

# ---------------------------------------------------------
# 8. EXACT DUPLICATE FEATURE ROWS
# ---------------------------------------------------------
print("\n[8] EXACT DUPLICATE ROWS")

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

# ---------------------------------------------------------
# 9. POSITIVE/NEGATIVE NEAREST DISTANCES
# ---------------------------------------------------------
print("\n[9] POSITIVE vs NEGATIVE SPATIAL SEPARATION")

if "latitude" in df.columns and "longitude" in df.columns:

    pos = df[df.label == 1][["latitude", "longitude"]].values
    neg = df[df.label == 0][["latitude", "longitude"]].values

    # Approximate km using latitude/longitude
    # sufficient for leakage audit
    mins = []

    for p in pos:

        dlat = (neg[:, 0] - p[0]) * 111.0
        dlon = (
            (neg[:, 1] - p[1])
            * 111.0
            * np.cos(np.radians(p[0]))
        )

        dist = np.sqrt(dlat**2 + dlon**2)

        mins.append(dist.min())

    mins = np.array(mins)

    print(
        pd.Series(mins).describe()
    )

    print("\nPositive points with negative neighbor within:")

    for km in [0.5, 1, 2, 3, 5, 10]:
        print(
            f"{km:>4} km:",
            int((mins <= km).sum()),
            "/",
            len(mins)
        )

# ---------------------------------------------------------
# 10. SPATIAL BLOCK ANALYSIS
# ---------------------------------------------------------
print("\n[10] SPATIAL BLOCK LABEL MIXING")

if "latitude" in df.columns and "longitude" in df.columns:

    block_size = 0.10

    df["_block_lat"] = np.floor(
        df.latitude / block_size
    ).astype(int)

    df["_block_lon"] = np.floor(
        df.longitude / block_size
    ).astype(int)

    blocks = (
        df.groupby(["_block_lat", "_block_lon"])
        .label.agg(["count", "sum", "mean"])
        .reset_index()
    )

    print("Number of blocks:", len(blocks))

    print(
        "\nBlocks by class composition:"
    )

    print(
        pd.Series(
            np.select(
                [
                    blocks["sum"] == 0,
                    blocks["sum"] == blocks["count"]
                ],
                [
                    "negative_only",
                    "positive_only"
                ],
                default="mixed"
            )
        ).value_counts()
    )

    print("\nMost positive-heavy blocks:")

    print(
        blocks.sort_values(
            "mean",
            ascending=False
        ).head(20).to_string(index=False)
    )

# ---------------------------------------------------------
# 11. CHECK WHETHER GEOLOGY PROXIES ARE LABEL-DERIVED
# ---------------------------------------------------------
print("\n[11] POSSIBLE LABEL-DERIVED GEOLOGY PROXIES")

for c in [
    "Sausar_Group_Proxy",
    "Metamorphic_Host_Proxy",
    "Geology_Unknown",
    "Geo_Boundary_Distance_km",
    "Geo_Boundary_Density_1km",
    "Geo_Boundary_Density_3km",
]:

    if c not in df.columns:
        continue

    a = df[df.label == 1][c].mean()
    b = df[df.label == 0][c].mean()

    print(
        f"{c:35s} "
        f"positive_mean={a:.4f} "
        f"negative_mean={b:.4f}"
    )

# ---------------------------------------------------------
# 12. SAVE AUC AUDIT
# ---------------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)

auc_df.to_csv(
    OUT,
    index=False
)

print("\nSaved:")
print(OUT)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
