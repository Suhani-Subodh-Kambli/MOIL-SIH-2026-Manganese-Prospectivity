from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# MOIL SIH 2026
# PHASE 5 - GEOLOGICAL ENRICHMENT VALIDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

GRID_FILE = OUTPUT_DIR / "phase5_prospectivity_grid.csv"

print("=" * 75)
print("MOIL SIH 2026 - GEOLOGICAL ENRICHMENT VALIDATION")
print("=" * 75)

# ============================================================
# STEP 1 - LOAD GRID
# ============================================================

print("\n" + "=" * 75)
print("STEP 1 - Loading prospectivity grid")
print("=" * 75)

df = pd.read_csv(GRID_FILE)

print(f"Grid shape: {df.shape}")

required = [
    "Prospectivity_Score",
    "Model_Probability",
    "geo_group",
    "geo_lithology",
    "geo_formation",
    "geo_age",
    "Sausar_Group_Proxy",
    "Metamorphic_Host_Proxy",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

# ============================================================
# STEP 2 - SORT BY PROSPECTIVITY
# ============================================================

print("\n" + "=" * 75)
print("STEP 2 - Ranking cells")
print("=" * 75)

df = df.sort_values(
    "Prospectivity_Score",
    ascending=False
).reset_index(drop=True)

N = len(df)

print(f"Total cells: {N}")


# ============================================================
# STEP 3 - DEFINE MODEL TARGET GROUPS
# ============================================================

groups = {
    "Top_1pct": max(1, int(N * 0.01)),
    "Top_5pct": max(1, int(N * 0.05)),
    "Top_10pct": max(1, int(N * 0.10)),
    "Top_20pct": max(1, int(N * 0.20)),
}


# ============================================================
# STEP 4 - SAUSAR ENRICHMENT
# ============================================================

print("\n" + "=" * 75)
print("STEP 4 - Sausar geological enrichment")
print("=" * 75)

overall_sausar = df[
    "Sausar_Group_Proxy"
].mean()

overall_meta = df[
    "Metamorphic_Host_Proxy"
].mean()

print(
    f"AOI mean Sausar proxy: "
    f"{overall_sausar:.4f}"
)

print(
    f"AOI mean metamorphic-host proxy: "
    f"{overall_meta:.4f}"
)

rows = []

for name, n in groups.items():

    subset = df.head(n)

    sausar = subset[
        "Sausar_Group_Proxy"
    ].mean()

    meta = subset[
        "Metamorphic_Host_Proxy"
    ].mean()

    sausar_enrichment = (
        sausar / overall_sausar
        if overall_sausar > 0
        else np.nan
    )

    meta_enrichment = (
        meta / overall_meta
        if overall_meta > 0
        else np.nan
    )

    print(f"\n{name}")

    print(
        f"  Sausar proxy       : {sausar:.4f}"
    )

    print(
        f"  Sausar enrichment  : "
        f"{sausar_enrichment:.3f}x"
    )

    print(
        f"  Metamorphic proxy  : {meta:.4f}"
    )

    print(
        f"  Metamorphic enrich : "
        f"{meta_enrichment:.3f}x"
    )

    rows.append({
        "Selection": name,
        "Cells": n,
        "Mean_Sausar_Proxy": sausar,
        "AOI_Mean_Sausar_Proxy": overall_sausar,
        "Sausar_Enrichment": sausar_enrichment,
        "Mean_Metamorphic_Host_Proxy": meta,
        "AOI_Mean_Metamorphic_Host_Proxy": overall_meta,
        "Metamorphic_Enrichment": meta_enrichment,
    })

proxy_df = pd.DataFrame(rows)


# ============================================================
# STEP 5 - GEOLOGICAL CATEGORY DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 75)
print("STEP 5 - Geological category enrichment")
print("=" * 75)

geo_columns = [
    "geo_group",
    "geo_lithology",
    "geo_formation",
    "geo_age",
    "geo_intrusive",
    "geo_supergroup",
]

geo_rows = []

for column in geo_columns:

    print("\n" + "-" * 70)
    print(column)
    print("-" * 70)

    overall_counts = (
        df[column]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
    )

    overall_pct = (
        overall_counts / N
    )

    # Focus on categories appearing in top 20%
    top20 = df.head(
        groups["Top_20pct"]
    )

    top20_counts = (
        top20[column]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
    )

    top20_pct = (
        top20_counts
        / len(top20)
    )

    comparison = pd.DataFrame({
        "Overall_Count": overall_counts,
        "Overall_Percent": overall_pct * 100,
        "Top20_Count": top20_counts,
        "Top20_Percent": top20_pct * 100,
    }).fillna(0)

    comparison["Enrichment"] = np.where(
        comparison["Overall_Percent"] > 0,
        comparison["Top20_Percent"]
        / comparison["Overall_Percent"],
        np.nan
    )

    comparison = comparison.sort_values(
        "Top20_Percent",
        ascending=False
    )

    print(
        comparison.head(15).to_string()
    )

    comparison_out = comparison.reset_index()

    comparison_out.insert(
        0,
        "Geology_Feature",
        column
    )

    geo_rows.append(
        comparison_out
    )


geo_enrichment_df = pd.concat(
    geo_rows,
    ignore_index=True
)


# ============================================================
# STEP 6 - SAUSAR / METAMORPHIC THRESHOLD TEST
# ============================================================

print("\n" + "=" * 75)
print("STEP 6 - Geological indicator threshold test")
print("=" * 75)

thresholds = [
    0.25,
    0.50,
    0.75,
    0.90,
]

threshold_rows = []

for feature in [
    "Sausar_Group_Proxy",
    "Metamorphic_Host_Proxy"
]:

    print(f"\n{feature}")

    overall = df[feature].mean()

    for threshold in thresholds:

        overall_rate = (
            df[feature] >= threshold
        ).mean()

        print(
            f"  >= {threshold:.2f}: "
            f"{overall_rate * 100:.2f}% AOI"
        )

        for name, n in groups.items():

            subset = df.head(n)

            target_rate = (
                subset[feature] >= threshold
            ).mean()

            enrichment = (
                target_rate / overall_rate
                if overall_rate > 0
                else np.nan
            )

            threshold_rows.append({
                "Feature": feature,
                "Threshold": threshold,
                "Selection": name,
                "Target_Rate": target_rate,
                "AOI_Rate": overall_rate,
                "Enrichment": enrichment,
            })


threshold_df = pd.DataFrame(
    threshold_rows
)


# ============================================================
# STEP 7 - UNKNOWN GEOLOGY AUDIT
# ============================================================

print("\n" + "=" * 75)
print("STEP 7 - Unknown geology audit")
print("=" * 75)

unknown_rows = []

for column in geo_columns:

    overall_unknown = (
        df[column]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .isin([
            "UNKNOWN",
            "NAN",
            "",
            "NONE"
        ])
        .mean()
    )

    for name, n in groups.items():

        subset = df.head(n)

        target_unknown = (
            subset[column]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            .isin([
                "UNKNOWN",
                "NAN",
                "",
                "NONE"
            ])
            .mean()
        )

        unknown_rows.append({
            "Feature": column,
            "Selection": name,
            "AOI_Unknown_Rate": overall_unknown,
            "Target_Unknown_Rate": target_unknown,
        })

        if name == "Top_5pct":

            print(
                f"{column}: "
                f"AOI unknown = "
                f"{overall_unknown * 100:.2f}%, "
                f"Top 5% unknown = "
                f"{target_unknown * 100:.2f}%"
            )


unknown_df = pd.DataFrame(
    unknown_rows
)


# ============================================================
# STEP 8 - TOP 100 GEOLOGICAL AUDIT
# ============================================================

print("\n" + "=" * 75)
print("STEP 8 - Top 100 geological audit")
print("=" * 75)

top100 = df.head(100)

for column in geo_columns:

    print(f"\n{column}")

    print(
        top100[column]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
        .head(15)
        .to_string()
    )


# ============================================================
# STEP 9 - SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 75)
print("STEP 9 - Saving outputs")
print("=" * 75)

proxy_file = (
    OUTPUT_DIR
    / "phase5_geological_proxy_enrichment.csv"
)

geo_file = (
    OUTPUT_DIR
    / "phase5_geological_category_enrichment.csv"
)

threshold_file = (
    OUTPUT_DIR
    / "phase5_geological_threshold_enrichment.csv"
)

unknown_file = (
    OUTPUT_DIR
    / "phase5_geology_unknown_audit.csv"
)

proxy_df.to_csv(
    proxy_file,
    index=False
)

geo_enrichment_df.to_csv(
    geo_file,
    index=False
)

threshold_df.to_csv(
    threshold_file,
    index=False
)

unknown_df.to_csv(
    unknown_file,
    index=False
)

print(f"Saved: {proxy_file}")
print(f"Saved: {geo_file}")
print(f"Saved: {threshold_file}")
print(f"Saved: {unknown_file}")


# ============================================================
# STEP 10 - FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL GEOLOGICAL ENRICHMENT SUMMARY")
print("=" * 75)

print(
    proxy_df[
        [
            "Selection",
            "Mean_Sausar_Proxy",
            "Sausar_Enrichment",
            "Mean_Metamorphic_Host_Proxy",
            "Metamorphic_Enrichment"
        ]
    ].to_string(index=False)
)

print("\nInterpretation guide:")
print("""
Enrichment > 1.0
    Target group contains more of the geological indicator
    than the full AOI.

Enrichment ~ 1.0
    Similar to the AOI.

Enrichment < 1.0
    Target group contains less of the indicator.

This is an audit of geological plausibility.
It is NOT a claim that these geological units contain
economically mineable manganese.
""")

print("\n" + "=" * 75)
print("GEOLOGICAL ENRICHMENT VALIDATION COMPLETE")
print("=" * 75)