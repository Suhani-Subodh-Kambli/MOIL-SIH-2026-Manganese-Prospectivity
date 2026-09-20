"""
scripts/build_india_manganese_occurrences.py

Consolidates India-wide manganese occurrences, deposits, and active mining leases
from Geological Survey of India (GSI) and National Geoscience Data Repository (NGDR).

Outputs:
  - data/gsi/india_manganese_occurrences.csv
  - data/gsi/india_manganese_occurrences.geojson
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import polygonize, unary_union


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
GSI_DIR = BASE_DIR / "data" / "gsi"

GSI_XLS = GSI_DIR / "Manganese_Ore_1.xls"
NGDR_DEPOSITS_FILE = GSI_DIR / "NGDR_Mineral_Deposits.parquet"
NGDR_LEASES_FILE = GSI_DIR / "NGDR_Major_Mining_Leases_2022.parquet"
METALLOGENIC_ZONES_FILE = GSI_DIR / "NGSR_Metallozenic_Zones.parquet"

OUTPUT_CSV = GSI_DIR / "india_manganese_occurrences.csv"
OUTPUT_GEOJSON = GSI_DIR / "india_manganese_occurrences.geojson"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def _download_if_needed(file_path: Path, url: str):
    if not file_path.exists():
        import urllib.request
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {file_path.name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(file_path, "wb") as f:
            f.write(resp.read())
        print(f"Downloaded {file_path.name}.")


def main():
    print("=" * 75)
    print("CONSOLIDATING INDIA-WIDE MANGANESE OCCURRENCES & MINING LEASES")
    print("=" * 75)

    # Ensure required NGDR files exist
    base_url = "https://github.com/ramSeraph/indian_land_features/releases/download/mining"
    _download_if_needed(NGDR_DEPOSITS_FILE, f"{base_url}/NGDR_Mineral_Deposits.parquet")
    _download_if_needed(NGDR_LEASES_FILE, f"{base_url}/NGDR_Major_Mining_Leases_2022.parquet")
    _download_if_needed(METALLOGENIC_ZONES_FILE, f"{base_url}/NGSR_Metallozenic_Zones.parquet")

    all_records = []

    # ------------------------------------------------------------
    # 1. GSI Excel Dataset (Manganese_Ore_1.xls)
    # ------------------------------------------------------------
    if GSI_XLS.exists():
        print(f"\n[1] Reading GSI Excel: {GSI_XLS.name}")
        df_xls = pd.read_excel(GSI_XLS)
        for _, row in df_xls.iterrows():
            lat = float(row.get("LATDD", np.nan))
            lon = float(row.get("LONDD", np.nan))
            if np.isnan(lat) or np.isnan(lon):
                continue
            all_records.append({
                "name": str(row.get("LOCALITY", "Unknown")).strip().upper(),
                "state": str(row.get("STATE", "Unknown")).strip().title(),
                "district": "Unknown",
                "latitude": lat,
                "longitude": lon,
                "host_rock": str(row.get("HOSTROCK", "Unknown")).strip(),
                "formation": str(row.get("FORMATION", "Unknown")).strip(),
                "source": "GSI Mineral Occurrence",
                "deposit_type": "Primary Deposit",
                "reserve": "Documented GSI Occurrence",
                "grade": "Historical GSI Record"
            })
        print(f"   Loaded {len(df_xls)} records from GSI Excel.")
    else:
        print(f"   Warning: {GSI_XLS} not found.")

    # ------------------------------------------------------------
    # 2. NGDR Mineral Deposits Parquet
    # ------------------------------------------------------------
    if NGDR_DEPOSITS_FILE.exists():
        print(f"\n[2] Reading NGDR Deposits: {NGDR_DEPOSITS_FILE.name}")
        gdf_dep = gpd.read_parquet(NGDR_DEPOSITS_FILE)
        mn_dep = gdf_dep[gdf_dep["mineral_or"].str.contains("Mn", case=False, na=False)]
        for _, row in mn_dep.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lon, lat = geom.x, geom.y
            all_records.append({
                "name": str(row.get("locality", "Unknown")).strip().upper(),
                "state": str(row.get("state", "Unknown")).strip().title(),
                "district": "Unknown",
                "latitude": float(lat),
                "longitude": float(lon),
                "host_rock": str(row.get("hostrock", "Unknown")).strip(),
                "formation": str(row.get("formation", "Unknown")).strip(),
                "source": "NGDR Mineral Deposit",
                "deposit_type": "Mineral Deposit",
                "reserve": str(row.get("reserve", "Unknown")).strip(),
                "grade": str(row.get("grade", "Unknown")).strip()
            })
        print(f"   Loaded {len(mn_dep)} records from NGDR Mineral Deposits.")

    # ------------------------------------------------------------
    # 3. NGDR Major Mining Leases 2022
    # ------------------------------------------------------------
    if NGDR_LEASES_FILE.exists():
        print(f"\n[3] Reading NGDR Major Mining Leases: {NGDR_LEASES_FILE.name}")
        gdf_leases = gpd.read_parquet(NGDR_LEASES_FILE)
        mn_leases = gdf_leases[gdf_leases["mineral_na"].str.contains("Manganese", case=False, na=False)].copy()
        
        # Calculate centroids accurately in projected CRS (EPSG:3857)
        projected = mn_leases.to_crs(epsg=3857)
        centroids = projected.geometry.centroid.to_crs(epsg=4326)
        
        for idx, row in mn_leases.iterrows():
            centroid = centroids.loc[idx]
            lon, lat = centroid.x, centroid.y
            mine_name = str(row.get("mine_name", "Unknown")).strip().upper()
            lessee = str(row.get("name_of_le", "Unknown")).strip().title()
            district = str(row.get("district", "Unknown")).strip().title()
            state = str(row.get("state", "Unknown")).strip().title()
            lease_area = row.get("lease_area", np.nan)
            
            all_records.append({
                "name": mine_name if mine_name != "UNKNOWN" else f"Manganese Lease ({lessee})",
                "state": state,
                "district": district,
                "latitude": float(lat),
                "longitude": float(lon),
                "host_rock": "Manganiferous Host Formation",
                "formation": "Active Mining Area",
                "source": "NGDR Major Mining Lease (2022)",
                "deposit_type": "Active Mining Lease",
                "reserve": f"Lease Area: {lease_area} ha" if pd.notna(lease_area) else "Active Lease",
                "grade": f"Lessee: {lessee}"
            })
        print(f"   Loaded {len(mn_leases)} records from NGDR Active Mining Leases.")

    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    print(f"\nTotal raw occurrences gathered: {len(df)}")

    # Clean state names
    state_mapping = {
        "Maharastra": "Maharashtra",
        "Orissa": "Odisha",
        "Chattisgarh": "Chhattisgarh"
    }
    df["state"] = df["state"].replace(state_mapping)

    # ------------------------------------------------------------
    # 4. Spatial Tagging with NGSR Metallogenic Provinces
    # ------------------------------------------------------------
    if METALLOGENIC_ZONES_FILE.exists():
        print(f"\n[4] Tagging points with NGSR Metallogenic Provinces...")
        gdf_zones = gpd.read_parquet(METALLOGENIC_ZONES_FILE)
        mn_zones = gdf_zones[gdf_zones["commodity"].str.contains("Mn", case=False, na=False)].copy()
        
        polygon_zones = []
        for _, r in mn_zones.iterrows():
            polys = list(polygonize(r.geometry))
            if not polys:
                poly = unary_union(r.geometry).convex_hull
            else:
                poly = polys[0]
            # Buffer by 0.15 deg (~16 km) to capture peripheral deposits
            buffered_poly = poly.buffer(0.15)
            polygon_zones.append({
                "belt_name": r["name"],
                "belt_zone_no": r["zone_no"],
                "geometry": buffered_poly
            })
            
        gdf_poly_zones = gpd.GeoDataFrame(polygon_zones, crs="EPSG:4326")
        
        gdf_points = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
            crs="EPSG:4326"
        )
        
        joined = gpd.sjoin(gdf_points, gdf_poly_zones, how="left", predicate="intersects")
        joined = joined.groupby(joined.index).first()
        
        df["metallogenic_belt"] = joined["belt_name"].fillna("Other Regional Mn Belt")
        df["metallogenic_zone_no"] = joined["belt_zone_no"].fillna("N/A")
    else:
        df["metallogenic_belt"] = "Other Regional Mn Belt"
        df["metallogenic_zone_no"] = "N/A"

    # ------------------------------------------------------------
    # 5. Deduplication (Group points closer than 500m)
    # ------------------------------------------------------------
    print("\n[5] Deduplicating overlapping points within 500m...")
    sorted_df = df.copy()
    keep_indices = []
    
    for i, row in sorted_df.iterrows():
        lat_i, lon_i = row["latitude"], row["longitude"]
        is_duplicate = False
        for k in keep_indices:
            lat_k, lon_k = sorted_df.loc[k, "latitude"], sorted_df.loc[k, "longitude"]
            dist_km = haversine_km(lat_i, lon_i, lat_k, lon_k)
            if dist_km < 0.5:  # within 500 meters
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)

    dedup_df = sorted_df.loc[keep_indices].reset_index(drop=True)
    dedup_df["occurrence_id"] = [f"MN_IND_{i+1:04d}" for i in range(len(dedup_df))]
    print(f"   Deduplicated count: {len(dedup_df)} unique manganese sites across India.")

    # Summary by state
    print("\nState-wise Breakdown of Known Manganese Sites:")
    state_counts = dedup_df["state"].value_counts()
    for state, count in state_counts.items():
        print(f"   {state:20s}: {count:3d}")

    # Summary by metallogenic belt
    print("\nMetallogenic Belt Breakdown:")
    belt_counts = dedup_df["metallogenic_belt"].value_counts()
    for belt, count in belt_counts.items():
        print(f"   {belt:35s}: {count:3d}")

    # ------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dedup_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV: {OUTPUT_CSV}")

    gdf_out = gpd.GeoDataFrame(
        dedup_df,
        geometry=[Point(xy) for xy in zip(dedup_df["longitude"], dedup_df["latitude"])],
        crs="EPSG:4326"
    )
    gdf_out.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"Saved GeoJSON: {OUTPUT_GEOJSON}")
    print("=" * 75)


if __name__ == "__main__":
    main()
