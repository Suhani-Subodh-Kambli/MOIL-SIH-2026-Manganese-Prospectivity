"""
core/geology_index.py

High-performance spatial geology indexer and structural feature calculator
for India-wide manganese prospectivity mapping.

Integrates:
- High-resolution 1:50,000 lithology for pilot/belt regions (Balaghat, etc.)
- National 1:2,000,000 NGDR geology covering all of India
- Rapid STRtree spatial indexing for sub-millisecond point queries
- Exact structural proxy derivation (boundary distance, density, diversity)
"""

import math
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape, Polygon
from shapely.strtree import STRtree
from shapely.ops import nearest_points


# ============================================================
# CONSTANTS
# ============================================================

EARTH_KM_PER_DEG = 111.32
RADIUS_1KM_DEG = 1.0 / EARTH_KM_PER_DEG
RADIUS_3KM_DEG = 3.0 / EARTH_KM_PER_DEG

METAMORPHIC_KEYWORDS = [
    "SCHIST",
    "GNEISS",
    "AMPHIBOLITE",
    "QUARTZITE",
    "GRANULITE",
    "PHYLLITE",
    "MARBLE",
    "GONDITE",
    "KHONDALITE",
    "CHARNOCKITE"
]


class GeologyIndexer:
    """
    Unified spatial geology query engine with hierarchical resolution:
    1. Check local high-res 1:50k lithology (if within coverage area)
    2. Fallback seamlessly to national 1:2M NGDR geology
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parents[1]
        else:
            self.base_dir = Path(base_dir)

        self.data_dir = self.base_dir / "data" / "geology"

        # National 2M dataset
        self.national_file = self.data_dir / "NGDR_Geology_2M.parquet"

        # Local Balaghat 50k dataset
        self.balaghat_file = self.data_dir / "balaghat" / "balaghat_lithology.geojsonl"

        # Data stores
        self._nat_gdf = None
        self._nat_tree = None
        self._nat_boundaries = None
        self._nat_boundary_tree = None

        self._local_geoms = []
        self._local_props = []
        self._local_tree = None
        self._local_boundaries = []
        self._local_boundary_tree = None
        self._local_bounds = (79.50, 21.30, 80.80, 22.40)  # Balaghat AOI

        self._load_data()

    def _load_data(self):
        # 1. Load National 2M Geology (with auto-download fallback)
        if not self.national_file.exists():
            try:
                print(f"National geology file not found. Auto-downloading {self.national_file.name} (30MB)...")
                import urllib.request
                self.national_file.parent.mkdir(parents=True, exist_ok=True)
                url = "https://github.com/ramSeraph/indian_land_features/releases/download/geology/NGDR_Geology_2M.parquet"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp, open(self.national_file, "wb") as f:
                    f.write(resp.read())
                print(f"Downloaded {self.national_file.name} successfully.")
            except Exception as e:
                print(f"Warning: Could not auto-download national geology: {e}")

        if self.national_file.exists():
            self._nat_gdf = gpd.read_parquet(self.national_file)
            # Ensure OGC:CRS84 / EPSG:4326
            if self._nat_gdf.crs and self._nat_gdf.crs.to_epsg() != 4326:
                self._nat_gdf = self._nat_gdf.to_crs(epsg=4326)

            self._nat_tree = STRtree(self._nat_gdf.geometry)
            self._nat_boundaries = [geom.boundary for geom in self._nat_gdf.geometry]
            self._nat_boundary_tree = STRtree(self._nat_boundaries)
        else:
            print(f"Warning: National geology file {self.national_file} not found.")

        # 2. Load Local Balaghat 50k Lithology (if available)
        if self.balaghat_file.exists():
            with open(self.balaghat_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        feat = json.loads(line)
                        geom_data = feat.get("geometry")
                        if not geom_data:
                            continue
                        geom = shape(geom_data)
                        if geom.is_empty:
                            continue
                        if not geom.is_valid:
                            geom = geom.buffer(0)
                        if geom.is_empty:
                            continue
                        self._local_geoms.append(geom)
                        self._local_props.append(feat.get("properties", {}))
                    except Exception:
                        continue

            if self._local_geoms:
                self._local_tree = STRtree(self._local_geoms)
                self._local_boundaries = [g.boundary for g in self._local_geoms]
                self._local_boundary_tree = STRtree(self._local_boundaries)

    def _is_in_balaghat(self, lon: float, lat: float) -> bool:
        min_lon, min_lat, max_lon, max_lat = self._local_bounds
        return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat) and (self._local_tree is not None)

    def query_geology(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Identify geological formation, group, lithology, and stratigraphy at (lat, lon).
        """
        lat, lon = float(latitude), float(longitude)
        point = Point(lon, lat)

        # A) Try High-Res Balaghat 50k first if within bounds
        if self._is_in_balaghat(lon, lat):
            candidates = self._local_tree.query(point)
            for idx in candidates:
                geom = self._local_geoms[idx]
                if geom.intersects(point):
                    p = self._local_props[idx]
                    return {
                        "geo_age": str(p.get("age", "UNKNOWN")).upper().strip(),
                        "geo_supergroup": str(p.get("supergroup", "UNKNOWN")).upper().strip(),
                        "geo_group": str(p.get("group_name", "UNKNOWN")).upper().strip(),
                        "geo_formation": str(p.get("formation", "UNKNOWN")).upper().strip(),
                        "geo_lithology": str(p.get("lithologic", "UNKNOWN")).upper().strip(),
                        "geo_intrusive": str(p.get("intrusive", "UNKNOWN")).upper().strip(),
                        "geo_stratigraphy": str(p.get("stratigraphy_new", "UNKNOWN")).upper().strip(),
                        "resolution": "1:50,000 (Local NGDR)"
                    }

        # B) Fallback to National 2M Geology
        if self._nat_tree is not None:
            candidates = self._nat_tree.query(point)
            for idx in candidates:
                geom = self._nat_gdf.geometry.iloc[idx]
                if geom.intersects(point):
                    row = self._nat_gdf.iloc[idx]
                    strat = str(row.get("stratigrap_new", row.get("stratigrap", "UNKNOWN"))).upper().strip()
                    group = str(row.get("group_", "UNKNOWN")).upper().strip()
                    supergroup = str(row.get("supergroup", "UNKNOWN")).upper().strip()
                    age = str(row.get("age", "UNKNOWN")).upper().strip()

                    # Infer broad lithology from stratigraphy string if not explicit
                    lithology = "UNKNOWN"
                    for kw in METAMORPHIC_KEYWORDS + ["BASALT", "GRANITE", "LATERITE", "SHALE", "SANDSTONE", "LIMESTONE"]:
                        if kw in strat:
                            lithology = kw
                            break

                    return {
                        "geo_age": age,
                        "geo_supergroup": supergroup,
                        "geo_group": group,
                        "geo_formation": group,  # At 2M, group/formation are correlated
                        "geo_lithology": lithology,
                        "geo_intrusive": "UNKNOWN",
                        "geo_stratigraphy": strat,
                        "resolution": "1:2,000,000 (National NGDR)"
                    }

        # C) Default / Unknown
        return {
            "geo_age": "UNKNOWN",
            "geo_supergroup": "UNKNOWN",
            "geo_group": "UNKNOWN",
            "geo_formation": "UNKNOWN",
            "geo_lithology": "UNKNOWN",
            "geo_intrusive": "UNKNOWN",
            "geo_stratigraphy": "UNKNOWN",
            "resolution": "Unmapped / Water"
        }

    def query_structural_features(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Compute structural contact distance, contact density, and diversity metrics.
        """
        lat, lon = float(latitude), float(longitude)
        point = Point(lon, lat)

        # Select target boundary tree and polygon tree
        if self._is_in_balaghat(lon, lat):
            boundary_tree = self._local_boundary_tree
            boundaries = self._local_boundaries
            poly_tree = self._local_tree
            geoms = self._local_geoms
            props = self._local_props
            is_local = True
        else:
            boundary_tree = self._nat_boundary_tree
            boundaries = self._nat_boundaries
            poly_tree = self._nat_tree
            geoms = self._nat_gdf.geometry if self._nat_gdf is not None else []
            props = self._nat_gdf if self._nat_gdf is not None else []
            is_local = False

        # If no index available, return defaults
        if boundary_tree is None:
            return {
                "Geo_Boundary_Distance_km": 5.0,
                "Geo_Boundary_Density_1km": 0,
                "Geo_Boundary_Density_3km": 0,
                "Lithology_Diversity_3km": 1,
                "Formation_Diversity_3km": 1,
                "Sausar_Group_Proxy": 0,
                "Metamorphic_Host_Proxy": 0,
                "Geology_Unknown": 1
            }

        # 1. Nearest boundary distance
        nearest_res = boundary_tree.nearest(point)
        try:
            nearest_idx = int(nearest_res)
        except (TypeError, ValueError):
            nearest_idx = 0

        nearest_boundary = boundaries[nearest_idx]
        try:
            _, nearest_pt = nearest_points(point, nearest_boundary)
            dx = (nearest_pt.x - lon) * EARTH_KM_PER_DEG * math.cos(math.radians(lat))
            dy = (nearest_pt.y - lat) * EARTH_KM_PER_DEG
            distance_km = math.sqrt(dx * dx + dy * dy)
        except Exception:
            distance_km = 5.0

        # 2. Boundary density within 1 km
        search_1km = point.buffer(RADIUS_1KM_DEG)
        candidates_1km = boundary_tree.query(search_1km)
        count_1km = len(candidates_1km)

        # 3. Boundary density within 3 km
        search_3km = point.buffer(RADIUS_3KM_DEG)
        candidates_3km = boundary_tree.query(search_3km)
        count_3km = len(candidates_3km)

        # 4. Geological diversity within 3 km
        poly_candidates = poly_tree.query(search_3km)
        lithologies = set()
        formations = set()

        for cand in poly_candidates:
            try:
                idx = int(cand)
            except (TypeError, ValueError):
                continue

            if is_local:
                p = props[idx]
                lith = p.get("lithologic")
                form = p.get("formation")
            else:
                row = self._nat_gdf.iloc[idx]
                lith = row.get("stratigrap_new", row.get("stratigrap"))
                form = row.get("group_")

            if lith:
                lithologies.add(str(lith).upper().strip())
            if form:
                formations.add(str(form).upper().strip())

        lith_diversity = max(1, len(lithologies))
        form_diversity = max(1, len(formations))

        # Base geology query for proxies
        geo = self.query_geology(lat, lon)
        group_text = geo["geo_group"]
        lith_text = geo["geo_lithology"]

        sausar_proxy = 1 if "SAUSAR" in group_text else 0
        metamorphic_proxy = 1 if any(kw in lith_text or kw in geo["geo_stratigraphy"] for kw in METAMORPHIC_KEYWORDS) else 0
        geology_unknown = 1 if group_text == "UNKNOWN" or lith_text == "UNKNOWN" else 0

        return {
            "Geo_Boundary_Distance_km": float(distance_km),
            "Geo_Boundary_Density_1km": int(count_1km),
            "Geo_Boundary_Density_3km": int(count_3km),
            "Lithology_Diversity_3km": int(lith_diversity),
            "Formation_Diversity_3km": int(form_diversity),
            "Sausar_Group_Proxy": int(sausar_proxy),
            "Metamorphic_Host_Proxy": int(metamorphic_proxy),
            "Geology_Unknown": int(geology_unknown),
            "resolution": geo.get("resolution", "Unknown")
        }


# Singleton instance for quick access
_INDEXER_INSTANCE = None

def get_geology_indexer() -> GeologyIndexer:
    global _INDEXER_INSTANCE
    if _INDEXER_INSTANCE is None:
        _INDEXER_INSTANCE = GeologyIndexer()
    return _INDEXER_INSTANCE
