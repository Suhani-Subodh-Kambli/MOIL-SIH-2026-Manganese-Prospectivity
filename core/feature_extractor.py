"""
core/feature_extractor.py

Feature engineering engine for MOIL Manganese Prospectivity.
Calculates all 44 numeric features and 7 categorical geological features
required by the Phase 4B pipeline and future India-wide models.

Strict Leakage Prevention:
- NEVER uses Distance_Mn as a model feature.
- NEVER uses latitude or longitude as model features.
- Uses exact numerical definitions matching Phase 4B schema.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from core.geology_index import get_geology_indexer, GeologyIndexer


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe division avoiding zero-division and inf."""
    return a / (b + 1e-6)


class ProspectivityFeatureExtractor:
    """
    Transforms raw satellite bands, radar, terrain, and coordinates
    into the exact 51-column raw feature dataframe required by the preprocessor.
    """

    NUMERIC_COLUMNS = [
        "Aspect", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
        "Elevation", "NDVI", "NIR_Red_Ratio", "Red_Green_Ratio", "SWIR_NIR_Ratio",
        "SWIR_Ratio", "Slope", "VH", "VV", "VV_VH_Difference", "NDMI", "NBR", "NDRE",
        "Iron_Oxide_Index", "Clay_Alteration_Index", "Ferrous_Index", "SWIR_Red_Ratio",
        "SWIR_Green_Ratio", "NIR_SWIR2_Ratio", "BSI", "B11_B12_NormDiff", "B8_B12_NormDiff",
        "B4_B2_NormDiff", "Sausar_Group_Proxy", "Metamorphic_Host_Proxy", "Geology_Unknown",
        "Geo_Boundary_Distance_km", "Geo_Boundary_Density_1km", "Geo_Boundary_Density_3km",
        "Lithology_Diversity_3km", "Formation_Diversity_3km", "Aspect_Sin", "Aspect_Cos"
    ]

    CATEGORICAL_COLUMNS = [
        "geo_age", "geo_supergroup", "geo_group", "geo_formation",
        "geo_lithology", "geo_intrusive", "geo_stratigraphy"
    ]

    def __init__(self, indexer: Optional[GeologyIndexer] = None):
        self.indexer = indexer or get_geology_indexer()

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame containing at least:
        ['latitude', 'longitude']
        and optionally satellite/terrain bands ['B2'..'B12', 'VV', 'VH', 'Elevation', 'Slope', 'Aspect'],
        and adds all structural, spectral, radar, and geological features.
        """
        df = df.copy()

        # 1. Attach Geology & Structural Features if not already present
        if "Geo_Boundary_Distance_km" not in df.columns or "geo_group" not in df.columns:
            geo_ages = []
            geo_supergroups = []
            geo_groups = []
            geo_formations = []
            geo_lithologies = []
            geo_intrusives = []
            geo_stratigraphies = []

            boundary_dist = []
            boundary_dens_1km = []
            boundary_dens_3km = []
            lith_div = []
            form_div = []
            sausar_proxies = []
            metamorphic_proxies = []
            geology_unknowns = []

            for _, row in df.iterrows():
                lat = float(row["latitude"])
                lon = float(row["longitude"])

                geo = self.indexer.query_geology(lat, lon)
                struct = self.indexer.query_structural_features(lat, lon)

                geo_ages.append(geo["geo_age"])
                geo_supergroups.append(geo["geo_supergroup"])
                geo_groups.append(geo["geo_group"])
                geo_formations.append(geo["geo_formation"])
                geo_lithologies.append(geo["geo_lithology"])
                geo_intrusives.append(geo["geo_intrusive"])
                geo_stratigraphies.append(geo["geo_stratigraphy"])

                boundary_dist.append(struct["Geo_Boundary_Distance_km"])
                boundary_dens_1km.append(struct["Geo_Boundary_Density_1km"])
                boundary_dens_3km.append(struct["Geo_Boundary_Density_3km"])
                lith_div.append(struct["Lithology_Diversity_3km"])
                form_div.append(struct["Formation_Diversity_3km"])
                sausar_proxies.append(struct["Sausar_Group_Proxy"])
                metamorphic_proxies.append(struct["Metamorphic_Host_Proxy"])
                geology_unknowns.append(struct["Geology_Unknown"])

            df["geo_age"] = geo_ages
            df["geo_supergroup"] = geo_supergroups
            df["geo_group"] = geo_groups
            df["geo_formation"] = geo_formations
            df["geo_lithology"] = geo_lithologies
            df["geo_intrusive"] = geo_intrusives
            df["geo_stratigraphy"] = geo_stratigraphies

            df["Geo_Boundary_Distance_km"] = boundary_dist
            df["Geo_Boundary_Density_1km"] = boundary_dens_1km
            df["Geo_Boundary_Density_3km"] = boundary_dens_3km
            df["Lithology_Diversity_3km"] = lith_div
            df["Formation_Diversity_3km"] = form_div
            df["Sausar_Group_Proxy"] = sausar_proxies
            df["Metamorphic_Host_Proxy"] = metamorphic_proxies
            df["Geology_Unknown"] = geology_unknowns

        # 2. Aspect Circular Features
        if "Aspect" in df.columns:
            aspect_rad = np.deg2rad(df["Aspect"].fillna(0).astype(float))
            df["Aspect_Sin"] = np.sin(aspect_rad)
            df["Aspect_Cos"] = np.cos(aspect_rad)

        # 3. Radar Differences
        if "VV" in df.columns and "VH" in df.columns:
            if "VV_VH_Difference" not in df.columns:
                df["VV_VH_Difference"] = df["VV"] - df["VH"]

        # 4. Spectral Indices & Ratios (Phase 4B.3)
        if "B8" in df.columns and "B4" in df.columns and "NDVI" not in df.columns:
            df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + 1e-6)

        if "B8" in df.columns and "B11" in df.columns and "NDMI" not in df.columns:
            df["NDMI"] = (df["B8"] - df["B11"]) / (df["B8"] + df["B11"] + 1e-6)

        if "B8" in df.columns and "B12" in df.columns and "NBR" not in df.columns:
            df["NBR"] = (df["B8"] - df["B12"]) / (df["B8"] + df["B12"] + 1e-6)

        if "B8A" in df.columns and "B5" in df.columns and "NDRE" not in df.columns:
            df["NDRE"] = (df["B8A"] - df["B5"]) / (df["B8A"] + df["B5"] + 1e-6)

        if "B4" in df.columns and "B2" in df.columns and "Iron_Oxide_Index" not in df.columns:
            df["Iron_Oxide_Index"] = safe_ratio(df["B4"], df["B2"])

        if "B11" in df.columns and "B12" in df.columns and "Clay_Alteration_Index" not in df.columns:
            df["Clay_Alteration_Index"] = safe_ratio(df["B11"], df["B12"])

        if "B12" in df.columns and "B8" in df.columns and "Ferrous_Index" not in df.columns:
            df["Ferrous_Index"] = safe_ratio(df["B12"], df["B8"])

        if "B11" in df.columns and "B4" in df.columns and "SWIR_Red_Ratio" not in df.columns:
            df["SWIR_Red_Ratio"] = safe_ratio(df["B11"], df["B4"])

        if "B11" in df.columns and "B3" in df.columns and "SWIR_Green_Ratio" not in df.columns:
            df["SWIR_Green_Ratio"] = safe_ratio(df["B11"], df["B3"])

        if "B8" in df.columns and "B12" in df.columns and "NIR_SWIR2_Ratio" not in df.columns:
            df["NIR_SWIR2_Ratio"] = safe_ratio(df["B8"], df["B12"])

        if "B8" in df.columns and "B4" in df.columns and "NIR_Red_Ratio" not in df.columns:
            df["NIR_Red_Ratio"] = safe_ratio(df["B8"], df["B4"])

        if "B11" in df.columns and "B8" in df.columns and "SWIR_NIR_Ratio" not in df.columns:
            df["SWIR_NIR_Ratio"] = safe_ratio(df["B11"], df["B8"])

        if "B11" in df.columns and "B12" in df.columns and "SWIR_Ratio" not in df.columns:
            df["SWIR_Ratio"] = safe_ratio(df["B11"], df["B12"])

        if "B4" in df.columns and "B3" in df.columns and "Red_Green_Ratio" not in df.columns:
            df["Red_Green_Ratio"] = safe_ratio(df["B4"], df["B3"])

        if all(k in df.columns for k in ["B11", "B4", "B8", "B2"]) and "BSI" not in df.columns:
            df["BSI"] = (
                ((df["B11"] + df["B4"]) - (df["B8"] + df["B2"])) /
                ((df["B11"] + df["B4"]) + (df["B8"] + df["B2"]) + 1e-6)
            )

        if "B11" in df.columns and "B12" in df.columns and "B11_B12_NormDiff" not in df.columns:
            df["B11_B12_NormDiff"] = (df["B11"] - df["B12"]) / (df["B11"] + df["B12"] + 1e-6)

        if "B8" in df.columns and "B12" in df.columns and "B8_B12_NormDiff" not in df.columns:
            df["B8_B12_NormDiff"] = (df["B8"] - df["B12"]) / (df["B8"] + df["B12"] + 1e-6)

        if "B4" in df.columns and "B2" in df.columns and "B4_B2_NormDiff" not in df.columns:
            df["B4_B2_NormDiff"] = (df["B4"] - df["B2"]) / (df["B4"] + df["B2"] + 1e-6)

        return df

    def get_model_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame containing exactly the 44 numeric and 7 categorical
        columns required as input to phase4b_preprocessor.joblib.
        """
        enriched = self.enrich_dataframe(df)

        # Check required columns
        for col in self.NUMERIC_COLUMNS:
            if col not in enriched.columns:
                enriched[col] = np.nan

        for col in self.CATEGORICAL_COLUMNS:
            if col not in enriched.columns:
                enriched[col] = "UNKNOWN"
            else:
                enriched[col] = enriched[col].fillna("UNKNOWN").astype(str)

        all_cols = self.NUMERIC_COLUMNS + self.CATEGORICAL_COLUMNS
        return enriched[all_cols].copy()

