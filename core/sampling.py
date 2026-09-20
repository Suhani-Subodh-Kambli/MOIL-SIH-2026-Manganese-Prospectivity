"""
core/sampling.py

Spatially balanced Positive-Unlabelled (PU) sampling pipeline
for India-wide manganese prospectivity modeling.

Guarantees:
- Enforces minimum buffer distance (>= 3 km) from known positives for all background samples
- Spatial grid-block grouping (spatial_block) to prevent spatial leakage during CV
- Balanced positive-to-background ratio
- Strict exclusion of Distance_Mn and coordinates as training features
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from shapely.geometry import Point


# ============================================================
# DISTANCE HELPER
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))


class SpatiallyBalancedSampler:
    """
    Constructs spatially separated positive and background samples
    across India's mineralized belts for cross-validation and training.
    """

    def __init__(
        self,
        min_background_distance_km: float = 3.0,
        background_ratio: float = 2.0,
        block_size_deg: float = 0.10,
        seed: int = 42
    ):
        self.min_background_distance_km = min_background_distance_km
        self.background_ratio = background_ratio
        self.block_size_deg = block_size_deg
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def assign_spatial_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign discrete spatial block IDs based on geographic coordinates."""
        df = df.copy()
        lat = df["latitude"].astype(float)
        lon = df["longitude"].astype(float)

        block_lat = (lat // self.block_size_deg) * self.block_size_deg
        block_lon = (lon // self.block_size_deg) * self.block_size_deg

        df["spatial_block"] = (
            "block_"
            + block_lat.round(4).astype(str)
            + "_"
            + block_lon.round(4).astype(str)
        )
        return df

    def filter_background_points(
        self,
        candidate_bg_df: pd.DataFrame,
        positive_df: pd.DataFrame,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter background points to ensure all are >= min_background_distance_km
        from ALL known positive manganese occurrences.
        """
        pos_lats = positive_df["latitude"].to_numpy(dtype=float)
        pos_lons = positive_df["longitude"].to_numpy(dtype=float)

        valid_indices = []
        min_distances = []

        for idx, row in candidate_bg_df.iterrows():
            bg_lat = float(row["latitude"])
            bg_lon = float(row["longitude"])

            dists = haversine_km(bg_lat, bg_lon, pos_lats, pos_lons)
            min_dist = float(np.min(dists))

            if min_dist >= self.min_background_distance_km:
                valid_indices.append(idx)
                min_distances.append(min_dist)

        filtered = candidate_bg_df.loc[valid_indices].copy()
        filtered["nearest_positive_km"] = min_distances
        filtered["label"] = 0

        # Subsample if max_samples is specified
        if max_samples and len(filtered) > max_samples:
            filtered = filtered.sample(n=max_samples, random_state=self.seed).copy()

        return filtered

    def create_balanced_dataset(
        self,
        positive_df: pd.DataFrame,
        candidate_bg_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge positives and filtered background into a single balanced dataset
        with spatial_block grouping.
        """
        pos = positive_df.copy()
        pos["label"] = 1
        pos["nearest_positive_km"] = 0.0

        n_target_bg = int(len(pos) * self.background_ratio)
        bg = self.filter_background_points(
            candidate_bg_df,
            positive_df,
            max_samples=n_target_bg
        )

        combined = pd.concat([pos, bg], ignore_index=True)
        # Assign spatial blocks
        combined = self.assign_spatial_blocks(combined)

        return combined
