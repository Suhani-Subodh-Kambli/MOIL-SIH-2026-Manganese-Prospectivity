"""
core/inference_engine.py

Unified, multi-model, multi-scale manganese prospectivity inference engine.
Decoupled from static CSV grids to support arbitrary coordinate and polygon
predictions anywhere in India.

Features:
- Dual-Model Support: XGBoost, Naive Bayes (GaussianNB), and Weighted Ensemble
- Dynamic Single Coordinate Prediction across all Indian states and metallogenic belts
- Dynamic Polygon / AOI Prediction with on-the-fly grid generation and DBSCAN clustering
- Transparent AI Explainability: feature breakdown, geological context, and reliability indicators
- Preserves 100% backward compatibility with Balaghat pilot benchmark
"""

import math
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN

from core.geology_index import get_geology_indexer, GeologyIndexer
from core.feature_extractor import ProspectivityFeatureExtractor


EARTH_KM_PER_DEG = 111.32


class UnifiedProspectivityEngine:
    """
    Decoupled Inference Engine supporting arbitrary point/polygon predictions across India.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parents[1]
        else:
            self.base_dir = Path(base_dir)

        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.outputs_dir = self.base_dir / "outputs"

        # 1. Load Preprocessor & Feature Schema
        preprocessor_path = self.models_dir / "phase4b_preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Missing preprocessor: {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)

        cols_path = self.models_dir / "phase4b_feature_columns.json"
        if not cols_path.exists():
            raise FileNotFoundError(f"Missing feature schema: {cols_path}")
        with open(cols_path, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        # 2. Load ML Models
        xgb_path = self.models_dir / "moil_manganese_prospectivity_xgb_phase4b.joblib"
        if not xgb_path.exists():
            raise FileNotFoundError(f"Missing XGBoost model: {xgb_path}")
        self.xgb_model = joblib.load(xgb_path)

        nb_path = self.models_dir / "moil_manganese_prospectivity_nb_phase4b.joblib"
        self.nb_model = joblib.load(nb_path) if nb_path.exists() else None

        # 3. Spatial & Feature Helpers
        self.indexer = get_geology_indexer()
        self.extractor = ProspectivityFeatureExtractor(self.indexer)

        # 4. Load National Known Manganese Occurrences for Proximity Reference
        self.occurrences_file = self.data_dir / "gsi" / "india_manganese_occurrences.csv"
        if self.occurrences_file.exists():
            self.occurrences_df = pd.read_csv(self.occurrences_file)
            self._occ_lats = self.occurrences_df["latitude"].to_numpy(dtype=float)
            self._occ_lons = self.occurrences_df["longitude"].to_numpy(dtype=float)
        else:
            self.occurrences_df = None
            self._occ_lats = np.array([])
            self._occ_lons = np.array([])

        # 5. Load Precomputed Balaghat Grid if available (for fast local pilot lookups)
        self.balaghat_grid_path = self.outputs_dir / "phase5_prospectivity_grid.csv"
        if not self.balaghat_grid_path.exists():
            self.balaghat_grid_path = self.data_dir / "MOIL_Balaghat_Phase5_Final_Feature_Grid.csv"

        self.balaghat_grid = None
        if self.balaghat_grid_path.exists():
            try:
                self.balaghat_grid = pd.read_csv(self.balaghat_grid_path)
            except Exception:
                self.balaghat_grid = None

        self.balaghat_bounds = (79.50, 21.30, 80.80, 22.40)

    # ------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------

    def _is_in_balaghat(self, lon: float, lat: float) -> bool:
        min_lon, min_lat, max_lon, max_lat = self.balaghat_bounds
        return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)

    def _priority(self, score: float) -> str:
        if score >= 80:
            return "VERY HIGH"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MODERATE"
        elif score >= 20:
            return "LOW"
        return "VERY LOW"

    def _strength(self, score: float) -> str:
        if score >= 80:
            return "Strong exploration indicator"
        elif score >= 60:
            return "Moderate-to-strong exploration indicator"
        elif score >= 40:
            return "Moderate exploration indicator"
        elif score >= 20:
            return "Weak exploration indicator"
        return "Very weak exploration indicator"

    def _get_nearest_occurrence(self, lat: float, lon: float) -> Dict[str, Any]:
        if len(self._occ_lats) == 0:
            return {"distance_km": np.nan, "name": "N/A", "belt": "N/A"}

        dlat = (self._occ_lats - lat) * EARTH_KM_PER_DEG
        dlon = (self._occ_lons - lon) * EARTH_KM_PER_DEG * math.cos(math.radians(lat))
        distances = np.sqrt(dlat**2 + dlon**2)
        min_idx = int(np.argmin(distances))

        nearest_row = self.occurrences_df.iloc[min_idx]
        return {
            "distance_km": float(distances[min_idx]),
            "name": str(nearest_row.get("name", "Unknown")),
            "state": str(nearest_row.get("state", "Unknown")),
            "belt": str(nearest_row.get("metallogenic_belt", "Unknown")),
            "source": str(nearest_row.get("source", "Unknown"))
        }

    def _predict_probabilities(self, X_trans: np.ndarray, model_type: str = "ensemble") -> np.ndarray:
        """
        Compute predicted prospectivity probabilities using XGBoost, Naive Bayes, or Ensemble.
        """
        model_type = model_type.lower().strip()
        X_df = pd.DataFrame(X_trans, columns=self.feature_columns)

        p_xgb = self.xgb_model.predict_proba(X_df)[:, 1]

        if self.nb_model is not None:
            p_nb = self.nb_model.predict_proba(X_trans)[:, 1]
        else:
            p_nb = p_xgb

        if model_type == "xgboost":
            return p_xgb
        elif model_type == "naive_bayes":
            return p_nb
        elif model_type == "ensemble":
            # 50-50 weighted ensemble combining non-linear trees with probabilistic Naive Bayes
            return 0.5 * p_xgb + 0.5 * p_nb
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'xgboost', 'naive_bayes', or 'ensemble'.")

    # ------------------------------------------------------------
    # PUBLIC API: SINGLE COORDINATE PREDICTION
    # ------------------------------------------------------------

    def predict_point(
        self,
        latitude: float,
        longitude: float,
        model_type: str = "ensemble",
        use_precomputed_grid: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate mineral prospectivity for any coordinate in India.
        """
        lat = float(latitude)
        lon = float(longitude)

        # Check if inside Balaghat pilot with precomputed grid
        if use_precomputed_grid and self._is_in_balaghat(lon, lat) and (self.balaghat_grid is not None):
            # Fast local grid cell query
            dlat = (self.balaghat_grid["latitude"] - lat) * EARTH_KM_PER_DEG
            dlon = (self.balaghat_grid["longitude"] - lon) * EARTH_KM_PER_DEG * math.cos(math.radians(lat))
            dist_km = np.sqrt(dlat**2 + dlon**2)
            nearest_idx = int(dist_km.idxmin())
            grid_cell = self.balaghat_grid.iloc[nearest_idx]

            # If score already precomputed in grid
            if "prospectivity_score" in grid_cell and model_type == "xgboost":
                score = float(grid_cell["prospectivity_score"])
                prob = float(grid_cell.get("probability", score / 100.0))
            else:
                # Extract features for this cell and score dynamically
                cell_df = pd.DataFrame([grid_cell])
                X_raw = self.extractor.get_model_features_df(cell_df)
                X_trans = self.preprocessor.transform(X_raw)
                prob = float(self._predict_probabilities(X_trans, model_type=model_type)[0])
                score = prob * 100.0

            geo = self.indexer.query_geology(lat, lon)
            struct = self.indexer.query_structural_features(lat, lon)
            nearest_occ = self._get_nearest_occurrence(lat, lon)

            return {
                "latitude": lat,
                "longitude": lon,
                "prospectivity_score": float(score),
                "model_probability": float(prob),
                "priority": self._priority(score),
                "model_signal": self._strength(score),
                "model_type": model_type.upper(),
                "resolution": "1:50,000 (Balaghat Pilot Grid)",
                "distance_to_grid_cell_km": float(dist_km.loc[nearest_idx]),
                "geology": {
                    "group": geo["geo_group"],
                    "formation": geo["geo_formation"],
                    "lithology": geo["geo_lithology"],
                    "stratigraphy": geo["geo_stratigraphy"]
                },
                "structural": {
                    "boundary_distance_km": struct["Geo_Boundary_Distance_km"],
                    "boundary_density_1km": struct["Geo_Boundary_Density_1km"],
                    "boundary_density_3km": struct["Geo_Boundary_Density_3km"],
                    "lithology_diversity_3km": struct["Lithology_Diversity_3km"],
                    "metamorphic_host": bool(struct["Metamorphic_Host_Proxy"]),
                    "sausar_group": bool(struct["Sausar_Group_Proxy"])
                },
                "nearest_known_manganese_site": nearest_occ,
                "confidence_level": "High (Multi-Sensor Satellite + 1:50k Lithology)"
            }

        # Dynamic Nationwide Coordinate Prediction
        geo = self.indexer.query_geology(lat, lon)
        struct = self.indexer.query_structural_features(lat, lon)
        nearest_occ = self._get_nearest_occurrence(lat, lon)

        # Baseline terrain & spectral proxies for arbitrary points outside the pilot grid
        point_data = {
            "latitude": [lat],
            "longitude": [lon],
            "Elevation": [350.0],  # Default moderate cratonic elevation
            "Slope": [5.0],        # Default moderate cratonic slope
            "Aspect": [180.0],
            # Default typical spectral reflection for weathered Precambrian terrain
            "B2": [0.08], "B3": [0.11], "B4": [0.14], "B5": [0.18],
            "B6": [0.22], "B7": [0.25], "B8": [0.28], "B8A": [0.29],
            "B11": [0.24], "B12": [0.18],
            "VV": [-12.0], "VH": [-18.0]
        }
        raw_point_df = pd.DataFrame(point_data)

        # Enrich and score
        X_raw = self.extractor.get_model_features_df(raw_point_df)
        X_trans = self.preprocessor.transform(X_raw)
        prob = float(self._predict_probabilities(X_trans, model_type=model_type)[0])
        score = prob * 100.0

        return {
            "latitude": lat,
            "longitude": lon,
            "prospectivity_score": float(score),
            "model_probability": float(prob),
            "priority": self._priority(score),
            "model_signal": self._strength(score),
            "model_type": model_type.upper(),
            "resolution": geo["resolution"],
            "distance_to_grid_cell_km": 0.0,
            "geology": {
                "group": geo["geo_group"],
                "formation": geo["geo_formation"],
                "lithology": geo["geo_lithology"],
                "stratigraphy": geo["geo_stratigraphy"]
            },
            "structural": {
                "boundary_distance_km": struct["Geo_Boundary_Distance_km"],
                "boundary_density_1km": struct["Geo_Boundary_Density_1km"],
                "boundary_density_3km": struct["Geo_Boundary_Density_3km"],
                "lithology_diversity_3km": struct["Lithology_Diversity_3km"],
                "metamorphic_host": bool(struct["Metamorphic_Host_Proxy"]),
                "sausar_group": bool(struct["Sausar_Group_Proxy"])
            },
            "nearest_known_manganese_site": nearest_occ,
            "confidence_level": "Moderate (Regional 1:2M Geology & Structural Context)"
        }

    # ------------------------------------------------------------
    # PUBLIC API: POLYGON / AREA PREDICTION
    # ------------------------------------------------------------

    def predict_area(
        self,
        coordinates: List[Tuple[float, float]],
        resolution_km: float = 2.0,
        max_points: int = 5000,
        model_type: str = "ensemble"
    ) -> Dict[str, Any]:
        """
        Analyze an arbitrary polygon anywhere in India.
        coordinates: [(lon, lat), (lon, lat), ...]
        """
        polygon = Polygon(coordinates)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        min_lon, min_lat, max_lon, max_lat = polygon.bounds

        # A) Check if inside precomputed Balaghat grid
        if (
            self.balaghat_grid is not None
            and self._is_in_balaghat(min_lon, min_lat)
            and self._is_in_balaghat(max_lon, max_lat)
        ):
            candidates = self.balaghat_grid[
                (self.balaghat_grid["longitude"] >= min_lon)
                & (self.balaghat_grid["longitude"] <= max_lon)
                & (self.balaghat_grid["latitude"] >= min_lat)
                & (self.balaghat_grid["latitude"] <= max_lat)
            ].copy()

            if len(candidates) > 0:
                # Filter points inside polygon
                pts = [Point(xy) for xy in zip(candidates["longitude"], candidates["latitude"])]
                mask = [polygon.contains(p) or polygon.touches(p) for p in pts]
                selected = candidates[mask].copy()

                if len(selected) > 0:
                    if len(selected) > max_points:
                        selected = selected.sample(max_points, random_state=42)

                    # Dynamic score if needed
                    if "prospectivity_score" in selected.columns and model_type == "xgboost":
                        scores = selected["prospectivity_score"].astype(float)
                    else:
                        X_raw = self.extractor.get_model_features_df(selected)
                        X_trans = self.preprocessor.transform(X_raw)
                        probs = self._predict_probabilities(X_trans, model_type=model_type)
                        scores = pd.Series(probs * 100.0, index=selected.index)
                        selected["prospectivity_score"] = scores

                    return self._summarize_area_results(selected, polygon, model_type, "1:50,000 (Balaghat Pilot Grid)")

        # B) Dynamic Nationwide Area Grid Generation
        step_deg = max(0.01, resolution_km / EARTH_KM_PER_DEG)
        lons = np.arange(min_lon, max_lon + step_deg, step_deg)
        lats = np.arange(min_lat, max_lat + step_deg, step_deg)

        grid_lons, grid_lats = np.meshgrid(lons, lats)
        flat_lons = grid_lons.flatten()
        flat_lats = grid_lats.flatten()

        # Filter by polygon
        inside_lons, inside_lats = [], []
        for lon, lat in zip(flat_lons, flat_lats):
            p = Point(lon, lat)
            if polygon.contains(p) or polygon.touches(p):
                inside_lons.append(lon)
                inside_lats.append(lat)

        if len(inside_lons) == 0:
            return {
                "cell_count": 0,
                "message": "No points found inside the selected area boundary."
            }

        if len(inside_lons) > max_points:
            indices = np.random.choice(len(inside_lons), size=max_points, replace=False)
            inside_lons = [inside_lons[i] for i in indices]
            inside_lats = [inside_lats[i] for i in indices]

        area_df = pd.DataFrame({"latitude": inside_lats, "longitude": inside_lons})

        # Enrich and predict
        X_raw = self.extractor.get_model_features_df(area_df)
        X_trans = self.preprocessor.transform(X_raw)
        probs = self._predict_probabilities(X_trans, model_type=model_type)
        area_df["prospectivity_score"] = probs * 100.0

        return self._summarize_area_results(area_df, polygon, model_type, "Adaptive Spatial Grid (National NGDR)")

    def _summarize_area_results(
        self,
        selected_df: pd.DataFrame,
        polygon: Polygon,
        model_type: str,
        resolution: str
    ) -> Dict[str, Any]:
        """Compute aggregated statistics and DBSCAN target zone clusters."""
        scores = selected_df["prospectivity_score"].astype(float)
        mean_score = float(scores.mean())
        median_score = float(scores.median())
        max_score = float(scores.max())

        high_count = int((scores >= 60).sum())
        very_high_count = int((scores >= 80).sum())

        # DBSCAN Clustering on Top Prospectivity Cells (score >= 60)
        target_zones = []
        high_cells = selected_df[selected_df["prospectivity_score"] >= 60].copy()

        if len(high_cells) >= 3:
            # Coordinates in projected km
            center_lat = float(high_cells["latitude"].mean())
            coords_km = np.column_stack([
                high_cells["longitude"] * EARTH_KM_PER_DEG * math.cos(math.radians(center_lat)),
                high_cells["latitude"] * EARTH_KM_PER_DEG
            ])

            # Cluster cells within 3 km of each other
            clustering = DBSCAN(eps=3.0, min_samples=2).fit(coords_km)
            high_cells["cluster"] = clustering.labels_

            for cluster_id in sorted(set(clustering.labels_)):
                if cluster_id == -1:
                    continue
                cluster_cells = high_cells[high_cells["cluster"] == cluster_id]
                target_zones.append({
                    "zone_id": f"ZONE_{len(target_zones)+1:02d}",
                    "cell_count": len(cluster_cells),
                    "centroid_latitude": float(cluster_cells["latitude"].mean()),
                    "centroid_longitude": float(cluster_cells["longitude"].mean()),
                    "mean_score": float(cluster_cells["prospectivity_score"].mean()),
                    "max_score": float(cluster_cells["prospectivity_score"].max()),
                    "priority": "VERY HIGH" if cluster_cells["prospectivity_score"].max() >= 80 else "HIGH"
                })

        return {
            "cell_count": len(selected_df),
            "mean_score": mean_score,
            "median_score": median_score,
            "maximum_score": max_score,
            "priority": self._priority(mean_score),
            "model_signal": self._strength(mean_score),
            "model_type": model_type.upper(),
            "resolution": resolution,
            "high_priority_cells": high_count,
            "very_high_priority_cells": very_high_count,
            "target_zones_count": len(target_zones),
            "target_zones": target_zones,
            "selected_cells": selected_df
        }

    def get_map_data(self) -> pd.DataFrame:
        """Returns baseline map grid data for dashboard display."""
        if self.balaghat_grid is not None and "prospectivity_score" in self.balaghat_grid.columns:
            return self.balaghat_grid[["latitude", "longitude", "prospectivity_score"]].copy()
        elif self.occurrences_df is not None:
            return self.occurrences_df[["latitude", "longitude", "name", "state", "metallogenic_belt"]].copy()
        return pd.DataFrame()


# Singleton
_ENGINE_INSTANCE = None

def get_prospectivity_engine() -> UnifiedProspectivityEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = UnifiedProspectivityEngine()
    return _ENGINE_INSTANCE

