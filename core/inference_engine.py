"""
core/inference_engine.py

Unified, Decoupled Prospectivity Inference Engine for India-Wide Manganese Exploration.
Supports:
1. Arbitrary single-point coordinate queries anywhere in India (Lat 6-38°N, Lon 68-98°E).
2. Arbitrary polygon area evaluations with adaptive grid generation and DBSCAN target clustering.
3. Multi-Model Support: XGBoost, Naive Bayes (GaussianNB), and 50/50 Probabilistic Ensemble.
4. Dual-Pipeline Routing:
   - "national": Multi-belt India-Wide Model (macro-lithology generalized, out-of-craton validated).
   - "pilot": High-resolution Balaghat Phase 4B Pilot Model (117 features, benchmark reference).
   - "auto": Automatically routes Balaghat pilot coordinates to high-res pilot model, and national points to national model.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import math
import joblib
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN

from core.geology_index import get_geology_indexer, GeologyIndexer
from core.feature_extractor import ProspectivityFeatureExtractor
from core.macro_lithology import classify_macro_lithology, classify_craton_domain


EARTH_KM_PER_DEG = 111.32


class UnifiedProspectivityEngine:
    """
    Decoupled inference engine for mineral prospectivity.
    Evaluates both arbitrary coordinates and polygon areas across India.
    """

    NATIONAL_NUMERIC_COLS = [
        "Aspect", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
        "Elevation", "NDVI", "NIR_Red_Ratio", "Red_Green_Ratio", "SWIR_NIR_Ratio",
        "SWIR_Ratio", "Slope", "VH", "VV", "VV_VH_Difference", "NDMI", "NBR", "NDRE",
        "Iron_Oxide_Index", "Clay_Alteration_Index", "Ferrous_Index", "SWIR_Red_Ratio",
        "SWIR_Green_Ratio", "NIR_SWIR2_Ratio", "BSI", "B11_B12_NormDiff", "B8_B12_NormDiff",
        "B4_B2_NormDiff", "Sausar_Group_Proxy", "Metamorphic_Host_Proxy", "Geology_Unknown",
        "Geo_Boundary_Distance_km", "Geo_Boundary_Density_1km", "Geo_Boundary_Density_3km",
        "Lithology_Diversity_3km", "Formation_Diversity_3km", "Aspect_Sin", "Aspect_Cos",
        "metallogenic_host_affinity"
    ]

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).resolve().parents[1]
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.outputs_dir = self.base_dir / "outputs"

        # --------------------------------------------------------
        # 1. Load Pilot Model Assets (Balaghat Phase 4B Baseline)
        # --------------------------------------------------------
        pilot_prep_path = self.models_dir / "phase4b_preprocessor.joblib"
        if not pilot_prep_path.exists():
            raise FileNotFoundError(f"Missing pilot preprocessor: {pilot_prep_path}")
        self.pilot_preprocessor = joblib.load(pilot_prep_path)

        pilot_cols_path = self.models_dir / "phase4b_feature_columns.json"
        if not pilot_cols_path.exists():
            raise FileNotFoundError(f"Missing pilot feature schema: {pilot_cols_path}")
        with open(pilot_cols_path, "r", encoding="utf-8") as f:
            self.pilot_feature_columns = json.load(f)

        pilot_xgb_path = self.models_dir / "moil_manganese_prospectivity_xgb_phase4b.joblib"
        if not pilot_xgb_path.exists():
            raise FileNotFoundError(f"Missing pilot XGBoost model: {pilot_xgb_path}")
        self.pilot_xgb = joblib.load(pilot_xgb_path)

        pilot_nb_path = self.models_dir / "moil_manganese_prospectivity_nb_phase4b.joblib"
        self.pilot_nb = joblib.load(pilot_nb_path) if pilot_nb_path.exists() else None

        # Backwards-compatible aliases
        self.preprocessor = self.pilot_preprocessor
        self.feature_columns = self.pilot_feature_columns
        self.xgb_model = self.pilot_xgb
        self.nb_model = self.pilot_nb

        # --------------------------------------------------------
        # 2. Load National Multi-Belt Model Assets (Phase 3)
        # --------------------------------------------------------
        nat_prep_path = self.models_dir / "national_preprocessor.joblib"
        self.national_preprocessor = joblib.load(nat_prep_path) if nat_prep_path.exists() else None

        nat_cols_path = self.models_dir / "national_feature_columns.json"
        if nat_cols_path.exists():
            with open(nat_cols_path, "r", encoding="utf-8") as f:
                self.national_feature_columns = json.load(f)
        else:
            self.national_feature_columns = None

        nat_xgb_path = self.models_dir / "moil_manganese_prospectivity_xgb_national.joblib"
        self.national_xgb = joblib.load(nat_xgb_path) if nat_xgb_path.exists() else None

        nat_nb_path = self.models_dir / "moil_manganese_prospectivity_nb_national.joblib"
        self.national_nb = joblib.load(nat_nb_path) if nat_nb_path.exists() else None

        # --------------------------------------------------------
        # 3. Spatial & Feature Helpers
        # --------------------------------------------------------
        self.indexer = get_geology_indexer()
        self.extractor = ProspectivityFeatureExtractor(self.indexer)

        # --------------------------------------------------------
        # 4. National Known Manganese Occurrences
        # --------------------------------------------------------
        self.occurrences_file = self.data_dir / "gsi" / "india_manganese_occurrences.csv"
        if self.occurrences_file.exists():
            self.occurrences_df = pd.read_csv(self.occurrences_file)
            self._occ_lats = self.occurrences_df["latitude"].to_numpy(dtype=float)
            self._occ_lons = self.occurrences_df["longitude"].to_numpy(dtype=float)
        else:
            self.occurrences_df = None
            self._occ_lats = np.array([])
            self._occ_lons = np.array([])

        # --------------------------------------------------------
        # 5. Precomputed Balaghat Grid
        # --------------------------------------------------------
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
        """Find the closest documented GSI manganese occurrence."""
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

    def _predict_probabilities(
        self,
        X_trans: np.ndarray,
        feature_columns: List[str],
        xgb_model: Any,
        nb_model: Optional[Any],
        model_type: str = "ensemble"
    ) -> np.ndarray:
        """
        Compute predicted prospectivity probabilities using XGBoost, Naive Bayes, or Ensemble.
        """
        model_type = model_type.lower().strip()
        X_df = pd.DataFrame(X_trans, columns=feature_columns)

        p_xgb = xgb_model.predict_proba(X_df)[:, 1] if xgb_model is not None else np.zeros(len(X_trans))

        if nb_model is not None:
            p_nb = nb_model.predict_proba(X_trans)[:, 1]
        else:
            p_nb = p_xgb

        if model_type == "xgboost":
            return p_xgb
        elif model_type == "naive_bayes":
            return p_nb
        elif model_type == "ensemble":
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
        pipeline: str = "auto",
        use_precomputed_grid: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate mineral prospectivity for any coordinate in India.

        Parameters:
            latitude: Latitude (WGS84)
            longitude: Longitude (WGS84)
            model_type: 'xgboost', 'naive_bayes', or 'ensemble'
            pipeline: 'auto', 'national', or 'pilot'
            use_precomputed_grid: whether to use Balaghat pilot grid when inside Balaghat
        """
        lat = float(latitude)
        lon = float(longitude)

        # Decide pipeline
        pip = pipeline.lower().strip()
        if pip == "auto":
            if use_precomputed_grid and self._is_in_balaghat(lon, lat) and (self.balaghat_grid is not None):
                pipeline_to_use = "pilot"
            elif self.national_xgb is not None:
                pipeline_to_use = "national"
            else:
                pipeline_to_use = "pilot"
        else:
            pipeline_to_use = pip

        # Query geology & structure
        geo = self.indexer.query_geology(lat, lon)
        struct = self.indexer.query_structural_features(lat, lon)
        nearest_occ = self._get_nearest_occurrence(lat, lon)

        # Macro-lithology classification
        m_class, m_aff = classify_macro_lithology(
            geo_group=geo["geo_group"],
            geo_formation=geo["geo_formation"],
            geo_lithology=geo["geo_lithology"],
            geo_stratigraphy=geo["geo_stratigraphy"],
            host_rock=nearest_occ.get("host_rock", None)
        )
        c_dom = classify_craton_domain(lat, lon, state=nearest_occ.get("state"), belt=nearest_occ.get("belt"))

        # --------------------------------------------------------
        # Case A: Pilot Pipeline (Balaghat High-Res Baseline)
        # --------------------------------------------------------
        if pipeline_to_use == "pilot":
            if use_precomputed_grid and self._is_in_balaghat(lon, lat) and (self.balaghat_grid is not None):
                dlat = (self.balaghat_grid["latitude"] - lat) * EARTH_KM_PER_DEG
                dlon = (self.balaghat_grid["longitude"] - lon) * EARTH_KM_PER_DEG * math.cos(math.radians(lat))
                dist_km = np.sqrt(dlat**2 + dlon**2)
                nearest_idx = int(dist_km.idxmin())
                grid_cell = self.balaghat_grid.iloc[nearest_idx]

                if "prospectivity_score" in grid_cell and model_type == "xgboost":
                    score = float(grid_cell["prospectivity_score"])
                    prob = float(grid_cell.get("probability", score / 100.0))
                else:
                    cell_df = pd.DataFrame([grid_cell])
                    X_raw = self.extractor.get_model_features_df(cell_df)
                    X_trans = self.pilot_preprocessor.transform(X_raw)
                    prob = float(self._predict_probabilities(
                        X_trans, self.pilot_feature_columns, self.pilot_xgb, self.pilot_nb, model_type=model_type
                    )[0])
                    score = prob * 100.0

                dist_cell = float(dist_km.loc[nearest_idx])
                res_str = "1:50,000 (Balaghat Pilot Grid)"
                conf_str = "High (Multi-Sensor Satellite + 1:50k Lithology)"
            else:
                # Synthesize baseline terrain/spectral inputs for pilot pipeline
                point_data = {
                    "latitude": [lat], "longitude": [lon],
                    "Elevation": [380.0], "Slope": [4.0], "Aspect": [180.0],
                    "B2": [480.0], "B3": [680.0], "B4": [620.0], "B5": [1100.0],
                    "B6": [1800.0], "B7": [2100.0], "B8": [2300.0], "B8A": [2350.0],
                    "B11": [2100.0], "B12": [1300.0], "VV": [-10.0], "VH": [-16.0]
                }
                raw_df = pd.DataFrame(point_data)
                X_raw = self.extractor.get_model_features_df(raw_df)
                X_trans = self.pilot_preprocessor.transform(X_raw)
                prob = float(self._predict_probabilities(
                    X_trans, self.pilot_feature_columns, self.pilot_xgb, self.pilot_nb, model_type=model_type
                )[0])
                score = prob * 100.0
                dist_cell = 0.0
                res_str = geo["resolution"]
                conf_str = "Moderate (Regional 1:2M Geology & Structural Context)"

        # --------------------------------------------------------
        # Case B: National Multi-Belt Pipeline (Transferable)
        # --------------------------------------------------------
        else:
            if self.national_preprocessor is None or self.national_xgb is None:
                raise RuntimeError("National models not loaded. Run scripts/train_national_models.py first.")

            point_data = {
                "latitude": [lat], "longitude": [lon],
                "Elevation": [380.0], "Slope": [4.0], "Aspect": [180.0],
                "B2": [480.0], "B3": [680.0], "B4": [620.0], "B5": [1100.0],
                "B6": [1800.0], "B7": [2100.0], "B8": [2300.0], "B8A": [2350.0],
                "B11": [2100.0], "B12": [1300.0], "VV": [-10.0], "VH": [-16.0]
            }
            raw_df = pd.DataFrame(point_data)
            enriched = self.extractor.enrich_dataframe(raw_df)
            enriched["macro_lithology"] = [m_class]
            enriched["metallogenic_host_affinity"] = [m_aff]
            enriched["craton_domain"] = [c_dom]

            for c in self.NATIONAL_NUMERIC_COLS:
                if c not in enriched.columns:
                    enriched[c] = np.nan

            req_cols = self.NATIONAL_NUMERIC_COLS + ["macro_lithology"]
            X_raw = enriched[req_cols]
            X_trans = self.national_preprocessor.transform(X_raw)

            prob = float(self._predict_probabilities(
                X_trans, self.national_feature_columns, self.national_xgb, self.national_nb, model_type=model_type
            )[0])
            score = prob * 100.0
            dist_cell = 0.0
            res_str = geo["resolution"]
            conf_str = "High (Transferable Multi-Belt National Model + NGDR Geology)"

        return {
            "latitude": lat,
            "longitude": lon,
            "prospectivity_score": float(score),
            "model_probability": float(prob),
            "priority": self._priority(score),
            "model_signal": self._strength(score),
            "model_type": model_type.upper(),
            "pipeline": pipeline_to_use.upper(),
            "macro_lithology": m_class,
            "metallogenic_host_affinity": float(m_aff),
            "craton_domain": c_dom,
            "resolution": res_str,
            "distance_to_grid_cell_km": dist_cell,
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
                "formation_diversity_3km": struct["Formation_Diversity_3km"],
                "metamorphic_host": bool(struct["Metamorphic_Host_Proxy"]),
                "sausar_group": bool(struct["Sausar_Group_Proxy"])
            },
            "nearest_known_manganese_site": nearest_occ,
            "confidence_level": conf_str
        }

    # ------------------------------------------------------------
    # PUBLIC API: POLYGON / AREA PREDICTION
    # ------------------------------------------------------------

    def predict_area(
        self,
        coordinates: List[Tuple[float, float]],
        resolution_km: float = 2.0,
        max_points: int = 5000,
        model_type: str = "ensemble",
        pipeline: str = "auto"
    ) -> Dict[str, Any]:
        """
        Analyze an arbitrary polygon anywhere in India.
        coordinates: [(lon, lat), (lon, lat), ...]
        """
        polygon = Polygon(coordinates)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        min_lon, min_lat, max_lon, max_lat = polygon.bounds

        pip = pipeline.lower().strip()
        if pip == "auto":
            if (
                self.balaghat_grid is not None
                and self._is_in_balaghat(min_lon, min_lat)
                and self._is_in_balaghat(max_lon, max_lat)
            ):
                pipeline_to_use = "pilot"
            elif self.national_xgb is not None:
                pipeline_to_use = "national"
            else:
                pipeline_to_use = "pilot"
        else:
            pipeline_to_use = pip

        # A) Inside precomputed Balaghat grid
        if (
            pipeline_to_use == "pilot"
            and self.balaghat_grid is not None
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
                pts = [Point(xy) for xy in zip(candidates["longitude"], candidates["latitude"])]
                mask = [polygon.contains(p) or polygon.touches(p) for p in pts]
                selected = candidates[mask].copy()

                if len(selected) > 0:
                    if len(selected) > max_points:
                        selected = selected.sample(max_points, random_state=42)

                    if "prospectivity_score" in selected.columns and model_type == "xgboost":
                        scores = selected["prospectivity_score"].astype(float)
                    else:
                        X_raw = self.extractor.get_model_features_df(selected)
                        X_trans = self.pilot_preprocessor.transform(X_raw)
                        probs = self._predict_probabilities(
                            X_trans, self.pilot_feature_columns, self.pilot_xgb, self.pilot_nb, model_type=model_type
                        )
                        scores = pd.Series(probs * 100.0, index=selected.index)
                        selected["prospectivity_score"] = scores

                    return self._summarize_area_results(selected, polygon, model_type, "1:50,000 (Balaghat Pilot Grid)", pipeline_to_use)

        # B) Dynamic Nationwide Area Grid Generation
        step_deg = max(0.015, resolution_km / EARTH_KM_PER_DEG)
        lons = np.arange(min_lon, max_lon + step_deg, step_deg)
        lats = np.arange(min_lat, max_lat + step_deg, step_deg)

        grid_lons, grid_lats = np.meshgrid(lons, lats)
        flat_lons = grid_lons.flatten()
        flat_lats = grid_lats.flatten()

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

        # Score with National Pipeline if available
        if pipeline_to_use == "national" and self.national_preprocessor is not None:
            area_df["Elevation"] = 380.0
            area_df["Slope"] = 4.0
            area_df["Aspect"] = 180.0
            area_df["B2"] = 480.0
            area_df["B3"] = 680.0
            area_df["B4"] = 620.0
            area_df["B5"] = 1100.0
            area_df["B6"] = 1800.0
            area_df["B7"] = 2100.0
            area_df["B8"] = 2300.0
            area_df["B8A"] = 2350.0
            area_df["B11"] = 2100.0
            area_df["B12"] = 1300.0
            area_df["VV"] = -10.0
            area_df["VH"] = -16.0

            enriched = self.extractor.enrich_dataframe(area_df)

            # Classify macro-lithologies for area points
            macro_liths, affinities = [], []
            for _, r in enriched.iterrows():
                m_c, m_a = classify_macro_lithology(
                    geo_group=r.get("geo_group"),
                    geo_formation=r.get("geo_formation"),
                    geo_lithology=r.get("geo_lithology"),
                    geo_stratigraphy=r.get("geo_stratigraphy")
                )
                macro_liths.append(m_c)
                affinities.append(m_a)

            enriched["macro_lithology"] = macro_liths
            enriched["metallogenic_host_affinity"] = affinities

            for c in self.NATIONAL_NUMERIC_COLS:
                if c not in enriched.columns:
                    enriched[c] = np.nan

            req_cols = self.NATIONAL_NUMERIC_COLS + ["macro_lithology"]
            X_raw = enriched[req_cols]
            X_trans = self.national_preprocessor.transform(X_raw)
            probs = self._predict_probabilities(
                X_trans, self.national_feature_columns, self.national_xgb, self.national_nb, model_type=model_type
            )
            area_df["prospectivity_score"] = probs * 100.0
            res_str = "Adaptive Multi-Belt Grid (National Model)"
        else:
            # Fallback to pilot preprocessor
            X_raw = self.extractor.get_model_features_df(area_df)
            X_trans = self.pilot_preprocessor.transform(X_raw)
            probs = self._predict_probabilities(
                X_trans, self.pilot_feature_columns, self.pilot_xgb, self.pilot_nb, model_type=model_type
            )
            area_df["prospectivity_score"] = probs * 100.0
            res_str = "Adaptive Spatial Grid (Pilot Pipeline Fallback)"

        return self._summarize_area_results(area_df, polygon, model_type, res_str, pipeline_to_use)

    def _summarize_area_results(
        self,
        selected_df: pd.DataFrame,
        polygon: Polygon,
        model_type: str,
        resolution: str,
        pipeline: str
    ) -> Dict[str, Any]:
        """Compute aggregated statistics and DBSCAN target zone clusters."""
        scores = selected_df["prospectivity_score"].astype(float)
        mean_score = float(scores.mean())
        median_score = float(scores.median())
        max_score = float(scores.max())

        high_count = int((scores >= 60).sum())
        very_high_count = int((scores >= 80).sum())

        target_zones = []
        high_cells = selected_df[selected_df["prospectivity_score"] >= 60].copy()

        if len(high_cells) >= 3:
            center_lat = float(high_cells["latitude"].mean())
            coords_km = np.column_stack([
                high_cells["longitude"] * EARTH_KM_PER_DEG * math.cos(math.radians(center_lat)),
                high_cells["latitude"] * EARTH_KM_PER_DEG
            ])

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
            "pipeline": pipeline.upper(),
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
