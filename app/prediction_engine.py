"""
app/prediction_engine.py

Adapter connecting the Streamlit user interface to the core UnifiedProspectivityEngine.
Provides backward compatibility with the interim prototype while unlocking
arbitrary coordinate prediction, arbitrary polygon analysis, and multi-model support
(XGBoost, Naive Bayes, Ensemble) across all of India.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.inference_engine import get_prospectivity_engine, UnifiedProspectivityEngine


class ProspectivityEngine:
    """
    User-facing Prediction Engine wrapper used by app/app.py.
    """

    def __init__(self):
        self.engine: UnifiedProspectivityEngine = get_prospectivity_engine()

        # For backwards compatibility with direct attribute queries in legacy code
        if self.engine.balaghat_grid is not None:
            self.grid = self.engine.balaghat_grid
            self.lat_col = "latitude" if "latitude" in self.grid.columns else "lat"
            self.lon_col = "longitude" if "longitude" in self.grid.columns else "lon"
            self.score_col = "prospectivity_score" if "prospectivity_score" in self.grid.columns else "score"
        else:
            self.grid = pd.DataFrame(columns=["latitude", "longitude", "prospectivity_score"])
            self.lat_col = "latitude"
            self.lon_col = "longitude"
            self.score_col = "prospectivity_score"

        self.probability_col = "model_probability"

    def predict_point(
        self,
        latitude: float,
        longitude: float,
        model_type: str = "ensemble",
        pipeline: str = "auto"
    ) -> Dict[str, Any]:
        """
        Evaluate prospectivity at (latitude, longitude).
        Supports arbitrary coordinates anywhere in India.
        """
        res = self.engine.predict_point(
            latitude,
            longitude,
            model_type=model_type,
            pipeline=pipeline
        )

        # Ensure compatibility with app.py legacy dictionary keys
        res["grid_latitude"] = res.get("grid_latitude", float(latitude))
        res["grid_longitude"] = res.get("grid_longitude", float(longitude))

        return res

    def predict_area(
        self,
        coordinates: List[Tuple[float, float]],
        max_points: int = 10000,
        model_type: str = "ensemble",
        pipeline: str = "auto"
    ) -> Dict[str, Any]:
        """
        Evaluate prospectivity within an arbitrary polygon [(lon, lat), ...].
        """
        return self.engine.predict_area(
            coordinates,
            resolution_km=2.0,
            max_points=max_points,
            model_type=model_type,
            pipeline=pipeline
        )

    def get_map_data(self) -> pd.DataFrame:
        """
        Returns map grid data for dashboard display.
        """
        data = self.engine.get_map_data()
        if "prospectivity_score" not in data.columns and "name" in data.columns:
            data["prospectivity_score"] = 85.0
        return data