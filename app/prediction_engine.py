from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID_PATH = PROJECT_ROOT / "outputs" / "phase5_prospectivity_grid.csv"

# Fallback to the original GEE-exported grid if needed
if not GRID_PATH.exists():
    GRID_PATH = (
        PROJECT_ROOT
        / "data"
        / "MOIL_Balaghat_Phase5_Final_Feature_Grid.csv"
    )


class ProspectivityEngine:
    def __init__(self):
        if not GRID_PATH.exists():
            raise FileNotFoundError(
                f"Prediction grid not found:\n{GRID_PATH}"
            )

        self.grid = pd.read_csv(GRID_PATH)

        self.lat_col = self._find_column(
            ["latitude", "lat", "Latitude", "LATITUDE"]
        )

        self.lon_col = self._find_column(
            ["longitude", "lon", "Longitude", "LONGITUDE"]
        )

        self.score_col = self._find_column(
            [
                "prospectivity_score",
                "Prospectivity_Score",
                "score",
                "Score",
                "prospectivity",
            ]
        )

        # Optional probability column
        self.probability_col = self._find_optional_column(
            [
                "probability",
                "prediction_probability",
                "prospectivity_probability",
                "model_probability",
            ]
        )

        self.grid[self.lat_col] = pd.to_numeric(
            self.grid[self.lat_col], errors="coerce"
        )

        self.grid[self.lon_col] = pd.to_numeric(
            self.grid[self.lon_col], errors="coerce"
        )

        self.grid[self.score_col] = pd.to_numeric(
            self.grid[self.score_col], errors="coerce"
        )

        self.grid = self.grid.dropna(
            subset=[self.lat_col, self.lon_col, self.score_col]
        ).copy()

    def _find_column(self, candidates):
        for column in candidates:
            if column in self.grid.columns:
                return column

        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns:\n{list(self.grid.columns)}"
        )

    def _find_optional_column(self, candidates):
        for column in candidates:
            if column in self.grid.columns:
                return column
        return None

    def _priority(self, score):
        if score >= 80:
            return "VERY HIGH"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MODERATE"
        elif score >= 20:
            return "LOW"
        return "VERY LOW"

    def _strength(self, score):
        """
        This is deliberately NOT called statistical confidence.
        The current model has not undergone probability calibration.
        """

        if score >= 80:
            return "Strong model signal"
        elif score >= 60:
            return "Moderate-to-strong model signal"
        elif score >= 40:
            return "Moderate model signal"
        elif score >= 20:
            return "Weak model signal"
        return "Very weak model signal"

    def predict_point(self, latitude, longitude):
        """
        Return the model prediction for the nearest precomputed
        prediction-grid cell.
        """

        latitude = float(latitude)
        longitude = float(longitude)

        # Fast approximate geographic distance.
        lat_scale = 111.0
        lon_scale = 111.0 * np.cos(np.radians(latitude))

        dlat = (self.grid[self.lat_col] - latitude) * lat_scale
        dlon = (self.grid[self.lon_col] - longitude) * lon_scale

        distance_km = np.sqrt(dlat**2 + dlon**2)

        index = distance_km.idxmin()

        row = self.grid.loc[index]

        score = float(row[self.score_col])

        result = {
            "latitude": latitude,
            "longitude": longitude,
            "grid_latitude": float(row[self.lat_col]),
            "grid_longitude": float(row[self.lon_col]),
            "distance_to_grid_cell_km": float(distance_km.loc[index]),
            "prospectivity_score": score,
            "priority": self._priority(score),
            "model_signal": self._strength(score),
        }

        if self.probability_col:
            result["model_probability"] = float(
                row[self.probability_col]
            )

        return result

    def predict_area(
        self,
        coordinates,
        max_points=10000,
    ):
        """
        Analyze all prediction-grid cells falling inside
        the selected polygon.

        coordinates:
            [(lon, lat), (lon, lat), ...]
        """

        polygon = Polygon(coordinates)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        min_lon, min_lat, max_lon, max_lat = polygon.bounds

        candidates = self.grid[
            (self.grid[self.lon_col] >= min_lon)
            & (self.grid[self.lon_col] <= max_lon)
            & (self.grid[self.lat_col] >= min_lat)
            & (self.grid[self.lat_col] <= max_lat)
        ].copy()

        if len(candidates) == 0:
            return {
                "cell_count": 0,
                "message": "No prediction-grid cells were found inside this area."
            }

        points_inside = []

        for index, row in candidates.iterrows():
            point = Point(
                float(row[self.lon_col]),
                float(row[self.lat_col]),
            )

            if polygon.contains(point) or polygon.touches(point):
                points_inside.append(index)

        selected = candidates.loc[points_inside].copy()

        if len(selected) == 0:
            return {
                "cell_count": 0,
                "message": "No prediction-grid cells were found inside this area."
            }

        if len(selected) > max_points:
            selected = selected.sample(
                max_points,
                random_state=42,
            )

        scores = selected[self.score_col].astype(float)

        mean_score = float(scores.mean())
        median_score = float(scores.median())
        max_score = float(scores.max())

        high_count = int((scores >= 60).sum())
        very_high_count = int((scores >= 80).sum())

        return {
            "cell_count": len(selected),
            "mean_score": mean_score,
            "median_score": median_score,
            "maximum_score": max_score,
            "priority": self._priority(mean_score),
            "model_signal": self._strength(mean_score),
            "high_priority_cells": high_count,
            "very_high_priority_cells": very_high_count,
            "selected_cells": selected,
        }

    def get_map_data(self):
        return self.grid[
            [
                self.lat_col,
                self.lon_col,
                self.score_col,
            ]
        ].copy()