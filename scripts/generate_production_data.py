"""
scripts/generate_production_data.py

Generates a realistic, operationally calibrated historical mining telemetry dataset
for MOIL Limited's 8 operating manganese mines over 36 months (January 2023 - December 2025).

Parameters modeled:
- Mine typology: Opencast vs Underground
- Monthly production targets & actual extraction (tonnes)
- Ore grade: target vs actual (% Mn) and grade deficit
- Central India seasonal monsoon dynamics (June-September peak rainfall)
- Heavy Earth Moving Machinery (HEMM) fleet availability & utilization (%)
- Unplanned maintenance downtime (hours)
- Haul road degradation index
- Stripping ratio lag (actual vs planned waste:ore)
- Underground vertical shaft hoisting utilization (%)
- Primary root cause attribution

Outputs:
- data/production/moil_mine_operations_history.csv
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = BASE_DIR / "data" / "production"
OUTPUT_FILE = DATA_DIR / "moil_mine_operations_history.csv"

# MOIL 8 Operating Mines Profile
MINE_PROFILES = [
    {
        "name": "Balaghat Mine",
        "type": "Underground",
        "state": "Madhya Pradesh",
        "base_target": 40000,
        "base_grade": 44.5,
        "shaft_capacity": 45000,
        "vulnerability_monsoon": 0.25, # Underground less affected by surface rain directly
        "vulnerability_equipment": 0.40,
        "vulnerability_shaft": 0.35
    },
    {
        "name": "Dongri Buzurg Mine",
        "type": "Opencast",
        "state": "Maharashtra",
        "base_target": 32000,
        "base_grade": 41.0,
        "shaft_capacity": 0,
        "vulnerability_monsoon": 0.70, # Opencast severely affected by pit inundation
        "vulnerability_equipment": 0.20,
        "vulnerability_shaft": 0.10
    },
    {
        "name": "Tirodi Mine",
        "type": "Opencast",
        "state": "Madhya Pradesh",
        "base_target": 25000,
        "base_grade": 39.0,
        "shaft_capacity": 0,
        "vulnerability_monsoon": 0.60,
        "vulnerability_equipment": 0.25,
        "vulnerability_shaft": 0.15
    },
    {
        "name": "Chikla Mine",
        "type": "Underground",
        "state": "Maharashtra",
        "base_target": 22000,
        "base_grade": 42.5,
        "shaft_capacity": 25000,
        "vulnerability_monsoon": 0.20,
        "vulnerability_equipment": 0.45,
        "vulnerability_shaft": 0.35
    },
    {
        "name": "Gumgaon Mine",
        "type": "Underground",
        "state": "Maharashtra",
        "base_target": 15000,
        "base_grade": 37.5,
        "shaft_capacity": 18000,
        "vulnerability_monsoon": 0.20,
        "vulnerability_equipment": 0.45,
        "vulnerability_shaft": 0.35
    },
    {
        "name": "Ukwa Mine",
        "type": "Underground",
        "state": "Madhya Pradesh",
        "base_target": 13000,
        "base_grade": 43.5,
        "shaft_capacity": 16000,
        "vulnerability_monsoon": 0.25,
        "vulnerability_equipment": 0.45,
        "vulnerability_shaft": 0.30
    },
    {
        "name": "Kandri Mine",
        "type": "Opencast",
        "state": "Maharashtra",
        "base_target": 18000,
        "base_grade": 38.0,
        "shaft_capacity": 0,
        "vulnerability_monsoon": 0.65,
        "vulnerability_equipment": 0.25,
        "vulnerability_shaft": 0.10
    },
    {
        "name": "Mansar Mine",
        "type": "Mixed",
        "state": "Maharashtra",
        "base_target": 17000,
        "base_grade": 39.5,
        "shaft_capacity": 10000,
        "vulnerability_monsoon": 0.50,
        "vulnerability_equipment": 0.30,
        "vulnerability_shaft": 0.20
    }
]

# Central India Monthly Rainfall & Weather Profile (Balaghat-Nagpur belt)
MONTHLY_WEATHER = {
    1:  {"rain_mean": 12.0,  "rain_std": 8.0,   "rain_days": 1,  "season": "Winter"},
    2:  {"rain_mean": 18.0,  "rain_std": 10.0,  "rain_days": 2,  "season": "Winter"},
    3:  {"rain_mean": 22.0,  "rain_std": 12.0,  "rain_days": 2,  "season": "Pre-Monsoon"},
    4:  {"rain_mean": 15.0,  "rain_std": 10.0,  "rain_days": 1,  "season": "Summer (Extreme Heat)"},
    5:  {"rain_mean": 25.0,  "rain_std": 15.0,  "rain_days": 2,  "season": "Summer (Extreme Heat)"},
    6:  {"rain_mean": 185.0, "rain_std": 45.0,  "rain_days": 12, "season": "Monsoon Start"},
    7:  {"rain_mean": 380.0, "rain_std": 75.0,  "rain_days": 21, "season": "Peak Monsoon"},
    8:  {"rain_mean": 340.0, "rain_std": 65.0,  "rain_days": 19, "season": "Peak Monsoon"},
    9:  {"rain_mean": 190.0, "rain_std": 50.0,  "rain_days": 11, "season": "Monsoon Retreat"},
    10: {"rain_mean": 45.0,  "rain_std": 20.0,  "rain_days": 4,  "season": "Post-Monsoon"},
    11: {"rain_mean": 15.0,  "rain_std": 10.0,  "rain_days": 1,  "season": "Post-Monsoon"},
    12: {"rain_mean": 8.0,   "rain_std": 6.0,   "rain_days": 1,  "season": "Winter"}
}


def generate_telemetry_dataset() -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    records = []

    # 36 months from Jan 2023 to Dec 2025
    years = [2023, 2024, 2025]
    months = list(range(1, 13))

    record_id = 0
    for yr in years:
        for m in months:
            w_info = MONTHLY_WEATHER[m]

            for mine in MINE_PROFILES:
                record_id += 1

                # 1. Base Target with organic yearly growth
                growth_factor = 1.0 + (yr - 2023) * 0.05
                target = int(mine["base_target"] * growth_factor * rng.uniform(0.96, 1.04))

                # 2. Weather conditions
                rainfall_mm = max(0.0, float(rng.normal(w_info["rain_mean"], w_info["rain_std"])))
                rain_days = max(0, int(w_info["rain_days"] + rng.randint(-1, 3)))

                # 3. Fleet & Operational Telemetry
                # Fleet availability drops in summer heat and severe monsoon
                base_avail = 84.0
                if m in [4, 5]: # Summer heat breakdown
                    avail_penalty = rng.uniform(4.0, 9.0)
                elif m in [7, 8]: # Monsoon waterlogging
                    avail_penalty = rng.uniform(8.0, 16.0) if mine["type"] == "Opencast" else rng.uniform(2.0, 5.0)
                else:
                    avail_penalty = rng.uniform(0.0, 4.0)

                fleet_avail = float(np.clip(base_avail - avail_penalty + rng.normal(0, 1.5), 62.0, 95.0))
                fleet_util = float(np.clip(fleet_avail * rng.uniform(0.85, 0.95), 50.0, 90.0))

                unplanned_downtime_hrs = float(np.clip(rng.exponential(35.0) + (100.0 - fleet_avail) * 1.8, 8.0, 180.0))

                # Stripping ratio (waste : ore) for opencast
                if mine["type"] in ["Opencast", "Mixed"]:
                    planned_sr = round(float(rng.uniform(3.5, 5.0)), 2)
                    sr_lag = float(rng.normal(0.0, 0.4))
                    actual_sr = round(max(1.5, planned_sr - sr_lag), 2)
                    stripping_ratio_deficit = round(planned_sr - actual_sr, 2)
                else:
                    planned_sr = 0.0
                    actual_sr = 0.0
                    stripping_ratio_deficit = 0.0

                # Shaft utilization for underground
                if mine["type"] in ["Underground", "Mixed"]:
                    shaft_util = float(np.clip(rng.normal(82.0, 8.0), 55.0, 98.0))
                    shaft_bottleneck = 1 if shaft_util > 92.0 or unplanned_downtime_hrs > 80.0 else 0
                else:
                    shaft_util = 0.0
                    shaft_bottleneck = 0

                power_outage_hrs = float(np.clip(rng.exponential(12.0) + (rainfall_mm / 30.0), 2.0, 65.0))

                # Haul road condition (1 = Good, 2 = Degraded, 3 = Impassable)
                if rainfall_mm > 300.0 and mine["type"] == "Opencast":
                    haul_road = "Impassable"
                elif rainfall_mm > 120.0:
                    haul_road = "Degraded"
                else:
                    haul_road = "Good"

                # 4. Realistic Extraction Physics
                # Impact coefficients
                weather_impact = (rainfall_mm / 450.0) * mine["vulnerability_monsoon"]
                fleet_impact = ((90.0 - fleet_avail) / 35.0) * mine["vulnerability_equipment"]
                maint_impact = (unplanned_downtime_hrs / 200.0) * 0.15
                power_impact = (power_outage_hrs / 70.0) * 0.10
                sr_impact = (max(0.0, stripping_ratio_deficit) / 3.0) * 0.15

                total_loss_fraction = np.clip(
                    weather_impact + fleet_impact + maint_impact + power_impact + sr_impact + rng.normal(0, 0.03),
                    0.0, 0.65
                )

                actual_extraction = int(target * (1.0 - total_loss_fraction))
                shortfall_tonnes = target - actual_extraction
                shortfall_pct = round((shortfall_tonnes / target) * 100.0, 2)

                # 5. Grade Dilution Physics
                target_grade = mine["base_grade"]
                # Monsoon causes ore dilution with wall rock / slime
                dilution = 0.0
                if haul_road in ["Degraded", "Impassable"]:
                    dilution += rng.uniform(0.8, 2.8)
                if fleet_avail < 75.0: # Rushed selective mining causes dilution
                    dilution += rng.uniform(0.4, 1.5)

                actual_grade = round(float(np.clip(target_grade - dilution + rng.normal(0, 0.3), 28.0, 49.0)), 2)
                grade_deficit = round(target_grade - actual_grade, 2)

                # 6. Risk Tier Classification
                if shortfall_pct >= 25.0:
                    risk_tier = "CRITICAL"
                elif shortfall_pct >= 15.0:
                    risk_tier = "HIGH"
                elif shortfall_pct >= 6.0:
                    risk_tier = "MODERATE"
                else:
                    risk_tier = "ON TARGET"

                # 7. Primary Root Cause Attribution
                causes = {
                    "Monsoon Flooding / Water Logging": weather_impact,
                    "HEMM Fleet Breakdown / Unavailability": fleet_impact,
                    "Equipment Maintenance Backlog": maint_impact,
                    "Overburden Stripping Lag": sr_impact,
                    "Shaft / Logistics Bottleneck": 0.20 if shaft_bottleneck else 0.02,
                    "Power Supply Outage": power_impact
                }
                primary_cause = max(causes, key=causes.get)

                records.append({
                    "record_id": f"REC_{record_id:04d}",
                    "mine_name": mine["name"],
                    "mine_type": mine["type"],
                    "state": mine["state"],
                    "year": yr,
                    "month": m,
                    "season": w_info["season"],
                    "monthly_target_tonnes": target,
                    "actual_extracted_tonnes": actual_extraction,
                    "shortfall_tonnes": shortfall_tonnes,
                    "shortfall_pct": shortfall_pct,
                    "risk_tier": risk_tier,
                    "target_grade_pct": target_grade,
                    "actual_grade_pct": actual_grade,
                    "grade_deficit_pct": grade_deficit,
                    "grade_off_spec": 1 if grade_deficit > 1.5 else 0,
                    "rainfall_mm": round(rainfall_mm, 1),
                    "rain_days": rain_days,
                    "fleet_availability_pct": round(fleet_avail, 1),
                    "fleet_utilization_pct": round(fleet_util, 1),
                    "unplanned_downtime_hrs": round(unplanned_downtime_hrs, 1),
                    "planned_stripping_ratio": planned_sr,
                    "actual_stripping_ratio": actual_sr,
                    "stripping_ratio_deficit": stripping_ratio_deficit,
                    "shaft_utilization_pct": round(shaft_util, 1),
                    "power_outage_hrs": round(power_outage_hrs, 1),
                    "haul_road_condition": haul_road,
                    "primary_root_cause": primary_cause
                })

    df = pd.DataFrame(records)
    return df


def main():
    print("=" * 75)
    print("GENERATING MOIL MINE OPERATIONS HISTORICAL DATASET (2023-2025)")
    print("=" * 75)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_telemetry_dataset()

    print(f"Total Telemetry Records: {len(df)}")
    print(f"Mines Covered: {df['mine_name'].nunique()}")
    print("\nRecords by Risk Tier:")
    print(df["risk_tier"].value_counts())
    print("\nMean Shortfall % by Mine Type:")
    print(df.groupby("mine_type")["shortfall_pct"].mean().round(2))
    print("\nPrimary Root Causes Count:")
    print(df["primary_root_cause"].value_counts())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully generated and saved historical dataset: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
