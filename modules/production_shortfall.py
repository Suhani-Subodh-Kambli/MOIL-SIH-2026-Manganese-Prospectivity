"""
modules/production_shortfall.py

Comprehensive Production Intelligence and Shortfall Management Engine for MOIL Limited.
Addresses the second core mandate of SIH Problem Statement 26009:
"Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome Production Shortfalls."

Key Components:
1. ShortfallPredictor: ML-driven prediction of monthly production shortfall, risk tiers, and grade deficits.
2. RootCauseAttributor: Quantitative attribution of production losses across weather, fleet, maintenance, and logistics.
3. OreBlendingOptimizer: Linear programming solver determining optimal stockpile blend ratios to guarantee contract grade (% Mn).
4. CorrectiveActionEngine: Prescriptive decision support generating prioritized, quantified operational recovery actions.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
REGRESSOR_FILE = MODEL_DIR / "moil_production_shortfall_regressor.joblib"
CLASSIFIER_FILE = MODEL_DIR / "moil_production_shortfall_classifier.joblib"
SCHEMA_FILE = MODEL_DIR / "production_feature_schema.json"


class OreBlendingOptimizer:
    """
    Linear Programming Solver for Multi-Stockpile Manganese Ore Blending.
    Solves for the cost-minimal blend recipe that strictly meets customer / smelter
    metallurgical specifications (% Mn, % SiO2, % P).
    """

    def optimize_blend(
        self,
        stockpiles: List[Dict[str, Any]],
        target_tonnes: float = 10000.0,
        min_mn_grade: float = 42.0,
        max_sio2: float = 8.5,
        max_p: float = 0.12
    ) -> Dict[str, Any]:
        """
        Solves:
            min sum(cost_i * x_i)
        Subject to:
            sum(x_i) == target_tonnes
            sum(mn_i * x_i) >= min_mn_grade * target_tonnes  =>  -sum(mn_i * x_i) <= -min_mn_grade * target_tonnes
            sum(sio2_i * x_i) <= max_sio2 * target_tonnes
            sum(p_i * x_i) <= max_p * target_tonnes
            0 <= x_i <= available_tonnes_i
        """
        n = len(stockpiles)
        if n == 0:
            return {"status": "Error", "message": "No stockpiles provided."}

        costs = np.array([float(s.get("cost_per_ton", 2500.0)) for s in stockpiles])
        mn_grades = np.array([float(s.get("grade_mn", 40.0)) for s in stockpiles])
        sio2_grades = np.array([float(s.get("sio2", 8.0)) for s in stockpiles])
        p_grades = np.array([float(s.get("p", 0.10)) for s in stockpiles])
        avail_tonnes = np.array([float(s.get("available_tonnes", 100000.0)) for s in stockpiles])

        # Objective: minimize total cost
        c = costs

        # Inequality constraints: A_ub * x <= b_ub
        # 1. Mn grade: -sum(mn_i * x_i) <= -min_mn_grade * target_tonnes
        # 2. SiO2 grade: sum(sio2_i * x_i) <= max_sio2 * target_tonnes
        # 3. P grade: sum(p_i * x_i) <= max_p * target_tonnes
        A_ub = [
            -mn_grades,
            sio2_grades,
            p_grades
        ]
        b_ub = [
            -min_mn_grade * target_tonnes,
            max_sio2 * target_tonnes,
            max_p * target_tonnes
        ]

        # Equality constraint: sum(x_i) == target_tonnes
        A_eq = [np.ones(n)]
        b_eq = [target_tonnes]

        # Bounds: 0 <= x_i <= available_tonnes_i
        bounds = [(0, min(target_tonnes, avail)) for avail in avail_tonnes]

        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

        if not res.success:
            return {
                "status": "Infeasible",
                "message": "Unable to satisfy target grade constraints with current stockpile inventories.",
                "solver_message": res.message,
                "recipe": []
            }

        alloc_tonnes = res.x
        total_cost = float(res.fun)
        avg_cost_per_ton = total_cost / max(1.0, target_tonnes)

        final_mn = float(np.sum(mn_grades * alloc_tonnes) / target_tonnes)
        final_sio2 = float(np.sum(sio2_grades * alloc_tonnes) / target_tonnes)
        final_p = float(np.sum(p_grades * alloc_tonnes) / target_tonnes)

        # Baseline cost if only using highest grade stockpile
        max_grade_idx = int(np.argmax(mn_grades))
        pure_hg_cost = float(costs[max_grade_idx] * target_tonnes)
        cost_savings = max(0.0, pure_hg_cost - total_cost)

        recipe = []
        for i, s in enumerate(stockpiles):
            draw_tonnes = float(alloc_tonnes[i])
            pct = round((draw_tonnes / target_tonnes) * 100.0, 1)
            if draw_tonnes > 1e-3:
                recipe.append({
                    "stockpile_name": s.get("name", f"Stockpile {i+1}"),
                    "allocated_tonnes": round(draw_tonnes, 1),
                    "blend_percentage": pct,
                    "feed_mn": float(mn_grades[i]),
                    "feed_sio2": float(sio2_grades[i]),
                    "feed_p": float(p_grades[i]),
                    "subtotal_cost": round(draw_tonnes * costs[i], 2)
                })

        return {
            "status": "Optimal",
            "target_tonnes": float(target_tonnes),
            "blended_grade_mn": round(final_mn, 2),
            "blended_sio2": round(final_sio2, 2),
            "blended_p": round(final_p, 4),
            "total_blend_cost_inr": round(total_cost, 2),
            "average_cost_per_tonne_inr": round(avg_cost_per_ton, 2),
            "cost_savings_vs_pure_highgrade_inr": round(cost_savings, 2),
            "recipe": recipe
        }


class CorrectiveActionEngine:
    """
    Prescriptive Intelligence: Generates actionable, prioritized mitigation workflows
    to recover lost production tonnage and resolve operational bottlenecks.
    """

    def generate_recommendations(
        self,
        mine_name: str,
        mine_type: str,
        shortfall_tonnes: float,
        shortfall_pct: float,
        root_causes: Dict[str, float],
        grade_deficit: float
    ) -> List[Dict[str, Any]]:
        actions = []

        # Sort root causes by impact
        sorted_causes = sorted(root_causes.items(), key=lambda x: x[1], reverse=True)
        top_cause = sorted_causes[0][0] if sorted_causes else "Unknown"

        # 1. Weather / Monsoon Flooding Action
        if root_causes.get("Monsoon Flooding / Water Logging", 0) > 0.15:
            if mine_type in ["Opencast", "Mixed"]:
                actions.append({
                    "priority": "HIGH (IMMEDIATE)",
                    "category": "Dewatering & Infrastructure",
                    "intervention": "Deploy High-Head Submersible Dewatering Pumps in sump",
                    "details": "Mobilize 2x 150 HP high-head dewatering units to lower pit water level by 3.5 m; apply crushed quartzite capping on haul road ramps to restore dumper traction.",
                    "estimated_recovery_tonnes": round(shortfall_tonnes * 0.35),
                    "days_saved": 4.5,
                    "estimated_cost_inr": "INR 4.5 Lakh"
                })
            else:
                actions.append({
                    "priority": "HIGH (IMMEDIATE)",
                    "category": "Underground Dewatering",
                    "intervention": "Commission Auxiliary Underground Sump Pumps",
                    "details": "Activate standby 200 HP dewatering pumps on main haulage levels to prevent haulage track submergence.",
                    "estimated_recovery_tonnes": round(shortfall_tonnes * 0.25),
                    "days_saved": 3.0,
                    "estimated_cost_inr": "INR 3.0 Lakh"
                })

        # 2. Equipment Downtime / Fleet Action
        if root_causes.get("HEMM Fleet Breakdown / Unavailability", 0) > 0.15:
            actions.append({
                "priority": "HIGH (WITHIN 24 HRS)",
                "category": "Fleet & Equipment Reliability",
                "intervention": "Dynamic Shovel-Dumper Re-allocation & Fast-Track Maintenance",
                "details": "Reassign 2 idle 35T dumpers from overburden stripping to active ore loading benches; schedule critical hydraulic seal replacements during night shift non-working window.",
                "estimated_recovery_tonnes": round(shortfall_tonnes * 0.30),
                "days_saved": 3.5,
                "estimated_cost_inr": "INR 2.2 Lakh"
            })

        # 3. Stripping Ratio Lag Action
        if root_causes.get("Overburden Stripping Lag", 0) > 0.10 and mine_type in ["Opencast", "Mixed"]:
            actions.append({
                "priority": "MEDIUM (WITHIN 48 HRS)",
                "category": "Mine Planning & Stripping",
                "intervention": "Bench Pushback Acceleration & Contractual Excavation",
                "details": "Mobilize auxiliary contractual excavator to execute pushback on North Wall Bench 4; expose 18,000 tonnes of high-grade ore horizon for immediate extraction.",
                "estimated_recovery_tonnes": round(shortfall_tonnes * 0.28),
                "days_saved": 5.0,
                "estimated_cost_inr": "INR 6.0 Lakh"
            })

        # 4. Shaft / Logistics Bottleneck (Underground)
        if root_causes.get("Shaft / Logistics Bottleneck", 0) > 0.15 and mine_type in ["Underground", "Mixed"]:
            actions.append({
                "priority": "MEDIUM (PLANNED)",
                "category": "Shaft Hoisting Logistics",
                "intervention": "Winder Cycle Optimization & Skip Counterweight Balancing",
                "details": "Reduce skip cycle turnaround from 4.8 min to 3.9 min by optimizing tippler automation; utilize intermediate ore pass bins to buffer stoping surges.",
                "estimated_recovery_tonnes": round(shortfall_tonnes * 0.22),
                "days_saved": 2.5,
                "estimated_cost_inr": "INR 1.8 Lakh"
            })

        # 5. Ore Grade Dilution Action
        if grade_deficit > 1.0:
            actions.append({
                "priority": "HIGH (OPERATIONAL)",
                "category": "Grade Control & Blending",
                "intervention": "Selective Blasting & Mathematical Ore Blending",
                "details": f"Ore grade is {grade_deficit:.1f}% below contract spec. Implement face-sampling drill assays before charging; blend low-grade ROM with surface high-grade stockpile (Balaghat/Ukwa feed) using the LP Blending Solver.",
                "estimated_recovery_tonnes": round(shortfall_tonnes * 0.15),
                "days_saved": 2.0,
                "estimated_cost_inr": "INR 1.5 Lakh"
            })

        # 6. Shift Catch-up Acceleration
        if shortfall_pct >= 15.0:
            actions.append({
                "priority": "HIGH (SCHEDULE)",
                "category": "Operations Scheduling",
                "intervention": "Deploy Sunday / Overtime Production Shift",
                "details": "Authorize 2 compensatory weekend production shifts across drilling and hauling crews to recover monthly shortfall deficit before accounting cutoff.",
                "estimated_recovery_tonnes": round(shortfall_tonnes * 0.20),
                "days_saved": 2.0,
                "estimated_cost_inr": "INR 3.5 Lakh"
            })

        return actions


class ShortfallPredictor:
    """
    ML Prediction Engine for Mining Extraction Shortfall and Grade Deficit.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).resolve().parents[1]
        self.models_dir = self.base_dir / "models"
        self.regressor_path = self.models_dir / "moil_production_shortfall_regressor.joblib"
        self.classifier_path = self.models_dir / "moil_production_shortfall_classifier.joblib"
        self.schema_path = self.models_dir / "production_feature_schema.json"

        self.regressor = None
        self.classifier = None
        self.feature_columns = []

        if self.regressor_path.exists():
            self.regressor = joblib.load(self.regressor_path)
        if self.classifier_path.exists():
            self.classifier = joblib.load(self.classifier_path)
        if self.schema_path.exists():
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.feature_columns = json.load(f)

        self.blender = OreBlendingOptimizer()
        self.action_engine = CorrectiveActionEngine()

    def predict_shortfall(
        self,
        mine_name: str,
        mine_type: str,
        target_tonnes: float,
        rainfall_mm: float,
        fleet_availability_pct: float,
        unplanned_downtime_hrs: float,
        actual_grade_pct: Optional[float] = None,
        target_grade_pct: Optional[float] = None,
        stripping_ratio_deficit: float = 0.0,
        shaft_utilization_pct: float = 80.0
    ) -> Dict[str, Any]:
        """
        Predicts expected shortfall, risk category, and root cause attribution.
        """
        # Formulate feature vector
        is_opencast = 1 if mine_type.lower() == "opencast" else 0
        is_underground = 1 if mine_type.lower() == "underground" else 0
        is_mixed = 1 if mine_type.lower() == "mixed" else 0

        fleet_utilization = fleet_availability_pct * 0.90
        power_outage = max(2.0, rainfall_mm / 25.0)

        feature_dict = {
            "monthly_target_tonnes": target_tonnes,
            "rainfall_mm": rainfall_mm,
            "fleet_availability_pct": fleet_availability_pct,
            "fleet_utilization_pct": fleet_utilization,
            "unplanned_downtime_hrs": unplanned_downtime_hrs,
            "stripping_ratio_deficit": stripping_ratio_deficit,
            "shaft_utilization_pct": shaft_utilization_pct if is_underground or is_mixed else 0.0,
            "power_outage_hrs": power_outage,
            "is_opencast": is_opencast,
            "is_underground": is_underground,
            "is_mixed": is_mixed
        }

        # Predict using trained regressor if available, else analytical physics formula
        if self.regressor is not None and len(self.feature_columns) > 0:
            X_df = pd.DataFrame([feature_dict])[self.feature_columns]
            pred_shortfall_pct = float(np.clip(self.regressor.predict(X_df)[0], 0.0, 75.0))
        else:
            # Analytical mining physics formula
            w_coeff = 0.70 if is_opencast else 0.25
            fl_coeff = 0.25 if is_opencast else 0.40
            weather_imp = (rainfall_mm / 450.0) * w_coeff
            fleet_imp = ((90.0 - fleet_availability_pct) / 35.0) * fl_coeff
            downtime_imp = (unplanned_downtime_hrs / 180.0) * 0.15
            sr_imp = (max(0.0, stripping_ratio_deficit) / 3.0) * 0.15
            loss_frac = np.clip(weather_imp + fleet_imp + downtime_imp + sr_imp, 0.0, 0.70)
            pred_shortfall_pct = round(float(loss_frac * 100.0), 2)

        pred_shortfall_tonnes = round(target_tonnes * (pred_shortfall_pct / 100.0))
        pred_extraction_tonnes = max(0, int(target_tonnes - pred_shortfall_tonnes))

        # Risk classification
        if pred_shortfall_pct >= 25.0:
            risk_tier = "CRITICAL"
            risk_badge = "🔴 CRITICAL DEFICIT RISK"
        elif pred_shortfall_pct >= 15.0:
            risk_tier = "HIGH"
            risk_badge = "🟠 HIGH SHORTFALL RISK"
        elif pred_shortfall_pct >= 6.0:
            risk_tier = "MODERATE"
            risk_badge = "🟡 MODERATE RISK"
        else:
            risk_tier = "ON TARGET"
            risk_badge = "🟢 ON TARGET"

        # Grade deficit calculation
        t_grade = target_grade_pct or 42.0
        a_grade = actual_grade_pct or (t_grade - (1.8 if rainfall_mm > 200 else 0.3))
        grade_deficit = round(max(0.0, t_grade - a_grade), 2)
        grade_deficit_prob = float(np.clip(0.15 + (rainfall_mm / 400.0) * 0.50 + ((85.0 - fleet_availability_pct) / 40.0) * 0.30, 0.05, 0.95))

        # Root Cause Attribution
        w_factor = (rainfall_mm / 450.0) * (0.65 if is_opencast else 0.20)
        f_factor = ((90.0 - fleet_availability_pct) / 35.0) * (0.25 if is_opencast else 0.45)
        m_factor = (unplanned_downtime_hrs / 180.0) * 0.15
        s_factor = (max(0.0, stripping_ratio_deficit) / 3.0) * 0.15
        sh_factor = 0.20 if (is_underground and shaft_utilization_pct > 90) else 0.03

        raw_causes = {
            "Monsoon Flooding / Water Logging": max(0.01, w_factor),
            "HEMM Fleet Breakdown / Unavailability": max(0.01, f_factor),
            "Equipment Maintenance Backlog": max(0.01, m_factor),
            "Overburden Stripping Lag": max(0.01, s_factor),
            "Shaft / Logistics Bottleneck": max(0.01, sh_factor)
        }
        total_cause_weight = sum(raw_causes.values())
        root_cause_pcts = {k: round((v / total_cause_weight) * 100.0, 1) for k, v in raw_causes.items()}
        primary_cause = max(root_cause_pcts, key=root_cause_pcts.get)

        # Generate Actionable Corrective Recommendations
        recommendations = self.action_engine.generate_recommendations(
            mine_name=mine_name,
            mine_type=mine_type,
            shortfall_tonnes=pred_shortfall_tonnes,
            shortfall_pct=pred_shortfall_pct,
            root_causes={k: v / 100.0 for k, v in root_cause_pcts.items()},
            grade_deficit=grade_deficit
        )

        total_recoverable = sum(r.get("estimated_recovery_tonnes", 0) for r in recommendations)
        recovered_pct = round((total_recoverable / target_tonnes) * 100.0, 1) if target_tonnes > 0 else 0.0

        return {
            "mine_name": mine_name,
            "mine_type": mine_type,
            "monthly_target_tonnes": float(target_tonnes),
            "predicted_actual_extraction_tonnes": int(pred_extraction_tonnes),
            "predicted_shortfall_tonnes": int(pred_shortfall_tonnes),
            "predicted_shortfall_pct": float(pred_shortfall_pct),
            "risk_tier": risk_tier,
            "risk_badge": risk_badge,
            "target_grade_pct": float(t_grade),
            "estimated_actual_grade_pct": float(a_grade),
            "grade_deficit_pct": float(grade_deficit),
            "grade_deficit_probability": round(grade_deficit_prob, 3),
            "primary_root_cause": primary_cause,
            "root_cause_attribution": root_cause_pcts,
            "recommendations": recommendations,
            "total_recoverable_tonnes": int(total_recoverable),
            "post_mitigation_shortfall_pct": round(max(0.0, pred_shortfall_pct - recovered_pct), 1)
        }


# Singleton instance
_SHORTFALL_PREDICTOR = None

def get_shortfall_predictor() -> ShortfallPredictor:
    global _SHORTFALL_PREDICTOR
    if _SHORTFALL_PREDICTOR is None:
        _SHORTFALL_PREDICTOR = ShortfallPredictor()
    return _SHORTFALL_PREDICTOR
