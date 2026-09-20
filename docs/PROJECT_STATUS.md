# Current Project Status

## Project

**MOIL AI Exploration Intelligence**

Smart India Hackathon 2026 — Problem Statement 26009:

> Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome Production Shortfalls.

Current development pilot: **Balaghat, Madhya Pradesh**

The Balaghat implementation is the exploration-prospectivity pilot.
The final system must be generalized to **India-wide coverage**.

---

# 1. Completed Work

## Phase 1 — Sentinel-2 Pipeline

- Google Earth Engine (GEE) pipeline created.
- Sentinel-2 Surface Reflectance Harmonized data used.
- 2024 Balaghat imagery processed.
- Cloud-filtered image collection created.
- Sentinel-2 spectral bands extracted.
- Initial feature grid exported from GEE.

---

## Phase 2 — GSI Manganese Occurrences

- GSI manganese occurrence dataset integrated.
- Known manganese occurrence coordinates used as positive geological evidence.
- Occurrence data used for spatial sampling and independent validation.

Important:

`Distance_Mn` is NOT used as a model feature because it would leak the label-generation process.

---

## Phase 3 — Initial Multimodal Feature Dataset

Integrated:

- Sentinel-2 spectral bands
- NDVI
- Spectral ratios
- Sentinel-1 VV
- Sentinel-1 VH
- VV/VH difference
- SRTM elevation
- Slope
- Aspect
- Manganese occurrence information
- Labels
- Latitude/longitude for spatial processing

Initial Balaghat dataset:

- 2,066 samples
- 600 positive
- 1,466 background samples

---

# 2. Phase 4B — Geological and ML Improvements

## Phase 4B.1 — Geological/Lithological Data

NGDR geological/lithological data integrated.

Source:

- Geological Survey of India / NGDR
- Indian lithology 1:50,000 dataset

Balaghat extraction:

- ~3.97 million source records processed
- 14,443 Balaghat geological polygons extracted
- Invalid geometries handled/repaired where required

Important geological attributes include:

- Age
- Supergroup
- Group
- Formation
- Member
- Lithology
- Intrusive type
- Stratigraphy

The large India-wide NGDR source files are NOT intended to be stored directly in normal Git history.

---

## Phase 4B.2 — Geological Boundaries and Structural Features

Added spatial geological features:

- Geological boundary distance
- Geological boundary density within 1 km
- Geological boundary density within 3 km
- Lithological diversity within 3 km
- Formation diversity within 3 km
- Geological group proxies
- Metamorphic-host proxy
- Sausar-group proxy
- Aspect sine/cosine representation

These features provide geological and structural/contextual information to the model.

---

## Phase 4B.3 — Spectral Features

Additional Sentinel-2 spectral features created:

- NDMI
- NBR
- NDRE
- Iron Oxide Index
- Clay Alteration Index
- Ferrous Index
- SWIR/Red ratio
- SWIR/Green ratio
- NIR/SWIR2 ratio
- BSI
- B11/B12 normalized difference
- B8/B12 normalized difference
- B4/B2 normalized difference

These complement the original Sentinel-2 bands and ratios.

---

## Phase 4B.4 — Spatially Balanced Sampling

Positive/background sampling improved.

Current balanced training dataset:

- 360 samples
- 120 positive
- 240 background
- 0 duplicate coordinates

Spatial grouping is used to reduce spatial leakage.

Unknown geological areas are not automatically interpreted as proven negatives.

---

## Phase 4B.5 — XGBoost Model

Current model:

**XGBoost manganese prospectivity model**

Raw feature groups:

- 44 numeric features
- 7 categorical geological features

After preprocessing:

**117 transformed model features**

Saved model:

`models/moil_manganese_prospectivity_xgb_phase4b.joblib`

Saved preprocessing pipeline:

`models/phase4b_preprocessor.joblib`

Saved transformed feature schema:

`models/phase4b_feature_columns.json`

The preprocessing pipeline must be reused during inference.
Do NOT manually recreate or reorder the transformed feature columns.

---

## Phase 4B.6 — 5-Fold Spatial Validation

5-fold `StratifiedGroupKFold` spatial validation completed.

Mean results:

- ROC-AUC: **0.6210 ± 0.2085**
- PR-AUC: **0.4746 ± 0.1422**
- Accuracy: **0.6556 ± 0.0514**
- Precision: **0.3743 ± 0.2626**
- Recall: **0.1833 ± 0.1733**
- F1: **0.2391 ± 0.2065**

The relatively high fold-to-fold variation indicates that the current model should be treated as a prototype rather than a production-grade exploration model.

---

## Phase 4B.7 — SHAP Explainability

SHAP analysis completed.

Important contributing feature groups include:

- Geological boundary distance
- Geological boundary density
- Elevation
- Slope
- Lithological categories
- Sentinel-1 VV/VH relationships
- Iron oxide spectral index
- Sausar-group proxy
- Geological diversity
- Spectral ratios

SHAP outputs:

- `outputs/phase4b_shap_summary.png`
- `outputs/phase4b_shap_bar.png`
- `outputs/phase4b_shap_importance.csv`
- `outputs/phase4b_shap_values.csv`

SHAP is used to explain model behaviour, not to prove that a geological feature guarantees manganese mineralization.

---

# 3. Phase 5 — Balaghat Prospectivity Mapping

Final Balaghat prediction grid generated.

Grid:

- **14,061 prediction cells**
- **1,407 top-priority cells**
- **94 target clusters/zones**

Current model output is converted to a:

**0–100 Exploration Priority Score**

The score is a relative ranking of locations within the prediction domain.

It is NOT:

- manganese concentration
- ore grade
- percentage probability of manganese
- proven reserve
- economic extraction probability
- drilling confirmation

---

# 4. Independent Validation

GSI manganese occurrences were used for an independent spatial validation of the final Balaghat map.

There are only a small number of known local occurrences in the current validation area, so the validation is directional rather than definitive.

Current validation indicates:

- stronger broad-scale association around approximately 10–20 km distances
- weak local association at approximately 1–5 km
- current model should NOT be presented as accurately locating individual deposits

Therefore:

**The current system is an exploration-prioritization tool, not a deposit-confirmation system.**

---

# 5. Geological Enrichment Audit

Geological composition of high-score areas was checked against the complete Balaghat prediction grid.

Findings:

- No strong Sausar-group enrichment was demonstrated.
- Some metamorphic-host and specific lithology/group features show enrichment.
- Several high-score areas contain geological categories such as:
  - Amgaon Gneissic Complex
  - Granite Gneiss/Migmatite
  - Gneiss/Migmatite
  - Mansar formation
  - Archaean units

This audit prevents unsupported claims such as:

> "The model has learned that all manganese occurs in the Sausar Group."

Such claims must NOT be made.

---

# 6. Current Interactive Application

A Streamlit-based interactive exploration interface has been implemented.

Application:

`app/app.py`

Prediction engine:

`app/prediction_engine.py`

Current capabilities:

### Single Location

User can enter:

- Latitude
- Longitude

and receive:

- Prospectivity score
- Exploration priority
- Model signal
- Map location
- AI interpretation

### Area Selection

User can interact with a map and draw:

- Rectangle
- Polygon

The application can analyse prediction cells inside the selected area and provide:

- Mean prospectivity
- Maximum prospectivity
- High-priority cells
- Very-high-priority cells
- Overall exploration priority
- Model signal

The prediction result is stored using Streamlit session state so it persists across UI reruns.

---

# 7. Current Geographic Limitation

IMPORTANT:

The current ML model and prediction grid were developed for the **Balaghat pilot**.

The application is NOT yet a true India-wide prediction system.

The current interactive UI must therefore NOT be described as an India-wide trained model.

The next major objective is to generalize the existing pipeline so that the same architecture can operate across India.

---

# 8. Current Exploration Architecture

DO NOT change this architecture without strong scientific/technical justification.

```text
GSI manganese occurrences
        +
Sentinel-2
        +
Sentinel-1
        +
SRTM terrain
        +
Geological / lithological evidence
        +
Structural / geological spatial features
        +
Spectral features
        ↓
Feature dataset
        ↓
XGBoost
        ↓
Spatial validation
        ↓
Prospectivity score
        ↓
Target zones
        ↓
Interactive dashboard
```

---

# 9. Phase 6 — India-Wide Data & Feature Pipeline (COMPLETED)

Branch: `feature/india-wide-data`

### 9.1 National Ground-Truth Manganese Dataset
Consolidated national manganese occurrences and active mining leases from GSI and NGDR:
- `data/gsi/india_manganese_occurrences.csv`
- `data/gsi/india_manganese_occurrences.geojson`
- **187 unique verified manganese sites** across 11 states:
  - Madhya Pradesh (56), Odisha (33), Andhra Pradesh (32), Maharashtra (22), Karnataka (17), Telangana (12), Goa (7), Jharkhand (4), Chhattisgarh (2), Gujarat (1), Rajasthan (1).
- Tagged with official NGSR Metallogenic Belts:
  - Sausar Mn Zone (64 sites)
  - Bonai-Noamundi-Jamda Fe-Mn Belt (35 sites)
  - Eastern Ghats Al-Mn Province (30 sites)
  - Chitradurga Polymetallic Province (9 sites)
  - Goa Fe-Mn Belt (8 sites)
  - Sandur Fe-Mn-Au Province (5 sites)
  - Kundremukh-Shimoga Belt (1 site)
  - Other Regional Belts (35 sites)

### 9.2 National Geological Indexer
Implemented in `core/geology_index.py`:
- Integrates `data/geology/NGDR_Geology_2M.parquet` (national 1:2M geology, 4,531 polygons covering all of India) with local 1:50k pilot lithology (`balaghat_lithology.geojsonl`).
- Rapid `STRtree` spatial indexing providing **2–4 millisecond point queries** for any arbitrary coordinate across India.
- Computes structural proxy features: geological boundary distance (km), boundary density (1 km and 3 km), lithological and formation diversity (3 km), Sausar group proxy, and metamorphic host proxy.

### 9.3 Reusable Feature Engineering Engine
Implemented in `core/feature_extractor.py`:
- Enriches any coordinate or raster grid with all 44 numeric features and 7 categorical features.
- Strict leakage prevention: coordinates and `Distance_Mn` are completely excluded from model features.
- Guaranteed 100% schema compatibility: outputs exact 51 raw features that transform seamlessly through `phase4b_preprocessor.joblib` into the 117 XGBoost model input features.

### 9.4 Spatially Balanced Sampling Engine
Implemented in `core/sampling.py`:
- Implements Positive-Unlabelled (PU) sampling logic.
- Enforces strict minimum distance buffer (`>= 3.0 km`) from all known positives for background samples.
- Assigns spatial block IDs (`spatial_block`) for leak-free `StratifiedGroupKFold` spatial cross-validation.

### 9.5 Automated Verification Suite
- `tests/test_geology_index.py`: Verifies spatial lookups across Balaghat, Sandur, Bonai, Goa, and Vizianagaram.
- `tests/test_feature_extractor.py`: Verifies feature enrichment, preprocessor transformation, and XGBoost inference.
- `tests/test_sampling.py`: Verifies positive buffering and spatial block partitioning.
- `tests/run_all_tests.py`: Runs all test suites (3/3 passing).
- `tests/run_all_tests.py`: Runs all test suites.

---

# 10. Phase 7 — Inference Engine Decoupling & Naive Bayes Model (COMPLETED)

Branch: `feature/inference-engine`

### 10.1 Naive Bayes Prospectivity Model
Trained and spatially validated a classical probabilistic **Gaussian Naive Bayes model** on the exact same 117 transformed features and 5-fold `StratifiedGroupKFold` spatial blocks:
- Script: `scripts/train_phase4b_nb.py`
- Model artifact: `models/moil_manganese_prospectivity_nb_phase4b.joblib`
- Metrics: `outputs/phase4b_nb_spatial_metrics.csv`
- Model Benchmark comparison: `outputs/phase4b_model_comparison.csv`

**Spatial Validation Benchmark (XGBoost vs. Naive Bayes):**

| Metric | XGBoost (Phase 4B) | Naive Bayes (GaussianNB) | Improvement / Takeaway |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.6210 ± 0.2085 | 0.6066 ± 0.0646 | Naive Bayes std is 3x lower (higher spatial stability) |
| **PR-AUC** | 0.4746 ± 0.1422 | **0.5294 ± 0.0675** | **+5.48% higher PR-AUC** |
| **Accuracy** | 0.6556 ± 0.0514 | **0.6889 ± 0.0400** | **+3.33% higher overall accuracy** |
| **Precision** | 0.3743 ± 0.2626 | **0.5923 ± 0.1615** | **+21.80% higher precision** |
| **Recall** | 0.1833 ± 0.1733 | **0.3167 ± 0.1003** | **+13.34% higher positive recall** |
| **F1 Score** | 0.2391 ± 0.2065 | **0.3985 ± 0.0840** | **+15.94% higher F1** |
| **Precision@Top 5%** | 0.5000 ± 0.3953 | **0.8000 ± 0.2092** | **80% precision at peak decile** |
| **Precision@Top 10%** | 0.4250 ± 0.1896 | **0.7500 ± 0.1250** | **75% precision in top 10%** |
| **Precision@Top 20%** | 0.4400 ± 0.1535 | **0.5333 ± 0.0471** | **53.3% precision in top 20%** |

### 10.2 Decoupled Unified Inference Engine
Implemented in `core/inference_engine.py`:
- **Arbitrary Point Prediction**: Dynamically derives geology, contacts, and proxies anywhere in India (e.g. Sandur, Bellary; Bonai-Keonjhar, Odisha; Goa; Vizianagaram, AP) while retaining exact 1:50k pilot resolution in Balaghat.
- **Arbitrary Polygon Prediction**: Generates adaptive spatial grids on-the-fly inside user polygons anywhere in India, extracts features, computes scores, and runs **DBSCAN spatial clustering** to output contiguous high-priority **Target Zones**.
- **Multi-Model Support**: Allows runtime selection between:
  - `xgboost`: Non-linear gradient boosted trees.
  - `naive_bayes`: Probabilistic conditional independence model.
  - `ensemble`: 50/50 weighted combination combining decision trees with Bayesian likelihood.

### 10.3 Streamlit Application Upgrade
Updated `app/app.py` and `app/prediction_engine.py`:
- Unlocked coordinates across all of India (Lat 6.00°N–38.00°N, Lon 68.00°E–98.00°E).
- Added quick-jump preset selector for major Indian manganese hubs (Balaghat, Ukwa, Tirodi, Dongri Buzurg, Sandur, Bonai, Garbham, Colamba).
- Added AI Model Selector in the sidebar (Ensemble, XGBoost, Naive Bayes).
- Added Geological & Structural diagnostics panel (Formation, Group, Boundary Distance, Contact Density 1km/3km, Lithology Diversity).
- Added nearest known GSI manganese occurrence reference and metallogenic belt distance.
- Preserved 100% backward compatibility for Balaghat pilot verification.

### 10.4 Automated Verification Suite
- `tests/test_inference_engine.py`: Tests point prediction in Balaghat, Sandur, Koira; tests model comparison; tests polygon prediction and DBSCAN clustering.
- `tests/run_all_tests.py`: Runs all 4 test suites (4/4 passing).