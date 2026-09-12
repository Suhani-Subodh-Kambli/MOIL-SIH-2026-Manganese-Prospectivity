# MOIL SIH 2026 — AI-Assisted Manganese Exploration & Production Intelligence

## Smart India Hackathon 2026

**Problem Statement ID:** 26009  
**Organization:** MOIL Limited  
**Theme:** AI/ML + Space Technology  
**Domain:** Mining / Mineral Exploration / Production Intelligence

> **Problem Statement:** Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome Production Shortfalls.

---

# 1. Project Overview

This project develops an **AI-assisted manganese exploration and production intelligence platform** for supporting mineral exploration and mine-planning decisions.

The platform combines:

- Satellite remote sensing
- Geological and lithological information
- Known manganese occurrences
- Terrain information
- Sentinel-1 SAR data
- Machine learning
- Spatial validation
- Explainable AI
- Interactive geospatial visualization

The current implementation focuses on **AI-assisted manganese mineral prospectivity mapping**.

The exploration module identifies locations with relatively stronger combinations of geological, spectral, radar, terrain and spatial indicators and converts these into **exploration-priority zones**.

The next major development stage is to generalize the exploration system from the current **Balaghat pilot area to India-wide operation**.

A separate production-intelligence module will subsequently address:

- Production shortfall prediction
- Equipment downtime
- Weather effects
- Operational delays
- Mine scheduling
- Corrective-action recommendations

---

# 2. Problem We Are Solving

Manganese exploration and production planning involve several sources of information:

- Geological maps
- Known mineral occurrences
- Remote sensing imagery
- Terrain
- Structural information
- Historical exploration data
- Equipment performance
- Weather
- Mine operations

These datasets are often difficult to analyze together manually.

The proposed platform provides an AI-assisted workflow that:

1. Integrates geological and remote-sensing information.
2. Extracts useful spatial features.
3. Uses known manganese occurrences as training information.
4. Trains a machine-learning prospectivity model.
5. Performs spatial validation.
6. Generates a relative prospectivity score.
7. Identifies exploration-priority cells and zones.
8. Provides an interactive map for exploration analysis.
9. Eventually extends the same platform to production-shortfall prediction and corrective-action planning.

---

# 3. Core Objective

The exploration system is designed to answer:

> **"Where should geologists investigate first?"**

It is **not** designed to answer:

> "Has manganese definitely been found underground at this location?"

Satellite and geological data provide exploration indicators. Actual mineralization must still be confirmed through geological investigation, sampling, trenching, drilling and other appropriate exploration methods.

---

# 4. Current Architecture

```text
                    GSI Manganese Occurrences
                              +
                    Sentinel-2 Optical Data
                              +
                    Sentinel-1 SAR Data
                              +
                       SRTM Terrain
                              +
                Geological / Lithological Data
                              |
                              v
                    Feature Engineering
                              |
                              v
                         XGBoost
                              |
                              v
                    Spatial Validation
                              |
                              v
                     SHAP Explainability
                              |
                              v
                    Prospectivity Score
                              |
                              v
                       Target Cells
                              |
                              v
                     Target-Zone Clustering
                              |
                              v
                     Interactive Dashboard


# 5. Current Development Status
Completed
Phase 1 — Remote Sensing Pipeline
 Sentinel-2 integration
 Sentinel-2 image filtering
 Sentinel-2 feature extraction
 Initial Balaghat feature dataset
 Google Earth Engine pipeline

Phase 2 — Geological Occurrence Integration
 GSI manganese occurrence dataset
 Occurrence filtering
 Balaghat / Madhya Pradesh occurrence preparation
 Training-label preparation

Phase 3 — Multi-Sensor Feature Integration
 Sentinel-1 integration
 SRTM integration
 Combined remote-sensing feature grid
 Initial terrain features
 Radar features

Phase 4B — Improved Prospectivity Model
 Geological/lithological integration
 Structural proxy features
 Spectral feature expansion
 Spatially controlled sample balancing
 XGBoost retraining
 5-fold spatial validation
 SHAP explainability
 Leakage audit

Phase 5 — Prospectivity Mapping
 Balaghat prediction grid
 Prospectivity scoring
 Target-cell generation
 Target-zone clustering
 CSV output
 GeoJSON output
 Prospectivity map
 Independent GSI occurrence validation
 Geological enrichment audit

Interactive Application
 Streamlit prototype
 Single-location prediction
 Rectangle AOI selection
 Polygon AOI selection
 Prospectivity visualization
 Target-zone visualization

# 6. Phase 4B Spatial Validation
The current Phase 4B model was evaluated using 5-fold spatially grouped validation.
Results

| Metric            |                Mean |
| ----------------- | ------------------: |
| ROC-AUC           | **0.6210 ± 0.2085** |
| PR-AUC            | **0.4746 ± 0.1422** |
| Precision         | **0.3743 ± 0.2626** |
| Recall            | **0.1833 ± 0.1733** |
| F1                | **0.2391 ± 0.2065** |
| Accuracy          | **0.6556 ± 0.0514** |
| Precision@Top 5%  |   **0.50 ± 0.3953** |
| Precision@Top 10% |  **0.425 ± 0.1896** |
| Precision@Top 20% |   **0.44 ± 0.1535** |

# 7. Phase 4B Spatial Validation
The Phase 4B model identified contributions from both geological and remote-sensing information.
Important features include:

Geological lithology
Geological group
Geological formation
Stratigraphy
Geological boundary density
Geological boundary distance
Elevation
Slope
Lithological diversity
Spectral ratios
NDMI
Radar VV/VH information

Some of the strongest model-level features include:

geo_lithology_MUSCOVITE SCHIST
geo_group_AMGAON GNEISSIC COMPLEX
geo_formation_CHORBAOLI
geo_lithology_QUARTZITE
geo_stratigraphy_PALAEOPROTEROZOIC-BIOTITE-SCHIST
Geo_Boundary_Density_3km
geo_lithology_BIOTITE SCHIST
geo_age_ARCHAEAN
Geo_Boundary_Density_1km
Elevation
NIR_SWIR2_Ratio
Geo_Boundary_Distance_km
Lithology_Diversity_3km
NDMI

# 8. SHAP Explainability

SHAP analysis was performed on the current 117-feature model.

Important SHAP contributors include:

Geo_Boundary_Distance_km
Geo_Boundary_Density_3km
Elevation
Slope
GNEISS/MIGMATITE
VV_VH_Difference
Iron_Oxide_Index
Sausar_Group_Proxy
MANSAR formation
Red_Green_Ratio
Lithology_Diversity_3km
Geo_Boundary_Density_1km

The SHAP results indicate that the model is using a combination of:

Geological information
Geological boundaries
Terrain
Spectral information
Radar information

# 9. Data Sources

The project uses publicly available geological, satellite, terrain and mineral-occurrence datasets. The current Balaghat implementation uses these datasets for model development and validation, while the next stage will generalize the same data architecture to India-wide operation.

---

## 9.1 Geological Survey of India (GSI) — Manganese Occurrences

**Source:** Geological Survey of India (GSI) / Open Government Data (OGD)

**Dataset:** Location of Manganese Ore Deposits in India and its Salient Features

**Official source:**

https://www.data.gov.in/catalog/location-manganese-ore-deposits-india-and-its-salient-features

The dataset provides known manganese occurrence/locality information across India.

Relevant fields include:

- Locality / occurrence name
- State
- Toposheet information
- Latitude
- Longitude
- Host rock
- Stratigraphic position
- Metallogenesis
- Morphogenesis

### Project usage

The occurrence data is used to:

- Identify known manganese-bearing locations.
- Construct positive training samples.
- Support spatially controlled background sampling.
- Validate prospectivity predictions.
- Provide geological context for manganese mineralization.

### Important limitation

Known occurrences represent **observed mineralization locations**, not the complete distribution of manganese deposits.

Unknown locations must therefore not automatically be interpreted as true negative samples.

---

## 9.2 GSI / NGDR — Geological and Lithological Data

**Source:** Geological Survey of India (GSI) / National Geoscience Data Repository (NGDR)

**NGDR portal:**

https://geodataindia.gov.in/

**GSI Open Government Data:**

https://www.data.gov.in/ministrydepartment/Geological%20Survey%20of%20India

NGDR provides access to standardized geoscientific datasets that can be used for GIS, spatial analysis and AI/ML applications.

### Project usage

The project uses geological/lithological information to represent the geological environment surrounding potential manganese mineralization.

The current feature pipeline incorporates attributes such as:

- Geological age
- Supergroup
- Group
- Formation
- Lithology
- Intrusive information
- Stratigraphy

Derived spatial features include:

- Geological boundary distance
- Geological boundary density
- Lithological diversity
- Formation diversity
- Geological proxies

### Balaghat pilot

For the Balaghat pilot, the NGDR-derived lithological dataset was spatially filtered to the study area.

The local extraction contains approximately:

**14,443 geological/lithological records**

after processing the relevant full dataset.

### Important limitation

Geological data coverage and attribute completeness can vary between regions.

Large raw NGDR datasets are intentionally **not stored in this GitHub repository** because of their size.

---

## 9.3 GSI Geological Maps

**Source:** Geological Survey of India

GSI geological maps provide information about:

- Rock types
- Geological formations
- Lithology
- Structural features
- Faults
- Geological boundaries
- Mineral occurrences

A publicly available geological maps dataset is also listed through the National Water Data Portal:

https://nwdp.nwic.in/dataset/geological-maps

### Project usage

Geological maps support:

- Geological feature extraction
- Lithological classification
- Geological boundary analysis
- Structural proxy generation
- Geological interpretation of high-prospectivity zones

---

## 9.4 Sentinel-2 Multispectral Imagery

**Source:** Copernicus Sentinel-2 / European Space Agency

**Google Earth Engine dataset:**

```text
COPERNICUS/S2_SR_HARMONIZED

## 9.5 GitHub — Indian Land Features / NGDR-derived Spatial Data

A public GitHub project providing processed Indian spatial datasets derived from government geospatial sources:

https://github.com/ramSeraph/indian_land_features

Relevant releases include NGDR-derived geological/lithological datasets such as:

NGDR_Geology_2M.geojsonl
NGDR_Lithology_50k.

## 9.6 Sentinel-1 SAR Data

**Source:** Copernicus Sentinel-1 / European Space Agency

**Google Earth Engine dataset:**

```text
COPERNICUS/S1_GRD

## 9.7 SRTM Digital Elevation Data

Source: NASA / USGS

Google Earth Engine dataset:

USGS/SRTMGL1_003

SRTM provides Digital Elevation Model (DEM) information.

Project usage

Terrain features derived from SRTM include:

Elevation
Slope
Aspect

Terrain information is used to provide geomorphological and spatial context for prospectivity modelling.

Derived terrain features are treated as supporting indicators rather than direct evidence of manganese mineralization.

Official documentation:

https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003

# 10. Project Structure
MOIL_SIH_2026/
│
├── app/
│   ├── app.py
│   └── prediction_engine.py
│
├── data/
│   ├── geology/
│   ├── gsi/
│   └── *.csv
│
├── docs/
│   └── PROJECT_STATUS.md
│
├── models/
│   ├── moil_manganese_prospectivity_xgb.joblib
│   ├── moil_manganese_prospectivity_xgb_phase4b.joblib
│   ├── phase4b_preprocessor.joblib
│   └── phase4b_feature_columns.json
│
├── outputs/
│   ├── phase4b_*.csv
│   ├── phase4b_*.png
│   ├── phase5_*.csv
│   ├── phase5_*.geojson
│   └── phase5_*.png
│
├── scripts/
│   ├── attach_geology.py
│   ├── balance_phase4b_samples.py
│   ├── build_phase4b_features.py
│   ├── extract_balaghat_lith.py
│   ├── inspect_balaghat_attributes.py
│   ├── inspect_ngdr.py
│   ├── phase4b_leakage_audit.py
│   ├── phase4b_real_leakage_audit.py
│   ├── phase4b_shap.py
│   ├── phase5_baseline_validation.py
│   ├── phase5_final_prospectivity.py
│   ├── phase5_geology_enrichment.py
│   ├── phase5_validation.py
│   └── train_phase4b.py
│
├── README.md
├── requirements.txt
└── .gitignore

# 11. Project Architecture
                         ┌───────────────────────┐
                         │     PUBLIC DATA       │
                         └───────────┬───────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
             v                       v                       v
        GSI / NGDR              Sentinel Data             SRTM
        Geology & Mn            S1 + S2                   Terrain
        Occurrences
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     v
                         ┌───────────────────────┐
                         │ DATA PREPROCESSING    │
                         │ & SPATIAL ALIGNMENT   │
                         └───────────┬───────────┘
                                     │
                                     v
                         ┌───────────────────────┐
                         │ FEATURE ENGINEERING   │
                         │                       │
                         │ Geological            │
                         │ Spectral              │
                         │ Radar                 │
                         │ Terrain               │
                         │ Structural Proxies    │
                         └───────────┬───────────┘
                                     │
                                     v
                         ┌───────────────────────┐
                         │ TRAINING DATASET      │
                         │                       │
                         │ Occurrence-based      │
                         │ Positive Samples      │
                         │ +                     │
                         │ Spatial Background    │
                         └───────────┬───────────┘
                                     │
                                     v
                         ┌───────────────────────┐
                         │ LEAKAGE AUDIT         │
                         │                       │
                         │ Coordinates removed   │
                         │ Distance-to-Mn        │
                         │ removed               │
                         │ Spatial controls      │
                         └───────────┬───────────┘
                                     │
                                     v
                         ┌───────────────────────┐
                         │       XGBOOST         │
                         │  PROSPECTIVITY MODEL  │
                         └───────────┬───────────┘
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                         v                       v
                ┌────────────────┐      ┌────────────────┐
                │ SPATIAL        │      │ SHAP           │
                │ VALIDATION     │      │ EXPLAINABILITY │
                └───────┬────────┘      └───────┬────────┘
                        │                       │
                        └───────────┬───────────┘
                                    │
                                    v
                         ┌───────────────────────┐
                         │ SPATIAL PREDICTION    │
                         │ GRID                  │
                         └───────────┬───────────┘
                                     │
                                     v
                         ┌───────────────────────┐
                         │ PROSPECTIVITY SCORE   │
                         │       0 - 100         │
                         └───────────┬───────────┘
                                     │
                                     v
                         ┌───────────────────────┐
                         │ HIGH-PRIORITY CELLS   │
                         └───────────┬───────────┘
                                     │
                                     v
                         ┌───────────────────────┐
                         │ TARGET-ZONE           │
                         │ CLUSTERING            │
                         └───────────┬───────────┘
                                     │
                                     v
                         ┌───────────────────────┐
                         │ GEOJSON / MAP /       │
                         │ STREAMLIT DASHBOARD   │
                         └───────────┬───────────┘
                                     │
                                     v
                         ┌───────────────────────┐
                         │ EXPLORATION           │
                         │ DECISION SUPPORT      │
                         └───────────────────────┘


                    FUTURE PRODUCTION MODULE
                    =========================

             Production + Equipment + Weather
                            │
                            v
                   Feature Engineering
                            │
                            v
                   Shortfall Prediction
                            │
                            v
                     Risk Analysis
                            │
                            v
                 Corrective Recommendations
                            │
                            v
                   Production Decision
                       Support