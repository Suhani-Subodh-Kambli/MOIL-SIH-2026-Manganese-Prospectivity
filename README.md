# MOIL SIH 2026 — AI-Assisted Manganese Exploration & Production Intelligence

## Smart India Hackathon 2026

### Problem Statement

Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome Production Shortfalls.

Organization: MOIL Limited

Problem Statement ID: 26009

---

# Project Overview

This project develops an AI-assisted exploration and production intelligence platform for manganese mining.

The system combines:

- Satellite remote sensing
- Geological and lithological information
- Known manganese occurrences
- Terrain information
- Radar information
- Machine learning
- Spatial validation
- Explainable AI

The current exploration module generates manganese mineral prospectivity maps and exploration-priority zones.

The next development stage will add production shortfall prediction and corrective-action recommendations.

---

# Current Architecture

GSI manganese occurrences
        +
Sentinel-2
        +
Sentinel-1
        +
SRTM terrain
        +
Geological/lithological evidence
        ↓
Feature engineering
        ↓
XGBoost
        ↓
Spatial validation
        ↓
SHAP explainability
        ↓
Prospectivity score
        ↓
Target zones
        ↓
Dashboard

---

# Current Status

## Completed

- Sentinel-2 feature pipeline
- Sentinel-1 feature integration
- SRTM terrain features
- GSI manganese occurrence integration
- NGDR geological/lithological integration
- Geological feature engineering
- Spectral feature engineering
- Spatial sample balancing
- XGBoost prospectivity model
- 5-fold spatial validation
- SHAP explainability
- Balaghat prospectivity prediction
- Target-cell generation
- Target-zone clustering
- CSV outputs
- GeoJSON outputs
- Prospectivity visualization

## Pending

- Production shortfall prediction
- Equipment-performance integration
- Weather/operational feature integration
- Corrective-action recommendation engine
- India-wide deployment
- Final web dashboard
- End-to-end application integration

---

# Phase 4B Model

Current model:

XGBoost

Transformed feature count:

117

Spatial validation:

5-fold spatial validation

Mean ROC-AUC:

0.6210 ± 0.2085

Mean PR-AUC:

0.4746 ± 0.1422

The current model demonstrates useful signal but spatial performance is variable. It should therefore be treated as an exploration-prioritization model rather than a definitive mineral-deposit detector.

---

# Phase 5 Output

Current Balaghat prediction grid:

14,061 cells

Current exploration-priority cells:

1,407

Current target zones:

94

The prospectivity score is a relative ranking from 0–100.

It is NOT:

- manganese concentration
- ore grade
- proven reserves
- probability of economic extraction

High-priority areas require geological field validation and drilling.

---

# Data Sources

## Sentinel-2

Google Earth Engine:

COPERNICUS/S2_SR_HARMONIZED

## Sentinel-1

Google Earth Engine:

COPERNICUS/S1_GRD

## SRTM

Google Earth Engine:

USGS/SRTMGL1_003

## Manganese occurrences

Geological Survey of India / Open Government Data.

## Geological data

NGDR/GSI-derived geological and lithological data.

Large raw geological datasets are intentionally not stored in this repository.

---

# Project Structure

```text
MOIL_SIH_2026/
│
├── data/
├── models/
├── scripts/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore