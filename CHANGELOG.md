# Changelog
## Machine Health Monitoring & Fault Diagnosis System

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2025-06-20

### Added
- **Data ingestion** (`src/ingestion/download_data.py`)
  - AI4I 2020 Predictive Maintenance dataset from UCI ML Repository
  - NASA Bearing dataset loader (requires Kaggle API token)
  - Generic CSV loader for custom datasets

- **Preprocessing** (`src/preprocessing/clean_data.py`)
  - Duplicate removal, missing value imputation (median strategy)
  - IQR-based outlier flagging (non-destructive)
  - Min-max and z-score normalisation with scaler persistence
  - Signal segmentation with configurable window size and overlap
  - Derived features: `power_W`, `temp_diff_K`, `fault_type` label

- **Feature extraction** (`src/features/feature_extraction.py`)
  - Time domain: RMS, kurtosis, crest factor, peak-to-peak, skewness, variance
  - Frequency domain: FFT peak frequency/amplitude, band energies (low/mid/high), spectral entropy
  - Rolling statistical features on tabular sensor data

- **Health scoring** (`src/modeling/health_scorer.py`)
  - Asset Health Score (AHS): 0–100, weighted combination of feature scores
  - Isolation Forest anomaly detector (configurable contamination)
  - Status classification: Good / Warning / Degraded / Critical
  - Rule-based maintenance recommendation engine

- **Fault classifier** (`src/modeling/fault_classifier.py`)
  - Random Forest classifier with balanced class weights
  - 5-fold cross-validation and full classification report
  - Feature importance ranking
  - Joblib model persistence

- **SQLite database** (`src/database/db_manager.py`, `sql/schema.sql`)
  - 6-table schema: assets, inspections, sensor_readings, feature_records, health_scores, recommendations
  - Analytical views: `v_latest_health`, `v_open_recommendations`
  - Bulk-load from DataFrame

- **Analytical SQL queries** (`sql/analytical_queries.sql`)
  - Fleet overview KPIs
  - Asset detail drillthrough queries
  - Health trend and anomaly queries
  - Maintenance backlog and activity reports

- **Visualisation** (`src/visualization/plots.py`)
  - Time-domain waveform plot
  - FFT spectrum with peak annotation
  - Feature trend over time (coloured by health status)
  - Health score distribution + status pie chart
  - Correlation heatmap
  - Fault distribution bar chart
  - 4-panel dashboard summary figure

- **HTML technical report** (`reports/report_template.html`, `reports/generate_report.py`)
  - Jinja2-templated report with KPI cards, tables, embedded plots
  - Auto-populated from SQLite + scored CSV

- **Portfolio web page** (`web/index.html`)
  - 7-step pipeline diagram
  - Diagnostic feature cards
  - Sample health score display
  - Deliverables checklist

- **MATLAB script** (`matlab/vibration_analysis.m`)
  - FFT, envelope analysis (Hilbert demodulation), KM-based health rule

- **R reliability analysis** (`r/reliability_analysis.R`)
  - Descriptive statistics
  - Kaplan-Meier survival curve (tool wear as time proxy)
  - Wilcoxon rank-sum tests (normal vs fault)
  - Correlation matrix (corrplot)
  - Box plots and health score trend (ggplot2)

- **Power BI guide** (`powerbi/powerbi_guide.md`)
  - ODBC and CSV connection instructions
  - 15+ DAX measures (KPIs, trends, reliability)
  - 4-page dashboard layout specification
  - Custom colour theme JSON

- **Power BI export** (`src/database/export_for_powerbi.py`)
  - Exports all tables, views, and 6 pre-aggregated query results as CSV
  - Optional multi-sheet Excel workbook

- **Jupyter notebooks** (`notebooks/`)
  - `01_exploratory_analysis.ipynb`: distributions, correlations, fault analysis, EDA
  - `02_modeling.ipynb`: model comparison, confusion matrix, ROC, learning curves, Isolation Forest tuning

- **Utilities** (`src/utils/helpers.py`)
  - Logger factory, `@timed` decorator, `PipelineRun` context manager
  - Path helpers, data validation, safe division, rolling z-score

- **Unit tests** (`tests/test_pipeline.py`)
  - 30+ pytest cases covering feature extraction, preprocessing, health scoring, utilities

- **Pipeline runner** (`main.py`)
  - Orchestrates all 8 steps with timing and error handling
  - `--skip-download` and `--steps` arguments

- **Makefile**
  - `make run`, `make data`, `make model`, `make report`, `make export`, `make test`, `make clean`

- **Configuration** (`config.yaml`)
  - Centralised settings for paths, features, model hyperparameters, health thresholds

---

## Planned — v1.1.0

- [ ] Streamlit dashboard (web-based, no Power BI licence required)
- [ ] Email/SMS alert when asset drops below Critical threshold
- [ ] SMOTE oversampling for fault class imbalance
- [ ] Hyperparameter tuning with Optuna
- [ ] Docker container for portable deployment
- [ ] REST API endpoint (`/predict` and `/health-score`) via FastAPI
