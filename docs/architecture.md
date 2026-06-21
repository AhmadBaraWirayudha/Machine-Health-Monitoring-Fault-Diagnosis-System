# System Architecture
## Machine Health Monitoring & Fault Diagnosis System

---

## Overview

This system is a seven-layer Condition-Based Monitoring (CBM) pipeline
that ingests open-source sensor data, extracts diagnostic features, scores
asset health using machine learning, persists results to a structured
database, and surfaces insights through a REST API, Streamlit dashboard,
and Power BI.

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Data Sources                                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ AI4I 2020 (UCI) │  │  NASA Bearing    │  │  Custom CSV/DB   │  │
│  │ Tabular sensor  │  │  Vibration .csv  │  │  (plug-in)       │  │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬─────────┘  │
└───────────┼────────────────────┼─────────────────────┼────────────┘
            │                    │                      │
┌───────────▼────────────────────▼─────────────────────▼────────────┐
│  Layer 2 — Ingestion  (src/ingestion/)                             │
│  download_data.py — HTTP download, Kaggle API, CSV loader          │
└───────────────────────────────┬────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────┐
│  Layer 3 — Preprocessing  (src/preprocessing/)                     │
│  clean_data.py — dedup, impute, IQR flag, normalise, segment       │
└───────────────────────────────┬────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────┐
│  Layer 4 — Feature Extraction  (src/features/)                     │
│  feature_extraction.py — RMS, kurtosis, crest factor, FFT,         │
│                           band energy, spectral entropy             │
│  bearing_diagnostics.py — BPFO, BPFI, BSF, FTF, envelope order    │
└───────────────────────────────┬────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────┐
│  Layer 5 — Modelling  (src/modeling/)                              │
│  ┌─────────────────────┐   ┌──────────────────────────────────┐   │
│  │  health_scorer.py   │   │  fault_classifier.py             │   │
│  │  Isolation Forest   │   │  Random Forest (SMOTE, Optuna)   │   │
│  │  → AHS 0–100        │   │  → Fault type + confidence       │   │
│  └──────────┬──────────┘   └────────────────┬─────────────────┘   │
│             └──────────────┬────────────────┘                      │
│                 ┌──────────▼──────────┐                            │
│                 │  model_registry.py  │                            │
│                 │  Version + promote  │                            │
│                 └─────────────────────┘                            │
└───────────────────────────────┬────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────┐
│  Layer 6 — Storage  (src/database/ + sql/)                         │
│  db_manager.py — SQLite CRUD, views, bulk load                     │
│  schema.sql    — 6 tables, 2 views, indexes                        │
│  analytical_queries.sql — 15 analytical queries                    │
│  export_for_powerbi.py  — flat CSV/XLSX export                     │
└──────┬─────────────────┬──────────────────────────────────────────┘
       │                 │
┌──────▼──────┐   ┌──────▼──────────────────────────────────────────┐
│  Layer 7a   │   │  Layer 7b — Visualisation & Reporting           │
│  REST API   │   │  src/visualization/plots.py — matplotlib charts  │
│  src/api/   │   │  reports/report_template.html — Jinja2 report    │
│  FastAPI    │   │  powerbi/powerbi_guide.md — DAX + dashboard      │
│  /predict   │   │  src/dashboard/app.py — Streamlit (5 pages)      │
│  /assets    │   │  web/index.html — portfolio landing page         │
│  /fleet     │   └──────────────────────────────────────────────────┘
└─────────────┘
```

---

## Module Dependency Graph

```
main.py
  ├── src.ingestion.download_data
  ├── src.preprocessing.clean_data
  │     └── src.utils.helpers
  ├── src.features.feature_extraction
  │     └── src.features.bearing_diagnostics
  ├── src.modeling.health_scorer
  │     └── src.utils.helpers
  ├── src.modeling.fault_classifier
  │     └── src.modeling.model_registry
  ├── src.database.db_manager
  ├── src.visualization.plots
  ├── reports.generate_report
  └── src.alerts.notifier

src.api.main
  ├── src.database.db_manager
  └── src.modeling.fault_classifier (inference)

src.dashboard.app
  ├── src.database.db_manager
  └── src.modeling.fault_classifier (live predict)
```

---

## Data Flow

```
Raw CSV
  → pandas DataFrame (14 columns)
  → Cleaned DataFrame (+ is_outlier, power_W, temp_diff_K, fault_type)
  → Feature DataFrame (+ rolling stats, [rms, kurtosis, ...] for vibration)
  → Scored DataFrame  (+ health_score, health_status, recommendation)
  → Predicted DataFrame (+ predicted_fault, fault_confidence)
  → SQLite tables (assets, inspections, sensor_readings, features, health_scores, recs)
  → CSV export / Power BI
  → HTML report / Streamlit dashboard / FastAPI response
```

---

## Database Schema (ERD)

```
assets ──────────┬──── inspections ──┬── sensor_readings
  asset_id (PK)  │       asset_id    │       inspection_id
  asset_name     │       inspection_id(PK)   air_temp_K
  asset_type     │       inspect_date         process_temp_K
  location       │       method               rotational_speed_rpm
                 │       inspector            torque_Nm
                 │                    └── feature_records
                 │                            inspection_id
                 │                            rms, kurtosis
                 │                            crest_factor, ...
                 │
                 ├──── health_scores
                 │       asset_id
                 │       score_id (PK)
                 │       health_score  ──┐
                 │       health_status   │
                 │       fault_type      │
                 │                       │
                 └──── recommendations ──┘
                         asset_id
                         score_id (FK)
                         priority
                         action
                         status
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data ingestion | Python `requests`, Kaggle API | Download public datasets |
| Preprocessing | `pandas`, `sklearn.preprocessing` | Clean, normalise, segment |
| Feature extraction | `numpy`, `scipy.fft`, `scipy.signal` | Time + frequency features |
| Anomaly detection | `sklearn.IsolationForest` | Unsupervised outlier scoring |
| Fault classification | `sklearn.RandomForestClassifier` | Supervised fault type ID |
| Imbalance handling | `imbalanced-learn` (SMOTE) | Upsample minority fault classes |
| Hyperparameter tuning | `optuna` | Bayesian optimisation |
| Model registry | Custom JSON + joblib store | Versioning + promotion |
| Storage | `sqlite3`, SQL | Structured inspection history |
| REST API | `FastAPI`, `uvicorn` | JSON endpoints for integration |
| Dashboard | `streamlit`, `plotly` | Interactive web UI |
| Reporting | `jinja2`, `matplotlib` | HTML technical report |
| Statistical analysis | `R` (ggplot2, survival, corrplot) | Reliability + Weibull |
| Signal processing | MATLAB (FFT, envelope, Hilbert) | Vibration validation |
| CI/CD | GitHub Actions | Lint → test → smoke → Docker |
| Containerisation | Docker, Compose, Nginx | Portable deployment |

---

## Deployment Options

### Option A — Local (development)
```bash
pip install -r requirements.txt -r requirements-extra.txt
python main.py                          # pipeline
streamlit run src/dashboard/app.py     # dashboard
uvicorn src.api.main:app --reload      # API
```

### Option B — Docker Compose (production-like)
```bash
docker compose up api dashboard        # start services
docker compose run --rm pipeline       # run pipeline once
```

### Option C — Docker + Nginx (full stack)
```bash
# Uncomment nginx service in docker-compose.yml
docker compose up nginx api dashboard
# Dashboard: http://localhost/
# API:       http://localhost/api/
# Docs:      http://localhost/docs
```

---

## Configuration (`config.yaml`)

All tuneable parameters live in `config.yaml`:

| Section | Key | Default | Effect |
|---|---|---|---|
| `database` | `path` | `data/processed/cmdb.sqlite` | SQLite file location |
| `signal` | `sampling_rate` | `25600` Hz | FFT bin resolution |
| `signal` | `window_size` | `512` samples | Segment length |
| `signal` | `overlap` | `0.5` | 50% window overlap |
| `model.classifier` | `n_estimators` | `100` | RF tree count |
| `model.classifier` | `max_depth` | `10` | RF max depth |
| `model.anomaly` | `contamination` | `0.05` | Isolation Forest anomaly rate |
| `health_score.weights` | `rms` etc. | various | AHS component weights |
| `health_score.thresholds` | `good` | `80` | Status boundary |

---

## Security Considerations

- **No auth on API by default** — add `src/api/auth.py` (API key middleware) for production
- **SQLite** — fine for portfolio; swap for PostgreSQL for multi-user production use
- **Docker** — runs as non-root `cbmuser` (UID 1000)
- **`.env`** — never commit; contains SMTP/Slack credentials
- **Model files** — versioned via `model_registry.py`; hash-verified on load

---

## Performance Benchmarks (approximate)

| Step | Dataset size | Time |
|---|---|---|
| Data download | 10,000 rows | ~2 s |
| Preprocessing | 10,000 rows | ~0.3 s |
| Feature extraction (tabular) | 10,000 rows | ~0.5 s |
| Anomaly detection (train) | 10,000 rows | ~1 s |
| RF classifier (train) | 10,000 rows | ~3 s |
| DB bulk load | 10,000 rows | ~2 s |
| Plot generation | 7 charts | ~4 s |
| Full pipeline | 10,000 rows | ~15 s |
| API `/predict` latency | 1 reading | ~5 ms |
| Dashboard load | First render | ~3 s |
