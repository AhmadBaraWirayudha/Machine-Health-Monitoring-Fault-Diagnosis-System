# Machine Health Monitoring & Fault Diagnosis System

**Predictive Maintenance and Asset Health Monitoring System for Rotating Equipment**  
Using Python · SQL · Power BI · HTML

---

## Project Overview

This project simulates a real-world **Condition-Based Monitoring (CBM)** engineering workflow:

1. Ingest open-source vibration / sensor data from public repositories
2. Clean and extract diagnostic features (RMS, kurtosis, FFT peaks, etc.)
3. Score asset health and detect faults using ML
4. Store results in a structured SQL database
5. Visualize trends in Power BI dashboard
6. Deliver a professional HTML/PDF technical report

---

## Architecture

```
Website Dataset → Python Ingestion → Feature Extraction → SQL Database
      → Anomaly Detection / Fault Classifier → Power BI Dashboard → HTML Report
```

---

## Folder Structure

```
project/
├── data/
│   ├── raw/            ← original downloaded datasets
│   ├── processed/      ← cleaned data, feature tables, SQLite DB
│   └── external/       ← reference tables, fault codes
├── notebooks/          ← exploratory analysis
├── src/
│   ├── ingestion/      ← download & load data
│   ├── preprocessing/  ← cleaning, normalization
│   ├── features/       ← RMS, kurtosis, FFT, band energy
│   ├── modeling/       ← health scorer, fault classifier
│   ├── database/       ← SQLite manager (insert/query)
│   └── visualization/  ← matplotlib plots for reports
├── matlab/             ← MATLAB scripts for vibration analysis
├── sql/                ← raw SQL schema and queries
├── powerbi/            ← Power BI .pbix file (manual step)
├── reports/            ← HTML/PDF technical reports
├── web/                ← portfolio landing page
├── models/             ← saved ML models (.joblib)
├── config.yaml
├── main.py             ← full pipeline runner
└── requirements.txt
```

---

## Datasets

| Dataset | Type | Source |
|---|---|---|
| AI4I 2020 Predictive Maintenance | Tabular (sensor + failure labels) | UCI ML Repository |
| NASA Bearing Dataset | Time-series vibration | NASA Prognostics Center |
| Lab-Scale Vibration Dataset | Vibration (normal/fault conditions) | arXiv 2212.14732 |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python src/ingestion/download_data.py

# 3. Preprocess & extract features
python src/preprocessing/clean_data.py
python src/features/feature_extraction.py

# 4. Run health scoring & fault detection
python src/modeling/health_scorer.py
python src/modeling/fault_classifier.py

# 5. Push results to database
python src/database/db_manager.py

# 6. Generate HTML report
python src/visualization/plots.py

# Or run the full pipeline at once:
python main.py
```

---

## Key Features Extracted

| Feature | Domain | Description |
|---|---|---|
| RMS | Time | Root Mean Square — overall vibration level |
| Kurtosis | Time | Detects impulsive bearing faults |
| Crest Factor | Time | Peak / RMS — severity of shock events |
| Peak-to-Peak | Time | Max amplitude range |
| Skewness | Time | Signal asymmetry |
| FFT Peak | Frequency | Dominant frequency and amplitude |
| Band Energy | Frequency | Energy in specific frequency bands |
| Spectral Entropy | Frequency | Frequency complexity / disorder |

---

## Health Score Scale

| Score | Status | Action |
|---|---|---|
| 80 – 100 | 🟢 Good | Routine monitoring |
| 60 – 79 | 🟡 Warning | Increase inspection frequency |
| 40 – 59 | 🟠 Degraded | Schedule maintenance |
| 0 – 39 | 🔴 Critical | Immediate action required |

---

## Tech Stack

- **Python** — data pipeline, ML, report automation
- **SQL / SQLite** — structured storage of inspection history
- **Power BI** — operational health dashboard
- **HTML / CSS** — web portfolio and technical report
- **MATLAB** — vibration signal analysis and FFT validation

---

## Deliverables

- [x] Raw dataset from public website
- [x] Cleaned feature table (CSV)
- [x] SQLite database with full inspection history
- [x] Fault classifier model (`.joblib`)
- [x] Health score trend per asset
- [x] HTML technical report
- [x] Portfolio web page
- [ ] Power BI dashboard (manual — connect to `data/processed/cmdb.sqlite`)
