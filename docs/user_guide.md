# User Guide
## Machine Health Monitoring & Fault Diagnosis System

---

## Getting Started

### 1. Install

```bash
git clone https://github.com/your-username/cbm-health-monitoring
cd cbm-health-monitoring

# Automated setup (creates venv, installs deps, runs tests)
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh

# Or manual:
pip install -r requirements.txt -r requirements-extra.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This executes all 8 steps automatically:

| Step | What happens | Output |
|---|---|---|
| 1 | Download AI4I 2020 dataset | `data/raw/ai4i2020.csv` |
| 2 | Clean and normalise | `data/processed/ai4i_clean.csv` |
| 3 | Extract features | `data/processed/ai4i_features.csv` |
| 4 | Score asset health (0–100) | `data/processed/ai4i_health_scored.csv` |
| 5 | Predict fault types | `data/processed/ai4i_predictions.csv` |
| 6 | Load results to SQLite | `data/processed/cmdb.sqlite` |
| 7 | Generate diagnostic plots | `reports/plots/*.png` |
| 8 | Render HTML report | `reports/technical_report.html` |

---

## Running Individual Steps

```bash
# Download dataset only
python main.py --steps 1

# Preprocess + features (skip download if already done)
python main.py --skip-download --steps 2 3

# Re-run modelling + DB only
python main.py --skip-download --steps 4 5 6

# Regenerate report only
python main.py --skip-download --steps 7 8
```

---

## Streamlit Dashboard

The interactive dashboard requires no Power BI licence.

```bash
streamlit run src/dashboard/app.py
```

Open **http://localhost:8501** in your browser.

### Dashboard Pages

**🏠 Fleet Overview**
- KPI cards: total assets, average health score, critical count, availability
- Health score bar chart per asset
- Status donut chart
- Asset fleet table with colour-coded status

**📈 Asset Detail**
- Select an asset in the sidebar
- Health score gauge and trend chart
- Sensor reading history table
- Open maintenance recommendations

**⚠️ Fault Analysis**
- Fault type distribution bar + pie
- Average health score by fault type
- Observation scatter coloured by fault

**🔧 Maintenance Backlog**
- Open actions count, overdue count
- Priority-sorted action table with colour coding
- Completion trend

**🤖 Live Predict**
- Adjust sensor sliders
- Click Run Prediction
- Instant fault type + health score + recommendation

> **Sidebar Refresh** — click 🔄 Refresh Data to reload from the database.

---

## REST API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for Swagger UI.

### Key endpoints

**Predict fault + health score from sensor readings:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temp_K": 298.1,
    "process_temp_K": 309.5,
    "rotational_speed_rpm": 1498,
    "torque_Nm": 34.2,
    "tool_wear_min": 150
  }'
```

Response:
```json
{
  "fault_type": "Normal",
  "confidence": 0.9412,
  "health_score": 74.3,
  "health_status": "Warning",
  "recommendation": "Increase vibration check frequency. Monitor trend.",
  "power_W": 5364.2,
  "temp_diff_K": 11.4,
  "timestamp": "2025-06-20T14:22:31Z"
}
```

**Get fleet summary:**
```bash
curl http://localhost:8000/fleet/summary
```

**Get asset history:**
```bash
curl http://localhost:8000/assets/MTR-001/history?limit=50
```

**Batch prediction (up to 500 readings):**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"air_temp_K":298,"process_temp_K":309,...}, ...]'
```

### Adding API key authentication

```bash
# Generate a key
python src/api/auth.py generate --prefix cbm_prod

# Add to .env
echo "API_KEYS=cbm_prod_abc123..." >> .env

# Enable in src/api/main.py
# from src.api.auth import APIKeyMiddleware
# app.add_middleware(APIKeyMiddleware)

# Call with key
curl -H "X-API-Key: cbm_prod_abc123..." http://localhost:8000/predict ...
```

---

## Power BI Dashboard

See `powerbi/powerbi_guide.md` for complete setup.

**Quick start:**
1. Export flat files: `python src/database/export_for_powerbi.py`
2. In Power BI Desktop: Home → Get Data → Excel Workbook
3. Load `data/processed/powerbi_export/cmdb_export.xlsx`
4. Build visuals using the pre-built DAX measures in the guide

---

## Advanced Modelling

### SMOTE for imbalanced classes

Fault datasets are typically ~96% Normal. SMOTE synthesises additional
minority (fault) samples to improve recall:

```bash
python src/modeling/smote_trainer.py
```

Compares: Baseline | SMOTE | BorderlineSMOTE | ADASYN | Balanced weights.
Saves the best model to `models/fault_classifier.joblib`.

### Hyperparameter tuning (Optuna)

```bash
# 50 trials on Random Forest (≈5 min)
python src/modeling/hyperparameter_tuner.py --n-trials 50

# 100 trials on both RF and GBM
python src/modeling/hyperparameter_tuner.py --n-trials 100 --model all
```

Results saved to `models/optuna_rf_trials.csv` and `models/optuna_rf_history.png`.

### Ensemble classifiers

```bash
python src/modeling/ensemble.py
```

Compares: individual RF/GBM/SVM/KNN/LR vs Hard Voting, Soft Voting,
Stacking (LR meta), Stacking (RF meta). Saves best ensemble.

### Model registry

```python
from src.modeling.model_registry import ModelRegistry

registry = ModelRegistry()
version = registry.register(clf, "fault_classifier",
    metrics={"f1_macro": 0.94},
    tags={"strategy": "SMOTE"},
)
registry.promote(version, "fault_classifier", stage="production")
```

---

## Statistical Analysis (R)

```bash
Rscript r/reliability_analysis.R
```

Produces in `reports/r_plots/`:
- `km_survival_curve.png` — Kaplan-Meier reliability
- `fault_distribution.png` — Fault breakdown
- `correlation_matrix.png` — Feature correlation
- `health_score_trend.png` — Health trend with status colours
- `feature_boxplots.png` — Feature distributions by fault type
- `descriptive_stats.csv` — Summary statistics table

---

## Vibration Signal Analysis (MATLAB)

```matlab
% From MATLAB command window:
cd matlab
vibration_analysis   % FFT, envelope, kurtosis
```

For bearing fault frequency calculation:

```python
from src.features.bearing_diagnostics import BearingGeometry

bearing = BearingGeometry.from_catalogue("6205")
freqs   = bearing.fault_frequencies(shaft_rpm=1500)
print(freqs.summary(shaft_rpm=1500))
```

---

## Alerts

```bash
python src/alerts/notifier.py
```

Checks latest health scores from the database and fires alerts for
Critical and Degraded assets via:
- **Console / log** (always enabled)
- **Email** (set `SMTP_USER`, `SMTP_PASS`, `ALERT_RECIPIENTS` in `.env`)
- **Slack** (set `SLACK_WEBHOOK_URL` in `.env`)
- **Teams** (set `TEAMS_WEBHOOK_URL` in `.env`)

---

## Database Backups

```bash
# Create backup
./scripts/backup_db.sh backup

# List backups
./scripts/backup_db.sh list

# Restore latest
./scripts/backup_db.sh restore latest

# Clean backups older than 7 days
./scripts/backup_db.sh clean 7
```

---

## Docker Deployment

```bash
# Start API + dashboard
docker compose up api dashboard

# Or build and start everything including nginx proxy
docker compose up --build nginx api dashboard

# Access:
#   Dashboard: http://localhost/
#   API:       http://localhost/api/
#   API Docs:  http://localhost/docs
```

---

## Makefile Quick Reference

```bash
make setup       # install dependencies
make run         # full pipeline
make data        # download dataset
make model       # train models
make report      # generate HTML report
make export      # export to Power BI CSVs
make notebook    # launch Jupyter Lab
make r           # run R analysis
make test        # run pytest
make lint        # flake8
make clean       # remove generated files
make clean-all   # remove everything including data
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `FileNotFoundError: ai4i2020.csv` | Run `python main.py --steps 1` or `make data` |
| `No module named 'streamlit'` | Run `pip install -r requirements-extra.txt` |
| `No module named 'optuna'` | Run `pip install optuna imbalanced-learn` |
| Dashboard shows "Demo mode" | Run `python main.py` to populate the database |
| API returns 503 | Database not found — run `python main.py --steps 6` |
| Tests fail with `DataValidationError` | Ensure preprocessing has run first |
| Kaggle download fails | Place `~/.kaggle/kaggle.json` and retry |
| Power BI can't connect | Use CSV export: `python src/database/export_for_powerbi.py` |
