# =============================================================
# Makefile — Machine Health Monitoring & Fault Diagnosis System
# =============================================================
# Quick reference:
#   make setup       install Python dependencies
#   make run         run the full pipeline (all 8 steps)
#   make data        download dataset only
#   make features    preprocess + extract features
#   make model       train health scorer + fault classifier
#   make smote       train with SMOTE oversampling
#   make tune        Optuna hyperparameter search (30 trials)
#   make ensemble    train ensemble classifier
#   make registry    run model registry demo
#   make metrics     compute reliability metrics (MTBF, MTTF, OEE)
#   make db          load results into SQLite
#   make report      generate plots + HTML report
#   make export      export database to CSVs for Power BI
#   make dashboard   launch Streamlit dashboard
#   make api         start FastAPI server
#   make notebook    launch Jupyter Lab (all 6 notebooks)
#   make r           run R reliability analysis
#   make alerts      check and send asset health alerts
#   make backup      backup the SQLite database
#   make run-sh      run pipeline via shell script (with logging)
#   make test        run all pytest unit tests
#   make test-fast   run tests excluding slow ones
#   make lint        flake8 code style check
#   make format      black + isort auto-format
#   make clean       remove generated outputs (keep raw data + models)
#   make clean-all   remove everything including raw data + models
#   make help        show this help

PYTHON    := python
RSCRIPT   := Rscript
PIP       := pip
VENV      := .venv
PROC_DIR  := data/processed
MODELS    := models
REPORTS   := reports

.PHONY: all setup venv run data preprocess features model smote tune \
        ensemble registry metrics db plots report export dashboard api \
        notebook r alerts backup run-sh test test-fast lint format \
        clean clean-all help

# ── Default ───────────────────────────────────────────────────────────────────
all: run

# ── Environment ───────────────────────────────────────────────────────────────
setup:
	@echo "→ Installing core dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "→ Installing extra dependencies (API, dashboard, dev)..."
	$(PIP) install -r requirements-extra.txt
	@echo ""
	@echo "✓ Setup complete. Run 'make run' to start the pipeline."

setup-full: setup
	@echo "→ Installing optional ML packages..."
	$(PIP) install imbalanced-learn optuna
	@echo "✓ Full setup complete."

venv:
	@echo "→ Creating virtual environment in $(VENV)..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt -r requirements-extra.txt
	@echo "✓ Activate with: source $(VENV)/bin/activate"

dev-setup:
	@chmod +x scripts/setup_dev.sh
	@./scripts/setup_dev.sh

# ── Core pipeline ─────────────────────────────────────────────────────────────
run:
	@echo "→ Running full pipeline (steps 1–8)..."
	$(PYTHON) main.py

run-sh:
	@echo "→ Running pipeline via shell script (with structured logging)..."
	@chmod +x scripts/run_pipeline.sh
	@./scripts/run_pipeline.sh

data:
	@echo "→ Step 1 — Download dataset..."
	$(PYTHON) src/ingestion/download_data.py

preprocess:
	@echo "→ Step 2 — Preprocess data..."
	$(PYTHON) src/preprocessing/clean_data.py

features: preprocess
	@echo "→ Step 3 — Extract features..."
	$(PYTHON) src/features/feature_extraction.py

model: features
	@echo "→ Steps 4-5 — Health scoring + Fault classification..."
	$(PYTHON) src/modeling/health_scorer.py
	$(PYTHON) src/modeling/fault_classifier.py

db: model
	@echo "→ Step 6 — Load database..."
	$(PYTHON) src/database/db_manager.py

plots: db
	@echo "→ Step 7 — Generate diagnostic plots..."
	$(PYTHON) src/visualization/plots.py

report: plots
	@echo "→ Step 8 — Render HTML technical report..."
	$(PYTHON) reports/generate_report.py
	@echo "✓ Report: reports/technical_report.html"

# ── Advanced modelling ────────────────────────────────────────────────────────
smote:
	@echo "→ Running SMOTE imbalance correction comparison..."
	$(PYTHON) src/modeling/smote_trainer.py

tune:
	@echo "→ Running Optuna hyperparameter search (30 trials)..."
	$(PYTHON) src/modeling/hyperparameter_tuner.py --n-trials 30

tune-full:
	@echo "→ Running full Optuna search (100 trials, RF + GBM)..."
	$(PYTHON) src/modeling/hyperparameter_tuner.py --n-trials 100 --model all

ensemble:
	@echo "→ Training ensemble classifier (Voting + Stacking)..."
	$(PYTHON) src/modeling/ensemble.py

registry:
	@echo "→ Running model registry demo..."
	$(PYTHON) src/modeling/model_registry.py

metrics:
	@echo "→ Computing reliability metrics (MTBF, MTTF, Availability, OEE)..."
	$(PYTHON) src/utils/metrics.py

augment:
	@echo "→ Running signal augmentation demo..."
	$(PYTHON) src/preprocessing/augmentation.py

bearing:
	@echo "→ Running bearing diagnostics demo..."
	$(PYTHON) src/features/bearing_diagnostics.py

# ── Data & database ───────────────────────────────────────────────────────────
export: db
	@echo "→ Exporting to Power BI flat files..."
	$(PYTHON) src/database/export_for_powerbi.py
	@echo "✓ Files in: $(PROC_DIR)/powerbi_export/"

backup:
	@echo "→ Backing up database..."
	@chmod +x scripts/backup_db.sh
	@./scripts/backup_db.sh backup

backup-list:
	@./scripts/backup_db.sh list

# ── Services ──────────────────────────────────────────────────────────────────
dashboard:
	@echo "→ Starting Streamlit dashboard on http://localhost:8501"
	streamlit run src/dashboard/app.py --server.port 8501

api:
	@echo "→ Starting FastAPI server on http://localhost:8000"
	@echo "  Swagger docs: http://localhost:8000/docs"
	uvicorn src.api.main:app --reload --port 8000

api-prod:
	@echo "→ Starting FastAPI (production mode, 4 workers)..."
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

docker-up:
	@echo "→ Starting Docker services (API + Dashboard)..."
	docker compose up api dashboard

docker-build:
	@echo "→ Building Docker image..."
	docker compose build

docker-down:
	docker compose down

alerts:
	@echo "→ Checking asset health and firing alerts..."
	$(PYTHON) src/alerts/notifier.py

gen-api-key:
	@echo "→ Generating new API key..."
	$(PYTHON) src/api/auth.py generate --prefix cbm

# ── Notebooks & analysis ──────────────────────────────────────────────────────
notebook:
	@echo "→ Launching Jupyter Lab (notebooks/)..."
	jupyter lab notebooks/

notebook-convert:
	@echo "→ Converting notebooks to HTML..."
	@mkdir -p $(REPORTS)/notebooks
	jupyter nbconvert notebooks/*.ipynb --to html --output-dir $(REPORTS)/notebooks/
	@echo "✓ HTML notebooks in $(REPORTS)/notebooks/"

r:
	@echo "→ Running R reliability analysis..."
	$(RSCRIPT) r/reliability_analysis.R
	@echo "✓ R plots in: reports/r_plots/"

matlab:
	@echo "→ MATLAB scripts are in matlab/"
	@echo "  Open MATLAB and run: vibration_analysis or bearing_fault_detection"

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	@echo "→ Running full test suite..."
	pytest tests/ -v --tb=short

test-fast:
	@echo "→ Running fast tests (excluding slow ML fits)..."
	pytest tests/ -v --tb=short -m "not slow"

test-api:
	pytest tests/test_api.py -v --tb=short

test-db:
	pytest tests/test_database.py -v --tb=short

test-modeling:
	pytest tests/test_modeling.py -v --tb=short

test-features:
	pytest tests/test_features.py -v --tb=short

test-alerts:
	pytest tests/test_alerts.py -v --tb=short

test-cov:
	@echo "→ Running tests with coverage report..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "✓ Coverage report: htmlcov/index.html"

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	@echo "→ Linting with flake8..."
	flake8 src/ main.py --max-line-length=110 --ignore=E501,W503,E302 \
	  --exclude=__pycache__,.venv

format:
	@echo "→ Formatting with black + isort..."
	black src/ tests/ main.py --line-length 110
	isort src/ tests/ main.py

type-check:
	@echo "→ Type checking with mypy..."
	mypy src/ --ignore-missing-imports

# ── Cleaning ──────────────────────────────────────────────────────────────────
clean:
	@echo "→ Removing generated outputs..."
	rm -rf $(PROC_DIR)/ai4i_clean.csv \
	       $(PROC_DIR)/ai4i_features.csv \
	       $(PROC_DIR)/ai4i_health_scored.csv \
	       $(PROC_DIR)/ai4i_predictions.csv \
	       $(PROC_DIR)/cmdb.sqlite \
	       $(PROC_DIR)/powerbi_export \
	       $(PROC_DIR)/reliability_metrics.csv \
	       $(PROC_DIR)/smote_strategy_comparison.csv \
	       $(PROC_DIR)/ensemble_comparison.csv \
	       $(REPORTS)/plots \
	       $(REPORTS)/r_plots \
	       $(REPORTS)/notebooks \
	       $(REPORTS)/technical_report.html \
	       logs/ htmlcov/ .coverage coverage.xml
	@find . -name __pycache__ -type d | xargs rm -rf 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean complete (raw data and models kept)"

clean-models:
	@echo "→ Removing trained models..."
	rm -rf $(MODELS)/*.joblib $(MODELS)/registry $(MODELS)/optuna_*.db \
	       $(MODELS)/optuna_*.csv $(MODELS)/training_meta.json \
	       $(MODELS)/ensemble_meta.json
	@echo "✓ Models removed"

clean-all: clean clean-models
	@echo "→ Removing downloaded raw data..."
	rm -rf data/raw/ai4i2020.csv data/raw/nasa_bearing
	@echo "✓ Full clean complete"

# ── Documentation ─────────────────────────────────────────────────────────────
docs:
	@echo "Project documentation:"
	@echo "  docs/architecture.md   — system architecture + data flow"
	@echo "  docs/user_guide.md     — how to run and use the system"
	@echo "  docs/developer_guide.md — contributing, extending, testing"
	@echo "  docs/api_reference.md  — REST API endpoint reference"
	@echo ""
	@echo "  API interactive docs (when running): http://localhost:8000/docs"

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Machine Health Monitoring & Fault Diagnosis System"
	@echo "==================================================="
	@echo ""
	@echo "SETUP"
	@echo "  make setup           Install dependencies (pip)"
	@echo "  make setup-full      Install + optional (SMOTE, Optuna)"
	@echo "  make venv            Create virtual environment"
	@echo "  make dev-setup       Full dev setup via shell script"
	@echo ""
	@echo "PIPELINE"
	@echo "  make run             Full pipeline (all 8 steps)"
	@echo "  make run-sh          Pipeline via shell script (structured logging)"
	@echo "  make data            Step 1: download dataset"
	@echo "  make preprocess      Step 2: clean data"
	@echo "  make features        Step 3: extract features"
	@echo "  make model           Steps 4-5: health scoring + fault classifier"
	@echo "  make db              Step 6: load database"
	@echo "  make report          Steps 7-8: plots + HTML report"
	@echo ""
	@echo "ADVANCED MODELLING"
	@echo "  make smote           SMOTE oversampling comparison"
	@echo "  make tune            Optuna hyperparameter search (30 trials)"
	@echo "  make tune-full       Full Optuna search (100 trials, RF + GBM)"
	@echo "  make ensemble        Voting + Stacking ensemble"
	@echo "  make registry        Model registry demo"
	@echo "  make metrics         Reliability metrics (MTBF, MTTF, OEE)"
	@echo "  make augment         Signal augmentation demo"
	@echo "  make bearing         Bearing diagnostics demo"
	@echo ""
	@echo "SERVICES"
	@echo "  make dashboard       Streamlit dashboard (port 8501)"
	@echo "  make api             FastAPI server (port 8000)"
	@echo "  make docker-up       Docker Compose (API + Dashboard)"
	@echo "  make alerts          Check + send health alerts"
	@echo ""
	@echo "DATA"
	@echo "  make export          Export DB to Power BI CSVs"
	@echo "  make backup          Backup SQLite database"
	@echo ""
	@echo "ANALYSIS"
	@echo "  make notebook        Jupyter Lab"
	@echo "  make r               R reliability analysis"
	@echo ""
	@echo "TESTING"
	@echo "  make test            Full test suite"
	@echo "  make test-fast       Fast tests only (skip slow)"
	@echo "  make test-cov        Tests + HTML coverage report"
	@echo ""
	@echo "CODE QUALITY"
	@echo "  make lint            flake8 linting"
	@echo "  make format          black + isort auto-format"
	@echo ""
	@echo "CLEANUP"
	@echo "  make clean           Remove generated files"
	@echo "  make clean-all       Remove everything"
	@echo ""
