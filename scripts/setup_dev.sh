#!/usr/bin/env bash
# =============================================================
# scripts/setup_dev.sh
# Development environment setup for CBM Health Monitoring System
# =============================================================
# Usage:
#   chmod +x scripts/setup_dev.sh
#   ./scripts/setup_dev.sh

set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo ""
echo "=================================================="
echo "  CBM Health Monitoring — Development Setup"
echo "=================================================="
echo ""

# ── Check Python version ──────────────────────────────────────
echo "→ Checking Python version..."
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "  ✗ Python 3.10+ required (found $PY_VERSION)"
    exit 1
fi
echo "  ✓ Python $PY_VERSION"

# ── Create virtual environment ────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment in $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
    echo "  ✓ Virtual environment created"
else
    echo "→ Virtual environment already exists ($VENV_DIR)"
fi

# ── Activate ──────────────────────────────────────────────────
if [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win32"* ]]; then
    ACTIVATE="$VENV_DIR/Scripts/activate"
else
    ACTIVATE="$VENV_DIR/bin/activate"
fi
# shellcheck disable=SC1090
source "$ACTIVATE"
echo "  ✓ Virtual environment activated"

# ── Install dependencies ──────────────────────────────────────
echo ""
echo "→ Installing core dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "  ✓ Core dependencies installed"

echo "→ Installing extra dependencies (API, dashboard, dev tools)..."
pip install -r requirements-extra.txt --quiet
echo "  ✓ Extra dependencies installed"

# ── Optional: imbalanced-learn + optuna ──────────────────────
echo "→ Installing optional ML packages (SMOTE, Optuna)..."
pip install imbalanced-learn optuna --quiet && \
    echo "  ✓ imbalanced-learn + optuna installed" || \
    echo "  ⚠ Could not install optional ML packages (non-fatal)"

# ── Create .env from example ──────────────────────────────────
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo ""
    echo "→ Created .env from .env.example"
    echo "  ✎ Edit .env to add your SMTP / Slack / Kaggle credentials"
fi

# ── Create data directories ───────────────────────────────────
echo ""
echo "→ Creating project directories..."
mkdir -p data/raw data/processed data/external \
         models reports/plots reports/r_plots logs \
         notebooks/.ipynb_checkpoints
echo "  ✓ Directories ready"

# ── Run tests ─────────────────────────────────────────────────
echo ""
echo "→ Running test suite to verify installation..."
pytest tests/ -v --tb=short -q 2>&1 | tail -15
echo ""

# ── Summary ───────────────────────────────────────────────────
echo "=================================================="
echo "  Setup complete!"
echo ""
echo "  Activate environment:  source $ACTIVATE"
echo ""
echo "  Quick start commands:"
echo "    make run              # full pipeline"
echo "    make data             # download dataset only"
echo "    make test             # run tests"
echo "    make notebook         # Jupyter Lab"
echo "    streamlit run src/dashboard/app.py"
echo "    uvicorn src.api.main:app --reload"
echo ""
echo "  See README.md and docs/ for full documentation."
echo "=================================================="
