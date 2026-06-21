#!/usr/bin/env bash
# =============================================================
# scripts/run_pipeline.sh
# Production pipeline runner with logging and error handling
# =============================================================
# Usage:
#   ./scripts/run_pipeline.sh               # full pipeline
#   ./scripts/run_pipeline.sh --steps 2 3   # specific steps
#   ./scripts/run_pipeline.sh --dry-run     # show what would run
#   NOTIFY_ON_FAILURE=true ./scripts/run_pipeline.sh

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
STEPS="1 2 3 4 5 6 7 8"
DRY_RUN=false
NOTIFY_ON_FAILURE="${NOTIFY_ON_FAILURE:-false}"
SKIP_DOWNLOAD=false
START_TIME=$(date +%s)

# ── Parse args ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            shift
            STEPS=""
            while [[ $# -gt 0 ]] && [[ "$1" =~ ^[0-9]+$ ]]; do
                STEPS="$STEPS $1"
                shift
            done
            ;;
        --dry-run)    DRY_RUN=true;         shift ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--steps N [N...]] [--dry-run] [--skip-download]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Setup ───────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
mkdir -p data/raw data/processed models reports/plots

# ── Logging helpers ─────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_section() {
    local msg="── $* $(printf '─%.0s' {1..40})"
    log "$msg"
}

fail() {
    log "✗ PIPELINE FAILED: $*"
    log "  Log file: $LOG_FILE"
    if [[ "$NOTIFY_ON_FAILURE" == "true" ]] && [[ -n "${SMTP_USER:-}" ]]; then
        log "  Sending failure notification..."
        "$PYTHON" src/alerts/notifier.py 2>/dev/null || true
    fi
    exit 1
}

# ── Step runner ─────────────────────────────────────────────────
run_step() {
    local step_num="$1"
    local step_name="$2"
    local python_cmd="$3"

    log_section "STEP ${step_num}: ${step_name}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "  [DRY-RUN] Would run: $python_cmd"
        return 0
    fi

    local step_start=$(date +%s)
    if $PYTHON -c "$python_cmd" >> "$LOG_FILE" 2>&1; then
        local step_end=$(date +%s)
        local elapsed=$(( step_end - step_start ))
        log "  ✓ Step ${step_num} complete in ${elapsed}s"
    else
        fail "Step ${step_num} (${step_name}) failed — see $LOG_FILE"
    fi
}

# ── Header ──────────────────────────────────────────────────────
log "=================================================================="
log "  CBM Health Monitoring — Pipeline Runner"
log "  Start time : $(date '+%Y-%m-%d %H:%M:%S')"
log "  Log file   : $LOG_FILE"
log "  Python     : $($PYTHON --version 2>&1)"
log "  Steps      : $STEPS"
log "  Dry run    : $DRY_RUN"
log "=================================================================="

# ── Run steps ───────────────────────────────────────────────────
for step in $STEPS; do
    case $step in
        1)
            if [[ "$SKIP_DOWNLOAD" == "true" ]]; then
                log "  [SKIP] Step 1 — using existing raw data"
            else
                run_step 1 "Download dataset" \
                    "from src.ingestion.download_data import download_ai4i; download_ai4i()"
            fi
            ;;
        2)
            run_step 2 "Preprocess data" \
                "from src.preprocessing.clean_data import preprocess_ai4i; preprocess_ai4i(save=True)"
            ;;
        3)
            run_step 3 "Feature extraction" \
                "
import pandas as pd, yaml
from pathlib import Path
from src.features.feature_extraction import extract_tabular_features
ROOT = Path('.')
with open(ROOT/'config.yaml') as f: import yaml; cfg = yaml.safe_load(f)
df = pd.read_csv(ROOT/cfg['paths']['processed_data']/'ai4i_clean.csv')
out = extract_tabular_features(df)
out.to_csv(ROOT/cfg['paths']['processed_data']/'ai4i_features.csv', index=False)
print(f'Features saved: {out.shape}')
"
            ;;
        4)
            run_step 4 "Health scoring" \
                "
import pandas as pd, numpy as np, yaml
from pathlib import Path
from src.modeling.health_scorer import run_health_scoring
ROOT = Path('.')
with open(ROOT/'config.yaml') as f: import yaml; cfg = yaml.safe_load(f)
proc = ROOT/cfg['paths']['processed_data']
df = pd.read_csv(proc/'ai4i_features.csv')
if 'rms' not in df.columns:
    df['rms'] = df['rotational_speed_rpm'] / df['rotational_speed_rpm'].max()
    df['kurtosis'] = (df['torque_Nm'] / (df['torque_Nm'].max() + 1e-9)) * 4
    df['crest_factor'] = df['kurtosis'] * 1.5
df = run_health_scoring(df, train_if_needed=True)
df.to_csv(proc/'ai4i_health_scored.csv', index=False)
print(f'Scored: {df.shape}')
"
            ;;
        5)
            run_step 5 "Fault classification" \
                "
import pandas as pd, numpy as np, yaml
from pathlib import Path
from src.modeling.fault_classifier import train_classifier, predict_fault
ROOT = Path('.')
with open(ROOT/'config.yaml') as f: import yaml; cfg = yaml.safe_load(f)
proc = ROOT/cfg['paths']['processed_data']
for fname in ['ai4i_features.csv', 'ai4i_clean.csv']:
    p = proc/fname
    if p.exists(): df = pd.read_csv(p); break
if 'fault_type' not in df.columns:
    df['fault_type'] = np.where(df.get('machine_failure', pd.Series([0]*len(df))) == 1, 'Fault', 'Normal')
train_classifier(df)
result = predict_fault(df)
result[['uid','predicted_fault','fault_confidence']].to_csv(proc/'ai4i_predictions.csv', index=False)
"
            ;;
        6)
            run_step 6 "Database load" \
                "
import pandas as pd, yaml
from pathlib import Path
from src.database.db_manager import create_schema, bulk_load_health_scores, insert_asset
ROOT = Path('.')
with open(ROOT/'config.yaml') as f: import yaml; cfg = yaml.safe_load(f)
create_schema()
for aid, name, atype, loc in [
    ('MTR-001','Pump Motor #1','motor','Plant A – Bay 1'),
    ('MTR-002','Cooling Fan #2','fan','Plant A – Bay 2'),
    ('PMP-001','Feed Pump #1','pump','Plant B – Utilities'),
    ('CMP-001','Air Compressor','compressor','Plant B – Utilities'),
]:
    insert_asset(aid, name, atype, loc)
p = ROOT/cfg['paths']['processed_data']/'ai4i_health_scored.csv'
if p.exists():
    df = pd.read_csv(p)
    bulk_load_health_scores(df.head(500))
"
            ;;
        7)
            run_step 7 "Generate plots" \
                "
import numpy as np, pandas as pd, yaml
from pathlib import Path
from src.visualization.plots import (plot_waveform, plot_fft,
    plot_health_distribution, plot_fault_distribution, plot_dashboard_summary)
ROOT = Path('.')
with open(ROOT/'config.yaml') as f: import yaml; cfg = yaml.safe_load(f)
FS = cfg['signal']['sampling_rate']
np.random.seed(42)
t = np.linspace(0, 1, FS)
sig = np.sin(2*np.pi*60*t) + 0.3*np.random.randn(FS)
sig[np.random.choice(FS, 30, replace=False)] += 2.5
plot_waveform(sig); plot_fft(sig)
p = ROOT/cfg['paths']['processed_data']/'ai4i_health_scored.csv'
if p.exists():
    df = pd.read_csv(p)
    plot_health_distribution(df); plot_fault_distribution(df); plot_dashboard_summary(df)
print('Plots complete')
"
            ;;
        8)
            run_step 8 "Generate HTML report" \
                "from reports.generate_report import generate_report; generate_report()"
            ;;
        *)
            log "  [WARN] Unknown step: $step — skipping"
            ;;
    esac
done

# ── Run alerts check ──────────────────────────────────────────────
if [[ "$DRY_RUN" == "false" ]]; then
    log_section "POST-PIPELINE: Alert check"
    $PYTHON src/alerts/notifier.py >> "$LOG_FILE" 2>&1 || \
        log "  [WARN] Alert check encountered an error (non-fatal)"
fi

# ── Summary ──────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

log ""
log "=================================================================="
log "  ✓ Pipeline complete"
log "  Duration   : ${MINUTES}m ${SECONDS}s"
log "  Outputs:"
log "    data/processed/   — cleaned data, features, scores, DB"
log "    models/           — trained ML models"
log "    reports/          — HTML report + diagnostic plots"
log "  Log saved  : $LOG_FILE"
log "=================================================================="
