"""
Asset Health Scorer
====================
Computes a 0–100 Asset Health Score (AHS) from extracted features,
runs Isolation Forest anomaly detection, and classifies health status.

Health Score Scale
------------------
  80–100  Good       — continue routine monitoring
  60–79   Warning    — increase inspection frequency
  40–59   Degraded   — schedule maintenance soon
   0–39   Critical   — immediate action required

Usage:
    python src/modeling/health_scorer.py
"""

import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# ── Load config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR   = ROOT / CFG["paths"]["processed_data"]
MODELS_DIR = ROOT / CFG["paths"]["models"]
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Feature weights for health score (must sum to 1.0)
WEIGHTS = CFG["health_score"]["weights"]
THRESHOLDS = CFG["health_score"]["thresholds"]

# Features used in scoring (adjust to whichever are available)
SCORE_FEATURES = ["rms", "kurtosis", "crest_factor"]


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

def train_anomaly_detector(df: pd.DataFrame, feature_cols: list) -> IsolationForest:
    """
    Fit an Isolation Forest on the provided feature columns.

    Assumes healthy data dominates (contamination parameter controls
    the expected fraction of anomalies).
    """
    iso_cfg = CFG["model"]["anomaly"]
    model = IsolationForest(
        contamination=iso_cfg["contamination"],
        random_state=iso_cfg["random_state"],
        n_estimators=100,
    )
    X = df[feature_cols].dropna()
    model.fit(X)
    path = MODELS_DIR / "isolation_forest.joblib"
    joblib.dump(model, path)
    print(f"[model] Isolation Forest trained on {len(X):,} samples → saved")
    return model


def load_anomaly_detector() -> IsolationForest | None:
    path = MODELS_DIR / "isolation_forest.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def score_anomaly(df: pd.DataFrame, feature_cols: list, model: IsolationForest | None = None) -> pd.Series:
    """
    Returns a 0–1 normalised anomaly score per row.
      0 = most anomalous  |  1 = most normal
    """
    if model is None:
        model = load_anomaly_detector()
    if model is None:
        raise RuntimeError("No anomaly detector found — call train_anomaly_detector() first.")

    X = df[feature_cols].fillna(0)
    # decision_function returns negative scores for anomalies
    raw = model.decision_function(X)
    # Shift to 0–1 (higher = more normal)
    normalised = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    return pd.Series(normalised, index=df.index, name="anomaly_score")


# ─────────────────────────────────────────────────────────────────────────────
# Feature → component score mapping
# ─────────────────────────────────────────────────────────────────────────────

def _invert_and_scale(series: pd.Series) -> pd.Series:
    """Scale a feature so that higher raw value → lower health (0=worst, 1=best)."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(1.0, index=series.index)
    normalised = (series - mn) / (mx - mn)
    return 1.0 - normalised   # invert: high kurtosis → low health


def compute_component_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map each diagnostic feature to a 0–1 health component score.

    Convention: 1.0 = perfectly healthy, 0.0 = failed.
    """
    scores = pd.DataFrame(index=df.index)

    if "rms" in df.columns:
        scores["rms_score"] = _invert_and_scale(df["rms"])

    if "kurtosis" in df.columns:
        # kurtosis > 3 is normal; higher = worse
        scores["kurtosis_score"] = _invert_and_scale(df["kurtosis"])

    if "crest_factor" in df.columns:
        scores["crest_score"] = _invert_and_scale(df["crest_factor"])

    if "anomaly_score" in df.columns:
        scores["anomaly_health"] = df["anomaly_score"]   # already 0–1 (1=healthy)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Asset Health Score (AHS)
# ─────────────────────────────────────────────────────────────────────────────

def compute_health_score(df: pd.DataFrame) -> pd.Series:
    """
    Weighted combination of component scores → Asset Health Score (0–100).
    """
    comp = compute_component_scores(df)

    # Build weight vector over whichever component scores are available
    mapping = {
        "rms_score":      WEIGHTS.get("rms",           0.25),
        "kurtosis_score": WEIGHTS.get("kurtosis",       0.25),
        "crest_score":    WEIGHTS.get("crest_factor",   0.20),
        "anomaly_health": WEIGHTS.get("anomaly_score",  0.30),
    }
    available = {k: v for k, v in mapping.items() if k in comp.columns}

    # Re-normalise weights to sum to 1
    total_weight = sum(available.values())
    ahs = sum(comp[col] * (w / total_weight) for col, w in available.items())
    return (ahs * 100).clip(0, 100).rename("health_score")


def classify_health(score: float) -> str:
    """Convert a numeric health score to a status label."""
    if score >= THRESHOLDS["good"]:
        return "Good"
    elif score >= THRESHOLDS["warning"]:
        return "Warning"
    elif score >= THRESHOLDS["degraded"]:
        return "Degraded"
    else:
        return "Critical"


def generate_recommendation(row: pd.Series) -> str:
    """Simple rule-based maintenance recommendation from health data."""
    status = row.get("health_status", "Unknown")
    kurtosis_val = row.get("kurtosis", 0)
    crest_val = row.get("crest_factor", 0)

    if status == "Good":
        return "Continue routine monitoring per PM schedule."
    elif status == "Warning":
        if kurtosis_val > 6:
            return "Elevated kurtosis detected — inspect bearings at next opportunity."
        return "Increase vibration check frequency. Monitor trend."
    elif status == "Degraded":
        if crest_val > 8:
            return "High crest factor — probable bearing impacting. Schedule bearing inspection."
        return "Asset showing signs of degradation. Schedule maintenance within 2 weeks."
    else:  # Critical
        return "CRITICAL: Immediate inspection required. Consider shutting down asset."


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_health_scoring(df: pd.DataFrame, train_if_needed: bool = True) -> pd.DataFrame:
    """
    Full health scoring pipeline on a feature-enriched DataFrame.

    Steps:
      1. Train / load anomaly detector
      2. Compute anomaly scores
      3. Compute weighted health score
      4. Classify health status
      5. Generate recommendation
    """
    available_feats = [c for c in SCORE_FEATURES if c in df.columns]

    if not available_feats:
        print("[WARN] No diagnostic features found. Skipping health scoring.")
        return df

    # Step 1–2: anomaly detection
    model = load_anomaly_detector()
    if model is None and train_if_needed:
        model = train_anomaly_detector(df, available_feats)
    if model is not None:
        df["anomaly_score"] = score_anomaly(df, available_feats, model)

    # Step 3: health score
    df["health_score"] = compute_health_score(df)

    # Step 4: classify
    df["health_status"] = df["health_score"].apply(classify_health)

    # Step 5: recommendation
    df["recommendation"] = df.apply(generate_recommendation, axis=1)

    print(f"\n[health] Scoring complete — {len(df):,} assets")
    print(df["health_status"].value_counts().to_string())
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  Asset Health Scorer — Condition Monitoring Project")
    print("=" * 60)

    feat_path = PROC_DIR / "ai4i_features.csv"
    if not feat_path.exists():
        print("[WARN] Feature file not found. Run feature_extraction.py first.")
    else:
        df = pd.read_csv(feat_path)

        # Map AI4I sensor columns to vibration-like feature names for demo
        if "rotational_speed_rpm" in df.columns and "rms" not in df.columns:
            df["rms"]          = df["rotational_speed_rpm"] / df["rotational_speed_rpm"].max()
            df["kurtosis"]     = (df.get("torque_Nm", 0) / (df.get("torque_Nm", 1).max() + 1e-9)) * 4
            df["crest_factor"] = df["kurtosis"] * 1.5

        df = run_health_scoring(df)
        out = PROC_DIR / "ai4i_health_scored.csv"
        df.to_csv(out, index=False)
        print(f"\n[OK]   Health scored dataset → {out}")

        print("\nSample results:")
        print(df[["uid", "health_score", "health_status", "recommendation"]].head(10).to_string(index=False))
