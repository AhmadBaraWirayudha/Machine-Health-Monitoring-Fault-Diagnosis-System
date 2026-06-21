"""
Fault Classifier
================
Trains and evaluates a Random Forest classifier to identify fault type
from sensor / vibration features.

Fault classes (AI4I 2020):
  Normal, TWF (Tool Wear), HDF (Heat Dissipation), PWF (Power), OSF (Overstrain), RNF (Random)

Usage:
    python src/modeling/fault_classifier.py
"""

import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

# ── Load config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR   = ROOT / CFG["paths"]["processed_data"]
MODELS_DIR = ROOT / CFG["paths"]["models"]
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CFG = CFG["model"]["classifier"]

# Columns used as input features
FEATURE_COLS = [
    "air_temp_K", "process_temp_K",
    "rotational_speed_rpm", "torque_Nm", "tool_wear_min",
]

# Rolling/engineered features (if available)
OPTIONAL_COLS = [
    "power_W", "temp_diff_K",
    "rotational_speed_rpm_rolling_mean", "rotational_speed_rpm_rolling_std",
    "torque_Nm_rolling_mean", "torque_Nm_rolling_std",
    "tool_wear_min_rolling_max",
]

LABEL_COL = "fault_type"


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select and return only available feature columns."""
    all_cols = FEATURE_COLS + OPTIONAL_COLS
    cols = [c for c in all_cols if c in df.columns]
    return df[cols].fillna(0), cols


def train_classifier(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the cleaned feature dataset.

    Returns the fitted model (also saved to models/).
    """
    X, feature_cols = prepare_features(df)
    y_raw = df[LABEL_COL] if LABEL_COL in df.columns else df.get("machine_failure", None)

    if y_raw is None:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset.")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_CFG["test_size"],
        random_state=MODEL_CFG["random_state"],
        stratify=y,
    )

    print(f"\n[train] Training Random Forest …")
    print(f"        Features  : {len(feature_cols)}")
    print(f"        Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")
    print(f"        Classes   : {list(le.classes_)}")

    clf = RandomForestClassifier(
        n_estimators=MODEL_CFG["n_estimators"],
        max_depth=MODEL_CFG["max_depth"],
        random_state=MODEL_CFG["random_state"],
        n_jobs=-1,
        class_weight="balanced",   # handle imbalanced fault data
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[eval]  Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"5-Fold CV F1 (macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nTop 5 Features by Importance:")
    print(importance_df.head(5).to_string(index=False))

    # Save model
    model_path = MODELS_DIR / "fault_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"\n[OK]   Model saved → {model_path}")

    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier() -> tuple[RandomForestClassifier | None, LabelEncoder | None]:
    clf_path = MODELS_DIR / "fault_classifier.joblib"
    le_path  = MODELS_DIR / "label_encoder.joblib"
    clf = joblib.load(clf_path) if clf_path.exists() else None
    le  = joblib.load(le_path)  if le_path.exists()  else None
    return clf, le


def predict_fault(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the trained classifier to new data.

    Adds columns:
        predicted_fault   — fault type string
        fault_confidence  — probability of predicted class (0–1)
    """
    clf, le = load_classifier()
    if clf is None:
        raise RuntimeError("Classifier not found — run train_classifier() first.")

    X, _ = prepare_features(df)
    pred_idx = clf.predict(X)
    pred_proba = clf.predict_proba(X).max(axis=1)

    df = df.copy()
    df["predicted_fault"]   = le.inverse_transform(pred_idx) if le else pred_idx
    df["fault_confidence"]  = pred_proba
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix helper
# ─────────────────────────────────────────────────────────────────────────────

def print_confusion_matrix(df: pd.DataFrame) -> None:
    """Print confusion matrix if true labels are available."""
    if LABEL_COL not in df.columns or "predicted_fault" not in df.columns:
        print("[WARN] Need both true label and prediction columns.")
        return
    cm = confusion_matrix(df[LABEL_COL], df["predicted_fault"])
    _, le = load_classifier()
    classes = le.classes_ if le else sorted(df[LABEL_COL].unique())
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Fault Classifier — Condition Monitoring Project")
    print("=" * 60)

    feat_path = PROC_DIR / "ai4i_features.csv"
    if not feat_path.exists():
        feat_path = PROC_DIR / "ai4i_clean.csv"

    if not feat_path.exists():
        print("[ERROR] No feature file found. Run preprocessing and feature extraction first.")
    else:
        df = pd.read_csv(feat_path)

        if LABEL_COL not in df.columns:
            print(f"[WARN] Column '{LABEL_COL}' missing. Creating dummy labels for demo.")
            df[LABEL_COL] = np.where(df.get("machine_failure", 0) == 1, "Fault", "Normal")

        clf = train_classifier(df)
        df_pred = predict_fault(df)
        print_confusion_matrix(df_pred)

        out_path = PROC_DIR / "ai4i_predictions.csv"
        df_pred[["uid", "predicted_fault", "fault_confidence"]].to_csv(out_path, index=False)
        print(f"\n[OK]   Predictions saved → {out_path}")
