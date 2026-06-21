"""
SMOTE Trainer
=============
Handles severe class imbalance in fault detection using
Synthetic Minority Over-sampling Technique (SMOTE) and variants.

Fault datasets are typically heavily imbalanced:
  - Normal: ~96.6%
  - Any fault: ~3.4%

Without correction, classifiers predict "Normal" for nearly everything
and achieve high accuracy but near-zero recall on faults.

Strategies implemented:
  1. SMOTE           — interpolates between minority samples
  2. BorderlineSMOTE — focuses synthesis on hard boundary cases
  3. ADASYN          — adaptive synthesis (more samples where needed)
  4. Class weights   — cheaper alternative (no resampling)
  5. Comparison      — evaluates all strategies on the same data

Requirements:
    pip install imbalanced-learn

Usage:
    python src/modeling/smote_trainer.py
"""

import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed — install with: pip install imbalanced-learn")
    print("       Falling back to class_weight='balanced' strategy only.")

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR   = ROOT / CFG["paths"]["processed_data"]
MODELS_DIR = ROOT / CFG["paths"]["models"]
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "air_temp_K", "process_temp_K",
    "rotational_speed_rpm", "torque_Nm", "tool_wear_min",
    "power_W", "temp_diff_K",
]
LABEL_COL    = "fault_type"
RANDOM_STATE = CFG["model"]["classifier"]["random_state"]
N_SPLITS     = 5


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    """Load features and labels, returning (X, y, label_encoder, feature_cols)."""
    for fname in ["ai4i_features.csv", "ai4i_clean.csv"]:
        p = PROC_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("No feature file found. Run preprocessing first.")

    if LABEL_COL not in df.columns:
        fault_cols = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]
        if fault_cols:
            def _label(row):
                hits = [c for c in fault_cols if row[c] == 1]
                return hits[0] if hits else "Normal"
            df[LABEL_COL] = df.apply(_label, axis=1)
        else:
            df[LABEL_COL] = np.where(df.get("machine_failure", 0) == 1, "Fault", "Normal")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X  = df[feat_cols].fillna(0).values
    le = LabelEncoder()
    y  = le.fit_transform(df[LABEL_COL])

    print(f"[data] Loaded {len(X):,} samples — {len(feat_cols)} features")
    print(f"[data] Class distribution: {dict(Counter(df[LABEL_COL]))}")
    minority_pct = (y != le.transform(["Normal"])[0]).mean() * 100
    print(f"[data] Fault rate: {minority_pct:.2f}%  (imbalance ratio {100/minority_pct:.0f}:1)")
    return X, y, le, feat_cols


# ── Resampling strategies ─────────────────────────────────────────────────────

def get_strategies(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Return a dict of (strategy_name → (X_resampled, y_resampled)).
    Also includes the 'No resampling' baseline.
    """
    strategies = {"Baseline (no resample)": (X, y)}

    if not IMBLEARN_AVAILABLE:
        print("[WARN] imbalanced-learn not available — skipping SMOTE strategies")
        return strategies

    minority_count = Counter(y).most_common()[-1][1]

    # SMOTE needs at least k_neighbors+1 minority samples
    k = min(5, minority_count - 1)
    if k < 1:
        print("[WARN] Too few minority samples for SMOTE — using baseline only")
        return strategies

    try:
        sm   = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_sm, y_sm = sm.fit_resample(X, y)
        strategies["SMOTE"] = (X_sm, y_sm)
        print(f"[smote] SMOTE: {len(X_sm):,} samples after resampling")
    except Exception as e:
        print(f"[WARN] SMOTE failed: {e}")

    try:
        bl = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=k, kind="borderline-1")
        X_bl, y_bl = bl.fit_resample(X, y)
        strategies["BorderlineSMOTE"] = (X_bl, y_bl)
        print(f"[smote] BorderlineSMOTE: {len(X_bl):,} samples after resampling")
    except Exception as e:
        print(f"[WARN] BorderlineSMOTE failed: {e}")

    try:
        ada = ADASYN(random_state=RANDOM_STATE, n_neighbors=k)
        X_ada, y_ada = ada.fit_resample(X, y)
        strategies["ADASYN"] = (X_ada, y_ada)
        print(f"[smote] ADASYN: {len(X_ada):,} samples after resampling")
    except Exception as e:
        print(f"[WARN] ADASYN failed: {e}")

    return strategies


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_strategy(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    class_weight=None,
) -> dict:
    """5-fold CV evaluation of a single strategy."""
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weight,
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    acc  = cross_val_score(clf, X, y, cv=cv, scoring="accuracy",  n_jobs=-1)
    f1m  = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro",  n_jobs=-1)
    f1w  = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)

    result = {
        "strategy":     name,
        "acc_mean":     acc.mean(),
        "acc_std":      acc.std(),
        "f1_macro_mean": f1m.mean(),
        "f1_macro_std":  f1m.std(),
        "f1_weighted_mean": f1w.mean(),
        "f1_weighted_std":  f1w.std(),
        "n_samples":    len(X),
    }
    print(f"  {name:<25}  Acc={acc.mean():.4f}±{acc.std():.4f}  "
          f"F1-macro={f1m.mean():.4f}±{f1m.std():.4f}  "
          f"F1-weighted={f1w.mean():.4f}±{f1w.std():.4f}")
    return result


def compare_strategies(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Compare all resampling strategies and return a ranked summary DataFrame."""
    print("\n── Strategy Comparison ──────────────────────────────────────")
    print(f"  {'Strategy':<25}  {'Accuracy':<18}  {'F1-macro':<18}  F1-weighted")
    print("  " + "─" * 78)

    strategies = get_strategies(X, y)
    results = []

    for name, (X_res, y_res) in strategies.items():
        results.append(evaluate_strategy(name, X_res, y_res))

    # Class weights (no resampling, handled internally by sklearn)
    results.append(evaluate_strategy(
        "Balanced weights", X, y, class_weight="balanced"
    ))

    df = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)
    return df


# ── Train best model ──────────────────────────────────────────────────────────

def train_best(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    strategy: str = "SMOTE",
) -> RandomForestClassifier:
    """
    Resample with the chosen strategy, then train and save the best model.

    Parameters
    ----------
    strategy : one of 'SMOTE', 'BorderlineSMOTE', 'ADASYN',
               'Balanced weights', 'Baseline (no resample)'
    """
    from sklearn.model_selection import train_test_split

    strategies = get_strategies(X, y)
    chosen_key = next((k for k in strategies if strategy.lower() in k.lower()), None)

    if chosen_key is None:
        print(f"[WARN] Strategy '{strategy}' not available — using balanced class weights")
        X_res, y_res = X, y
        cw = "balanced"
    else:
        X_res, y_res = strategies[chosen_key]
        cw = None
        print(f"\n[train] Using strategy: {chosen_key} ({len(X_res):,} samples)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y_res,
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        class_weight=cw,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    f1     = f1_score(y_test, y_pred, average="macro")
    print(f"\n[eval]  Test F1 Macro: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model (overwrites the standard one)
    path = MODELS_DIR / "fault_classifier.joblib"
    joblib.dump(clf, path)
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    print(f"\n[OK]   Model saved → {path}")

    # Save strategy metadata
    meta = {
        "strategy":        chosen_key or "balanced_weights",
        "n_train":         len(X_train),
        "f1_macro_test":   round(f1, 4),
        "classes":         list(le.classes_),
        "feature_count":   X.shape[1],
    }
    import json
    (MODELS_DIR / "training_meta.json").write_text(json.dumps(meta, indent=2))
    return clf


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SMOTE Trainer — Imbalanced Fault Detection")
    print("=" * 60)

    X, y, le, feat_cols = load_data()

    print("\nStep 1: Compare all resampling strategies")
    comparison = compare_strategies(X, y)

    print("\n── Ranking (by F1-macro) ────────────────────────────────────")
    rank_cols = ["strategy", "f1_macro_mean", "f1_macro_std", "f1_weighted_mean", "n_samples"]
    print(comparison[rank_cols].round(4).to_string(index=False))

    best_strategy = comparison.iloc[0]["strategy"]
    print(f"\n[best] Best strategy: {best_strategy}")

    print("\nStep 2: Train final model with best strategy")
    train_best(X, y, le, strategy=best_strategy)

    # Save comparison table
    out = PROC_DIR / "smote_strategy_comparison.csv"
    comparison.to_csv(out, index=False)
    print(f"\n[OK]   Comparison table saved → {out}")
