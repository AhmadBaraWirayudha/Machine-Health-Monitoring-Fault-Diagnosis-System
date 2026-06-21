"""
Ensemble Classifier
====================
Combines multiple base classifiers using Voting and Stacking
to improve fault detection robustness.

Strategies:
  Hard Voting    — majority class vote across classifiers
  Soft Voting    — average predicted probabilities
  Stacking       — meta-learner trained on base classifier outputs

Usage:
    python src/modeling/ensemble.py
"""

import yaml
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR   = ROOT / CFG["paths"]["processed_data"]
MODELS_DIR = ROOT / CFG["paths"]["models"]
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = CFG["model"]["classifier"]["random_state"]
CV_FOLDS     = 5

FEATURE_COLS = [
    "air_temp_K", "process_temp_K",
    "rotational_speed_rpm", "torque_Nm", "tool_wear_min",
    "power_W", "temp_diff_K",
]
LABEL_COL = "fault_type"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    for fname in ["ai4i_features.csv", "ai4i_clean.csv"]:
        p = PROC_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("No feature file found. Run main.py --steps 1 2 3 first.")

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

    print(f"[data]  {len(X):,} samples  |  {len(feat_cols)} features  |  "
          f"{len(le.classes_)} classes: {list(le.classes_)}")
    return X, y, le, feat_cols


# ── Base estimators ───────────────────────────────────────────────────────────

def get_base_estimators() -> list[tuple[str, object]]:
    """Return a list of (name, estimator) pairs for use in ensemble."""
    return [
        ("rf", RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
        ("gbm", GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE,
        )),
        ("svm", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight="balanced",
                    random_state=RANDOM_STATE, probability=False),
            )),
        ])),
        ("knn", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        ])),
        ("lr", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=500,
                random_state=RANDOM_STATE,
            )),
        ])),
    ]


# ── Voting Classifier ─────────────────────────────────────────────────────────

def build_voting_classifier(voting: str = "soft") -> VotingClassifier:
    """
    Build a Voting ensemble.

    Parameters
    ----------
    voting : 'soft' (average probabilities) or 'hard' (majority vote)
             Soft voting generally performs better when base models are
             well-calibrated.
    """
    estimators = get_base_estimators()
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)


# ── Stacking Classifier ───────────────────────────────────────────────────────

def build_stacking_classifier(
    final_estimator=None,
    passthrough: bool = False,
) -> StackingClassifier:
    """
    Build a Stacking ensemble.

    The meta-learner (final_estimator) is trained on the out-of-fold
    predictions of the base classifiers.

    Parameters
    ----------
    final_estimator : meta-learner. Defaults to LogisticRegression.
    passthrough     : if True, also passes original features to meta-learner.
    """
    if final_estimator is None:
        final_estimator = LogisticRegression(
            class_weight="balanced", max_iter=500,
            random_state=RANDOM_STATE,
        )
    estimators = get_base_estimators()
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        passthrough=passthrough,
        n_jobs=-1,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_ensemble(
    name: str,
    clf,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Cross-validate an ensemble and return metrics dict."""
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy",  n_jobs=-1)
    f1m = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro",  n_jobs=-1)
    f1w = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"  {name:<25}  "
          f"Acc={acc.mean():.4f}±{acc.std():.4f}  "
          f"F1-macro={f1m.mean():.4f}±{f1m.std():.4f}  "
          f"F1-weighted={f1w.mean():.4f}±{f1w.std():.4f}")
    return {
        "name":               name,
        "acc_mean":           round(acc.mean(), 4),
        "f1_macro_mean":      round(f1m.mean(), 4),
        "f1_weighted_mean":   round(f1w.mean(), 4),
    }


def compare_ensembles(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Compare individual base models against ensemble strategies.
    """
    print("\n── Ensemble Comparison ──────────────────────────────────────")
    print(f"  {'Model':<25}  {'Accuracy':<22}  {'F1-macro':<22}  F1-weighted")
    print("  " + "─" * 82)

    results = []

    # Individual base models for baseline
    for name, clf in get_base_estimators():
        results.append(evaluate_ensemble(f"Base: {name.upper()}", clf, X, y))

    # Ensemble strategies
    for label, clf in [
        ("Hard Voting",      build_voting_classifier("hard")),
        ("Soft Voting",      build_voting_classifier("soft")),
        ("Stacking (LR)",    build_stacking_classifier()),
        ("Stacking (RF)",    build_stacking_classifier(
            final_estimator=RandomForestClassifier(
                n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1
            )
        )),
    ]:
        results.append(evaluate_ensemble(label, clf, X, y))

    df = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)
    return df


# ── Train & save best ensemble ────────────────────────────────────────────────

def train_best_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    strategy: str = "soft_voting",
) -> object:
    """
    Train the full ensemble on train split, evaluate on test split, save model.

    Parameters
    ----------
    strategy : 'soft_voting' | 'hard_voting' | 'stacking'
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    strategy_map = {
        "soft_voting": build_voting_classifier("soft"),
        "hard_voting": build_voting_classifier("hard"),
        "stacking":    build_stacking_classifier(),
    }
    clf = strategy_map.get(strategy, build_voting_classifier("soft"))

    print(f"\n[train] Fitting {strategy} ensemble on {len(X_train):,} samples …")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    f1     = f1_score(y_test, y_pred, average="macro")
    print(f"[eval]  Test F1 Macro: {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save
    model_path = MODELS_DIR / "ensemble_classifier.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(le,  MODELS_DIR / "label_encoder.joblib")

    meta = {
        "strategy":      strategy,
        "f1_macro_test": round(f1, 4),
        "trained_at":    datetime.now().isoformat(),
        "n_train":       len(X_train),
        "classes":       list(le.classes_),
    }
    (MODELS_DIR / "ensemble_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[OK]   Ensemble saved → {model_path}")
    return clf


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Ensemble Classifier — Condition Monitoring")
    print("=" * 60)

    X, y, le, feat_cols = load_data()

    print("\nStep 1: Compare all strategies (5-fold CV) …")
    comparison = compare_ensembles(X, y)
    print(f"\n── Ranking ──────────────────────────────────────────────────")
    print(comparison[["name", "f1_macro_mean", "f1_weighted_mean"]].to_string(index=False))

    best_row = comparison.iloc[0]
    print(f"\n[best] Best strategy: {best_row['name']} (F1={best_row['f1_macro_mean']:.4f})")

    # Map display name back to strategy key
    strategy_key = "soft_voting"
    if "Hard" in best_row["name"]:
        strategy_key = "hard_voting"
    elif "Stack" in best_row["name"]:
        strategy_key = "stacking"

    print(f"\nStep 2: Train final ensemble using '{strategy_key}' …")
    train_best_ensemble(X, y, le, strategy=strategy_key)

    # Save comparison
    out = PROC_DIR / "ensemble_comparison.csv"
    comparison.to_csv(out, index=False)
    print(f"\n[OK]   Comparison saved → {out}")
