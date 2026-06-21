"""
Hyperparameter Tuner (Optuna)
==============================
Optimises the Random Forest (and optionally Gradient Boosting)
classifier using Bayesian optimisation via Optuna.

Finds the best combination of:
  n_estimators, max_depth, min_samples_split, min_samples_leaf,
  max_features, class_weight, (for GBM: learning_rate, subsample)

Requirements:
    pip install optuna

Usage:
    # Quick search (30 trials, RF only)
    python src/modeling/hyperparameter_tuner.py

    # Full search
    python src/modeling/hyperparameter_tuner.py --n-trials 100 --model all
"""

import yaml
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARN] Optuna not installed — install with: pip install optuna")

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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    for fname in ["ai4i_features.csv", "ai4i_clean.csv"]:
        p = PROC_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("No feature file found.")

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
    return X, y, le, feat_cols


# ── Objective functions ───────────────────────────────────────────────────────

def rf_objective(trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for Random Forest — maximise F1 macro."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]),
        "class_weight":      trial.suggest_categorical("class_weight", ["balanced", None]),
        "n_jobs":            -1,
        "random_state":      RANDOM_STATE,
    }
    clf = RandomForestClassifier(**params)
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return scores.mean()


def gbm_objective(trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for Gradient Boosting — maximise F1 macro."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
        "max_depth":         trial.suggest_int("max_depth", 2, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state":      RANDOM_STATE,
    }
    clf = GradientBoostingClassifier(**params)
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return scores.mean()


# ── Tuning ────────────────────────────────────────────────────────────────────

def tune(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rf",
    n_trials: int = 50,
    timeout: int | None = None,
) -> dict:
    """
    Run Optuna study for the specified model type.

    Parameters
    ----------
    model_type : 'rf' | 'gbm'
    n_trials   : number of trials (more = better, slower)
    timeout    : stop after this many seconds (optional)

    Returns
    -------
    best_params : dict of best hyperparameters
    """
    if not OPTUNA_AVAILABLE:
        print("[WARN] Optuna not installed — returning default config.yaml params")
        return dict(CFG["model"]["classifier"])

    objective_fn = rf_objective if model_type == "rf" else gbm_objective
    direction    = "maximize"
    study_name   = f"cbm_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # SQLite storage for resumable studies
    storage_path = MODELS_DIR / "optuna_studies.db"
    storage      = f"sqlite:///{storage_path}"

    print(f"\n[optuna] Starting {model_type.upper()} study — {n_trials} trials")
    print(f"         Storage: {storage_path}")

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=False,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    # Progress callback
    def _callback(study, trial):
        if trial.number % 10 == 0 or trial.number < 5:
            print(f"  Trial {trial.number:>3d} — F1={trial.value:.4f}  "
                  f"[best so far: {study.best_value:.4f}]")

    study.optimize(
        lambda trial: objective_fn(trial, X, y),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[_callback],
        show_progress_bar=False,
        gc_after_trial=True,
    )

    print(f"\n[optuna] Best F1 macro: {study.best_value:.4f}")
    print(f"         Best params  : {study.best_params}")

    # Save results
    trials_df = study.trials_dataframe()
    out_csv   = MODELS_DIR / f"optuna_{model_type}_trials.csv"
    trials_df.to_csv(out_csv, index=False)
    print(f"[optuna] Trial history → {out_csv.name}")

    return study.best_params


# ── Plot optimization history ─────────────────────────────────────────────────

def plot_optimization_history(model_type: str = "rf") -> None:
    """Load saved trials and plot convergence."""
    import matplotlib.pyplot as plt

    csv_path = MODELS_DIR / f"optuna_{model_type}_trials.csv"
    if not csv_path.exists():
        print(f"[WARN] No trial CSV found for {model_type}")
        return

    trials = pd.read_csv(csv_path)
    if "value" not in trials.columns:
        return

    best_so_far = trials["value"].cummax()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Convergence
    axes[0].plot(trials.index, trials["value"], alpha=0.4, color="gray", linewidth=0.8, label="Trial F1")
    axes[0].plot(trials.index, best_so_far, color="#2980b9", linewidth=2, label="Best F1")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("F1 Macro (CV)")
    axes[0].set_title(f"{model_type.upper()} — Optimisation Convergence", fontweight="bold")
    axes[0].legend()
    axes[0].set_ylim([0, 1])

    # Parameter importance (top 6)
    param_cols = [c for c in trials.columns if c.startswith("params_")]
    if len(param_cols) > 1:
        corr = trials[param_cols + ["value"]].corr()["value"].drop("value").abs()
        corr = corr.sort_values(ascending=False).head(6)
        axes[1].barh(
            [c.replace("params_", "") for c in corr.index],
            corr.values,
            color="#2980b9", edgecolor="white",
        )
        axes[1].set_xlabel("|Correlation with F1|")
        axes[1].set_title("Parameter Sensitivity", fontweight="bold")
        axes[1].invert_yaxis()

    plt.suptitle(f"Optuna Hyperparameter Tuning — {model_type.upper()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = MODELS_DIR / f"optuna_{model_type}_history.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Optimization history → {out.name}")


# ── Retrain with best params ──────────────────────────────────────────────────

def retrain_with_best(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    best_params: dict,
    model_type: str = "rf",
) -> object:
    """Train final model on full dataset using tuned hyperparameters."""
    from sklearn.model_selection import train_test_split

    print(f"\n[train] Retraining {model_type.upper()} with best hyperparameters …")

    clf_params = {k: v for k, v in best_params.items()}
    clf_params["random_state"] = RANDOM_STATE

    if model_type == "rf":
        clf_params["n_jobs"] = -1
        clf = RandomForestClassifier(**clf_params)
    else:
        clf = GradientBoostingClassifier(**clf_params)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    f1     = f1_score(y_test, y_pred, average="macro")
    print(f"[eval]  Test F1 Macro (tuned): {f1:.4f}")

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save
    model_path = MODELS_DIR / "fault_classifier.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

    meta = {
        "model_type":        model_type,
        "tuned":             True,
        "best_params":       best_params,
        "f1_macro_test":     round(f1, 4),
        "trained_at":        datetime.now().isoformat(),
        "n_samples":         len(X),
    }
    (MODELS_DIR / "training_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK]   Model saved → {model_path}")
    return clf


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuner")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--model", choices=["rf", "gbm", "all"], default="rf",
                        help="Model type to tune (default: rf)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Stop tuning after N seconds")
    args = parser.parse_args()

    print("=" * 60)
    print("  Hyperparameter Tuner (Optuna)")
    print("=" * 60)

    X, y, le, feat_cols = load_data()
    print(f"\nFeatures: {feat_cols}")

    models_to_tune = ["rf", "gbm"] if args.model == "all" else [args.model]
    all_results    = {}

    for mtype in models_to_tune:
        print(f"\n{'─'*60}")
        print(f"  Tuning: {mtype.upper()}")
        print(f"{'─'*60}")

        best = tune(X, y, model_type=mtype, n_trials=args.n_trials, timeout=args.timeout)
        all_results[mtype] = best
        plot_optimization_history(mtype)

    # Train with overall best
    best_model_type = models_to_tune[0]
    print(f"\n[final] Training with best {best_model_type.upper()} params …")
    retrain_with_best(X, y, le, all_results[best_model_type], model_type=best_model_type)

    print("\n[done] Hyperparameter tuning complete.")
    print(f"       Best parameters saved to: {MODELS_DIR / 'training_meta.json'}")
