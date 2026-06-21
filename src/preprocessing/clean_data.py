"""
Data Preprocessing Module
==========================
Cleans raw sensor data, handles missing values, normalises signals,
and saves processed outputs to data/processed/.

Usage:
    python src/preprocessing/clean_data.py
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

# ── Load config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

RAW_DIR  = ROOT / CFG["paths"]["raw_data"]
PROC_DIR = ROOT / CFG["paths"]["processed_data"]
PROC_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLS = [
    "air_temp_K", "process_temp_K",
    "rotational_speed_rpm", "torque_Nm", "tool_wear_min",
]


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        print(f"  [clean] Removed {removed} duplicate rows")
    return df


def handle_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Fill numeric NaNs with column median (default) or mean.
    Drop rows where ALL values are NaN.
    """
    df = df.dropna(how="all")
    numeric_cols = df.select_dtypes(include="number").columns
    n_missing = df[numeric_cols].isna().sum().sum()

    if n_missing > 0:
        print(f"  [clean] Filling {n_missing} missing numeric values via '{strategy}'")
        if strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "forward_fill":
            df[numeric_cols] = df[numeric_cols].ffill()
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 3.0) -> pd.DataFrame:
    """
    Flag (not drop) outliers using IQR method.
    Adds a boolean column `is_outlier` for traceability.
    """
    outlier_mask = pd.Series(False, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        outlier_mask |= (df[col] < lower) | (df[col] > upper)

    n_outliers = outlier_mask.sum()
    df["is_outlier"] = outlier_mask
    if n_outliers:
        print(f"  [clean] Flagged {n_outliers} outlier rows (kept, marked is_outlier=True)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise(
    df: pd.DataFrame,
    cols: list,
    method: str = "minmax",
    save_scaler: bool = True,
) -> pd.DataFrame:
    """
    Normalise sensor columns and optionally save the scaler.

    Parameters
    ----------
    method : 'minmax' (0-1) or 'standard' (z-score)
    """
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    df = df.copy()
    valid_cols = [c for c in cols if c in df.columns]
    df[valid_cols] = scaler.fit_transform(df[valid_cols])

    if save_scaler:
        models_dir = ROOT / CFG["paths"]["models"]
        models_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = models_dir / f"scaler_{method}.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"  [norm] Scaler saved → {scaler_path.name}")

    print(f"  [norm] Normalised {len(valid_cols)} columns via '{method}'")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Derived / engineered columns for tabular data
# ─────────────────────────────────────────────────────────────────────────────

def engineer_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived columns useful for CBM analysis."""
    df = df.copy()

    # Power = Torque × Angular velocity  (ω = rpm × 2π/60)
    if "torque_Nm" in df.columns and "rotational_speed_rpm" in df.columns:
        omega = df["rotational_speed_rpm"] * 2 * np.pi / 60
        df["power_W"] = df["torque_Nm"] * omega

    # Temperature differential (process - air)
    if "process_temp_K" in df.columns and "air_temp_K" in df.columns:
        df["temp_diff_K"] = df["process_temp_K"] - df["air_temp_K"]

    # Fault label: combine all failure modes into one string
    fault_cols = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]
    if fault_cols:
        def _label(row):
            faults = [c for c in fault_cols if row[c] == 1]
            return faults[0] if faults else "Normal"
        df["fault_type"] = df.apply(_label, axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Signal segmentation  (for time-series vibration data)
# ─────────────────────────────────────────────────────────────────────────────

def segment_signal(
    signal: np.ndarray,
    window_size: int | None = None,
    overlap: float | None = None,
) -> list[np.ndarray]:
    """
    Slice a 1-D vibration signal into overlapping windows.

    Returns a list of numpy arrays, each of length `window_size`.
    """
    if window_size is None:
        window_size = CFG["signal"]["window_size"]
    if overlap is None:
        overlap = CFG["signal"]["overlap"]

    step = int(window_size * (1 - overlap))
    segments = []
    start = 0
    while start + window_size <= len(signal):
        segments.append(signal[start : start + window_size])
        start += step
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_ai4i(save: bool = True) -> pd.DataFrame:
    """Full cleaning pipeline for the AI4I 2020 dataset."""
    from src.ingestion.download_data import load_ai4i

    print("\n[STEP] Loading raw data …")
    df = load_ai4i()

    print("[STEP] Cleaning …")
    df = drop_duplicates(df)
    df = handle_missing(df)
    df = remove_outliers_iqr(df, cols=SENSOR_COLS)

    print("[STEP] Engineering features …")
    df = engineer_tabular_features(df)

    print("[STEP] Normalising sensor columns …")
    df = normalise(df, cols=SENSOR_COLS, method="minmax")

    if save:
        out_path = PROC_DIR / "ai4i_clean.csv"
        df.to_csv(out_path, index=False)
        print(f"\n[OK]   Clean dataset saved → {out_path}")

    print(f"\nFinal shape: {df.shape}")
    print(f"Fault distribution:\n{df['fault_type'].value_counts()}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  Data Preprocessing — Condition Monitoring Project")
    print("=" * 60)
    df_clean = preprocess_ai4i(save=True)
    print("\nSample:\n", df_clean[["uid", "fault_type", "power_W", "temp_diff_K"]].head(5).to_string(index=False))
