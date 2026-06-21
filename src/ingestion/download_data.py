"""
Data Ingestion Module
=====================
Downloads and loads open-source condition monitoring datasets.

Supported datasets:
  - AI4I 2020 Predictive Maintenance (UCI)       — tabular sensor + failure labels
  - NASA Bearing Dataset                          — time-series vibration signals

Usage:
    python src/ingestion/download_data.py
"""

import os
import yaml
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Load config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

RAW_DIR = ROOT / CFG["paths"]["raw_data"]
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. AI4I 2020  (tabular — fastest to get started)
# ─────────────────────────────────────────────────────────────────────────────

def download_ai4i(force: bool = False) -> Path:
    """Download the AI4I 2020 Predictive Maintenance dataset from UCI."""
    url  = CFG["datasets"]["ai4i_url"]
    dest = RAW_DIR / CFG["datasets"]["ai4i_filename"]

    if dest.exists() and not force:
        print(f"[INFO] AI4I dataset already exists: {dest}")
        return dest

    print(f"[INFO] Downloading AI4I 2020 dataset from UCI ...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))

    print(f"[OK]   Saved to {dest}")
    return dest


def load_ai4i(path: Path | None = None) -> pd.DataFrame:
    """
    Load and lightly rename the AI4I 2020 dataset.

    Columns returned
    ----------------
    uid, product_id, type, air_temp_K, process_temp_K,
    rotational_speed_rpm, torque_Nm, tool_wear_min,
    machine_failure, TWF, HDF, PWF, OSF, RNF
    """
    if path is None:
        path = RAW_DIR / CFG["datasets"]["ai4i_filename"]

    if not path.exists():
        download_ai4i()

    df = pd.read_csv(path)

    rename = {
        "UDI":                   "uid",
        "Product ID":            "product_id",
        "Type":                  "type",
        "Air temperature [K]":   "air_temp_K",
        "Process temperature [K]": "process_temp_K",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]":           "torque_Nm",
        "Tool wear [min]":       "tool_wear_min",
        "Machine failure":       "machine_failure",
        "TWF":                   "TWF",   # Tool Wear Failure
        "HDF":                   "HDF",   # Heat Dissipation Failure
        "PWF":                   "PWF",   # Power Failure
        "OSF":                   "OSF",   # Overstrain Failure
        "RNF":                   "RNF",   # Random Failures
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    print(f"[OK]   Loaded AI4I — {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"       Failure rate: {df['machine_failure'].mean():.1%}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. NASA Bearing Dataset  (time-series vibration)
# ─────────────────────────────────────────────────────────────────────────────
# The NASA Bearing dataset is hosted on Kaggle and requires a Kaggle API token.
# Instructions:
#   1. Create account at kaggle.com
#   2. Download API token → place at ~/.kaggle/kaggle.json
#   3. Uncomment and run download_nasa_bearing() below.
#
# Dataset URL: https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset

def download_nasa_bearing():
    """Download NASA Bearing Dataset via Kaggle CLI."""
    try:
        import subprocess
        dest = RAW_DIR / "nasa_bearing"
        if dest.exists():
            print("[INFO] NASA Bearing dataset already exists.")
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d",
             "vinayak123tyagi/bearing-dataset", "--unzip", "-p", str(dest)],
            check=True,
        )
        print(f"[OK]   NASA Bearing dataset saved to {dest}")
        return dest
    except FileNotFoundError:
        print("[WARN] Kaggle CLI not found. Install with: pip install kaggle")
        print("       Then place ~/.kaggle/kaggle.json and retry.")


def load_nasa_bearing(folder: Path | None = None) -> pd.DataFrame:
    """
    Load NASA Bearing vibration data from CSV files.
    Each file is a timestep; columns are four bearing channels (B1…B4).

    Returns a long-format DataFrame with columns:
        timestamp, bearing, channel, value
    """
    if folder is None:
        folder = RAW_DIR / "nasa_bearing"

    if not folder.exists():
        print("[WARN] NASA Bearing folder not found. Run download_nasa_bearing() first.")
        return pd.DataFrame()

    records = []
    for csv_file in sorted(folder.rglob("*.csv")):
        df = pd.read_csv(csv_file, sep="\t", header=None)
        df.columns = [f"B{i+1}" for i in range(df.shape[1])]
        df["file"] = csv_file.stem
        records.append(df)

    if not records:
        return pd.DataFrame()

    combined = pd.concat(records, ignore_index=True)
    print(f"[OK]   Loaded NASA Bearing — {combined.shape[0]:,} rows")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generic CSV loader (for custom datasets)
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    """Load any CSV from the raw data directory."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, **kwargs)
    print(f"[OK]   Loaded {filename} — {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Data Ingestion — Condition Monitoring Project")
    print("=" * 60)

    # Download & quick-peek at AI4I
    download_ai4i()
    df = load_ai4i()
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
    print(f"\nColumn types:\n{df.dtypes}")
