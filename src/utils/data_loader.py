"""
Data Loader
============
Unified interface for loading any dataset supported by the project.
Handles format detection, caching, validation, and summary printing.

Supported sources:
  - AI4I 2020 (CSV from UCI or local)
  - NASA Bearing (CSV files from Kaggle download)
  - Processed outputs (cleaned, features, scored)
  - SQLite database (health scores, inspection history)
  - Custom CSV / Excel / Parquet

Usage:
    from src.utils.data_loader import DataLoader
    loader = DataLoader()

    # Load raw or processed data
    df = loader.load("ai4i_raw")
    df = loader.load("ai4i_clean")
    df = loader.load("ai4i_scored")

    # Load from custom path
    df = loader.load_file("path/to/my_data.csv")

    # Load from database
    df = loader.from_db("health_scores", asset_id="MTR-001")
"""

import yaml
import sqlite3
import hashlib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

log = logging.getLogger(__name__)

RAW_DIR  = ROOT / CFG["paths"]["raw_data"]
PROC_DIR = ROOT / CFG["paths"]["processed_data"]
DB_PATH  = ROOT / CFG["database"]["path"]

# ── Known datasets ────────────────────────────────────────────────────────────

DATASET_MAP = {
    # Raw downloads
    "ai4i_raw":     RAW_DIR  / "ai4i2020.csv",
    # Processed pipeline outputs
    "ai4i_clean":   PROC_DIR / "ai4i_clean.csv",
    "ai4i_features":PROC_DIR / "ai4i_features.csv",
    "ai4i_enriched":PROC_DIR / "ai4i_features_enriched.csv",
    "ai4i_selected":PROC_DIR / "ai4i_features_selected.csv",
    "ai4i_scored":  PROC_DIR / "ai4i_health_scored.csv",
    "ai4i_preds":   PROC_DIR / "ai4i_predictions.csv",
    # Analysis outputs
    "reliability":  PROC_DIR / "reliability_metrics.csv",
    "smote_compare":PROC_DIR / "smote_strategy_comparison.csv",
    "ensemble_compare": PROC_DIR / "ensemble_comparison.csv",
    "eda_stats":    PROC_DIR / "eda_summary_stats.csv",
    "feat_importance": PROC_DIR / "feature_importance.csv",
}


# ── DataLoader class ──────────────────────────────────────────────────────────

class DataLoader:
    """
    Unified dataset loader with optional in-memory caching.

    Parameters
    ----------
    cache : bool
        Cache loaded DataFrames in memory (faster repeated loads, uses RAM).
    """

    def __init__(self, cache: bool = True):
        self._cache: dict[str, pd.DataFrame] = {}
        self._use_cache = cache

    # ── Named dataset loader ──────────────────────────────────────────────────

    def load(self, name: str, force_reload: bool = False, **read_kwargs) -> pd.DataFrame:
        """
        Load a named dataset from the project data directories.

        Parameters
        ----------
        name         : dataset key (see DATASET_MAP for valid names)
        force_reload : bypass cache even if available
        **read_kwargs: passed to pd.read_csv / pd.read_parquet etc.

        Returns
        -------
        pd.DataFrame
        """
        if name not in DATASET_MAP:
            available = "\n  ".join(sorted(DATASET_MAP.keys()))
            raise KeyError(
                f"Unknown dataset '{name}'. Available datasets:\n  {available}"
            )

        path = DATASET_MAP[name]

        if self._use_cache and name in self._cache and not force_reload:
            log.debug(f"[cache] Hit: {name}")
            return self._cache[name].copy()

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset '{name}' not found at: {path}\n"
                f"Run the pipeline to generate it: python main.py"
            )

        df = self.load_file(path, **read_kwargs)
        log.info(f"[load] {name}: {df.shape[0]:,} rows × {df.shape[1]} cols  ({path.name})")

        if self._use_cache:
            self._cache[name] = df.copy()

        return df

    # ── File loader ───────────────────────────────────────────────────────────

    def load_file(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """
        Load any supported file format by extension.

        Supported: .csv, .tsv, .parquet, .feather, .xlsx, .xls, .json
        """
        path = Path(path)
        ext  = path.suffix.lower()

        loaders = {
            ".csv":     lambda p: pd.read_csv(p, **kwargs),
            ".tsv":     lambda p: pd.read_csv(p, sep="\t", **kwargs),
            ".parquet": lambda p: pd.read_parquet(p, **kwargs),
            ".feather": lambda p: pd.read_feather(p, **kwargs),
            ".xlsx":    lambda p: pd.read_excel(p, **kwargs),
            ".xls":     lambda p: pd.read_excel(p, **kwargs),
            ".json":    lambda p: pd.read_json(p, **kwargs),
        }

        loader = loaders.get(ext)
        if loader is None:
            raise ValueError(
                f"Unsupported file format: {ext}\n"
                f"Supported: {list(loaders.keys())}"
            )

        df = loader(path)
        self._print_summary(df, path.name)
        return df

    # ── Database loaders ──────────────────────────────────────────────────────

    def from_db(
        self,
        table: str,
        asset_id: str | None = None,
        limit: int | None = None,
        where: str | None = None,
    ) -> pd.DataFrame:
        """
        Load a table or view from the SQLite database.

        Parameters
        ----------
        table    : table or view name (e.g. 'health_scores', 'v_latest_health')
        asset_id : optional filter on asset_id column
        limit    : optional row limit
        where    : raw WHERE clause (overrides asset_id filter)

        Returns
        -------
        pd.DataFrame
        """
        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"Database not found: {DB_PATH}\n"
                "Run: python main.py to initialise."
            )

        sql    = f"SELECT * FROM {table}"
        params: tuple = ()

        if where:
            sql += f" WHERE {where}"
        elif asset_id:
            sql += " WHERE asset_id = ?"
            params = (asset_id,)

        if limit:
            sql += f" LIMIT {limit}"

        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        log.info(f"[db] {table}: {len(df):,} rows")
        return df

    def asset_history(self, asset_id: str) -> pd.DataFrame:
        """Load full health score history for one asset."""
        return self.from_db("health_scores", asset_id=asset_id)

    def latest_health(self) -> pd.DataFrame:
        """Load the latest health score for each asset."""
        return self.from_db("v_latest_health")

    def open_recommendations(self) -> pd.DataFrame:
        """Load all open maintenance recommendations."""
        return self.from_db("v_open_recommendations")

    # ── NASA Bearing loader ───────────────────────────────────────────────────

    def load_nasa_bearing(
        self,
        folder: str | Path | None = None,
        channel: int = 0,
        max_files: int | None = None,
    ) -> pd.DataFrame:
        """
        Load NASA Bearing dataset from downloaded CSV files.

        Returns a DataFrame with columns:
            file_id, sample_idx, B1, B2, B3, B4
        """
        if folder is None:
            folder = RAW_DIR / "nasa_bearing"
        folder = Path(folder)

        if not folder.exists():
            raise FileNotFoundError(
                f"NASA Bearing folder not found: {folder}\n"
                "Download with: python src/ingestion/download_data.py (requires Kaggle API key)"
            )

        records = []
        csv_files = sorted(folder.rglob("*.csv"))
        if max_files:
            csv_files = csv_files[:max_files]

        for i, csv_file in enumerate(csv_files):
            df = pd.read_csv(csv_file, sep="\t", header=None)
            n_ch = df.shape[1]
            df.columns = [f"B{j+1}" for j in range(n_ch)]
            df["file_id"] = i
            df["sample_idx"] = range(len(df))
            records.append(df)

        if not records:
            return pd.DataFrame()

        combined = pd.concat(records, ignore_index=True)
        log.info(f"[nasa] Loaded {len(csv_files)} files → {combined.shape[0]:,} samples")
        return combined

    # ── Batch loading ─────────────────────────────────────────────────────────

    def load_many(self, names: list[str]) -> dict[str, pd.DataFrame]:
        """Load multiple named datasets at once."""
        return {name: self.load(name) for name in names}

    def load_pipeline_outputs(self) -> dict[str, pd.DataFrame]:
        """
        Load all available pipeline output DataFrames.
        Skips files that don't exist yet.
        """
        outputs = ["ai4i_clean", "ai4i_features", "ai4i_scored", "ai4i_preds"]
        result  = {}
        for name in outputs:
            try:
                result[name] = self.load(name)
            except FileNotFoundError:
                log.debug(f"[skip] {name} not yet generated")
        return result

    # ── Cache management ──────────────────────────────────────────────────────

    def clear_cache(self, name: str | None = None) -> None:
        """Clear one or all cached DataFrames."""
        if name:
            self._cache.pop(name, None)
            log.debug(f"[cache] Cleared: {name}")
        else:
            self._cache.clear()
            log.debug("[cache] All cleared")

    def cache_info(self) -> dict[str, str]:
        """Return info about cached DataFrames."""
        return {
            name: f"{df.shape[0]:,} rows × {df.shape[1]} cols"
            for name, df in self._cache.items()
        }

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(
        self,
        df: pd.DataFrame,
        required_cols: list[str],
        name: str = "DataFrame",
        min_rows: int = 1,
    ) -> bool:
        """
        Validate a DataFrame. Returns True if valid, raises ValueError otherwise.
        """
        if len(df) < min_rows:
            raise ValueError(f"{name}: need ≥{min_rows} rows, got {len(df)}")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
        return True

    # ── File integrity ────────────────────────────────────────────────────────

    @staticmethod
    def checksum(path: str | Path) -> str:
        """Return MD5 checksum of a file."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(df: pd.DataFrame, name: str) -> None:
        n_missing = df.isna().sum().sum()
        numeric   = df.select_dtypes("number").columns.tolist()
        log.info(
            f"[load] {name}: {df.shape[0]:,} rows × {df.shape[1]} cols  "
            f"| {len(numeric)} numeric  | {n_missing:,} missing values"
        )


# ── Module-level convenience instance ────────────────────────────────────────

_default_loader = DataLoader()

def load(name: str, **kwargs) -> pd.DataFrame:
    """Module-level shortcut: DataLoader().load(name)."""
    return _default_loader.load(name, **kwargs)

def load_file(path: str | Path, **kwargs) -> pd.DataFrame:
    """Module-level shortcut: DataLoader().load_file(path)."""
    return _default_loader.load_file(path, **kwargs)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("  Data Loader — Available Datasets")
    print("=" * 60)

    loader = DataLoader()

    print("\nRegistered dataset names:")
    for name, path in sorted(DATASET_MAP.items()):
        exists = "✓" if path.exists() else "✗"
        print(f"  [{exists}] {name:<25} → {path.name}")

    print("\nLoading available pipeline outputs...")
    outputs = loader.load_pipeline_outputs()
    for name, df in outputs.items():
        print(f"  {name:<20} {df.shape[0]:,} rows × {df.shape[1]} cols")

    if outputs:
        first = list(outputs.values())[0]
        print(f"\nSample (first dataset):\n{first.head(3).to_string()}")

    print("\nDatabase tables available:")
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            tables = conn.execute(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
            ).fetchall()
        for name, kind in tables:
            print(f"  {kind:<5} {name}")
    else:
        print("  (Database not initialised — run python main.py)")
