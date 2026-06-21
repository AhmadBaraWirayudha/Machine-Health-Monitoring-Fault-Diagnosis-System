"""
Shared Utilities
================
Helper functions used across the pipeline:
  - logging setup
  - config loading
  - path resolution
  - timing decorator
  - data validation
  - export helpers
"""

import yaml
import logging
import time
import json
import functools
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# ── Project root & config ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    """Load project config.yaml."""
    if path is None:
        path = ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger that writes to console + logs/pipeline.log.

    Usage:
        log = get_logger(__name__)
        log.info("Step complete")
        log.warning("Missing column: rms")
        log.error("Pipeline failed")
    """
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Timing decorator ──────────────────────────────────────────────────────────

def timed(fn):
    """Decorator: prints elapsed time after each function call."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        log = get_logger(fn.__module__)
        log.info(f"→ Starting {fn.__name__}")
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info(f"✓ {fn.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ── Path helpers ──────────────────────────────────────────────────────────────

def get_path(key: str) -> Path:
    """Resolve a path from config.yaml paths section."""
    rel = CFG["paths"].get(key)
    if rel is None:
        raise KeyError(f"Path key '{key}' not found in config.yaml")
    p = ROOT / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir()       -> Path: return get_path("raw_data")
def processed_dir() -> Path: return get_path("processed_data")
def models_dir()    -> Path: return get_path("models")
def reports_dir()   -> Path: return get_path("reports")
def plots_dir()     -> Path: return get_path("plots")


# ── Data validation ───────────────────────────────────────────────────────────

class DataValidationError(Exception):
    pass


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str = "DataFrame",
    min_rows: int = 1,
) -> None:
    """
    Raise DataValidationError if the DataFrame doesn't meet expectations.

    Checks:
      - Not empty (min_rows)
      - All required columns present
      - No fully-null required columns
    """
    if len(df) < min_rows:
        raise DataValidationError(
            f"{name}: expected ≥{min_rows} rows, got {len(df)}"
        )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{name}: missing required columns: {missing}"
        )
    for col in required_cols:
        if df[col].isna().all():
            raise DataValidationError(
                f"{name}: column '{col}' is entirely null"
            )


def check_feature_range(
    df: pd.DataFrame,
    col: str,
    lo: float,
    hi: float,
    warn_only: bool = True,
) -> pd.Series:
    """
    Flag rows where `col` falls outside [lo, hi].

    Returns a boolean mask of out-of-range rows.
    If warn_only=False, raises ValueError instead.
    """
    log = get_logger(__name__)
    mask = (df[col] < lo) | (df[col] > hi)
    n = mask.sum()
    if n > 0:
        msg = f"'{col}': {n} values outside [{lo}, {hi}]"
        if warn_only:
            log.warning(msg)
        else:
            raise ValueError(msg)
    return mask


# ── Numeric helpers ───────────────────────────────────────────────────────────

def safe_divide(a: float | np.ndarray, b: float | np.ndarray,
                fill: float = 0.0) -> float | np.ndarray:
    """Division that returns `fill` instead of Inf/NaN on zero denominator."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result if isinstance(a, np.ndarray) else float(result)


def moving_average(series: pd.Series, window: int = 10) -> pd.Series:
    """Centred moving average with min_periods=1."""
    return series.rolling(window, center=True, min_periods=1).mean()


def rolling_zscore(series: pd.Series, window: int = 50) -> pd.Series:
    """Z-score relative to a rolling window (useful for drift detection)."""
    roll_mean = series.rolling(window, min_periods=1).mean()
    roll_std  = series.rolling(window, min_periods=1).std().fillna(1.0)
    return (series - roll_mean) / roll_std


# ── Export helpers ────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, filename: str, subdir: str = "processed_data") -> Path:
    """Save a DataFrame to the processed data directory."""
    out = get_path(subdir) / filename
    df.to_csv(out, index=False)
    get_logger(__name__).info(f"Saved {filename} ({len(df):,} rows)")
    return out


def save_json(obj: dict | list, filename: str, subdir: str = "reports") -> Path:
    """Save a JSON object to the reports directory."""
    out = get_path(subdir) / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    get_logger(__name__).info(f"Saved {filename}")
    return out


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Return a Markdown table string from a DataFrame (for README embeds)."""
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False)


# ── Pipeline run record ───────────────────────────────────────────────────────

class PipelineRun:
    """
    Context manager that records a pipeline run to logs/runs.json.

    Usage:
        with PipelineRun("health_scoring") as run:
            run.log("rows_processed", 10000)
            do_work()
    """
    _log_path = ROOT / "logs" / "runs.json"

    def __init__(self, name: str):
        self.name   = name
        self.start  = None
        self.meta   = {}
        self.logger = get_logger(f"run.{name}")

    def __enter__(self):
        self.start = datetime.now()
        self.logger.info(f"Pipeline run started: {self.name}")
        return self

    def log(self, key: str, value):
        """Record a metadata value for this run."""
        self.meta[key] = value

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = datetime.now()
        record = {
            "name":      self.name,
            "started":   self.start.isoformat(),
            "ended":     end.isoformat(),
            "duration_s": round((end - self.start).total_seconds(), 2),
            "status":    "ERROR" if exc_type else "OK",
            "error":     str(exc_val) if exc_val else None,
            **self.meta,
        }
        # Append to runs.json
        self._log_path.parent.mkdir(exist_ok=True)
        runs = []
        if self._log_path.exists():
            with open(self._log_path) as f:
                runs = json.load(f)
        runs.append(record)
        with open(self._log_path, "w") as f:
            json.dump(runs[-100:], f, indent=2)  # keep last 100 runs

        status = "✓" if not exc_type else "✗"
        self.logger.info(f"{status} Run '{self.name}' finished in {record['duration_s']}s")
        return False   # don't suppress exceptions


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    log = get_logger("helpers.test")
    log.info("Logger OK")

    cfg = load_config()
    log.info(f"Config loaded: project = {cfg['project']['name']}")

    # Test timed decorator
    @timed
    def demo_fn():
        time.sleep(0.1)
        return 42
    demo_fn()

    # Test safe_divide
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, 2) == 5.0

    # Test PipelineRun
    with PipelineRun("self_test") as run:
        run.log("rows", 999)

    log.info("All helpers OK")
