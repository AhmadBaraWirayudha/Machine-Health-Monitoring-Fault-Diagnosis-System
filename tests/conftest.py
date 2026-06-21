"""
conftest.py — Shared pytest fixtures
=====================================
Available to all test files in tests/ automatically.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Signal fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fs():
    """Default sampling rate for all signal tests."""
    return 4096


@pytest.fixture(scope="session")
def healthy_signal(fs):
    """Clean sine wave — no impulsive content."""
    np.random.seed(0)
    t = np.linspace(0, 1, fs)
    return np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(fs)


@pytest.fixture(scope="session")
def fault_signal(fs):
    """Sine wave with heavy bearing-like impulses."""
    np.random.seed(1)
    t   = np.linspace(0, 1, fs)
    sig = np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(fs)
    idx = np.random.choice(fs, size=60, replace=False)
    sig[idx] += 6.0
    return sig


@pytest.fixture(scope="session")
def multi_freq_signal(fs):
    """Multi-component signal with shaft + bearing fault tones."""
    np.random.seed(2)
    t = np.linspace(0, 1, fs)
    return (
        np.sin(2 * np.pi * 50 * t) +           # shaft
        0.4 * np.sin(2 * np.pi * 120 * t) +    # bearing outer race
        0.2 * np.sin(2 * np.pi * 200 * t) +    # harmonic
        0.1 * np.random.randn(fs)               # noise
    )


# ── DataFrame fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small AI4I-like DataFrame, 200 rows."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "uid":                    range(1, n + 1),
        "air_temp_K":             np.random.normal(300, 2, n),
        "process_temp_K":         np.random.normal(310, 1.5, n),
        "rotational_speed_rpm":   np.random.normal(1500, 50, n),
        "torque_Nm":              np.random.normal(40, 10, n),
        "tool_wear_min":          np.random.uniform(0, 250, n),
        "machine_failure":        np.random.binomial(1, 0.05, n),
        "fault_type":             np.random.choice(
            ["Normal"] * 9 + ["HDF"], n
        ),
    })
    return df


@pytest.fixture
def scored_df(sample_df):
    """sample_df with health score columns appended."""
    df = sample_df.copy()
    np.random.seed(10)
    n  = len(df)
    df["rms"]           = np.random.uniform(0.1, 0.9, n)
    df["kurtosis"]      = np.random.uniform(2.0, 8.0, n)
    df["crest_factor"]  = np.random.uniform(1.5, 10.0, n)
    df["health_score"]  = np.random.uniform(20, 95, n)
    df["health_status"] = df["health_score"].apply(
        lambda s: "Good" if s >= 80 else ("Warning" if s >= 60 else ("Degraded" if s >= 40 else "Critical"))
    )
    df["recommendation"] = "Continue monitoring."
    return df


@pytest.fixture
def imbalanced_df(sample_df):
    """DataFrame with heavy class imbalance (2% fault rate)."""
    df = sample_df.copy()
    np.random.seed(7)
    df["machine_failure"] = np.random.binomial(1, 0.02, len(df))
    return df


# ── Temp SQLite DB fixture ────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path):
    """
    Temporary SQLite database, populated with the project schema.
    Automatically deleted after the test.
    """
    db_path = tmp_path / "test_cmdb.sqlite"

    # Patch DB path in db_manager
    import src.database.db_manager as dbm
    original_path = dbm.DB_PATH
    dbm.DB_PATH = db_path

    dbm.create_schema()
    dbm.insert_asset("TEST-001", "Test Motor", "motor", "Test Site")
    insp_id  = dbm.insert_inspection("TEST-001", method="vibration")
    score_id = dbm.insert_health_score("TEST-001", 72.5, "Warning", insp_id, anomaly_score=0.6)
    dbm.insert_recommendation("TEST-001", "Inspect bearings.", "Medium", score_id)

    yield db_path

    dbm.DB_PATH = original_path   # restore original path


# ── Model fixture (demo, no file I/O) ────────────────────────────────────────

@pytest.fixture
def trained_classifier(scored_df):
    """A RandomForest fitted on the scored_df fixture (fast, small)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    feature_cols = ["rotational_speed_rpm", "torque_Nm", "tool_wear_min"]
    X  = scored_df[feature_cols].fillna(0)
    le = LabelEncoder()
    y  = le.fit_transform(scored_df["fault_type"])

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf, le


# ── Markers ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skip with -m 'not slow')")
    config.addinivalue_line("markers", "integration: marks tests requiring the full pipeline output")
    config.addinivalue_line("markers", "db: marks tests requiring a live database")
