"""
Database Manager
================
Manages the SQLite condition monitoring database (cmdb.sqlite).

Tables
------
  assets            — registered machines / assets
  inspections       — inspection events per asset
  sensor_readings   — raw sensor values per inspection
  feature_records   — extracted diagnostic features
  health_scores     — computed health scores over time
  recommendations   — maintenance recommendations

Usage:
    python src/database/db_manager.py
"""

import sqlite3
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Load config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["database"]["path"]
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row     # return dict-like rows
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema creation
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Assets table
CREATE TABLE IF NOT EXISTS assets (
    asset_id        TEXT PRIMARY KEY,
    asset_name      TEXT NOT NULL,
    asset_type      TEXT,          -- e.g. 'motor', 'pump', 'fan'
    location        TEXT,
    manufacturer    TEXT,
    install_date    TEXT,
    rated_power_kW  REAL,
    rated_speed_rpm REAL,
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Inspection events
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id        TEXT NOT NULL REFERENCES assets(asset_id),
    inspect_date    TEXT NOT NULL,
    inspector       TEXT,
    method          TEXT,          -- 'vibration', 'temperature', 'oil', 'visual'
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Raw sensor readings per inspection
CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id       INTEGER NOT NULL REFERENCES inspections(inspection_id),
    air_temp_K          REAL,
    process_temp_K      REAL,
    rotational_speed_rpm REAL,
    torque_Nm           REAL,
    tool_wear_min       REAL,
    power_W             REAL,
    temp_diff_K         REAL,
    recorded_at         TEXT DEFAULT (datetime('now'))
);

-- Extracted diagnostic features
CREATE TABLE IF NOT EXISTS feature_records (
    feature_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id   INTEGER NOT NULL REFERENCES inspections(inspection_id),
    rms             REAL,
    kurtosis        REAL,
    crest_factor    REAL,
    peak_to_peak    REAL,
    skewness        REAL,
    variance        REAL,
    fft_peak_freq   REAL,
    fft_peak_amp    REAL,
    band_energy_low REAL,
    band_energy_mid REAL,
    band_energy_high REAL,
    spectral_entropy REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Health scores over time
CREATE TABLE IF NOT EXISTS health_scores (
    score_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id        TEXT NOT NULL REFERENCES assets(asset_id),
    inspection_id   INTEGER REFERENCES inspections(inspection_id),
    score_date      TEXT NOT NULL,
    health_score    REAL NOT NULL,
    health_status   TEXT NOT NULL,   -- Good / Warning / Degraded / Critical
    anomaly_score   REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Maintenance recommendations
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id        TEXT NOT NULL REFERENCES assets(asset_id),
    score_id        INTEGER REFERENCES health_scores(score_id),
    rec_date        TEXT NOT NULL,
    priority        TEXT,            -- High / Medium / Low
    action          TEXT NOT NULL,
    due_date        TEXT,
    status          TEXT DEFAULT 'Open',   -- Open / In Progress / Closed
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_hs_asset  ON health_scores(asset_id);
CREATE INDEX IF NOT EXISTS idx_hs_date   ON health_scores(score_date);
CREATE INDEX IF NOT EXISTS idx_rec_asset ON recommendations(asset_id);
CREATE INDEX IF NOT EXISTS idx_insp_asset ON inspections(asset_id);
"""


def create_schema() -> None:
    """Create all tables if they don't already exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
    print(f"[db]   Schema ready → {DB_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Insert helpers
# ─────────────────────────────────────────────────────────────────────────────

def insert_asset(
    asset_id: str,
    asset_name: str,
    asset_type: str = "",
    location: str = "",
    **kwargs,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO assets
               (asset_id, asset_name, asset_type, location,
                manufacturer, install_date, rated_power_kW, rated_speed_rpm, notes)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                asset_id, asset_name, asset_type, location,
                kwargs.get("manufacturer", ""),
                kwargs.get("install_date", ""),
                kwargs.get("rated_power_kW", None),
                kwargs.get("rated_speed_rpm", None),
                kwargs.get("notes", ""),
            ),
        )


def insert_inspection(
    asset_id: str,
    inspect_date: str | None = None,
    inspector: str = "",
    method: str = "vibration",
    notes: str = "",
) -> int:
    if inspect_date is None:
        inspect_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO inspections (asset_id, inspect_date, inspector, method, notes) VALUES (?,?,?,?,?)",
            (asset_id, inspect_date, inspector, method, notes),
        )
        return cur.lastrowid


def insert_sensor_reading(inspection_id: int, readings: dict) -> None:
    cols = [
        "air_temp_K", "process_temp_K", "rotational_speed_rpm",
        "torque_Nm", "tool_wear_min", "power_W", "temp_diff_K",
    ]
    vals = [readings.get(c) for c in cols]
    with get_connection() as conn:
        conn.execute(
            f"""INSERT INTO sensor_readings (inspection_id, {', '.join(cols)})
                VALUES (?, {', '.join(['?'] * len(cols))})""",
            [inspection_id] + vals,
        )


def insert_features(inspection_id: int, features: dict) -> None:
    cols = [
        "rms", "kurtosis", "crest_factor", "peak_to_peak",
        "skewness", "variance", "fft_peak_freq", "fft_peak_amp",
        "band_energy_low", "band_energy_mid", "band_energy_high", "spectral_entropy",
    ]
    vals = [features.get(c) for c in cols]
    with get_connection() as conn:
        conn.execute(
            f"""INSERT INTO feature_records (inspection_id, {', '.join(cols)})
                VALUES (?, {', '.join(['?'] * len(cols))})""",
            [inspection_id] + vals,
        )


def insert_health_score(
    asset_id: str,
    health_score: float,
    health_status: str,
    inspection_id: int | None = None,
    anomaly_score: float | None = None,
    score_date: str | None = None,
) -> int:
    if score_date is None:
        score_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO health_scores
               (asset_id, inspection_id, score_date, health_score, health_status, anomaly_score)
               VALUES (?,?,?,?,?,?)""",
            (asset_id, inspection_id, score_date, health_score, health_status, anomaly_score),
        )
        return cur.lastrowid


def insert_recommendation(
    asset_id: str,
    action: str,
    priority: str = "Medium",
    score_id: int | None = None,
    due_date: str | None = None,
) -> None:
    rec_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO recommendations
               (asset_id, score_id, rec_date, priority, action, due_date)
               VALUES (?,?,?,?,?,?)""",
            (asset_id, score_id, rec_date, priority, action, due_date),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Bulk-load from DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def bulk_load_health_scores(df: pd.DataFrame, asset_id_col: str = "uid") -> int:
    """
    Load a scored DataFrame into the health_scores table.
    Expects columns: health_score, health_status, [anomaly_score], [score_date].
    """
    create_schema()
    count = 0
    with get_connection() as conn:
        for _, row in df.iterrows():
            asset_id     = str(row.get(asset_id_col, "ASSET-001"))
            health_score = float(row.get("health_score", 0))
            health_status = str(row.get("health_status", "Unknown"))
            anomaly_score = float(row.get("anomaly_score", 0)) if "anomaly_score" in row else None
            score_date    = row.get("score_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Ensure asset exists
            conn.execute(
                "INSERT OR IGNORE INTO assets (asset_id, asset_name) VALUES (?,?)",
                (asset_id, f"Asset {asset_id}"),
            )
            conn.execute(
                """INSERT INTO health_scores
                   (asset_id, score_date, health_score, health_status, anomaly_score)
                   VALUES (?,?,?,?,?)""",
                (asset_id, score_date, health_score, health_status, anomaly_score),
            )
            count += 1
    print(f"[db]   Inserted {count:,} health score records")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────────

def query_asset_history(asset_id: str) -> pd.DataFrame:
    """Return full health score history for one asset."""
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM health_scores WHERE asset_id=? ORDER BY score_date",
            conn, params=(asset_id,),
        )


def query_critical_assets() -> pd.DataFrame:
    """Return assets currently in Critical or Degraded status."""
    with get_connection() as conn:
        return pd.read_sql_query(
            """SELECT asset_id, MAX(score_date) AS latest_date,
                      health_score, health_status
               FROM health_scores
               GROUP BY asset_id
               HAVING health_status IN ('Critical', 'Degraded')
               ORDER BY health_score ASC""",
            conn,
        )


def query_summary_stats() -> dict:
    """Return summary counts for the dashboard."""
    with get_connection() as conn:
        total_assets  = conn.execute("SELECT COUNT(DISTINCT asset_id) FROM assets").fetchone()[0]
        total_insp    = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        open_recs     = conn.execute("SELECT COUNT(*) FROM recommendations WHERE status='Open'").fetchone()[0]
        status_counts = dict(conn.execute(
            """SELECT health_status, COUNT(*) FROM
               (SELECT asset_id, health_status FROM health_scores
                GROUP BY asset_id HAVING MAX(score_date))
               GROUP BY health_status"""
        ).fetchall())
    return {
        "total_assets":  total_assets,
        "total_inspections": total_insp,
        "open_recommendations": open_recs,
        "status_counts": status_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main: seed demo data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Database Manager — Condition Monitoring Project")
    print("=" * 60)

    create_schema()

    # Seed demo assets
    demo_assets = [
        ("MTR-001", "Pump Motor #1",   "motor", "Plant A - Bay 1"),
        ("MTR-002", "Cooling Fan #2",  "fan",   "Plant A - Bay 2"),
        ("PMP-001", "Feed Pump #1",    "pump",  "Plant B - Utilities"),
        ("CMP-001", "Air Compressor",  "compressor", "Plant B - Utilities"),
    ]
    for a in demo_assets:
        insert_asset(*a, rated_speed_rpm=1500.0, rated_power_kW=7.5)

    # Seed demo inspection + health score
    insp_id = insert_inspection("MTR-001", method="vibration", inspector="Engineer A")
    insert_sensor_reading(insp_id, {"rotational_speed_rpm": 1498, "torque_Nm": 34.2, "air_temp_K": 298.1})
    insert_features(insp_id, {"rms": 0.82, "kurtosis": 4.1, "crest_factor": 3.9})
    score_id = insert_health_score("MTR-001", 72.5, "Warning", insp_id, anomaly_score=0.61)
    insert_recommendation("MTR-001", "Increase vibration check frequency. Monitor kurtosis trend.", "Medium", score_id)

    # Load scored CSV if it exists
    scored_path = ROOT / CFG["paths"]["processed_data"] / "ai4i_health_scored.csv"
    if scored_path.exists():
        df = pd.read_csv(scored_path)
        bulk_load_health_scores(df)

    stats = query_summary_stats()
    print(f"\n[db]   Summary statistics:")
    for k, v in stats.items():
        print(f"        {k}: {v}")

    print("\n[db]   Critical / Degraded assets:")
    critical = query_critical_assets()
    if critical.empty:
        print("        None currently")
    else:
        print(critical.to_string(index=False))
