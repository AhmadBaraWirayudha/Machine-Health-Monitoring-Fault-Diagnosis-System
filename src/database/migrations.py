"""
Database Migrations
====================
Lightweight schema migration system for the CBM SQLite database.
Tracks applied migrations in a `schema_migrations` table.

Migrations run in order; each is applied exactly once.
Safe to run repeatedly — already-applied migrations are skipped.

Usage:
    python src/database/migrations.py              # apply all pending
    python src/database/migrations.py --status     # show migration status
    python src/database/migrations.py --rollback 3 # rollback to migration 3

Adding a new migration:
    1. Add a function `migration_NNN(conn)` to the MIGRATIONS list below
    2. Run: python src/database/migrations.py
"""

import yaml
import sqlite3
import logging
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["database"]["path"]
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Migration functions
# Each function receives an open sqlite3.Connection.
# Name format: migration_NNN  (NNN = zero-padded number)
# ─────────────────────────────────────────────────────────────────────────────

def migration_001(conn: sqlite3.Connection) -> None:
    """Initial schema — creates all core tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS assets (
            asset_id        TEXT PRIMARY KEY,
            asset_name      TEXT NOT NULL,
            asset_type      TEXT,
            location        TEXT,
            manufacturer    TEXT,
            install_date    TEXT,
            rated_power_kW  REAL,
            rated_speed_rpm REAL,
            notes           TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS inspections (
            inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id      TEXT NOT NULL REFERENCES assets(asset_id),
            inspect_date  TEXT NOT NULL,
            inspector     TEXT,
            method        TEXT,
            notes         TEXT,
            created_at    TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sensor_readings (
            reading_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id        INTEGER NOT NULL REFERENCES inspections(inspection_id),
            air_temp_K           REAL,
            process_temp_K       REAL,
            rotational_speed_rpm REAL,
            torque_Nm            REAL,
            tool_wear_min        REAL,
            power_W              REAL,
            temp_diff_K          REAL,
            recorded_at          TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS feature_records (
            feature_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id    INTEGER NOT NULL REFERENCES inspections(inspection_id),
            rms              REAL,
            kurtosis         REAL,
            crest_factor     REAL,
            peak_to_peak     REAL,
            skewness         REAL,
            variance_val     REAL,
            fft_peak_freq    REAL,
            fft_peak_amp     REAL,
            band_energy_low  REAL,
            band_energy_mid  REAL,
            band_energy_high REAL,
            spectral_entropy REAL,
            created_at       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS health_scores (
            score_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id         TEXT NOT NULL REFERENCES assets(asset_id),
            inspection_id    INTEGER REFERENCES inspections(inspection_id),
            score_date       TEXT NOT NULL,
            health_score     REAL NOT NULL,
            health_status    TEXT NOT NULL,
            anomaly_score    REAL,
            fault_type       TEXT,
            fault_confidence REAL,
            created_at       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            rec_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id    TEXT NOT NULL REFERENCES assets(asset_id),
            score_id    INTEGER REFERENCES health_scores(score_id),
            rec_date    TEXT NOT NULL,
            priority    TEXT,
            action      TEXT NOT NULL,
            due_date    TEXT,
            status      TEXT DEFAULT 'Open',
            closed_date TEXT,
            closed_by   TEXT,
            notes       TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_hs_asset_date  ON health_scores(asset_id, score_date);
        CREATE INDEX IF NOT EXISTS idx_insp_asset     ON inspections(asset_id, inspect_date);
        CREATE INDEX IF NOT EXISTS idx_rec_status     ON recommendations(status, priority);
    """)
    log.info("  Created all core tables and indexes")


def migration_002(conn: sqlite3.Connection) -> None:
    """Add views v_latest_health and v_open_recommendations."""
    conn.executescript("""
        CREATE VIEW IF NOT EXISTS v_latest_health AS
        SELECT a.asset_id, a.asset_name, a.asset_type, a.location,
               h.score_date, h.health_score, h.health_status, h.fault_type
        FROM assets a
        JOIN health_scores h ON a.asset_id = h.asset_id
        WHERE h.score_date = (
            SELECT MAX(score_date) FROM health_scores WHERE asset_id = a.asset_id
        );

        CREATE VIEW IF NOT EXISTS v_open_recommendations AS
        SELECT r.rec_id, r.asset_id, a.asset_name, a.location,
               r.priority, r.action, r.due_date, r.rec_date
        FROM recommendations r
        JOIN assets a ON r.asset_id = a.asset_id
        WHERE r.status = 'Open'
        ORDER BY
            CASE r.priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
            r.due_date;
    """)
    log.info("  Created views: v_latest_health, v_open_recommendations")


def migration_003(conn: sqlite3.Connection) -> None:
    """Add bearing_features table for vibration-specific diagnostic data."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS bearing_features (
            bf_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id   INTEGER NOT NULL REFERENCES inspections(inspection_id),
            bearing_model   TEXT,
            shaft_rpm       REAL,
            bpfo_energy     REAL,
            bpfi_energy     REAL,
            bsf_energy      REAL,
            ftf_energy      REAL,
            bpfo_amplitude  REAL,
            bpfi_amplitude  REAL,
            n_fault_tones   INTEGER,
            diagnosis       TEXT,
            confidence      REAL,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_bf_insp ON bearing_features(inspection_id);
    """)
    log.info("  Created bearing_features table")


def migration_004(conn: sqlite3.Connection) -> None:
    """Add reliability_metrics table for MTBF/MTTF/OEE tracking."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reliability_metrics (
            metric_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id          TEXT NOT NULL REFERENCES assets(asset_id),
            calc_date         TEXT NOT NULL,
            n_inspections     INTEGER,
            n_failures        INTEGER,
            total_op_hours    REAL,
            mtbf_hours        REAL,
            mttf_hours        REAL,
            mttr_hours        REAL,
            availability      REAL,
            failure_rate_per_1000h REAL,
            oee_estimate      REAL,
            created_at        TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_rm_asset ON reliability_metrics(asset_id, calc_date);
    """)
    log.info("  Created reliability_metrics table")


def migration_005(conn: sqlite3.Connection) -> None:
    """Add api_keys table for API key management (optional auth)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash     TEXT UNIQUE NOT NULL,
            key_prefix   TEXT NOT NULL,
            label        TEXT,
            created_by   TEXT,
            created_at   TEXT DEFAULT (datetime('now')),
            expires_at   TEXT,
            last_used_at TEXT,
            is_active    INTEGER DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_ak_hash   ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS idx_ak_active ON api_keys(is_active);
    """)
    log.info("  Created api_keys table")


def migration_006(conn: sqlite3.Connection) -> None:
    """Add condition_found column to inspections (was missing from initial schema)."""
    try:
        conn.execute("ALTER TABLE inspections ADD COLUMN condition_found TEXT")
        log.info("  Added condition_found column to inspections")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            log.debug("  condition_found column already exists — skipping")
        else:
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Migration registry (ORDER MATTERS — do not reorder existing entries)
# ─────────────────────────────────────────────────────────────────────────────

MIGRATIONS: list[tuple[str, callable]] = [
    ("001_initial_schema",          migration_001),
    ("002_views",                   migration_002),
    ("003_bearing_features",        migration_003),
    ("004_reliability_metrics",     migration_004),
    ("005_api_keys",                migration_005),
    ("006_inspections_condition",   migration_006),
]


# ─────────────────────────────────────────────────────────────────────────────
# Migration runner
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at   TEXT NOT NULL,
            duration_ms  INTEGER
        )
    """)
    conn.commit()


def _applied_migrations(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT migration_id FROM schema_migrations")}


def apply_migrations(db_path: Path | None = None) -> int:
    """
    Apply all pending migrations in order.
    Returns the number of migrations applied.
    """
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        _ensure_migrations_table(conn)
        applied = _applied_migrations(conn)

    n_applied = 0
    for migration_id, fn in MIGRATIONS:
        if migration_id in applied:
            log.debug(f"[skip] {migration_id}")
            continue

        log.info(f"[migrate] Applying: {migration_id}")
        t0 = __import__("time").perf_counter()
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                fn(conn)
                elapsed_ms = int((__import__("time").perf_counter() - t0) * 1000)
                conn.execute(
                    "INSERT INTO schema_migrations (migration_id, applied_at, duration_ms) "
                    "VALUES (?, ?, ?)",
                    (migration_id, datetime.now().isoformat(), elapsed_ms),
                )
                conn.commit()
            print(f"  ✓ {migration_id} ({elapsed_ms}ms)")
            n_applied += 1
        except Exception as e:
            log.error(f"  ✗ FAILED: {migration_id}: {e}")
            raise

    return n_applied


def migration_status(db_path: Path | None = None) -> None:
    """Print the status of all migrations."""
    db_path = db_path or DB_PATH
    if not db_path.exists():
        print("Database not found — no migrations applied yet.")
        return

    with sqlite3.connect(db_path) as conn:
        try:
            _ensure_migrations_table(conn)
            rows = {row[0]: row[1] for row in
                    conn.execute("SELECT migration_id, applied_at FROM schema_migrations")}
        except Exception:
            rows = {}

    print(f"\nMigration status for: {db_path}")
    print(f"{'Status':<6}  {'Migration ID':<35}  {'Applied At'}")
    print("─" * 70)
    for mid, _ in MIGRATIONS:
        if mid in rows:
            print(f"  ✓    {mid:<35}  {rows[mid][:19]}")
        else:
            print(f"  ✗    {mid:<35}  (pending)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="CBM Database Migration Tool")
    parser.add_argument("--status",    action="store_true", help="Show migration status")
    parser.add_argument("--db",        type=str, default=None, help="Path to database file")
    args = parser.parse_args()

    db = Path(args.db) if args.db else DB_PATH

    if args.status:
        migration_status(db)
    else:
        print(f"\nApplying migrations to: {db}")
        n = apply_migrations(db)
        if n == 0:
            print("  All migrations already applied — database is up to date.")
        else:
            print(f"\n✓ Applied {n} migration(s) successfully.")
        migration_status(db)
