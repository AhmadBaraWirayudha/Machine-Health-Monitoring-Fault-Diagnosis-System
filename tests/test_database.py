"""
Database Unit Tests
====================
Tests for the SQLite schema, CRUD helpers, and analytical queries.

Uses the `temp_db` fixture from conftest.py which creates a
disposable in-memory SQLite database per test.

Run:
    pytest tests/test_database.py -v
    pytest tests/test_database.py -v -m db
"""

import sys
import sqlite3
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.database.db_manager as dbm


# ── Schema tests ──────────────────────────────────────────────────────────────

@pytest.mark.db
class TestSchema:

    def test_all_tables_exist(self, temp_db):
        with sqlite3.connect(temp_db) as conn:
            tables = {row[0] for row in
                      conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        expected = {"assets", "inspections", "sensor_readings",
                    "feature_records", "health_scores", "recommendations"}
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_views_exist(self, temp_db):
        with sqlite3.connect(temp_db) as conn:
            views = {row[0] for row in
                     conn.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()}
        assert "v_latest_health" in views
        assert "v_open_recommendations" in views

    def test_foreign_keys_enabled(self, temp_db):
        with sqlite3.connect(temp_db) as conn:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1, "Foreign keys should be enabled"

    def test_indexes_exist(self, temp_db):
        with sqlite3.connect(temp_db) as conn:
            indexes = {row[1] for row in
                       conn.execute("SELECT * FROM sqlite_master WHERE type='index'").fetchall()}
        assert any("health_scores" in idx for idx in indexes)

    def test_assets_columns(self, temp_db):
        with sqlite3.connect(temp_db) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(assets)").fetchall()}
        expected = {"asset_id", "asset_name", "asset_type", "location", "created_at"}
        assert expected.issubset(cols)

    def test_health_scores_columns(self, temp_db):
        with sqlite3.connect(temp_db) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(health_scores)").fetchall()}
        expected = {"score_id", "asset_id", "score_date", "health_score", "health_status"}
        assert expected.issubset(cols)


# ── Asset CRUD ────────────────────────────────────────────────────────────────

@pytest.mark.db
class TestAssetCRUD:

    def test_insert_asset(self, temp_db):
        dbm.DB_PATH = temp_db
        dbm.insert_asset("MTR-TEST", "Test Motor", "motor", "Site X")
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT * FROM assets WHERE asset_id='MTR-TEST'"
            ).fetchone()
        assert row is not None

    def test_insert_asset_idempotent(self, temp_db):
        """Inserting the same asset_id twice should not raise."""
        dbm.DB_PATH = temp_db
        dbm.insert_asset("DUP-001", "Dup Asset")
        dbm.insert_asset("DUP-001", "Dup Asset Again")   # OR IGNORE
        with sqlite3.connect(temp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM assets WHERE asset_id='DUP-001'"
            ).fetchone()[0]
        assert count == 1, "Duplicate asset_id should not create extra rows"

    def test_asset_name_stored(self, temp_db):
        dbm.DB_PATH = temp_db
        dbm.insert_asset("NM-001", "My Pump", "pump", "Bay 3")
        with sqlite3.connect(temp_db) as conn:
            name = conn.execute(
                "SELECT asset_name FROM assets WHERE asset_id='NM-001'"
            ).fetchone()[0]
        assert name == "My Pump"

    def test_rated_fields_stored(self, temp_db):
        dbm.DB_PATH = temp_db
        dbm.insert_asset("RT-001", "Rated Motor", rated_power_kW=7.5, rated_speed_rpm=1500)
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT rated_power_kW, rated_speed_rpm FROM assets WHERE asset_id='RT-001'"
            ).fetchone()
        assert row[0] == 7.5
        assert row[1] == 1500.0


# ── Inspection CRUD ───────────────────────────────────────────────────────────

@pytest.mark.db
class TestInspectionCRUD:

    def test_insert_inspection_returns_id(self, temp_db):
        dbm.DB_PATH = temp_db
        insp_id = dbm.insert_inspection("TEST-001", method="vibration")
        assert isinstance(insp_id, int)
        assert insp_id > 0

    def test_inspection_linked_to_asset(self, temp_db):
        dbm.DB_PATH = temp_db
        insp_id = dbm.insert_inspection("TEST-001", method="thermal")
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT asset_id, method FROM inspections WHERE inspection_id=?",
                (insp_id,)
            ).fetchone()
        assert row[0] == "TEST-001"
        assert row[1] == "thermal"

    def test_inspection_date_auto(self, temp_db):
        dbm.DB_PATH = temp_db
        insp_id = dbm.insert_inspection("TEST-001")
        with sqlite3.connect(temp_db) as conn:
            date = conn.execute(
                "SELECT inspect_date FROM inspections WHERE inspection_id=?",
                (insp_id,)
            ).fetchone()[0]
        assert date is not None and len(date) > 0


# ── Sensor readings ───────────────────────────────────────────────────────────

@pytest.mark.db
class TestSensorReadings:

    def test_insert_sensor_reading(self, temp_db):
        dbm.DB_PATH = temp_db
        insp_id = dbm.insert_inspection("TEST-001")
        dbm.insert_sensor_reading(insp_id, {
            "rotational_speed_rpm": 1498.0,
            "torque_Nm": 34.2,
            "air_temp_K": 298.1,
        })
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT rotational_speed_rpm FROM sensor_readings WHERE inspection_id=?",
                (insp_id,)
            ).fetchone()
        assert row is not None
        assert abs(row[0] - 1498.0) < 0.1

    def test_multiple_readings_per_inspection(self, temp_db):
        dbm.DB_PATH = temp_db
        insp_id = dbm.insert_inspection("TEST-001")
        for speed in [1490, 1500, 1510]:
            dbm.insert_sensor_reading(insp_id, {"rotational_speed_rpm": speed})
        with sqlite3.connect(temp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM sensor_readings WHERE inspection_id=?",
                (insp_id,)
            ).fetchone()[0]
        assert count == 3


# ── Health scores ─────────────────────────────────────────────────────────────

@pytest.mark.db
class TestHealthScores:

    def test_insert_health_score_returns_id(self, temp_db):
        dbm.DB_PATH = temp_db
        score_id = dbm.insert_health_score("TEST-001", 72.5, "Warning")
        assert isinstance(score_id, int)
        assert score_id > 0

    def test_health_score_value_stored(self, temp_db):
        dbm.DB_PATH = temp_db
        sid = dbm.insert_health_score("TEST-001", 55.0, "Degraded", anomaly_score=0.4)
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT health_score, health_status, anomaly_score FROM health_scores WHERE score_id=?",
                (sid,)
            ).fetchone()
        assert abs(row[0] - 55.0) < 0.01
        assert row[1] == "Degraded"
        assert abs(row[2] - 0.4) < 0.01

    def test_query_asset_history(self, temp_db):
        dbm.DB_PATH = temp_db
        for score in [80, 75, 70]:
            dbm.insert_health_score("TEST-001", float(score), "Warning")
        history = dbm.query_asset_history("TEST-001")
        assert len(history) >= 3
        assert "health_score" in history.columns

    def test_query_critical_assets(self, temp_db):
        dbm.DB_PATH = temp_db
        dbm.insert_asset("CRIT-001", "Critical Motor")
        dbm.insert_health_score("CRIT-001", 15.0, "Critical")
        critical = dbm.query_critical_assets()
        assert len(critical) >= 1
        assert "CRIT-001" in critical["asset_id"].values

    def test_bulk_load(self, temp_db, scored_df):
        dbm.DB_PATH = temp_db
        count = dbm.bulk_load_health_scores(scored_df.head(20))
        assert count == 20
        with sqlite3.connect(temp_db) as conn:
            n = conn.execute("SELECT COUNT(*) FROM health_scores").fetchone()[0]
        assert n >= 20


# ── Recommendations ───────────────────────────────────────────────────────────

@pytest.mark.db
class TestRecommendations:

    def test_insert_recommendation(self, temp_db):
        dbm.DB_PATH = temp_db
        dbm.insert_recommendation("TEST-001", "Inspect bearings", priority="High")
        with sqlite3.connect(temp_db) as conn:
            row = conn.execute(
                "SELECT action, priority, status FROM recommendations WHERE asset_id='TEST-001'"
            ).fetchone()
        assert row is not None
        assert row[0] == "Inspect bearings"
        assert row[1] == "High"
        assert row[2] == "Open"

    def test_recommendation_default_status_open(self, temp_db):
        dbm.DB_PATH = temp_db
        dbm.insert_recommendation("TEST-001", "Check alignment")
        with sqlite3.connect(temp_db) as conn:
            status = conn.execute(
                "SELECT status FROM recommendations ORDER BY rec_id DESC LIMIT 1"
            ).fetchone()[0]
        assert status == "Open"


# ── Summary statistics ────────────────────────────────────────────────────────

@pytest.mark.db
class TestSummaryStats:

    def test_summary_stats_structure(self, temp_db):
        dbm.DB_PATH = temp_db
        stats = dbm.query_summary_stats()
        assert isinstance(stats, dict)
        assert "total_assets" in stats
        assert "total_inspections" in stats
        assert "open_recommendations" in stats

    def test_summary_counts_correct(self, temp_db):
        dbm.DB_PATH = temp_db
        stats = dbm.query_summary_stats()
        # temp_db has 1 asset (TEST-001) seeded in conftest
        assert stats["total_assets"] >= 1
        assert stats["open_recommendations"] >= 1


# ── View queries ──────────────────────────────────────────────────────────────

@pytest.mark.db
class TestViews:

    def test_v_latest_health_returns_rows(self, temp_db):
        dbm.DB_PATH = temp_db
        with sqlite3.connect(temp_db) as conn:
            rows = conn.execute("SELECT * FROM v_latest_health").fetchall()
        assert len(rows) >= 1

    def test_v_open_recommendations_returns_rows(self, temp_db):
        dbm.DB_PATH = temp_db
        with sqlite3.connect(temp_db) as conn:
            rows = conn.execute("SELECT * FROM v_open_recommendations").fetchall()
        assert len(rows) >= 1

    def test_v_latest_health_has_score(self, temp_db):
        dbm.DB_PATH = temp_db
        with sqlite3.connect(temp_db) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM v_latest_health").fetchall()
        for row in rows:
            assert "health_score" in row.keys()
