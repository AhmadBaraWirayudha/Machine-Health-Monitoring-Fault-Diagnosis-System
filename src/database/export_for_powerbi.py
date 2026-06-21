"""
Power BI Export Script
======================
Exports all database tables + derived views to flat CSV/Excel files
so Power BI can connect without needing an ODBC driver.

Output folder: data/processed/powerbi_export/

Usage:
    python src/database/export_for_powerbi.py
"""

import yaml
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH   = ROOT / CFG["database"]["path"]
PROC_DIR  = ROOT / CFG["paths"]["processed_data"]
EXPORT_DIR = PROC_DIR / "powerbi_export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Tables and views to export ────────────────────────────────────────────────

TABLES = [
    "assets",
    "inspections",
    "sensor_readings",
    "feature_records",
    "health_scores",
    "recommendations",
]

VIEWS = [
    "v_latest_health",
    "v_open_recommendations",
]

# ── Custom queries (pre-aggregated for Power BI) ──────────────────────────────

CUSTOM_QUERIES = {
    "fleet_kpis": """
        SELECT
            COUNT(DISTINCT asset_id)                                        AS total_assets,
            ROUND(AVG(health_score), 1)                                     AS avg_health_score,
            SUM(CASE WHEN health_status = 'Good'      THEN 1 ELSE 0 END)   AS good_count,
            SUM(CASE WHEN health_status = 'Warning'   THEN 1 ELSE 0 END)   AS warning_count,
            SUM(CASE WHEN health_status = 'Degraded'  THEN 1 ELSE 0 END)   AS degraded_count,
            SUM(CASE WHEN health_status = 'Critical'  THEN 1 ELSE 0 END)   AS critical_count,
            ROUND(
                SUM(CASE WHEN health_status IN ('Good','Warning') THEN 1.0 ELSE 0 END)
                / COUNT(*) * 100, 1
            ) AS fleet_availability_pct
        FROM (
            SELECT asset_id, health_score, health_status
            FROM   health_scores h1
            WHERE  score_date = (
                SELECT MAX(score_date) FROM health_scores WHERE asset_id = h1.asset_id
            )
        )
    """,
    "health_trend_daily": """
        SELECT
            strftime('%Y-%m-%d', score_date) AS date,
            asset_id,
            ROUND(AVG(health_score), 2)       AS avg_health_score,
            MIN(health_score)                  AS min_health_score,
            MAX(health_score)                  AS max_health_score,
            COUNT(*)                           AS observations
        FROM health_scores
        GROUP BY date, asset_id
        ORDER BY date, asset_id
    """,
    "fault_summary": """
        SELECT
            fault_type,
            COUNT(*)                                        AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct,
            ROUND(AVG(health_score), 1)                     AS avg_health_score,
            ROUND(AVG(fault_confidence), 3)                 AS avg_confidence
        FROM health_scores
        WHERE fault_type IS NOT NULL AND fault_type != ''
        GROUP BY fault_type
        ORDER BY count DESC
    """,
    "inspections_monthly": """
        SELECT
            strftime('%Y-%m', inspect_date) AS month,
            method,
            COUNT(*)                         AS inspections,
            COUNT(DISTINCT asset_id)         AS unique_assets
        FROM inspections
        GROUP BY month, method
        ORDER BY month
    """,
    "recommendations_backlog": """
        SELECT
            r.rec_id,
            a.asset_id,
            a.asset_name,
            a.asset_type,
            a.location,
            r.priority,
            r.action,
            r.rec_date,
            r.due_date,
            r.status,
            h.health_score,
            h.health_status,
            CASE
                WHEN r.due_date < date('now') AND r.status = 'Open'
                THEN CAST(julianday('now') - julianday(r.due_date) AS INTEGER)
                ELSE 0
            END AS days_overdue
        FROM recommendations r
        JOIN assets       a ON r.asset_id = a.asset_id
        LEFT JOIN health_scores h ON r.score_id = h.score_id
        ORDER BY
            CASE r.priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
            r.due_date NULLS LAST
    """,
    "feature_stats_by_fault": """
        SELECT
            hs.fault_type,
            COUNT(*)                      AS n,
            ROUND(AVG(fr.rms), 4)         AS avg_rms,
            ROUND(AVG(fr.kurtosis), 4)    AS avg_kurtosis,
            ROUND(AVG(fr.crest_factor), 4)AS avg_crest_factor,
            ROUND(AVG(fr.fft_peak_freq),2)AS avg_fft_peak_freq,
            ROUND(AVG(fr.spectral_entropy),4) AS avg_spectral_entropy
        FROM health_scores  hs
        JOIN inspections    i  ON hs.inspection_id = i.inspection_id
        JOIN feature_records fr ON i.inspection_id  = fr.inspection_id
        WHERE hs.fault_type IS NOT NULL AND hs.fault_type != ''
        GROUP BY hs.fault_type
    """,
}


# ── Export logic ──────────────────────────────────────────────────────────────

def export_all() -> None:
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        print("        Run: python main.py --steps 1 2 3 4 6")
        return

    print("=" * 60)
    print("  Power BI Export")
    print(f"  Source DB : {DB_PATH}")
    print(f"  Output    : {EXPORT_DIR}")
    print("=" * 60)

    with sqlite3.connect(DB_PATH) as conn:

        # Export raw tables
        for table in TABLES:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                out = EXPORT_DIR / f"{table}.csv"
                df.to_csv(out, index=False)
                print(f"  [table] {table:<30} {len(df):>7,} rows  → {out.name}")
            except Exception as e:
                print(f"  [WARN]  {table}: {e}")

        # Export views
        for view in VIEWS:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {view}", conn)
                out = EXPORT_DIR / f"{view}.csv"
                df.to_csv(out, index=False)
                print(f"  [view]  {view:<30} {len(df):>7,} rows  → {out.name}")
            except Exception as e:
                print(f"  [WARN]  {view}: {e}")

        # Export custom queries
        for name, sql in CUSTOM_QUERIES.items():
            try:
                df = pd.read_sql_query(sql.strip(), conn)
                out = EXPORT_DIR / f"{name}.csv"
                df.to_csv(out, index=False)
                print(f"  [query] {name:<30} {len(df):>7,} rows  → {out.name}")
            except Exception as e:
                print(f"  [WARN]  {name}: {e}")

    # Write an Excel workbook with all sheets (one per table)
    xl_path = EXPORT_DIR / "cmdb_export.xlsx"
    try:
        with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:
            with sqlite3.connect(DB_PATH) as conn:
                for table in TABLES + list(CUSTOM_QUERIES.keys()):
                    sql = f"SELECT * FROM {table}" if table in TABLES \
                          else CUSTOM_QUERIES.get(table, "")
                    if not sql:
                        continue
                    try:
                        df = pd.read_sql_query(sql, conn)
                        sheet = table[:31]   # Excel sheet name max 31 chars
                        df.to_excel(writer, sheet_name=sheet, index=False)
                    except Exception:
                        pass
        print(f"\n  [xlsx]  All tables → {xl_path.name}")
    except ImportError:
        print("  [WARN]  openpyxl not installed — skipping Excel export")

    # Write a manifest
    manifest = {
        "exported_at": datetime.now().isoformat(),
        "source_db": str(DB_PATH),
        "files": [str(f.name) for f in sorted(EXPORT_DIR.glob("*.csv"))],
        "powerbi_connection_hint": (
            "In Power BI: Home → Get Data → Text/CSV → select each .csv file. "
            "Or use the .xlsx workbook: Home → Get Data → Excel Workbook."
        ),
    }
    import json
    (EXPORT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n[OK]   Export complete — {len(list(EXPORT_DIR.glob('*.csv')))} CSV files")
    print(f"       Open Power BI and load from: {EXPORT_DIR}")
    print("       See powerbi/powerbi_guide.md for dashboard setup.")


if __name__ == "__main__":
    export_all()
