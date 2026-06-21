"""
FastAPI REST API
================
Serves health score predictions and asset data over HTTP.

Endpoints:
  GET  /                        — API info
  GET  /health                  — service health check
  GET  /assets                  — list all assets
  GET  /assets/{asset_id}       — single asset detail
  GET  /assets/{asset_id}/history — health score history
  POST /predict                 — predict fault + health score from sensor readings
  POST /predict/batch           — batch prediction from CSV-style data
  GET  /fleet/summary           — fleet KPI summary
  GET  /recommendations         — open maintenance recommendations

Run:
    uvicorn src.api.main:app --reload --port 8000

Then open:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sqlite3
import numpy as np
import pandas as pd
import joblib
import yaml
from datetime import datetime
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("Install FastAPI: pip install fastapi uvicorn")

# ── Config ────────────────────────────────────────────────────────────────────
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH    = ROOT / CFG["database"]["path"]
MODELS_DIR = ROOT / CFG["paths"]["models"]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CBM Health Monitoring API",
    description=(
        "REST API for the Machine Health Monitoring & Fault Diagnosis System. "
        "Provides real-time fault prediction and asset health data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """A single set of sensor readings for one asset at one point in time."""
    air_temp_K:            float = Field(..., ge=280, le=320, example=298.1,
                                          description="Air temperature (Kelvin)")
    process_temp_K:        float = Field(..., ge=290, le=330, example=309.5,
                                          description="Process temperature (Kelvin)")
    rotational_speed_rpm:  float = Field(..., ge=500, le=3000, example=1500,
                                          description="Rotational speed (rpm)")
    torque_Nm:             float = Field(..., ge=0, le=100, example=40.0,
                                          description="Torque (Newton-metres)")
    tool_wear_min:         float = Field(..., ge=0, le=300, example=120,
                                          description="Cumulative tool wear (minutes)")

    class Config:
        json_schema_extra = {
            "example": {
                "air_temp_K": 298.1,
                "process_temp_K": 309.5,
                "rotational_speed_rpm": 1498,
                "torque_Nm": 34.2,
                "tool_wear_min": 150,
            }
        }


class PredictionResponse(BaseModel):
    fault_type:      str
    confidence:      float
    health_score:    float
    health_status:   str
    recommendation:  str
    power_W:         float
    temp_diff_K:     float
    timestamp:       str


class AssetSummary(BaseModel):
    asset_id:      str
    asset_name:    str
    asset_type:    Optional[str]
    location:      Optional[str]
    health_score:  Optional[float]
    health_status: Optional[str]
    fault_type:    Optional[str]


class FleetSummary(BaseModel):
    total_assets:        int
    avg_health_score:    float
    good_count:          int
    warning_count:       int
    degraded_count:      int
    critical_count:      int
    fleet_availability:  float
    open_recommendations: int


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Database not found. Run `python main.py` to initialise the pipeline.",
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(zip(row.keys(), row))


# ── Model loader ──────────────────────────────────────────────────────────────

_clf_cache = None
_le_cache  = None

def load_models():
    global _clf_cache, _le_cache
    clf_path = MODELS_DIR / "fault_classifier.joblib"
    le_path  = MODELS_DIR / "label_encoder.joblib"
    if clf_path.exists() and _clf_cache is None:
        _clf_cache = joblib.load(clf_path)
    if le_path.exists() and _le_cache is None:
        _le_cache = joblib.load(le_path)
    return _clf_cache, _le_cache


def classify_health(score: float) -> str:
    if score >= 80: return "Good"
    if score >= 60: return "Warning"
    if score >= 40: return "Degraded"
    return "Critical"


def recommend(status: str, fault: str) -> str:
    if status == "Good":     return "Continue routine monitoring per PM schedule."
    if status == "Warning":  return f"Elevated indicators ({fault}). Increase monitoring frequency."
    if status == "Degraded": return f"Degradation detected ({fault}). Schedule maintenance within 2 weeks."
    return f"CRITICAL: {fault} fault detected. Immediate inspection required."


def predict_from_reading(reading: SensorReading) -> dict:
    clf, le = load_models()
    power     = reading.torque_Nm * (reading.rotational_speed_rpm * 2 * np.pi / 60)
    temp_diff = reading.process_temp_K - reading.air_temp_K

    row = pd.DataFrame([{
        "air_temp_K":           reading.air_temp_K,
        "process_temp_K":       reading.process_temp_K,
        "rotational_speed_rpm": reading.rotational_speed_rpm,
        "torque_Nm":            reading.torque_Nm,
        "tool_wear_min":        reading.tool_wear_min,
        "power_W":              power,
        "temp_diff_K":          temp_diff,
    }])

    if clf is not None and le is not None:
        feat_cols   = [c for c in clf.feature_names_in_ if c in row.columns]
        X           = row[feat_cols].fillna(0)
        pred_idx    = clf.predict(X)[0]
        pred_proba  = clf.predict_proba(X)[0]
        fault_type  = le.inverse_transform([pred_idx])[0]
        confidence  = float(pred_proba.max())
    else:
        fault_type = "Normal"
        confidence = 0.75

    # Simple health score
    rms_proxy   = reading.torque_Nm / 80.0
    kurt_proxy  = min(reading.tool_wear_min / 250.0, 1.0) * 6
    cf_proxy    = max(reading.rotational_speed_rpm - 1400, 0) / 600 * 5
    health_score = float(np.clip(
        (1 - rms_proxy) * 0.3 + (1 - kurt_proxy / 6) * 0.4 + (1 - cf_proxy / 5) * 0.3, 0, 1
    ) * 100)
    status = classify_health(health_score)

    return {
        "fault_type":     fault_type,
        "confidence":     round(confidence, 4),
        "health_score":   round(health_score, 2),
        "health_status":  status,
        "recommendation": recommend(status, fault_type),
        "power_W":        round(power, 2),
        "temp_diff_K":    round(temp_diff, 2),
        "timestamp":      datetime.utcnow().isoformat() + "Z",
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    """API information and available endpoints."""
    return {
        "name":    "CBM Health Monitoring API",
        "version": "1.0.0",
        "docs":    "/docs",
        "redoc":   "/redoc",
        "endpoints": {
            "GET  /assets":                "List all assets with current health",
            "GET  /assets/{id}":           "Single asset detail",
            "GET  /assets/{id}/history":   "Health score history",
            "POST /predict":               "Predict fault + health from sensor reading",
            "POST /predict/batch":         "Batch prediction",
            "GET  /fleet/summary":         "Fleet KPI summary",
            "GET  /recommendations":       "Open maintenance recommendations",
        },
    }


@app.get("/health", tags=["Info"])
def health_check():
    """Service liveness check."""
    db_ok = DB_PATH.exists()
    clf_ok = (MODELS_DIR / "fault_classifier.joblib").exists()
    return {
        "status":   "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "not found",
        "models":   "loaded" if clf_ok else "not found (demo mode)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/assets", response_model=list[AssetSummary], tags=["Assets"])
def list_assets():
    """Return all assets with their current health score."""
    with get_conn() as conn:
        try:
            rows = conn.execute("SELECT * FROM v_latest_health").fetchall()
        except Exception:
            rows = conn.execute("SELECT asset_id, asset_name, asset_type, location FROM assets").fetchall()
    return [dict(r) for r in rows]


@app.get("/assets/{asset_id}", tags=["Assets"])
def get_asset(asset_id: str):
    """Return detail for a single asset."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found")

        hs = conn.execute(
            """SELECT health_score, health_status, fault_type, score_date
               FROM health_scores WHERE asset_id = ?
               ORDER BY score_date DESC LIMIT 1""",
            (asset_id,),
        ).fetchone()

        insp_count = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE asset_id = ?", (asset_id,)
        ).fetchone()[0]

        open_recs = conn.execute(
            "SELECT COUNT(*) FROM recommendations WHERE asset_id = ? AND status = 'Open'",
            (asset_id,),
        ).fetchone()[0]

    return {
        **dict(row),
        "current_health": dict(hs) if hs else None,
        "inspection_count": insp_count,
        "open_recommendations": open_recs,
    }


@app.get("/assets/{asset_id}/history", tags=["Assets"])
def get_asset_history(
    asset_id: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Max records to return"),
):
    """Return health score history for an asset."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT score_date, health_score, health_status, fault_type, anomaly_score
               FROM health_scores WHERE asset_id = ?
               ORDER BY score_date DESC LIMIT ?""",
            (asset_id, limit),
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No history for asset '{asset_id}'")
    return {"asset_id": asset_id, "records": [dict(r) for r in rows]}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(reading: SensorReading):
    """
    Predict fault type and compute health score from a single sensor reading.

    Returns fault classification, confidence, health score (0–100),
    status label, and maintenance recommendation.
    """
    return predict_from_reading(reading)


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(readings: list[SensorReading]):
    """Predict fault type and health score for a list of sensor readings (max 500)."""
    if len(readings) > 500:
        raise HTTPException(status_code=400, detail="Max 500 readings per batch request")
    return [predict_from_reading(r) for r in readings]


@app.get("/fleet/summary", response_model=FleetSummary, tags=["Fleet"])
def fleet_summary():
    """Return KPI summary for the entire monitored fleet."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT asset_id, health_score, health_status
               FROM health_scores h1
               WHERE score_date = (
                   SELECT MAX(score_date) FROM health_scores WHERE asset_id = h1.asset_id
               )"""
        ).fetchall()
        open_recs = conn.execute(
            "SELECT COUNT(*) FROM recommendations WHERE status = 'Open'"
        ).fetchone()[0]

    data = [dict(r) for r in rows]
    scores = [d["health_score"] for d in data if d["health_score"] is not None]

    status_counts = {}
    for d in data:
        s = d.get("health_status", "Unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    avail = sum(1 for d in data if d.get("health_status") in ("Good", "Warning")) / max(len(data), 1)

    return {
        "total_assets":         len(data),
        "avg_health_score":     round(float(np.mean(scores)), 2) if scores else 0.0,
        "good_count":           status_counts.get("Good", 0),
        "warning_count":        status_counts.get("Warning", 0),
        "degraded_count":       status_counts.get("Degraded", 0),
        "critical_count":       status_counts.get("Critical", 0),
        "fleet_availability":   round(avail * 100, 2),
        "open_recommendations": open_recs,
    }


@app.get("/recommendations", tags=["Maintenance"])
def get_recommendations(
    priority: Optional[str] = Query(default=None, description="Filter by priority: High, Medium, Low"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Return open maintenance recommendations, optionally filtered by priority."""
    with get_conn() as conn:
        sql = "SELECT * FROM v_open_recommendations"
        params: tuple = ()
        if priority:
            sql += " WHERE priority = ?"
            params = (priority,)
        sql += f" LIMIT {limit}"
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE status='Open' LIMIT ?", (limit,)
            ).fetchall()
    return {"open_recommendations": len(rows), "records": [dict(r) for r in rows]}
