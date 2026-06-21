"""
Reliability Metrics
====================
Computes standard CBM / reliability engineering metrics from
inspection history and health score data.

Metrics:
  MTBF  — Mean Time Between Failures
  MTTF  — Mean Time To Failure (non-repairable)
  MTTR  — Mean Time To Repair
  Availability — A = MTBF / (MTBF + MTTR)
  OEE   — Overall Equipment Effectiveness (simplified)
  Failure Rate λ — λ = 1 / MTBF
  FMEA  — Failure Mode table from prediction history

Usage:
    from src.utils.metrics import ReliabilityMetrics
    rm = ReliabilityMetrics(df_scored)
    print(rm.summary())
"""

import yaml
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["database"]["path"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AssetReliability:
    """Reliability summary for a single asset."""
    asset_id:         str
    asset_name:       str = ""
    n_inspections:    int = 0
    n_failures:       int = 0
    total_op_time:    float = 0.0   # minutes
    total_repair_time: float = 0.0  # minutes (estimated)
    mtbf:             float = 0.0   # minutes
    mttf:             float = 0.0   # minutes
    mttr:             float = 0.0   # minutes
    availability:     float = 0.0   # 0–1
    failure_rate:     float = 0.0   # failures per hour
    current_health:   float = 0.0
    current_status:   str = "Unknown"

    @property
    def availability_pct(self) -> str:
        return f"{self.availability:.2%}"

    @property
    def mtbf_hours(self) -> float:
        return self.mtbf / 60.0

    @property
    def failure_rate_per_1000h(self) -> float:
        return self.failure_rate * 1000

    def as_dict(self) -> dict:
        return {
            "asset_id":       self.asset_id,
            "asset_name":     self.asset_name,
            "n_inspections":  self.n_inspections,
            "n_failures":     self.n_failures,
            "total_op_h":     round(self.total_op_time / 60, 1),
            "mtbf_h":         round(self.mtbf_hours, 1),
            "mttf_h":         round(self.mttf / 60, 1),
            "mttr_h":         round(self.mttr / 60, 1),
            "availability":   round(self.availability, 4),
            "failure_rate_per_1000h": round(self.failure_rate_per_1000h, 4),
            "current_health": round(self.current_health, 1),
            "current_status": self.current_status,
        }


@dataclass
class FleetReliability:
    """Aggregated reliability summary for the entire fleet."""
    n_assets:          int = 0
    avg_availability:  float = 0.0
    avg_mtbf_h:        float = 0.0
    total_failures:    int = 0
    fleet_failure_rate: float = 0.0
    critical_assets:   list = field(default_factory=list)
    avg_health_score:  float = 0.0
    oee_estimate:      float = 0.0   # simplified OEE


# ── Core calculations ─────────────────────────────────────────────────────────

def mtbf(operational_time_min: float, n_failures: int) -> float:
    """MTBF = total operational time / number of failures."""
    if n_failures == 0:
        return operational_time_min   # no failures = full op time as lower bound
    return operational_time_min / n_failures


def mttf(time_to_failures: list[float]) -> float:
    """MTTF = mean of time-to-failure observations."""
    if not time_to_failures:
        return 0.0
    return float(np.mean(time_to_failures))


def mttr(repair_times_min: list[float]) -> float:
    """MTTR = mean of repair/restoration times."""
    if not repair_times_min:
        return 0.0
    return float(np.mean(repair_times_min))


def availability(mtbf_val: float, mttr_val: float) -> float:
    """Inherent availability: A = MTBF / (MTBF + MTTR)."""
    denom = mtbf_val + mttr_val
    return mtbf_val / denom if denom > 0 else 1.0


def failure_rate(mtbf_min: float) -> float:
    """Failure rate λ = 1/MTBF [failures/hour]."""
    if mtbf_min <= 0:
        return 0.0
    return 1.0 / (mtbf_min / 60.0)


def oee(availability_val: float,
        performance: float = 0.95,
        quality: float = 0.99) -> float:
    """
    Simplified OEE = Availability × Performance × Quality.

    For CBM purposes, performance and quality are assumed from
    industry benchmarks unless measured values are available.
    """
    return availability_val * performance * quality


# ── Health-score-based failure detection ──────────────────────────────────────

def detect_failures_from_health(
    df: pd.DataFrame,
    health_col: str = "health_score",
    threshold: float = 40.0,
    min_duration: int = 3,
) -> pd.DataFrame:
    """
    Identify failure events from health score crossing the critical threshold.

    A 'failure' is defined as `min_duration` consecutive observations
    below `threshold`.

    Returns DataFrame of failure events with columns:
        start_idx, end_idx, duration_obs, min_health_score
    """
    if health_col not in df.columns:
        return pd.DataFrame()

    below    = df[health_col] < threshold
    events   = []
    in_event = False
    start    = 0

    for i, is_below in enumerate(below):
        if is_below and not in_event:
            in_event = True
            start = i
        elif not is_below and in_event:
            duration = i - start
            if duration >= min_duration:
                events.append({
                    "start_idx":       start,
                    "end_idx":         i - 1,
                    "duration_obs":    duration,
                    "min_health_score": df[health_col].iloc[start:i].min(),
                })
            in_event = False

    if in_event:
        duration = len(df) - start
        if duration >= min_duration:
            events.append({
                "start_idx":       start,
                "end_idx":         len(df) - 1,
                "duration_obs":    duration,
                "min_health_score": df[health_col].iloc[start:].min(),
            })

    return pd.DataFrame(events)


# ── Main calculator ───────────────────────────────────────────────────────────

class ReliabilityMetrics:
    """
    Computes reliability metrics from a health-scored DataFrame
    or directly from the SQLite database.

    Parameters
    ----------
    df : pd.DataFrame with columns health_score, health_status,
         [uid / asset_id], [tool_wear_min]
    assumed_op_time_per_obs : float
        Assumed operating time (minutes) represented by each observation.
        Default 60 min (i.e. one inspection per hour).
    assumed_mttr : float
        Assumed mean time to repair per failure event (minutes).
        Default 240 min (4 hours).
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        assumed_op_time_per_obs: float = 60.0,
        assumed_mttr: float = 240.0,
    ):
        self.df = df
        self.assumed_op_per_obs = assumed_op_time_per_obs
        self.assumed_mttr = assumed_mttr
        self._asset_metrics: dict[str, AssetReliability] = {}

    # ── From DataFrame ────────────────────────────────────────────────────────

    def compute_from_df(self) -> dict[str, AssetReliability]:
        """Compute per-asset reliability from the scored DataFrame."""
        if self.df is None:
            raise ValueError("No DataFrame provided.")

        id_col = "uid" if "uid" in self.df.columns else "asset_id"
        if id_col not in self.df.columns:
            self.df[id_col] = "ASSET-001"

        results = {}
        for asset_id, grp in self.df.groupby(id_col):
            grp = grp.reset_index(drop=True)
            n_obs   = len(grp)
            op_time = n_obs * self.assumed_op_per_obs

            # Detect failures
            failure_events = detect_failures_from_health(grp)
            n_fail = len(failure_events)

            # TTF from tool_wear_min if available
            if "tool_wear_min" in grp.columns:
                ttf_vals = grp.loc[
                    grp["health_score"] < 40, "tool_wear_min"
                ].dropna().tolist() if "health_score" in grp.columns else []
            else:
                ttf_vals = []

            mtbf_val = mtbf(op_time, n_fail)
            mttf_val = mttf(ttf_vals) if ttf_vals else mtbf_val
            mttr_val = self.assumed_mttr
            avail    = availability(mtbf_val, mttr_val)
            fr       = failure_rate(mtbf_val)

            current_health = grp["health_score"].iloc[-1] if "health_score" in grp.columns else 0
            current_status = grp["health_status"].iloc[-1] if "health_status" in grp.columns else "Unknown"

            results[str(asset_id)] = AssetReliability(
                asset_id=str(asset_id),
                n_inspections=n_obs,
                n_failures=n_fail,
                total_op_time=op_time,
                total_repair_time=n_fail * mttr_val,
                mtbf=mtbf_val,
                mttf=mttf_val,
                mttr=mttr_val,
                availability=avail,
                failure_rate=fr,
                current_health=current_health,
                current_status=str(current_status),
            )

        self._asset_metrics = results
        return results

    # ── From database ─────────────────────────────────────────────────────────

    def compute_from_db(self) -> dict[str, AssetReliability]:
        """Compute reliability metrics from the SQLite database."""
        if not DB_PATH.exists():
            raise FileNotFoundError(f"Database not found: {DB_PATH}")

        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                """SELECT h.asset_id, a.asset_name,
                          h.score_date, h.health_score, h.health_status
                   FROM health_scores h
                   JOIN assets a ON h.asset_id = a.asset_id
                   ORDER BY h.asset_id, h.score_date""",
                conn,
            )
            insp = pd.read_sql_query(
                "SELECT asset_id, COUNT(*) AS n_insp FROM inspections GROUP BY asset_id",
                conn,
            )

        insp_map = dict(zip(insp["asset_id"], insp["n_insp"]))
        results  = {}

        for asset_id, grp in df.groupby("asset_id"):
            grp = grp.reset_index(drop=True)
            n_obs   = len(grp)
            op_time = n_obs * self.assumed_op_per_obs

            failure_events = detect_failures_from_health(grp)
            n_fail = len(failure_events)

            mtbf_val = mtbf(op_time, n_fail)
            avail    = availability(mtbf_val, self.assumed_mttr)
            fr       = failure_rate(mtbf_val)

            results[str(asset_id)] = AssetReliability(
                asset_id      = str(asset_id),
                asset_name    = grp["asset_name"].iloc[0] if "asset_name" in grp.columns else "",
                n_inspections = insp_map.get(str(asset_id), n_obs),
                n_failures    = n_fail,
                total_op_time = op_time,
                mtbf          = mtbf_val,
                mttf          = mtbf_val,
                mttr          = self.assumed_mttr,
                availability  = avail,
                failure_rate  = fr,
                current_health = grp["health_score"].iloc[-1],
                current_status = grp["health_status"].iloc[-1],
            )

        self._asset_metrics = results
        return results

    # ── Fleet summary ─────────────────────────────────────────────────────────

    def fleet_summary(self) -> FleetReliability:
        """Aggregate per-asset metrics into a fleet summary."""
        if not self._asset_metrics:
            try:
                self.compute_from_db()
            except Exception:
                if self.df is not None:
                    self.compute_from_df()
                else:
                    return FleetReliability()

        metrics = list(self._asset_metrics.values())
        if not metrics:
            return FleetReliability()

        availabilities  = [m.availability for m in metrics]
        mtbf_hours      = [m.mtbf_hours   for m in metrics]
        total_failures  = sum(m.n_failures for m in metrics)
        health_scores   = [m.current_health for m in metrics]
        critical        = [m.asset_id for m in metrics if m.current_status == "Critical"]

        avg_avail = float(np.mean(availabilities))
        avg_oee   = oee(avg_avail)

        total_op_h = sum(m.total_op_time / 60 for m in metrics)
        fleet_fr   = total_failures / max(total_op_h, 1)

        return FleetReliability(
            n_assets          = len(metrics),
            avg_availability  = avg_avail,
            avg_mtbf_h        = float(np.mean(mtbf_hours)),
            total_failures    = total_failures,
            fleet_failure_rate = fleet_fr,
            critical_assets   = critical,
            avg_health_score  = float(np.mean(health_scores)),
            oee_estimate      = avg_oee,
        )

    def summary(self) -> pd.DataFrame:
        """Return per-asset reliability as a sorted DataFrame."""
        if not self._asset_metrics:
            try:
                self.compute_from_db()
            except Exception:
                if self.df is not None:
                    self.compute_from_df()

        rows = [m.as_dict() for m in self._asset_metrics.values()]
        if not rows:
            return pd.DataFrame()
        return (pd.DataFrame(rows)
                .sort_values("availability", ascending=True)
                .reset_index(drop=True))

    def fmea_table(self) -> pd.DataFrame:
        """
        Generate a simple FMEA (Failure Mode and Effects Analysis) table
        from the prediction history in the scored DataFrame.
        """
        if self.df is None:
            return pd.DataFrame()

        fault_col = "fault_type" if "fault_type" in self.df.columns else None
        pred_col  = "predicted_fault" if "predicted_fault" in self.df.columns else None
        col = fault_col or pred_col
        if col is None:
            return pd.DataFrame()

        failure_modes = {
            "TWF": ("Tool Wear Failure",    "H", "Gradual tool degradation past limit",
                    "Tool replacement, wear monitoring"),
            "HDF": ("Heat Dissipation Failure", "H", "Insufficient cooling, process temp excess",
                    "Cooling system check, temp monitoring"),
            "PWF": ("Power Failure",        "H", "Power outside operating range",
                    "Electrical check, power quality monitoring"),
            "OSF": ("Overstrain Failure",   "M", "Excessive torque / force on components",
                    "Load monitoring, torque limits"),
            "RNF": ("Random Failure",       "L", "Unclassified random event",
                    "General PM, root cause analysis"),
        }

        counts = self.df[col].value_counts()
        rows   = []
        for mode, (name, severity, cause, mitigation) in failure_modes.items():
            count = int(counts.get(mode, 0))
            if count > 0:
                rows.append({
                    "Failure Mode": mode,
                    "Description":  name,
                    "Severity":     severity,
                    "Occurrences":  count,
                    "Probable Cause": cause,
                    "Recommended Action": mitigation,
                })

        return pd.DataFrame(rows).sort_values("Occurrences", ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Reliability Metrics")
    print("=" * 60)

    # Try from scored CSV
    scored_path = ROOT / CFG["paths"]["processed_data"] / "ai4i_health_scored.csv"
    if scored_path.exists():
        df = pd.read_csv(scored_path)
        rm = ReliabilityMetrics(df, assumed_op_time_per_obs=60, assumed_mttr=240)
        rm.compute_from_df()

        print("\nPer-Asset Reliability Summary:")
        summary_df = rm.summary()
        if not summary_df.empty:
            print(summary_df.to_string(index=False))

        fleet = rm.fleet_summary()
        print(f"\nFleet Summary:")
        print(f"  Assets              : {fleet.n_assets}")
        print(f"  Avg Availability    : {fleet.avg_availability:.2%}")
        print(f"  Avg MTBF            : {fleet.avg_mtbf_h:.1f} hours")
        print(f"  Total Failures      : {fleet.total_failures}")
        print(f"  Fleet Failure Rate  : {fleet.fleet_failure_rate:.4f} failures/hour")
        print(f"  Avg Health Score    : {fleet.avg_health_score:.1f}")
        print(f"  OEE Estimate        : {fleet.oee_estimate:.2%}")
        if fleet.critical_assets:
            print(f"  Critical Assets     : {fleet.critical_assets}")

        print("\nFMEA Table:")
        fmea = rm.fmea_table()
        if not fmea.empty:
            print(fmea.to_string(index=False))
        else:
            print("  No failure modes detected in current dataset.")

        # Save
        out = ROOT / CFG["paths"]["processed_data"] / "reliability_metrics.csv"
        summary_df.to_csv(out, index=False)
        print(f"\n[OK]   Metrics saved → {out}")
    else:
        print("[WARN] No scored dataset found — run python main.py first.")
