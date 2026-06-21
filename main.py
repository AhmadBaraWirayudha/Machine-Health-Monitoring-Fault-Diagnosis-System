"""
Main Pipeline Runner
=====================
Orchestrates the full CBM pipeline end-to-end:

  Step 1 → Download dataset (AI4I 2020 from UCI)
  Step 2 → Clean and preprocess data
  Step 3 → Extract diagnostic features
  Step 4 → Score asset health + anomaly detection
  Step 5 → Classify fault types
  Step 6 → Push results to SQLite database
  Step 7 → Generate diagnostic plots
  Step 8 → Render HTML technical report

Usage:
    python main.py

To skip the download (use existing raw data):
    python main.py --skip-download

To run only up to a specific step:
    python main.py --steps 1 2 3
"""

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_step(n: int, name: str, fn, *args, **kwargs):
    """Run one pipeline step with timing and error handling."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  STEP {n} — {name}")
    print(sep)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n  ✓ Completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        print(f"\n  ✗ FAILED: {e}")
        raise


def main(skip_download: bool = False, steps: list[int] | None = None):
    all_steps = steps is None

    print("=" * 60)
    print("  Machine Health Monitoring & Fault Diagnosis Pipeline")
    print("=" * 60)
    t_start = time.time()

    # ── Step 1: Ingest ────────────────────────────────────────────
    if all_steps or 1 in steps:
        from src.ingestion.download_data import download_ai4i, load_ai4i
        if not skip_download:
            run_step(1, "Data Ingestion", download_ai4i)
        else:
            print("\n[SKIP] Step 1 — using existing raw data")

    # ── Step 2: Preprocess ────────────────────────────────────────
    df_clean = None
    if all_steps or 2 in steps:
        from src.preprocessing.clean_data import preprocess_ai4i
        df_clean = run_step(2, "Data Preprocessing", preprocess_ai4i, save=True)

    # ── Step 3: Feature extraction ────────────────────────────────
    df_feat = None
    if all_steps or 3 in steps:
        from src.features.feature_extraction import extract_tabular_features
        import pandas as pd
        from pathlib import Path
        import yaml
        with open(ROOT / "config.yaml") as f:
            import yaml; cfg = yaml.safe_load(f)
        proc_dir = ROOT / cfg["paths"]["processed_data"]
        clean_path = proc_dir / "ai4i_clean.csv"

        def _feat_step():
            src = df_clean if df_clean is not None else pd.read_csv(clean_path)
            result = extract_tabular_features(src)
            out = proc_dir / "ai4i_features.csv"
            result.to_csv(out, index=False)
            print(f"[feat] Saved → {out}")
            return result

        df_feat = run_step(3, "Feature Extraction", _feat_step)

    # ── Step 4: Health scoring ────────────────────────────────────
    df_scored = None
    if all_steps or 4 in steps:
        from src.modeling.health_scorer import run_health_scoring
        import pandas as pd
        import yaml
        with open(ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        proc_dir = ROOT / cfg["paths"]["processed_data"]

        def _score_step():
            src = df_feat
            if src is None:
                feat_p = proc_dir / "ai4i_features.csv"
                if feat_p.exists():
                    src = pd.read_csv(feat_p)
                else:
                    src = pd.read_csv(proc_dir / "ai4i_clean.csv")

            # Create proxy vibration features from tabular data
            if "rotational_speed_rpm" in src.columns and "rms" not in src.columns:
                src = src.copy()
                src["rms"]          = src["rotational_speed_rpm"] / src["rotational_speed_rpm"].max()
                src["kurtosis"]     = (src.get("torque_Nm", 0) / (src.get("torque_Nm", 1).max() + 1e-9)) * 4
                src["crest_factor"] = src["kurtosis"] * 1.5

            result = run_health_scoring(src, train_if_needed=True)
            out = proc_dir / "ai4i_health_scored.csv"
            result.to_csv(out, index=False)
            print(f"[health] Saved → {out}")
            return result

        df_scored = run_step(4, "Asset Health Scoring", _score_step)

    # ── Step 5: Fault classification ─────────────────────────────
    if all_steps or 5 in steps:
        from src.modeling.fault_classifier import train_classifier, predict_fault
        import pandas as pd
        import yaml
        with open(ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        proc_dir = ROOT / cfg["paths"]["processed_data"]
        LABEL_COL = "fault_type"

        def _classify_step():
            src = df_scored
            if src is None:
                scored_p = proc_dir / "ai4i_health_scored.csv"
                clean_p  = proc_dir / "ai4i_clean.csv"
                src = pd.read_csv(scored_p if scored_p.exists() else clean_p)

            import numpy as np
            if LABEL_COL not in src.columns:
                src[LABEL_COL] = np.where(src.get("machine_failure", 0) == 1, "Fault", "Normal")

            train_classifier(src)
            result = predict_fault(src)
            out = proc_dir / "ai4i_predictions.csv"
            result[["uid", "predicted_fault", "fault_confidence"]].to_csv(out, index=False)
            print(f"[clf] Predictions saved → {out}")
            return result

        run_step(5, "Fault Classification", _classify_step)

    # ── Step 6: Database ──────────────────────────────────────────
    if all_steps or 6 in steps:
        from src.database.db_manager import create_schema, bulk_load_health_scores, insert_asset
        import pandas as pd
        import yaml
        with open(ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        proc_dir = ROOT / cfg["paths"]["processed_data"]

        def _db_step():
            create_schema()
            for aid, name, atype, loc in [
                ("MTR-001", "Pump Motor #1",    "motor",       "Plant A – Bay 1"),
                ("MTR-002", "Cooling Fan #2",   "fan",         "Plant A – Bay 2"),
                ("PMP-001", "Feed Pump #1",     "pump",        "Plant B – Utilities"),
                ("CMP-001", "Air Compressor #1","compressor",  "Plant B – Utilities"),
            ]:
                insert_asset(aid, name, atype, loc)

            scored_p = proc_dir / "ai4i_health_scored.csv"
            if scored_p.exists():
                df = pd.read_csv(scored_p)
                bulk_load_health_scores(df.head(500))   # limit for demo

        run_step(6, "Database Load", _db_step)

    # ── Step 7: Plots ─────────────────────────────────────────────
    if all_steps or 7 in steps:
        from src.visualization.plots import (
            plot_waveform, plot_fft, plot_health_distribution,
            plot_fault_distribution, plot_dashboard_summary,
        )
        import numpy as np
        import pandas as pd
        import yaml
        with open(ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        FS = cfg["signal"]["sampling_rate"]
        proc_dir = ROOT / cfg["paths"]["processed_data"]

        def _plot_step():
            np.random.seed(42)
            t = np.linspace(0, 1, FS)
            sig = np.sin(2 * np.pi * 60 * t) + 0.3 * np.random.randn(FS)
            sig[np.random.choice(FS, 30, replace=False)] += 2.5
            plot_waveform(sig)
            plot_fft(sig)

            scored_p = proc_dir / "ai4i_health_scored.csv"
            if scored_p.exists():
                df = pd.read_csv(scored_p)
                plot_health_distribution(df)
                plot_fault_distribution(df)
                plot_dashboard_summary(df)

        run_step(7, "Generate Plots", _plot_step)

    # ── Step 8: Report ────────────────────────────────────────────
    if all_steps or 8 in steps:
        from reports.generate_report import generate_report
        run_step(8, "Generate HTML Report", generate_report)

    # ── Summary ───────────────────────────────────────────────────
    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {total:.1f}s")
    print(f"  Outputs:")
    print(f"    data/processed/    — cleaned data, features, scores")
    print(f"    models/            — trained ML models")
    print(f"    reports/           — HTML report + plots")
    print(f"    web/index.html     — portfolio page")
    print(f"  Open reports/technical_report.html to view results.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBM Pipeline Runner")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--steps", nargs="+", type=int, help="Run only these step numbers")
    args = parser.parse_args()
    main(skip_download=args.skip_download, steps=args.steps)
