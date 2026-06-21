"""
Model Registry
==============
Tracks all trained model versions with metadata, metrics, and lineage.
Lightweight alternative to MLflow for portfolio use.

Features:
  - Save / load versioned models with metadata
  - Promote a version to 'production'
  - Compare model versions by metric
  - Rollback to a previous version
  - Export registry to JSON / CSV

Usage:
    from src.modeling.model_registry import ModelRegistry
    registry = ModelRegistry()

    # Register a new model version
    version = registry.register(
        model=clf,
        model_type="fault_classifier",
        params=best_params,
        metrics={"f1_macro": 0.94, "accuracy": 0.97},
        tags={"strategy": "SMOTE", "dataset": "ai4i_2020"},
    )

    # Promote to production
    registry.promote(version, "production")

    # Load production model
    clf, meta = registry.load_production("fault_classifier")
"""

import json
import shutil
import joblib
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

REGISTRY_DIR = ROOT / CFG["paths"]["models"] / "registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = REGISTRY_DIR / "registry.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _file_hash(path: Path) -> str:
    """MD5 hash of a file (for integrity checks)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_registry() -> list[dict]:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return []


def _save_registry(records: list[dict]) -> None:
    REGISTRY_FILE.write_text(json.dumps(records, indent=2, default=str))


# ── ModelRegistry ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Lightweight model registry backed by a local JSON file.

    Directory layout:
        models/registry/
            registry.json              — index of all versions
            fault_classifier/
                v1/
                    model.joblib
                    label_encoder.joblib
                    meta.json
                v2/
                    ...
            isolation_forest/
                v1/
                    ...
    """

    def __init__(self):
        self._records = _load_registry()

    # ── Core operations ───────────────────────────────────────────────────────

    def register(
        self,
        model: Any,
        model_type: str,
        params: dict | None = None,
        metrics: dict | None = None,
        tags: dict | None = None,
        extra_artifacts: dict[str, Any] | None = None,
        description: str = "",
    ) -> str:
        """
        Save a trained model with metadata and return its version ID.

        Parameters
        ----------
        model           : fitted sklearn estimator (or any joblib-serialisable object)
        model_type      : logical name, e.g. 'fault_classifier', 'isolation_forest'
        params          : hyperparameters used
        metrics         : evaluation metrics dict, e.g. {'f1_macro': 0.94}
        tags            : arbitrary key-value tags
        extra_artifacts : additional objects to save alongside the model
                          e.g. {'label_encoder': le, 'scaler': scaler}
        description     : human-readable note

        Returns
        -------
        version_id : str, e.g. 'v3'
        """
        # Determine next version number
        existing = [r for r in self._records if r["model_type"] == model_type]
        version_num = len(existing) + 1
        version_id  = f"v{version_num}"

        # Create version directory
        version_dir = REGISTRY_DIR / model_type / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = version_dir / "model.joblib"
        joblib.dump(model, model_path)
        model_hash = _file_hash(model_path)

        # Save extra artifacts
        artifact_paths = {}
        if extra_artifacts:
            for name, obj in extra_artifacts.items():
                art_path = version_dir / f"{name}.joblib"
                joblib.dump(obj, art_path)
                artifact_paths[name] = str(art_path.relative_to(ROOT))

        # Build metadata record
        record = {
            "version_id":    version_id,
            "model_type":    model_type,
            "description":   description,
            "stage":         "staging",    # staging | production | archived
            "params":        params or {},
            "metrics":       metrics or {},
            "tags":          tags or {},
            "artifacts":     artifact_paths,
            "model_path":    str(model_path.relative_to(ROOT)),
            "model_hash":    model_hash,
            "registered_at": _now(),
            "promoted_at":   None,
        }

        # Save per-version meta.json
        (version_dir / "meta.json").write_text(json.dumps(record, indent=2, default=str))

        # Update registry index
        self._records.append(record)
        _save_registry(self._records)

        print(f"[registry] Registered {model_type} {version_id} "
              f"(stage=staging, metrics={metrics})")
        return version_id

    def promote(self, version_id: str, model_type: str, stage: str = "production") -> None:
        """
        Promote a version to a lifecycle stage.

        Stages: staging → production → archived

        Promoting to 'production' automatically archives the previous
        production version of the same model_type.
        """
        assert stage in ("staging", "production", "archived"), \
            f"Unknown stage '{stage}'"

        if stage == "production":
            # Archive previous production
            for rec in self._records:
                if rec["model_type"] == model_type and rec["stage"] == "production":
                    rec["stage"] = "archived"
                    print(f"[registry] Archived previous production: {model_type} {rec['version_id']}")

        target = self._get(version_id, model_type)
        target["stage"] = stage
        if stage == "production":
            target["promoted_at"] = _now()
            # Copy to canonical path for easy loading
            self._copy_to_canonical(version_id, model_type)

        _save_registry(self._records)
        print(f"[registry] Promoted {model_type} {version_id} → {stage}")

    def load(
        self, version_id: str, model_type: str
    ) -> tuple[Any, dict]:
        """Load a specific model version. Returns (model, metadata)."""
        rec = self._get(version_id, model_type)
        model_path = ROOT / rec["model_path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Integrity check
        actual_hash = _file_hash(model_path)
        if actual_hash != rec["model_hash"]:
            raise ValueError(
                f"Model file hash mismatch for {model_type} {version_id}. "
                "File may have been corrupted or tampered with."
            )

        model = joblib.load(model_path)
        return model, rec

    def load_production(self, model_type: str) -> tuple[Any, dict]:
        """Load the current production model for the given model_type."""
        prod = [r for r in self._records
                if r["model_type"] == model_type and r["stage"] == "production"]
        if not prod:
            raise LookupError(
                f"No production model for '{model_type}'. "
                "Register and promote a version first."
            )
        return self.load(prod[-1]["version_id"], model_type)

    def rollback(self, model_type: str) -> str:
        """
        Rollback: promote the previous production version back to production
        and archive the current one.
        Returns the rolled-back version_id.
        """
        history = [r for r in self._records
                   if r["model_type"] == model_type and r["stage"] in ("production", "archived")]
        history.sort(key=lambda r: r.get("promoted_at") or r["registered_at"])

        if len(history) < 2:
            raise LookupError(f"No previous version to roll back to for '{model_type}'")

        current  = history[-1]
        previous = history[-2]

        current["stage"]  = "archived"
        previous["stage"] = "production"
        previous["promoted_at"] = _now()
        _save_registry(self._records)
        self._copy_to_canonical(previous["version_id"], model_type)

        print(f"[registry] Rolled back {model_type}: "
              f"{current['version_id']} → archived, "
              f"{previous['version_id']} → production")
        return previous["version_id"]

    # ── Comparison & reporting ────────────────────────────────────────────────

    def compare(
        self,
        model_type: str,
        metric: str = "f1_macro",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Return a sorted DataFrame comparing all versions by a metric."""
        records = [r for r in self._records if r["model_type"] == model_type]
        if not records:
            print(f"[registry] No records for model_type='{model_type}'")
            return pd.DataFrame()

        rows = []
        for r in records:
            rows.append({
                "version_id":    r["version_id"],
                "stage":         r["stage"],
                metric:          r["metrics"].get(metric, np.nan),
                "registered_at": r["registered_at"],
                "promoted_at":   r.get("promoted_at"),
                **{f"param_{k}": v for k, v in list(r["params"].items())[:4]},
            })

        df = pd.DataFrame(rows).sort_values(metric, ascending=False).head(top_n)
        return df

    def summary(self) -> pd.DataFrame:
        """Return a summary of all registered models across all types."""
        if not self._records:
            return pd.DataFrame(columns=["model_type", "version_id", "stage",
                                          "f1_macro", "registered_at"])
        rows = []
        for r in self._records:
            rows.append({
                "model_type":    r["model_type"],
                "version_id":    r["version_id"],
                "stage":         r["stage"],
                "f1_macro":      r["metrics"].get("f1_macro", "—"),
                "accuracy":      r["metrics"].get("accuracy", "—"),
                "registered_at": r["registered_at"],
                "tags":          str(r.get("tags", {})),
            })
        return pd.DataFrame(rows)

    def delete_version(self, version_id: str, model_type: str) -> None:
        """Permanently delete a version (cannot be production)."""
        rec = self._get(version_id, model_type)
        if rec["stage"] == "production":
            raise ValueError("Cannot delete the production version. Archive it first.")
        version_dir = REGISTRY_DIR / model_type / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)
        self._records = [r for r in self._records
                         if not (r["model_type"] == model_type and r["version_id"] == version_id)]
        _save_registry(self._records)
        print(f"[registry] Deleted {model_type} {version_id}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get(self, version_id: str, model_type: str) -> dict:
        matches = [r for r in self._records
                   if r["model_type"] == model_type and r["version_id"] == version_id]
        if not matches:
            raise LookupError(f"Version '{version_id}' not found for '{model_type}'")
        return matches[-1]

    def _copy_to_canonical(self, version_id: str, model_type: str) -> None:
        """Copy a version's model to the canonical path used by the pipeline."""
        rec = self._get(version_id, model_type)
        src = ROOT / rec["model_path"]
        dst = ROOT / CFG["paths"]["models"] / f"{model_type}.joblib"
        shutil.copy2(src, dst)

        # Copy extra artifacts too
        for name, rel_path in rec.get("artifacts", {}).items():
            src_art = ROOT / rel_path
            dst_art = ROOT / CFG["paths"]["models"] / f"{name}.joblib"
            if src_art.exists():
                shutil.copy2(src_art, dst_art)

    def export(self, path: Path | None = None) -> Path:
        """Export registry to a CSV file."""
        df = self.summary()
        out = path or (REGISTRY_DIR / "registry_export.csv")
        df.to_csv(out, index=False)
        print(f"[registry] Exported {len(df)} records → {out}")
        return out


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import numpy as np

    print("=" * 60)
    print("  Model Registry — Demo")
    print("=" * 60)

    registry = ModelRegistry()

    # Simulate registering two model versions
    X, y = make_classification(n_samples=500, n_features=7, random_state=42)

    for i, (n_est, depth) in enumerate([(50, 5), (100, 10)], start=1):
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
        clf.fit(X, y)
        f1 = float(np.random.uniform(0.88, 0.96))

        vid = registry.register(
            model=clf,
            model_type="fault_classifier",
            params={"n_estimators": n_est, "max_depth": depth},
            metrics={"f1_macro": round(f1, 4), "accuracy": round(f1 + 0.02, 4)},
            tags={"run": i, "dataset": "demo"},
            description=f"Demo training run {i}",
        )
        print(f"  Registered: {vid}")

    # Promote best version
    registry.promote("v2", "fault_classifier", stage="production")

    # Compare
    print("\nVersion Comparison:")
    print(registry.compare("fault_classifier", metric="f1_macro").to_string(index=False))

    # Summary
    print("\nRegistry Summary:")
    print(registry.summary().to_string(index=False))

    # Export
    registry.export()
