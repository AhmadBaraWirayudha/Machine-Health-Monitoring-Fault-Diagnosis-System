"""
Modeling Unit Tests
====================
Tests for health_scorer, fault_classifier, ensemble, and model_registry.

Run:
    pytest tests/test_modeling.py -v
    pytest tests/test_modeling.py -v -k "not slow"
"""

import sys
import json
import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Health Scorer ─────────────────────────────────────────────────────────────

class TestClassifyHealth:

    def test_good_threshold(self):
        from src.modeling.health_scorer import classify_health
        assert classify_health(100) == "Good"
        assert classify_health(80)  == "Good"

    def test_warning_threshold(self):
        from src.modeling.health_scorer import classify_health
        assert classify_health(79)  == "Warning"
        assert classify_health(60)  == "Warning"

    def test_degraded_threshold(self):
        from src.modeling.health_scorer import classify_health
        assert classify_health(59)  == "Degraded"
        assert classify_health(40)  == "Degraded"

    def test_critical_threshold(self):
        from src.modeling.health_scorer import classify_health
        assert classify_health(39)  == "Critical"
        assert classify_health(0)   == "Critical"

    def test_boundary_values(self):
        from src.modeling.health_scorer import classify_health
        for score, expected in [(80,"Good"), (60,"Warning"), (40,"Degraded")]:
            assert classify_health(score) == expected

    @pytest.mark.parametrize("score", [0, 20, 40, 60, 80, 100])
    def test_all_scores_have_status(self, score):
        from src.modeling.health_scorer import classify_health
        result = classify_health(score)
        assert result in ("Good", "Warning", "Degraded", "Critical")


class TestGenerateRecommendation:

    def test_good_status_recommendation(self):
        from src.modeling.health_scorer import generate_recommendation
        row = pd.Series({"health_status": "Good", "kurtosis": 2.5, "crest_factor": 2.0})
        rec = generate_recommendation(row)
        assert isinstance(rec, str)
        assert len(rec) > 10

    def test_critical_status_urgent(self):
        from src.modeling.health_scorer import generate_recommendation
        row = pd.Series({"health_status": "Critical", "kurtosis": 9.0, "crest_factor": 12.0})
        rec = generate_recommendation(row)
        assert any(word in rec.lower() for word in ("immediate", "critical", "inspect"))

    def test_all_statuses_return_string(self):
        from src.modeling.health_scorer import generate_recommendation
        for status in ["Good", "Warning", "Degraded", "Critical"]:
            row = pd.Series({"health_status": status, "kurtosis": 3.0, "crest_factor": 3.0})
            rec = generate_recommendation(row)
            assert isinstance(rec, str) and len(rec) > 5


class TestComputeHealthScore:

    def test_score_in_range(self, scored_df):
        from src.modeling.health_scorer import compute_health_score
        scores = compute_health_score(scored_df)
        assert scores.between(0, 100).all(), "Health scores must be 0–100"

    def test_high_features_low_score(self):
        """Higher kurtosis/crest_factor should produce lower health score."""
        from src.modeling.health_scorer import compute_health_score
        df_bad  = pd.DataFrame({"rms": [0.95], "kurtosis": [9.0], "crest_factor": [12.0]})
        df_good = pd.DataFrame({"rms": [0.1],  "kurtosis": [2.5], "crest_factor": [1.8]})
        s_bad  = compute_health_score(df_bad).iloc[0]
        s_good = compute_health_score(df_good).iloc[0]
        assert s_bad <= s_good, \
            f"Bad features ({s_bad:.1f}) should score ≤ good features ({s_good:.1f})"

    def test_anomaly_score_column_used(self, scored_df):
        from src.modeling.health_scorer import compute_health_score
        df = scored_df.copy()
        df["anomaly_score"] = 0.9   # high = healthy
        scores_high = compute_health_score(df).mean()
        df["anomaly_score"] = 0.1   # low = anomalous
        scores_low  = compute_health_score(df).mean()
        assert scores_high >= scores_low


class TestRunHealthScoring:

    def test_adds_health_score_column(self, scored_df):
        from src.modeling.health_scorer import run_health_scoring
        df = scored_df.drop(columns=["health_score", "health_status"], errors="ignore")
        result = run_health_scoring(df, train_if_needed=True)
        assert "health_score" in result.columns

    def test_adds_health_status_column(self, scored_df):
        from src.modeling.health_scorer import run_health_scoring
        df = scored_df.drop(columns=["health_status"], errors="ignore")
        result = run_health_scoring(df, train_if_needed=True)
        assert "health_status" in result.columns

    def test_adds_recommendation_column(self, scored_df):
        from src.modeling.health_scorer import run_health_scoring
        df = scored_df.drop(columns=["recommendation"], errors="ignore")
        result = run_health_scoring(df, train_if_needed=True)
        assert "recommendation" in result.columns

    def test_no_missing_scores(self, scored_df):
        from src.modeling.health_scorer import run_health_scoring
        result = run_health_scoring(scored_df.copy(), train_if_needed=True)
        assert result["health_score"].notna().all()

    def test_preserves_row_count(self, scored_df):
        from src.modeling.health_scorer import run_health_scoring
        original_len = len(scored_df)
        result = run_health_scoring(scored_df.copy(), train_if_needed=True)
        assert len(result) == original_len


# ── Fault Classifier ──────────────────────────────────────────────────────────

class TestFaultClassifier:

    def test_predict_returns_dataframe(self, trained_classifier, scored_df):
        from src.modeling.fault_classifier import predict_fault
        import joblib
        clf, le = trained_classifier
        MODELS_DIR = ROOT / "models"
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(clf, MODELS_DIR / "fault_classifier.joblib")
        joblib.dump(le,  MODELS_DIR / "label_encoder.joblib")

        feature_cols = ["rotational_speed_rpm", "torque_Nm", "tool_wear_min"]
        result = predict_fault(scored_df[["uid"] + feature_cols + ["fault_type"]])
        assert isinstance(result, pd.DataFrame)

    def test_prediction_columns_present(self, trained_classifier, scored_df):
        from src.modeling.fault_classifier import predict_fault
        import joblib
        clf, le = trained_classifier
        MODELS_DIR = ROOT / "models"
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(clf, MODELS_DIR / "fault_classifier.joblib")
        joblib.dump(le,  MODELS_DIR / "label_encoder.joblib")

        result = predict_fault(scored_df)
        assert "predicted_fault" in result.columns
        assert "fault_confidence" in result.columns

    def test_confidence_in_range(self, trained_classifier, scored_df):
        from src.modeling.fault_classifier import predict_fault
        import joblib
        clf, le = trained_classifier
        MODELS_DIR = ROOT / "models"
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(clf, MODELS_DIR / "fault_classifier.joblib")
        joblib.dump(le,  MODELS_DIR / "label_encoder.joblib")

        result = predict_fault(scored_df)
        assert result["fault_confidence"].between(0, 1).all()


# ── Anomaly Detector ──────────────────────────────────────────────────────────

class TestAnomalyDetector:

    def test_train_returns_model(self, scored_df):
        from src.modeling.health_scorer import train_anomaly_detector
        feature_cols = ["rms", "kurtosis", "crest_factor"]
        model = train_anomaly_detector(scored_df, feature_cols=feature_cols)
        assert model is not None

    def test_anomaly_scores_in_range(self, scored_df):
        from src.modeling.health_scorer import train_anomaly_detector, score_anomaly
        feature_cols = ["rms", "kurtosis", "crest_factor"]
        model = train_anomaly_detector(scored_df, feature_cols=feature_cols)
        scores = score_anomaly(scored_df, feature_cols=feature_cols, model=model)
        assert scores.between(0, 1).all(), "Anomaly scores must be 0–1"

    def test_anomaly_score_is_series(self, scored_df):
        from src.modeling.health_scorer import train_anomaly_detector, score_anomaly
        feature_cols = ["rms", "kurtosis", "crest_factor"]
        model = train_anomaly_detector(scored_df, feature_cols=feature_cols)
        scores = score_anomaly(scored_df, feature_cols=feature_cols, model=model)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(scored_df)


# ── Ensemble ──────────────────────────────────────────────────────────────────

class TestEnsemble:

    def test_voting_clf_builds(self):
        from src.modeling.ensemble import build_voting_classifier
        clf = build_voting_classifier("soft")
        assert clf is not None

    def test_stacking_clf_builds(self):
        from src.modeling.ensemble import build_stacking_classifier
        clf = build_stacking_classifier()
        assert clf is not None

    @pytest.mark.slow
    def test_soft_voting_fits_and_predicts(self, sample_df):
        from src.modeling.ensemble import build_voting_classifier
        from sklearn.preprocessing import LabelEncoder

        X = sample_df[["rotational_speed_rpm", "torque_Nm", "tool_wear_min"]].fillna(0).values
        le = LabelEncoder()
        y  = le.fit_transform(sample_df["fault_type"])

        clf = build_voting_classifier("soft")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset(set(y))

    def test_base_estimators_list(self):
        from src.modeling.ensemble import get_base_estimators
        estimators = get_base_estimators()
        assert len(estimators) >= 3
        for name, clf in estimators:
            assert isinstance(name, str)
            assert hasattr(clf, "fit")
            assert hasattr(clf, "predict")


# ── Model Registry ────────────────────────────────────────────────────────────

class TestModelRegistry:

    def test_register_returns_version_id(self, tmp_path, trained_classifier):
        from src.modeling.model_registry import ModelRegistry
        import src.modeling.model_registry as mr
        mr.REGISTRY_DIR = tmp_path / "registry"
        mr.REGISTRY_FILE = mr.REGISTRY_DIR / "registry.json"

        registry = ModelRegistry()
        clf, le  = trained_classifier
        vid = registry.register(clf, "test_model", metrics={"f1_macro": 0.9})
        assert vid == "v1"

    def test_register_increments_version(self, tmp_path, trained_classifier):
        from src.modeling.model_registry import ModelRegistry
        import src.modeling.model_registry as mr
        mr.REGISTRY_DIR = tmp_path / "registry2"
        mr.REGISTRY_FILE = mr.REGISTRY_DIR / "registry.json"

        registry = ModelRegistry()
        clf, _ = trained_classifier
        v1 = registry.register(clf, "counter_model", metrics={})
        v2 = registry.register(clf, "counter_model", metrics={})
        assert v1 == "v1"
        assert v2 == "v2"

    def test_promote_changes_stage(self, tmp_path, trained_classifier):
        from src.modeling.model_registry import ModelRegistry
        import src.modeling.model_registry as mr
        mr.REGISTRY_DIR = tmp_path / "registry3"
        mr.REGISTRY_FILE = mr.REGISTRY_DIR / "registry.json"

        registry = ModelRegistry()
        clf, _ = trained_classifier
        registry.register(clf, "stage_model")
        registry.promote("v1", "stage_model", stage="production")

        summary = registry.summary()
        row = summary[summary["version_id"] == "v1"]
        assert row["stage"].iloc[0] == "production"

    def test_summary_returns_dataframe(self, tmp_path, trained_classifier):
        from src.modeling.model_registry import ModelRegistry
        import src.modeling.model_registry as mr
        mr.REGISTRY_DIR = tmp_path / "registry4"
        mr.REGISTRY_FILE = mr.REGISTRY_DIR / "registry.json"

        registry = ModelRegistry()
        clf, _ = trained_classifier
        registry.register(clf, "sum_model", metrics={"f1_macro": 0.88})
        df = registry.summary()
        assert isinstance(df, pd.DataFrame)
        assert "model_type" in df.columns
        assert "version_id" in df.columns
        assert "stage" in df.columns

    def test_compare_sorts_by_metric(self, tmp_path, trained_classifier):
        from src.modeling.model_registry import ModelRegistry
        import src.modeling.model_registry as mr
        mr.REGISTRY_DIR = tmp_path / "registry5"
        mr.REGISTRY_FILE = mr.REGISTRY_DIR / "registry.json"

        registry = ModelRegistry()
        clf, _ = trained_classifier
        registry.register(clf, "cmp_model", metrics={"f1_macro": 0.80})
        registry.register(clf, "cmp_model", metrics={"f1_macro": 0.93})
        df = registry.compare("cmp_model", metric="f1_macro")
        assert df.iloc[0]["f1_macro"] >= df.iloc[1]["f1_macro"]
