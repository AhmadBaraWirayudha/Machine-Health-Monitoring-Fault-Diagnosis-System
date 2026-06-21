"""
Unit Tests
==========
Tests for the core pipeline modules.

Run with:
    pytest tests/ -v
    make test
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_signal():
    """1-second synthetic vibration signal at 25600 Hz."""
    np.random.seed(42)
    fs = 25600
    t  = np.linspace(0, 1, fs)
    sig = (
        np.sin(2 * np.pi * 60 * t) +
        0.3 * np.sin(2 * np.pi * 120 * t) +
        0.1 * np.random.randn(fs)
    )
    return sig, fs


@pytest.fixture
def healthy_signal():
    """Clean sinusoid — no impulses."""
    np.random.seed(0)
    fs = 4096
    t  = np.linspace(0, 1, fs)
    return np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(fs), fs


@pytest.fixture
def fault_signal():
    """Signal with heavy impulses — simulates bearing fault."""
    np.random.seed(1)
    fs  = 4096
    t   = np.linspace(0, 1, fs)
    sig = np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(fs)
    impulse_idx = np.random.choice(fs, size=50, replace=False)
    sig[impulse_idx] += 5.0
    return sig, fs


@pytest.fixture
def sample_df():
    """Small DataFrame mimicking the AI4I dataset."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "uid":                    range(1, n + 1),
        "air_temp_K":             np.random.normal(300, 2, n),
        "process_temp_K":         np.random.normal(310, 1.5, n),
        "rotational_speed_rpm":   np.random.normal(1500, 50, n),
        "torque_Nm":              np.random.normal(40, 10, n),
        "tool_wear_min":          np.random.uniform(0, 250, n),
        "machine_failure":        np.random.binomial(1, 0.05, n),
        "fault_type":             np.random.choice(
                                      ["Normal", "TWF", "HDF", "Normal", "Normal"], n
                                  ),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeDomainFeatures:

    def test_rms_pure_sine(self):
        """RMS of sin(t) over full period = 1/√2 ≈ 0.707."""
        from src.features.feature_extraction import rms
        t = np.linspace(0, 2 * np.pi, 10000)
        sig = np.sin(t)
        assert abs(rms(sig) - 1 / np.sqrt(2)) < 0.01

    def test_rms_zero_signal(self):
        from src.features.feature_extraction import rms
        assert rms(np.zeros(100)) == 0.0

    def test_rms_positive(self, synthetic_signal):
        from src.features.feature_extraction import rms
        sig, _ = synthetic_signal
        assert rms(sig) > 0

    def test_kurtosis_gaussian(self):
        """Kurtosis of a Gaussian signal ≈ 3."""
        from src.features.feature_extraction import kurtosis
        np.random.seed(99)
        sig = np.random.randn(50000)
        assert abs(kurtosis(sig) - 3.0) < 0.2

    def test_kurtosis_fault_higher_than_healthy(self, healthy_signal, fault_signal):
        from src.features.feature_extraction import kurtosis
        k_healthy = kurtosis(healthy_signal[0])
        k_fault   = kurtosis(fault_signal[0])
        assert k_fault > k_healthy, \
            f"Fault kurtosis ({k_fault:.2f}) should exceed healthy ({k_healthy:.2f})"

    def test_crest_factor_positive(self, synthetic_signal):
        from src.features.feature_extraction import crest_factor
        sig, _ = synthetic_signal
        cf = crest_factor(sig)
        assert cf > 0

    def test_crest_factor_pure_sine(self):
        """Crest factor of pure sine = √2 ≈ 1.414."""
        from src.features.feature_extraction import crest_factor
        t = np.linspace(0, 2 * np.pi, 10000)
        sig = np.sin(t)
        assert abs(crest_factor(sig) - np.sqrt(2)) < 0.05

    def test_peak_to_peak_range(self):
        from src.features.feature_extraction import peak_to_peak
        sig = np.array([-3.0, 0, 1, 2, 5.0])
        assert peak_to_peak(sig) == pytest.approx(8.0)

    def test_extract_time_features_returns_dict(self, synthetic_signal):
        from src.features.feature_extraction import extract_time_features
        sig, _ = synthetic_signal
        feats = extract_time_features(sig)
        expected_keys = {"rms", "kurtosis", "crest_factor", "peak_to_peak", "skewness", "variance"}
        assert expected_keys.issubset(feats.keys())
        assert all(isinstance(v, float) for v in feats.values())


class TestFrequencyDomainFeatures:

    def test_fft_peak_freq_detection(self):
        """FFT should detect the dominant frequency in a sine wave."""
        from src.features.feature_extraction import fft_peak
        fs  = 10000
        f0  = 250   # Hz
        t   = np.linspace(0, 1, fs)
        sig = np.sin(2 * np.pi * f0 * t)
        peak_freq, peak_amp = fft_peak(sig, fs=fs)
        assert abs(peak_freq - f0) < 5, f"Expected ~{f0} Hz, got {peak_freq:.1f} Hz"
        assert peak_amp > 0.5

    def test_band_energy_positive(self, synthetic_signal):
        from src.features.feature_extraction import band_energy
        sig, fs = synthetic_signal
        be = band_energy(sig, f_low=0, f_high=fs / 2, fs=fs)
        assert be > 0

    def test_spectral_entropy_bounded(self, synthetic_signal):
        from src.features.feature_extraction import spectral_entropy
        sig, fs = synthetic_signal
        se = spectral_entropy(sig, fs=fs)
        assert se >= 0

    def test_spectral_entropy_pure_sine_low(self):
        """Pure sine has very low spectral entropy (energy at one frequency)."""
        from src.features.feature_extraction import spectral_entropy
        fs  = 8192
        t   = np.linspace(0, 1, fs)
        sig = np.sin(2 * np.pi * 100 * t)
        se  = spectral_entropy(sig, fs=fs)
        assert se < 2.0, f"Expected low entropy for sine, got {se:.3f}"

    def test_extract_all_features_complete(self, synthetic_signal):
        from src.features.feature_extraction import extract_all_features
        sig, fs = synthetic_signal
        feats = extract_all_features(sig, fs=fs, label="test")
        assert "rms" in feats
        assert "fft_peak_freq" in feats
        assert "spectral_entropy" in feats
        assert feats["label"] == "test"


class TestSignalSegmentation:

    def test_segment_count(self):
        from src.preprocessing.clean_data import segment_signal
        sig = np.zeros(2048)
        segs = segment_signal(sig, window_size=512, overlap=0.5)
        # step = 256; starts: 0, 256, 512, 768, 1024, 1280, 1536 → 7 segments
        assert len(segs) == 7

    def test_segment_length(self):
        from src.preprocessing.clean_data import segment_signal
        sig  = np.random.randn(4096)
        segs = segment_signal(sig, window_size=512, overlap=0.0)
        assert all(len(s) == 512 for s in segs)

    def test_no_overlap(self):
        from src.preprocessing.clean_data import segment_signal
        sig  = np.arange(1000, dtype=float)
        segs = segment_signal(sig, window_size=100, overlap=0.0)
        assert len(segs) == 10
        # First segment should start from 0
        assert segs[0][0] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_drop_duplicates(self, sample_df):
        from src.preprocessing.clean_data import drop_duplicates
        df_duped = pd.concat([sample_df, sample_df.head(10)], ignore_index=True)
        df_clean = drop_duplicates(df_duped)
        assert len(df_clean) == len(sample_df)

    def test_handle_missing_fills(self):
        from src.preprocessing.clean_data import handle_missing
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
        df_filled = handle_missing(df)
        assert df_filled.isna().sum().sum() == 0

    def test_outlier_flagging(self, sample_df):
        from src.preprocessing.clean_data import remove_outliers_iqr
        df = remove_outliers_iqr(sample_df, cols=["torque_Nm"])
        assert "is_outlier" in df.columns

    def test_normalise_range(self, sample_df):
        from src.preprocessing.clean_data import normalise
        cols = ["torque_Nm", "tool_wear_min"]
        df_norm = normalise(sample_df, cols=cols, method="minmax", save_scaler=False)
        for col in cols:
            assert df_norm[col].min() >= -0.01
            assert df_norm[col].max() <=  1.01


# ─────────────────────────────────────────────────────────────────────────────
# Health Scorer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthScorer:

    def test_classify_health_thresholds(self):
        from src.modeling.health_scorer import classify_health
        assert classify_health(90)  == "Good"
        assert classify_health(70)  == "Warning"
        assert classify_health(50)  == "Degraded"
        assert classify_health(20)  == "Critical"
        assert classify_health(80)  == "Good"
        assert classify_health(60)  == "Warning"
        assert classify_health(40)  == "Degraded"

    def test_health_score_range(self, sample_df):
        from src.modeling.health_scorer import run_health_scoring
        df = sample_df.copy()
        df["rms"]          = np.random.rand(len(df))
        df["kurtosis"]     = np.random.uniform(1, 8, len(df))
        df["crest_factor"] = np.random.uniform(1, 10, len(df))
        df_scored = run_health_scoring(df, train_if_needed=True)
        assert "health_score" in df_scored.columns
        assert df_scored["health_score"].between(0, 100).all()

    def test_health_status_labels(self, sample_df):
        from src.modeling.health_scorer import run_health_scoring
        df = sample_df.copy()
        df["rms"] = np.random.rand(len(df))
        df["kurtosis"] = 3.0
        df["crest_factor"] = 2.0
        df_scored = run_health_scoring(df, train_if_needed=True)
        valid_statuses = {"Good", "Warning", "Degraded", "Critical"}
        assert set(df_scored["health_status"].unique()).issubset(valid_statuses)

    def test_recommendation_non_empty(self):
        from src.modeling.health_scorer import generate_recommendation
        for status in ["Good", "Warning", "Degraded", "Critical"]:
            row = pd.Series({"health_status": status, "kurtosis": 5.0, "crest_factor": 7.0})
            rec = generate_recommendation(row)
            assert isinstance(rec, str) and len(rec) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Utilities Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_safe_divide_zero(self):
        from src.utils.helpers import safe_divide
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, fill=-1) == -1.0

    def test_safe_divide_normal(self):
        from src.utils.helpers import safe_divide
        assert safe_divide(10, 2) == 5.0

    def test_rolling_zscore_mean_zero(self, sample_df):
        from src.utils.helpers import rolling_zscore
        s = sample_df["rotational_speed_rpm"]
        z = rolling_zscore(s, window=50)
        assert abs(z.iloc[50:].mean()) < 1.0   # roughly centred

    def test_validate_dataframe_passes(self, sample_df):
        from src.utils.helpers import validate_dataframe
        validate_dataframe(sample_df, required_cols=["uid", "torque_Nm"])

    def test_validate_dataframe_missing_col(self, sample_df):
        from src.utils.helpers import validate_dataframe, DataValidationError
        with pytest.raises(DataValidationError):
            validate_dataframe(sample_df, required_cols=["nonexistent_column"])

    def test_validate_dataframe_empty(self):
        from src.utils.helpers import validate_dataframe, DataValidationError
        with pytest.raises(DataValidationError):
            validate_dataframe(pd.DataFrame(), required_cols=[], min_rows=1)
