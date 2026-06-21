"""
Feature & Augmentation Tests
==============================
Tests for bearing_diagnostics and signal augmentation modules.

Run:
    pytest tests/test_features.py -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Bearing Geometry ──────────────────────────────────────────────────────────

class TestBearingGeometry:

    def test_from_catalogue_6205(self):
        from src.features.bearing_diagnostics import BearingGeometry
        b = BearingGeometry.from_catalogue("6205")
        assert b.n_balls == 9
        assert abs(b.ball_dia_mm - 7.94) < 0.01

    def test_from_catalogue_case_insensitive(self):
        from src.features.bearing_diagnostics import BearingGeometry
        b = BearingGeometry.from_catalogue("skf6205")
        assert b is not None

    def test_unknown_bearing_raises(self):
        from src.features.bearing_diagnostics import BearingGeometry
        with pytest.raises(KeyError):
            BearingGeometry.from_catalogue("XXXX-9999")

    def test_fault_frequencies_positive(self):
        from src.features.bearing_diagnostics import BearingGeometry
        b = BearingGeometry.from_catalogue("6205")
        ff = b.fault_frequencies(shaft_rpm=1500)
        assert ff.shaft_hz > 0
        assert ff.BPFO > 0
        assert ff.BPFI > 0
        assert ff.BSF > 0
        assert ff.FTF > 0

    def test_bpfi_greater_than_bpfo(self):
        """BPFI > BPFO for standard deep-groove bearings."""
        from src.features.bearing_diagnostics import BearingGeometry
        b = BearingGeometry.from_catalogue("6205")
        ff = b.fault_frequencies(1500)
        assert ff.BPFI > ff.BPFO

    def test_shaft_hz_correct(self):
        from src.features.bearing_diagnostics import BearingGeometry
        b = BearingGeometry.from_catalogue("6205")
        ff = b.fault_frequencies(1500)
        assert abs(ff.shaft_hz - 25.0) < 0.01

    def test_fault_frequencies_scale_with_rpm(self):
        """All fault frequencies should double when shaft speed doubles."""
        from src.features.bearing_diagnostics import BearingGeometry
        b = BearingGeometry.from_catalogue("6205")
        ff1 = b.fault_frequencies(1000)
        ff2 = b.fault_frequencies(2000)
        for attr in ("shaft_hz", "BPFO", "BPFI", "BSF", "FTF"):
            ratio = getattr(ff2, attr) / getattr(ff1, attr)
            assert abs(ratio - 2.0) < 0.01, \
                f"{attr}: expected ratio=2.0, got {ratio:.4f}"

    def test_harmonics_count(self):
        from src.features.bearing_diagnostics import BearingGeometry
        b  = BearingGeometry.from_catalogue("6205")
        ff = b.fault_frequencies(1500)
        h  = ff.harmonics("BPFO", n=5)
        assert len(h) == 5
        assert abs(h[1] - ff.BPFO * 2) < 0.01

    def test_summary_dataframe(self):
        from src.features.bearing_diagnostics import BearingGeometry
        import pandas as pd
        b   = BearingGeometry.from_catalogue("6205")
        ff  = b.fault_frequencies(1500)
        df  = ff.summary(1500)
        assert isinstance(df, pd.DataFrame)
        assert "frequency" in df.columns
        assert "Hz" in df.columns
        assert len(df) == 5


# ── Spectral Analysis ─────────────────────────────────────────────────────────

class TestSpectralAnalysis:

    def test_find_peaks_returns_dataframe(self, healthy_signal, fs):
        from src.features.bearing_diagnostics import find_spectral_peaks
        import pandas as pd
        peaks = find_spectral_peaks(healthy_signal, fs=fs)
        assert isinstance(peaks, pd.DataFrame)
        assert "frequency_Hz" in peaks.columns
        assert "amplitude" in peaks.columns

    def test_find_peaks_positive_frequencies(self, healthy_signal, fs):
        from src.features.bearing_diagnostics import find_spectral_peaks
        peaks = find_spectral_peaks(healthy_signal, fs=fs)
        assert (peaks["frequency_Hz"] > 0).all()

    def test_find_peaks_positive_amplitudes(self, healthy_signal, fs):
        from src.features.bearing_diagnostics import find_spectral_peaks
        peaks = find_spectral_peaks(healthy_signal, fs=fs)
        assert (peaks["amplitude"] >= 0).all()

    def test_diagnose_returns_dict(self, healthy_signal, fs):
        from src.features.bearing_diagnostics import BearingGeometry, diagnose_spectrum
        b      = BearingGeometry.from_catalogue("6205")
        result = diagnose_spectrum(healthy_signal, fs, b, shaft_rpm=1500)
        assert isinstance(result, dict)
        assert "diagnosis" in result
        assert "confidence" in result
        assert "detected_faults" in result

    def test_diagnose_confidence_in_range(self, healthy_signal, fs):
        from src.features.bearing_diagnostics import BearingGeometry, diagnose_spectrum
        b      = BearingGeometry.from_catalogue("6205")
        result = diagnose_spectrum(healthy_signal, fs, b, shaft_rpm=1500)
        assert 0 <= result["confidence"] <= 1

    def test_diagnose_outer_race_detected(self, fs):
        """Outer-race fault signal should trigger BPFO detection."""
        from src.features.bearing_diagnostics import BearingGeometry, diagnose_spectrum
        b  = BearingGeometry.from_catalogue("6205")
        ff = b.fault_frequencies(1500)

        # Synthesise strong BPFO signal
        t   = np.linspace(0, 1, fs, endpoint=False)
        sig = np.sin(2 * np.pi * ff.shaft_hz * t)
        for ti in np.arange(0, 1, 1 / ff.BPFO):
            idx = int(ti * fs)
            if idx < fs - 20:
                sig[idx:idx+20] += 5.0 * np.exp(-np.linspace(0, 5, 20))

        result = diagnose_spectrum(sig, fs, b, shaft_rpm=1500, n_harmonics=2)
        assert result["fault_counts"].get("BPFO", 0) >= 1, \
            "Expected BPFO to be detected in outer-race fault signal"

    def test_order_spectrum_returns_dataframe(self, healthy_signal, fs):
        from src.features.bearing_diagnostics import order_spectrum
        import pandas as pd
        orders = order_spectrum(healthy_signal, fs=fs, shaft_rpm=1500)
        assert isinstance(orders, pd.DataFrame)
        assert "order" in orders.columns
        assert (orders["order"] > 0).all()

    def test_extract_bearing_features_keys(self, healthy_signal, fs):
        from src.features.bearing_diagnostics import BearingGeometry, extract_bearing_features
        b     = BearingGeometry.from_catalogue("6205")
        feats = extract_bearing_features(healthy_signal, fs, b, shaft_rpm=1500)
        for key in ("bpfo_energy", "bpfi_energy", "bsf_energy", "n_fault_tones", "confidence"):
            assert key in feats, f"Missing key: {key}"

    def test_extract_bearing_features_non_negative(self, healthy_signal, fs):
        from src.features.bearing_diagnostics import BearingGeometry, extract_bearing_features
        b     = BearingGeometry.from_catalogue("6205")
        feats = extract_bearing_features(healthy_signal, fs, b, shaft_rpm=1500)
        for key in ("bpfo_energy", "bpfi_energy", "bsf_energy"):
            assert feats[key] >= 0, f"{key} must be non-negative"


# ── Signal Augmentation ───────────────────────────────────────────────────────

class TestSignalAugmentation:

    def test_add_noise_same_length(self, healthy_signal):
        from src.preprocessing.augmentation import add_noise
        aug = add_noise(healthy_signal, snr_db=20)
        assert len(aug) == len(healthy_signal)

    def test_add_noise_changes_signal(self, healthy_signal):
        from src.preprocessing.augmentation import add_noise
        aug = add_noise(healthy_signal, snr_db=15)
        assert not np.allclose(aug, healthy_signal), "Noise should change the signal"

    def test_time_shift_same_length(self, healthy_signal):
        from src.preprocessing.augmentation import time_shift
        aug = time_shift(healthy_signal, max_shift_pct=0.1)
        assert len(aug) == len(healthy_signal)

    def test_amplitude_scale_ratio(self, healthy_signal):
        from src.preprocessing.augmentation import amplitude_scale
        np.random.seed(0)
        aug = amplitude_scale(healthy_signal, low=1.5, high=1.5)
        ratio = np.std(aug) / np.std(healthy_signal)
        assert abs(ratio - 1.5) < 0.01

    def test_phase_randomise_same_spectrum(self, healthy_signal, fs):
        """Phase randomisation should preserve the power spectrum."""
        from src.preprocessing.augmentation import phase_randomise
        aug    = phase_randomise(healthy_signal)
        assert len(aug) == len(healthy_signal)
        orig_power = np.abs(np.fft.rfft(healthy_signal))
        aug_power  = np.abs(np.fft.rfft(aug))
        # Power spectrum should be nearly identical
        assert np.allclose(orig_power, aug_power, rtol=1e-5)

    def test_frequency_warp_same_length(self, healthy_signal):
        from src.preprocessing.augmentation import frequency_warp
        aug = frequency_warp(healthy_signal, warp_factor=1.05)
        assert len(aug) == len(healthy_signal)

    def test_window_jitter_same_length(self, healthy_signal):
        from src.preprocessing.augmentation import window_jitter
        aug = window_jitter(healthy_signal, max_jitter=16)
        assert len(aug) == len(healthy_signal)

    def test_mixup_same_length(self, healthy_signal, fault_signal):
        from src.preprocessing.augmentation import mixup
        mixed = mixup(healthy_signal, fault_signal[0] if isinstance(fault_signal, tuple) else fault_signal)
        assert len(mixed) == len(healthy_signal)

    def test_augment_dataset_increases_size(self):
        from src.preprocessing.augmentation import augment_dataset
        n = 50
        X = np.random.randn(n, 512)
        y = np.array([0] * 40 + [1] * 10)
        X_aug, y_aug = augment_dataset(X, y, n_augmented=2, balance=False, random_seed=0)
        assert len(X_aug) > len(X)
        assert len(y_aug) == len(X_aug)

    def test_augment_dataset_balance(self):
        """After balanced augmentation, all classes should have same count."""
        from src.preprocessing.augmentation import augment_dataset
        X = np.random.randn(110, 256)
        y = np.array([0]*100 + [1]*10)
        X_aug, y_aug = augment_dataset(X, y, n_augmented=0, balance=True, random_seed=42)
        counts = {c: (y_aug == c).sum() for c in np.unique(y_aug)}
        assert counts[0] == counts[1], f"Expected balanced classes, got {counts}"

    def test_augment_preserves_labels(self):
        """No new class labels should be introduced by augmentation."""
        from src.preprocessing.augmentation import augment_dataset
        X = np.random.randn(60, 256)
        y = np.array([0]*30 + [1]*20 + [2]*10)
        X_aug, y_aug = augment_dataset(X, y, n_augmented=1, random_seed=0)
        assert set(y_aug).issubset({0, 1, 2})

    def test_apply_random_augmentation(self, healthy_signal):
        from src.preprocessing.augmentation import apply_random_augmentation
        aug = apply_random_augmentation(healthy_signal, techniques=["noise", "shift"], n_apply=1)
        assert len(aug) == len(healthy_signal)

    def test_augment_tabular(self, sample_df):
        from src.preprocessing.augmentation import augment_tabular
        df_aug = augment_tabular(
            sample_df,
            label_col="fault_type",
            numeric_cols=["rotational_speed_rpm", "torque_Nm"],
            n_augmented=2,
            balance=True,
        )
        assert len(df_aug) >= len(sample_df)
        assert "fault_type" in df_aug.columns
