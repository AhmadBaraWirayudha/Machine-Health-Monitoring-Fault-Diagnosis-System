"""
Feature Extraction Module
==========================
Extracts time-domain and frequency-domain diagnostic features
from sensor / vibration data.

Features computed
-----------------
Time domain  : RMS, kurtosis, crest factor, peak-to-peak, skewness, variance
Frequency    : FFT peak frequency, FFT peak amplitude, band energies, spectral entropy

Usage:
    python src/features/feature_extraction.py
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.fft import rfft, rfftfreq

# ── Load config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR = ROOT / CFG["paths"]["processed_data"]
FS = CFG["signal"]["sampling_rate"]   # Hz


# ─────────────────────────────────────────────────────────────────────────────
# Time-domain features
# ─────────────────────────────────────────────────────────────────────────────

def rms(signal: np.ndarray) -> float:
    """Root Mean Square — overall vibration energy level."""
    return float(np.sqrt(np.mean(signal ** 2)))


def kurtosis(signal: np.ndarray) -> float:
    """
    Statistical kurtosis — sensitive to impulsive bearing faults.
    Healthy bearing: ~3. Fault onset: >4–6.
    """
    return float(stats.kurtosis(signal, fisher=False))


def crest_factor(signal: np.ndarray) -> float:
    """
    Peak / RMS — indicator of shock events.
    Healthy: 2–6. Severe fault: >10.
    """
    rms_val = rms(signal)
    return float(np.max(np.abs(signal)) / rms_val) if rms_val != 0 else 0.0


def peak_to_peak(signal: np.ndarray) -> float:
    """Maximum amplitude range of the signal."""
    return float(np.max(signal) - np.min(signal))


def skewness(signal: np.ndarray) -> float:
    """Signal asymmetry — useful for detecting partial faults."""
    return float(stats.skew(signal))


def variance(signal: np.ndarray) -> float:
    """Signal variance."""
    return float(np.var(signal))


def extract_time_features(signal: np.ndarray) -> dict:
    """Compute all time-domain features for one signal segment."""
    return {
        "rms":           rms(signal),
        "kurtosis":      kurtosis(signal),
        "crest_factor":  crest_factor(signal),
        "peak_to_peak":  peak_to_peak(signal),
        "skewness":      skewness(signal),
        "variance":      variance(signal),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain features
# ─────────────────────────────────────────────────────────────────────────────

def fft_analysis(signal: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute single-sided amplitude spectrum.

    Returns
    -------
    freqs : 1D array of frequency bins (Hz)
    amps  : 1D array of amplitudes
    """
    N = len(signal)
    yf = np.abs(rfft(signal)) * 2 / N
    xf = rfftfreq(N, d=1.0 / fs)
    return xf, yf


def fft_peak(signal: np.ndarray, fs: int = FS) -> tuple[float, float]:
    """Dominant frequency and its amplitude."""
    xf, yf = fft_analysis(signal, fs)
    idx = np.argmax(yf)
    return float(xf[idx]), float(yf[idx])


def band_energy(signal: np.ndarray, f_low: float, f_high: float, fs: int = FS) -> float:
    """
    Total spectral energy within [f_low, f_high] Hz.
    Useful for isolating bearing fault frequency bands.
    """
    xf, yf = fft_analysis(signal, fs)
    mask = (xf >= f_low) & (xf <= f_high)
    return float(np.sum(yf[mask] ** 2))


def spectral_entropy(signal: np.ndarray, fs: int = FS) -> float:
    """
    Shannon entropy of the normalised power spectrum.
    Low = narrow-band fault tone. High = broadband / noise.
    """
    _, yf = fft_analysis(signal, fs)
    power = yf ** 2
    p_norm = power / (np.sum(power) + 1e-12)
    return float(-np.sum(p_norm * np.log2(p_norm + 1e-12)))


def extract_frequency_features(signal: np.ndarray, fs: int = FS) -> dict:
    """Compute all frequency-domain features for one signal segment."""
    peak_freq, peak_amp = fft_peak(signal, fs)

    # Frequency band boundaries (adjust to your machine's characteristics)
    f_max = fs / 2
    return {
        "fft_peak_freq":      peak_freq,
        "fft_peak_amplitude": peak_amp,
        "band_energy_low":    band_energy(signal, 0,           f_max * 0.1,  fs),
        "band_energy_mid":    band_energy(signal, f_max * 0.1, f_max * 0.4,  fs),
        "band_energy_high":   band_energy(signal, f_max * 0.4, f_max,        fs),
        "spectral_entropy":   spectral_entropy(signal, fs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(signal: np.ndarray, fs: int = FS, label: str = "") -> dict:
    """Extract all time + frequency features for a single segment."""
    feats = {}
    feats.update(extract_time_features(signal))
    feats.update(extract_frequency_features(signal, fs))
    if label:
        feats["label"] = label
    return feats


def extract_features_from_signals(
    signals: list[np.ndarray],
    labels: list[str] | None = None,
    fs: int = FS,
) -> pd.DataFrame:
    """
    Extract features from a list of signal segments.

    Parameters
    ----------
    signals : list of 1D numpy arrays
    labels  : optional list of fault labels per segment
    fs      : sampling rate (Hz)

    Returns
    -------
    pd.DataFrame — one row per segment
    """
    rows = []
    for i, sig in enumerate(signals):
        lbl = labels[i] if labels and i < len(labels) else ""
        rows.append(extract_all_features(sig, fs=fs, label=lbl))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction on the tabular AI4I dataset
# (uses sensor readings as proxies for vibration features)
# ─────────────────────────────────────────────────────────────────────────────

def extract_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For the AI4I tabular dataset, compute rolling statistical features
    over the sensor columns — simulates signal feature extraction.
    """
    sensor_cols = [
        "rotational_speed_rpm", "torque_Nm",
        "air_temp_K", "process_temp_K", "tool_wear_min",
    ]
    result = df.copy()
    window = 50   # rolling window (rows)

    for col in sensor_cols:
        if col not in df.columns:
            continue
        s = df[col]
        result[f"{col}_rolling_mean"] = s.rolling(window, min_periods=1).mean()
        result[f"{col}_rolling_std"]  = s.rolling(window, min_periods=1).std().fillna(0)
        result[f"{col}_rolling_max"]  = s.rolling(window, min_periods=1).max()

    print(f"[feat] Tabular features extracted — shape: {result.shape}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main: demo with synthetic vibration signal
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Feature Extraction — Condition Monitoring Project")
    print("=" * 60)

    # ── Synthetic demo signal: sine + noise + impulse ──
    np.random.seed(42)
    t  = np.linspace(0, 1, FS)          # 1-second window at FS Hz
    f1, f2 = 50, 120                     # Hz — shaft + bearing fault frequency
    signal = (
        np.sin(2 * np.pi * f1 * t) +
        0.3 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.random.randn(FS)
    )
    # Add impulses to simulate bearing fault
    impulse_times = np.random.choice(FS, size=20, replace=False)
    signal[impulse_times] += 3.0

    print(f"\nSynthetic signal — length: {len(signal)} samples @ {FS} Hz")
    feats = extract_all_features(signal, label="bearing_fault")

    print("\nExtracted features:")
    for k, v in feats.items():
        if k != "label":
            print(f"  {k:<25} {v:>10.4f}")

    # ── Also run on cleaned AI4I data ──
    ai4i_path = PROC_DIR / "ai4i_clean.csv"
    if ai4i_path.exists():
        df = pd.read_csv(ai4i_path)
        df_feat = extract_tabular_features(df)
        out = PROC_DIR / "ai4i_features.csv"
        df_feat.to_csv(out, index=False)
        print(f"\n[OK]   Features saved → {out}")
