"""
Bearing Diagnostics
====================
Calculates bearing fault frequencies, performs order analysis,
and identifies fault tones in FFT spectra.

Fault frequencies depend on bearing geometry and shaft speed:
  BPFO  — Ball Pass Frequency, Outer race
  BPFI  — Ball Pass Frequency, Inner race
  BSF   — Ball Spin Frequency
  FTF   — Fundamental Train Frequency (cage)

Usage:
    from src.features.bearing_diagnostics import BearingGeometry, diagnose_spectrum

    bearing = BearingGeometry(
        n_balls=9, ball_dia_mm=7.94,
        pitch_dia_mm=38.5, contact_angle_deg=0.0
    )
    freqs = bearing.fault_frequencies(shaft_rpm=1500)
    print(freqs)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.fft import rfft, rfftfreq
from typing import NamedTuple


# ── Common bearing catalogue ──────────────────────────────────────────────────
# Partial catalogue — add more as needed.
# Source: SKF bearing catalogue (typical values)

BEARING_CATALOGUE = {
    "6205": dict(n_balls=9,  ball_dia_mm=7.94,  pitch_dia_mm=38.50, contact_angle_deg=0.0),
    "6206": dict(n_balls=9,  ball_dia_mm=9.53,  pitch_dia_mm=46.00, contact_angle_deg=0.0),
    "6207": dict(n_balls=9,  ball_dia_mm=11.11, pitch_dia_mm=52.00, contact_angle_deg=0.0),
    "6305": dict(n_balls=7,  ball_dia_mm=10.32, pitch_dia_mm=42.00, contact_angle_deg=0.0),
    "6306": dict(n_balls=8,  ball_dia_mm=11.11, pitch_dia_mm=48.50, contact_angle_deg=0.0),
    "6308": dict(n_balls=8,  ball_dia_mm=15.08, pitch_dia_mm=63.50, contact_angle_deg=0.0),
    "6310": dict(n_balls=8,  ball_dia_mm=19.05, pitch_dia_mm=80.00, contact_angle_deg=0.0),
    "22210": dict(n_balls=14, ball_dia_mm=12.70, pitch_dia_mm=72.00, contact_angle_deg=15.0),
}


# ── Data classes ──────────────────────────────────────────────────────────────

class FaultFrequencies(NamedTuple):
    shaft_hz:  float   # 1X rotation frequency
    BPFO:      float   # Ball Pass Frequency, Outer race
    BPFI:      float   # Ball Pass Frequency, Inner race
    BSF:       float   # Ball Spin Frequency
    FTF:       float   # Fundamental Train Frequency (cage)

    def as_dict(self) -> dict:
        return self._asdict()

    def harmonics(self, freq_name: str, n: int = 5) -> list[float]:
        """Return n harmonics of a given fault frequency."""
        base = getattr(self, freq_name)
        return [base * i for i in range(1, n + 1)]

    def summary(self, shaft_rpm: float) -> pd.DataFrame:
        return pd.DataFrame([
            {"frequency": "Shaft (1X)",   "Hz": self.shaft_hz,  "CPM": self.shaft_hz * 60,
             "X factor": 1.0},
            {"frequency": "BPFO",          "Hz": self.BPFO,  "CPM": self.BPFO * 60,
             "X factor": round(self.BPFO / self.shaft_hz, 3)},
            {"frequency": "BPFI",          "Hz": self.BPFI,  "CPM": self.BPFI * 60,
             "X factor": round(self.BPFI / self.shaft_hz, 3)},
            {"frequency": "BSF",           "Hz": self.BSF,   "CPM": self.BSF * 60,
             "X factor": round(self.BSF  / self.shaft_hz, 3)},
            {"frequency": "FTF (cage)",    "Hz": self.FTF,   "CPM": self.FTF * 60,
             "X factor": round(self.FTF  / self.shaft_hz, 3)},
        ])


@dataclass
class BearingGeometry:
    """
    Bearing geometry descriptor for fault frequency calculation.

    Parameters
    ----------
    n_balls         : number of rolling elements
    ball_dia_mm     : rolling element diameter (mm)
    pitch_dia_mm    : pitch circle diameter (mm)
    contact_angle_deg : contact angle (degrees), 0 for deep-groove
    bearing_id      : optional label (e.g. 'SKF 6205-2RS')
    """
    n_balls:            int
    ball_dia_mm:        float
    pitch_dia_mm:       float
    contact_angle_deg:  float = 0.0
    bearing_id:         str   = ""

    @classmethod
    def from_catalogue(cls, model: str) -> "BearingGeometry":
        """Create from bearing catalogue by model number."""
        model_clean = model.upper().replace("SKF", "").replace(" ", "").replace("-", "").strip()
        # Try common variations
        for key, params in BEARING_CATALOGUE.items():
            if key in model_clean or model_clean in key:
                return cls(**params, bearing_id=model)
        raise KeyError(f"Bearing model '{model}' not in catalogue. "
                       f"Available: {list(BEARING_CATALOGUE.keys())}")

    def fault_frequencies(self, shaft_rpm: float) -> FaultFrequencies:
        """
        Calculate all fault frequencies for a given shaft speed.

        Parameters
        ----------
        shaft_rpm : shaft rotational speed in RPM

        Returns
        -------
        FaultFrequencies named tuple (all values in Hz)
        """
        shaft_hz = shaft_rpm / 60.0
        ratio    = (self.ball_dia_mm / self.pitch_dia_mm) * np.cos(
                       np.radians(self.contact_angle_deg)
                   )
        n = self.n_balls

        BPFO = shaft_hz * (n / 2) * (1 - ratio)
        BPFI = shaft_hz * (n / 2) * (1 + ratio)
        BSF  = shaft_hz * (self.pitch_dia_mm / (2 * self.ball_dia_mm)) * (1 - ratio ** 2)
        FTF  = shaft_hz * (1 - ratio) / 2

        return FaultFrequencies(
            shaft_hz=round(shaft_hz, 4),
            BPFO=round(BPFO, 4),
            BPFI=round(BPFI, 4),
            BSF=round(BSF,  4),
            FTF=round(FTF,  4),
        )

    def __str__(self) -> str:
        label = self.bearing_id or "Custom"
        return (f"BearingGeometry({label}: n={self.n_balls}, "
                f"ball={self.ball_dia_mm}mm, pitch={self.pitch_dia_mm}mm, "
                f"α={self.contact_angle_deg}°)")


# ── Spectral analysis ─────────────────────────────────────────────────────────

def find_spectral_peaks(
    signal: np.ndarray,
    fs: int,
    n_peaks: int = 10,
    min_freq: float = 5.0,
) -> pd.DataFrame:
    """
    Find the top N amplitude peaks in the FFT spectrum.

    Returns DataFrame with columns: frequency_Hz, amplitude, rank
    """
    from scipy.signal import find_peaks

    N  = len(signal)
    yf = np.abs(rfft(signal)) * 2 / N
    xf = rfftfreq(N, d=1.0 / fs)

    # Only look above min_freq
    mask     = xf >= min_freq
    xf_valid = xf[mask]
    yf_valid = yf[mask]

    # Find local maxima
    peaks_idx, props = find_peaks(yf_valid, prominence=yf_valid.std() * 0.5)

    if len(peaks_idx) == 0:
        # Fallback: top N raw values
        peaks_idx = np.argsort(yf_valid)[-n_peaks:][::-1]
    else:
        # Sort by amplitude
        peaks_idx = peaks_idx[np.argsort(yf_valid[peaks_idx])[::-1]]

    top_idx = peaks_idx[:n_peaks]
    return pd.DataFrame({
        "frequency_Hz": xf_valid[top_idx].round(2),
        "amplitude":    yf_valid[top_idx].round(6),
        "rank":         range(1, len(top_idx) + 1),
    }).reset_index(drop=True)


def diagnose_spectrum(
    signal: np.ndarray,
    fs: int,
    bearing: BearingGeometry,
    shaft_rpm: float,
    tolerance_pct: float = 5.0,
    n_harmonics: int = 3,
) -> dict:
    """
    Compare spectral peaks against known fault frequencies and return a diagnosis.

    Parameters
    ----------
    signal        : vibration signal (1D numpy array)
    fs            : sampling rate (Hz)
    bearing       : BearingGeometry object
    shaft_rpm     : current shaft speed (RPM)
    tolerance_pct : frequency matching tolerance (% of fault frequency)
    n_harmonics   : number of harmonics to check per fault type

    Returns
    -------
    dict with keys:
        fault_frequencies : FaultFrequencies
        spectral_peaks    : DataFrame
        detected_faults   : list of dicts with fault type, harmonic, frequency, amplitude
        diagnosis         : str summary
        confidence        : float 0–1 (ratio of matched harmonics)
    """
    ff     = bearing.fault_frequencies(shaft_rpm)
    peaks  = find_spectral_peaks(signal, fs)

    detected = []
    for fault_name in ("BPFO", "BPFI", "BSF", "FTF", "shaft_hz"):
        base_freq = getattr(ff, fault_name)
        for h in range(1, n_harmonics + 1):
            target = base_freq * h
            tol    = target * tolerance_pct / 100.0

            # Check if any spectral peak falls within tolerance
            matches = peaks[np.abs(peaks["frequency_Hz"] - target) <= tol]
            if not matches.empty:
                best = matches.loc[matches["amplitude"].idxmax()]
                detected.append({
                    "fault_type":  fault_name,
                    "harmonic":    h,
                    "target_Hz":   round(target, 2),
                    "found_Hz":    round(best["frequency_Hz"], 2),
                    "amplitude":   round(best["amplitude"], 6),
                    "deviation_pct": round(
                        abs(best["frequency_Hz"] - target) / target * 100, 2
                    ),
                })

    # Summarise
    fault_counts = {}
    for d in detected:
        ft = d["fault_type"]
        if ft not in ("shaft_hz",):
            fault_counts[ft] = fault_counts.get(ft, 0) + 1

    max_possible = len([f for f in ("BPFO", "BPFI", "BSF", "FTF")
                        if f in fault_counts]) * n_harmonics or 1
    confidence   = min(sum(fault_counts.values()) / (max_possible + 1e-9), 1.0)

    if not fault_counts:
        diagnosis = "No bearing fault frequencies detected. Signature appears normal."
    else:
        top_fault = max(fault_counts, key=fault_counts.get)
        names = {"BPFO": "Outer Race", "BPFI": "Inner Race", "BSF": "Ball/Roller", "FTF": "Cage"}
        diagnosis = (f"Probable {names.get(top_fault, top_fault)} fault "
                     f"({fault_counts[top_fault]} harmonic(s) matched). "
                     f"Confidence: {confidence:.0%}")

    return {
        "fault_frequencies": ff,
        "spectral_peaks":    peaks,
        "detected_faults":   detected,
        "fault_counts":      fault_counts,
        "diagnosis":         diagnosis,
        "confidence":        round(confidence, 4),
    }


# ── Order analysis ────────────────────────────────────────────────────────────

def order_spectrum(
    signal: np.ndarray,
    fs: int,
    shaft_rpm: float,
    max_order: int = 20,
) -> pd.DataFrame:
    """
    Convert FFT spectrum to order domain (multiples of shaft frequency).

    Useful when shaft speed varies — aligns spectra across different RPMs.

    Returns DataFrame with columns: order, amplitude, frequency_Hz
    """
    shaft_hz = shaft_rpm / 60.0
    N  = len(signal)
    yf = np.abs(rfft(signal)) * 2 / N
    xf = rfftfreq(N, d=1.0 / fs)

    # Convert frequency to order
    orders_raw = xf / shaft_hz
    mask       = (orders_raw > 0) & (orders_raw <= max_order)

    return pd.DataFrame({
        "order":        orders_raw[mask].round(3),
        "amplitude":    yf[mask].round(6),
        "frequency_Hz": xf[mask].round(2),
    }).reset_index(drop=True)


def extract_bearing_features(
    signal: np.ndarray,
    fs: int,
    bearing: BearingGeometry,
    shaft_rpm: float,
) -> dict:
    """
    Extract bearing-specific diagnostic features for a signal segment.

    Features
    --------
    bpfo_energy     : spectral energy in ±5% band around BPFO
    bpfi_energy     : spectral energy in ±5% band around BPFI
    bsf_energy      : spectral energy in ±5% band around BSF
    ftf_energy      : spectral energy in ±5% band around FTF
    bpfo_amplitude  : peak amplitude at BPFO
    bpfi_amplitude  : peak amplitude at BPFI
    n_fault_tones   : total number of fault harmonics detected
    diagnosis       : string diagnosis
    confidence      : float 0–1
    """
    ff  = bearing.fault_frequencies(shaft_rpm)
    N   = len(signal)
    yf  = np.abs(rfft(signal)) * 2 / N
    xf  = rfftfreq(N, d=1.0 / fs)

    def band_amp(center_hz, tol_pct=5.0) -> float:
        tol  = center_hz * tol_pct / 100
        mask = (xf >= center_hz - tol) & (xf <= center_hz + tol)
        return float(np.max(yf[mask])) if mask.any() else 0.0

    def band_energy(center_hz, tol_pct=5.0) -> float:
        tol  = center_hz * tol_pct / 100
        mask = (xf >= center_hz - tol) & (xf <= center_hz + tol)
        return float(np.sum(yf[mask] ** 2)) if mask.any() else 0.0

    diag = diagnose_spectrum(signal, fs, bearing, shaft_rpm)

    return {
        "bpfo_energy":    band_energy(ff.BPFO),
        "bpfi_energy":    band_energy(ff.BPFI),
        "bsf_energy":     band_energy(ff.BSF),
        "ftf_energy":     band_energy(ff.FTF),
        "bpfo_amplitude": band_amp(ff.BPFO),
        "bpfi_amplitude": band_amp(ff.BPFI),
        "bsf_amplitude":  band_amp(ff.BSF),
        "n_fault_tones":  sum(diag["fault_counts"].values()),
        "diagnosis":      diag["diagnosis"],
        "confidence":     diag["confidence"],
    }


# ── Main demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Bearing Diagnostics — Demo")
    print("=" * 60)

    # Bearing setup
    bearing = BearingGeometry.from_catalogue("6205")
    print(f"\nBearing: {bearing}")

    shaft_rpm = 1500
    ff = bearing.fault_frequencies(shaft_rpm)
    print(f"\nFault Frequencies @ {shaft_rpm} RPM:")
    print(ff.summary(shaft_rpm).to_string(index=False))

    # Synthesise outer-race fault signal
    fs  = 25600
    T   = 2.0
    t   = np.linspace(0, T, int(fs * T), endpoint=False)
    sig = np.sin(2 * np.pi * ff.shaft_hz * t) + 0.1 * np.random.randn(len(t))
    # Add BPFO impulses
    for ti in np.arange(0, T, 1 / ff.BPFO):
        idx = int(ti * fs)
        if idx < len(sig) - 50:
            sig[idx:idx+50] += 4.0 * np.exp(-np.linspace(0, 10, 50))

    print(f"\nDiagnosing outer-race fault signal ...")
    result = diagnose_spectrum(sig, fs, bearing, shaft_rpm)
    print(f"\nDiagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"\nDetected fault tones:")
    for d in result["detected_faults"][:5]:
        print(f"  {d['fault_type']} H{d['harmonic']}: "
              f"target={d['target_Hz']} Hz, found={d['found_Hz']} Hz "
              f"(dev={d['deviation_pct']}%)")

    print(f"\nBearing features:")
    feats = extract_bearing_features(sig, fs, bearing, shaft_rpm)
    for k, v in feats.items():
        if k not in ("diagnosis",):
            print(f"  {k:<20} {v}")
    print(f"  {'diagnosis':<20} {feats['diagnosis']}")
