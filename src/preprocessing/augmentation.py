"""
Signal Augmentation
====================
Expands vibration signal training datasets by applying controlled
perturbations that preserve fault characteristics.

Techniques:
  Noise injection      — adds Gaussian noise at specified SNR
  Time shifting        — circular shift of the signal window
  Amplitude scaling    — multiplies amplitude by a random factor
  Phase randomisation  — randomises FFT phase while keeping amplitudes
  Frequency warping    — stretches or compresses the time axis
  Window jitter        — small random offset on window start
  Mixup                — linear interpolation between two signals

Usage:
    from src.preprocessing.augmentation import augment_dataset
    X_aug, y_aug = augment_dataset(X, y, n_augmented=5, techniques=["noise","scale"])
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Callable


# ── Individual augmentation functions ────────────────────────────────────────

def add_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add white Gaussian noise at the specified signal-to-noise ratio.

    Parameters
    ----------
    snr_db : target SNR in dB. Lower = more noise (more augmentation).
             Typical range: 10–30 dB.
    """
    signal_power = np.mean(signal ** 2)
    if signal_power == 0:
        return signal.copy()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise       = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise


def time_shift(signal: np.ndarray, max_shift_pct: float = 0.1) -> np.ndarray:
    """
    Circular shift the signal by a random number of samples.

    Parameters
    ----------
    max_shift_pct : maximum shift as a fraction of signal length (0–1).
    """
    max_shift = int(len(signal) * max_shift_pct)
    shift     = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(signal, shift)


def amplitude_scale(signal: np.ndarray,
                    low: float = 0.8,
                    high: float = 1.2) -> np.ndarray:
    """Multiply signal amplitude by a uniform random factor in [low, high]."""
    factor = np.random.uniform(low, high)
    return signal * factor


def phase_randomise(signal: np.ndarray) -> np.ndarray:
    """
    Randomise the phase of each FFT bin while preserving amplitudes.
    Produces a surrogate signal with the same power spectrum but different
    temporal structure — useful for destroying artificial patterns.
    """
    ft     = np.fft.rfft(signal)
    angles = np.random.uniform(0, 2 * np.pi, len(ft))
    ft_new = np.abs(ft) * np.exp(1j * angles)
    return np.fft.irfft(ft_new, n=len(signal))


def frequency_warp(signal: np.ndarray,
                   warp_factor: float | None = None,
                   low: float = 0.95,
                   high: float = 1.05) -> np.ndarray:
    """
    Stretch or compress the time axis by resampling.
    Returns a signal of the same length as the input.

    Parameters
    ----------
    warp_factor : if None, samples uniformly from [low, high].
    """
    if warp_factor is None:
        warp_factor = np.random.uniform(low, high)
    n_orig = len(signal)
    n_new  = int(n_orig * warp_factor)
    if n_new == 0:
        return signal.copy()
    warped = np.interp(
        np.linspace(0, n_orig - 1, n_new),
        np.arange(n_orig),
        signal,
    )
    # Resize back to original length
    return np.interp(
        np.linspace(0, n_new - 1, n_orig),
        np.arange(n_new),
        warped,
    )


def window_jitter(signal: np.ndarray,
                  max_jitter: int = 32) -> np.ndarray:
    """
    Extract a slightly offset sub-window, simulating imprecise
    trigger placement in data acquisition.
    """
    n = len(signal)
    jitter = np.random.randint(0, max_jitter + 1)
    if jitter == 0:
        return signal.copy()
    padded = np.concatenate([signal, signal[:jitter]])
    return padded[jitter: jitter + n]


def mixup(signal_a: np.ndarray,
          signal_b: np.ndarray,
          alpha: float = 0.3) -> np.ndarray:
    """
    Linearly interpolate between two signals.
    λ ~ Beta(alpha, alpha); result = λ·a + (1-λ)·b.

    Note: applies only when both signals share the same label.
    """
    lam = np.random.beta(alpha, alpha)
    return lam * signal_a + (1 - lam) * signal_b


# ── Augmentation registry ─────────────────────────────────────────────────────

TECHNIQUES: dict[str, Callable] = {
    "noise":     lambda s: add_noise(s, snr_db=np.random.uniform(15, 30)),
    "shift":     lambda s: time_shift(s, max_shift_pct=0.1),
    "scale":     lambda s: amplitude_scale(s, 0.85, 1.15),
    "phase":     phase_randomise,
    "warp":      lambda s: frequency_warp(s, low=0.97, high=1.03),
    "jitter":    lambda s: window_jitter(s, max_jitter=16),
}


def apply_random_augmentation(
    signal: np.ndarray,
    techniques: list[str] | None = None,
    n_apply: int = 1,
) -> np.ndarray:
    """
    Apply n_apply randomly chosen augmentation techniques to a signal.

    Parameters
    ----------
    techniques : list of technique names (keys of TECHNIQUES dict).
                 If None, uses all available techniques.
    n_apply    : number of techniques to chain (1–3 recommended).
    """
    if techniques is None:
        techniques = list(TECHNIQUES.keys())

    chosen = np.random.choice(techniques, size=min(n_apply, len(techniques)),
                              replace=False)
    result = signal.copy()
    for tech in chosen:
        fn = TECHNIQUES.get(tech)
        if fn:
            result = fn(result)
    return result


# ── Dataset-level augmentation ────────────────────────────────────────────────

def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_augmented: int = 3,
    techniques: list[str] | None = None,
    n_apply: int = 1,
    balance: bool = True,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment a signal dataset by generating synthetic samples.

    Parameters
    ----------
    X           : (n_samples, n_features) or (n_samples, signal_length) array
    y           : (n_samples,) class labels
    n_augmented : number of augmented copies per original sample
    techniques  : augmentation techniques to use
    n_apply     : number of techniques to chain per augmentation
    balance     : if True, over-sample minority classes to match majority count
    random_seed : random seed for reproducibility

    Returns
    -------
    X_aug : augmented feature array (original + synthetic)
    y_aug : corresponding labels
    """
    np.random.seed(random_seed)

    if balance:
        counts = Counter(y)
        max_count = max(counts.values())
        aug_per_class = {cls: max(0, max_count - cnt) for cls, cnt in counts.items()}
    else:
        aug_per_class = None

    X_new, y_new = [], []

    for cls in np.unique(y):
        idx    = np.where(y == cls)[0]
        if balance:
            n_to_gen = aug_per_class.get(cls, 0) + n_augmented * len(idx)
        else:
            n_to_gen = n_augmented * len(idx)

        for _ in range(n_to_gen):
            src = X[np.random.choice(idx)]
            aug = apply_random_augmentation(src, techniques, n_apply)
            X_new.append(aug)
            y_new.append(cls)

    if not X_new:
        return X, y

    X_aug = np.vstack([X, np.array(X_new)])
    y_aug = np.concatenate([y, np.array(y_new)])

    orig_counts = Counter(y)
    aug_counts  = Counter(y_aug)
    print(f"[augment] Original : {dict(orig_counts)}")
    print(f"[augment] Augmented: {dict(aug_counts)}")
    print(f"[augment] Total samples: {len(X)} → {len(X_aug)}")
    return X_aug, y_aug


def augment_tabular(
    df: pd.DataFrame,
    label_col: str,
    numeric_cols: list[str],
    n_augmented: int = 3,
    noise_std_pct: float = 0.02,
    balance: bool = True,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Augment a tabular (non-signal) dataset by adding small Gaussian noise
    to numeric features. Suitable for AI4I-style tabular sensor data.

    Parameters
    ----------
    noise_std_pct : noise standard deviation as a fraction of each column's std.
    balance       : over-sample minority classes to match majority.
    """
    np.random.seed(random_seed)

    counts = Counter(df[label_col])
    if balance:
        max_count = max(counts.values())
        target = {cls: max_count for cls in counts}
    else:
        target = {cls: cnt + n_augmented * cnt for cls, cnt in counts.items()}

    new_rows = []
    for cls, grp in df.groupby(label_col):
        n_needed = max(0, target.get(cls, 0) - len(grp))
        for _ in range(n_needed):
            row = grp.sample(1, random_state=np.random.randint(0, 99999)).iloc[0].copy()
            for col in numeric_cols:
                if col in row.index:
                    col_std = df[col].std() * noise_std_pct
                    row[col] += np.random.normal(0, max(col_std, 1e-6))
            new_rows.append(row)

    if not new_rows:
        return df

    df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    print(f"[augment] Tabular augmentation: {len(df)} → {len(df_aug)} rows")
    print(f"[augment] Class balance: {dict(Counter(df_aug[label_col]))}")
    return df_aug


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Signal Augmentation — Demo")
    print("=" * 60)

    # Synthetic demo signal
    np.random.seed(42)
    fs  = 4096
    t   = np.linspace(0, 1, fs)
    sig = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(fs)

    print("\nApplying individual techniques:")
    for name, fn in TECHNIQUES.items():
        aug   = fn(sig)
        delta = np.std(aug - sig)
        print(f"  {name:<12} → std(original)={sig.std():.4f}  "
              f"std(augmented)={aug.std():.4f}  delta_std={delta:.4f}")

    # Dataset augmentation
    print("\nDataset augmentation (3 classes, 100 samples each):")
    n = 100
    X = np.random.randn(n * 3, fs)
    y = np.array([0]*n + [1]*n + [2]*n)
    X_aug, y_aug = augment_dataset(X, y, n_augmented=2,
                                   techniques=["noise", "shift", "scale"])
    print(f"  X shape: {X.shape} → {X_aug.shape}")

    # Imbalanced example
    print("\nImbalanced dataset (80/15/5 split, balance=True):")
    y_imbal = np.array([0]*80 + [1]*15 + [2]*5)
    X_imbal = np.random.randn(100, 512)
    X_b, y_b = augment_dataset(X_imbal, y_imbal, n_augmented=1, balance=True)
