"""
Visualization Module
=====================
Generates diagnostic plots for technical reports and the HTML dashboard.

Outputs are saved to reports/plots/ as PNG files.

Usage:
    python src/visualization/plots.py
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.fft import rfft, rfftfreq

# ── Load config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR  = ROOT / CFG["paths"]["processed_data"]
PLOTS_DIR = ROOT / CFG["paths"]["plots"]
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "Good":     "#2ecc71",
    "Warning":  "#f39c12",
    "Degraded": "#e67e22",
    "Critical": "#e74c3c",
    "primary":  "#2c3e50",
    "accent":   "#3498db",
}
FS = CFG["signal"]["sampling_rate"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Time-waveform plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform(
    signal: np.ndarray,
    fs: int = FS,
    title: str = "Vibration Waveform",
    label: str = "",
    save: bool = True,
) -> None:
    t = np.linspace(0, len(signal) / fs, len(signal))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t * 1000, signal, color=COLORS["accent"], linewidth=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title + (f" — {label}" if label else ""), fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        out = PLOTS_DIR / "waveform.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Waveform saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FFT Spectrum
# ─────────────────────────────────────────────────────────────────────────────

def plot_fft(
    signal: np.ndarray,
    fs: int = FS,
    title: str = "Frequency Spectrum (FFT)",
    save: bool = True,
) -> None:
    N  = len(signal)
    yf = np.abs(rfft(signal)) * 2 / N
    xf = rfftfreq(N, d=1.0 / fs)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(xf, yf, color=COLORS["primary"], linewidth=0.8)
    # Mark peak
    peak_idx = np.argmax(yf)
    ax.axvline(xf[peak_idx], color=COLORS["Critical"], linestyle="--", linewidth=1,
               label=f"Peak: {xf[peak_idx]:.1f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save:
        out = PLOTS_DIR / "fft_spectrum.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] FFT spectrum saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature trend over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_trend(
    df: pd.DataFrame,
    feature: str,
    time_col: str = "uid",
    health_col: str = "health_status",
    save: bool = True,
) -> None:
    if feature not in df.columns:
        print(f"[WARN] Column '{feature}' not in DataFrame")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    color_map = {s: COLORS[s] for s in COLORS if s in df[health_col].values} if health_col in df.columns else {}

    if color_map:
        for status, group in df.groupby(health_col):
            c = COLORS.get(status, "gray")
            ax.scatter(group.index, group[feature], c=c, s=8, label=status, alpha=0.7)
    else:
        ax.plot(df.index, df[feature], color=COLORS["accent"], linewidth=1)

    ax.set_xlabel("Observation index")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature.replace('_', ' ').title()} Trend", fontsize=13, fontweight="bold")
    if color_map:
        ax.legend(title="Health Status", markerscale=2)
    fig.tight_layout()
    if save:
        out = PLOTS_DIR / f"trend_{feature}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Trend plot saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Health Score distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_health_distribution(df: pd.DataFrame, save: bool = True) -> None:
    if "health_score" not in df.columns:
        print("[WARN] 'health_score' column not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Histogram
    axes[0].hist(df["health_score"], bins=30, color=COLORS["accent"], edgecolor="white")
    for threshold, label in [(80, "Good"), (60, "Warning"), (40, "Degraded")]:
        axes[0].axvline(threshold, color=COLORS[label], linestyle="--", linewidth=1.2, label=label)
    axes[0].set_xlabel("Health Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Health Score Distribution", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)

    # Status pie chart
    if "health_status" in df.columns:
        counts = df["health_status"].value_counts()
        pie_colors = [COLORS.get(s, "gray") for s in counts.index]
        axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%",
                    colors=pie_colors, startangle=140)
        axes[1].set_title("Asset Health Status", fontsize=12, fontweight="bold")

    fig.suptitle("Asset Fleet Health Overview", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save:
        out = PLOTS_DIR / "health_distribution.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Health distribution saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, cols: list | None = None, save: bool = True) -> None:
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in cols if "id" not in c.lower() and "uid" not in c.lower()][:12]

    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        out = PLOTS_DIR / "correlation_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Correlation heatmap saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fault type bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_fault_distribution(df: pd.DataFrame, col: str = "fault_type", save: bool = True) -> None:
    if col not in df.columns:
        col = "predicted_fault" if "predicted_fault" in df.columns else None
    if col is None:
        print("[WARN] No fault column found")
        return

    counts = df[col].value_counts()
    bar_colors = [COLORS.get("Critical", COLORS["accent"]) if v != "Normal" else COLORS["Good"]
                  for v in counts.index]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_xlabel("Fault Type")
    ax.set_ylabel("Count")
    ax.set_title("Fault Type Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        out = PLOTS_DIR / "fault_distribution.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Fault distribution saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dashboard summary figure (multi-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard_summary(df: pd.DataFrame, save: bool = True) -> None:
    """4-panel overview figure suitable for reports."""
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: health score histogram
    ax0 = fig.add_subplot(gs[0, 0])
    if "health_score" in df.columns:
        ax0.hist(df["health_score"], bins=25, color=COLORS["accent"], edgecolor="white")
        for thr, lbl in [(80, "Good"), (60, "Warning"), (40, "Degraded")]:
            ax0.axvline(thr, color=COLORS[lbl], linestyle="--", linewidth=1.2)
        ax0.set_title("Health Score Distribution", fontweight="bold")
        ax0.set_xlabel("Score")
        ax0.set_ylabel("Count")

    # Panel B: status pie
    ax1 = fig.add_subplot(gs[0, 1])
    if "health_status" in df.columns:
        counts = df["health_status"].value_counts()
        pie_colors = [COLORS.get(s, "gray") for s in counts.index]
        ax1.pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=pie_colors, startangle=140)
        ax1.set_title("Health Status Breakdown", fontweight="bold")

    # Panel C: fault distribution
    ax2 = fig.add_subplot(gs[1, 0])
    fault_col = "fault_type" if "fault_type" in df.columns else "predicted_fault" if "predicted_fault" in df.columns else None
    if fault_col:
        counts2 = df[fault_col].value_counts().head(8)
        bar_cols = [COLORS["Good"] if v == "Normal" else COLORS["Critical"] for v in counts2.index]
        ax2.barh(counts2.index[::-1], counts2.values[::-1], color=bar_cols[::-1])
        ax2.set_title("Fault Type Distribution", fontweight="bold")
        ax2.set_xlabel("Count")

    # Panel D: sensor trend (rotational speed)
    ax3 = fig.add_subplot(gs[1, 1])
    spd_col = "rotational_speed_rpm"
    if spd_col in df.columns:
        sample = df[spd_col].head(300)
        ax3.plot(sample.values, color=COLORS["primary"], linewidth=0.8)
        ax3.set_title("Rotational Speed Trend (first 300 obs.)", fontweight="bold")
        ax3.set_xlabel("Observation")
        ax3.set_ylabel("Speed (rpm)")

    fig.suptitle(
        f"Condition Monitoring Dashboard — {CFG['project']['name']}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    if save:
        out = PLOTS_DIR / "dashboard_summary.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Dashboard summary saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Visualization — Condition Monitoring Project")
    print("=" * 60)

    # Demo with synthetic signal
    np.random.seed(0)
    t = np.linspace(0, 1, FS)
    demo_sig = np.sin(2 * np.pi * 60 * t) + 0.3 * np.random.randn(FS)
    demo_sig[np.random.choice(FS, 30, replace=False)] += 2.5   # impulses

    plot_waveform(demo_sig, title="Demo Vibration Waveform — Bearing Fault Simulation")
    plot_fft(demo_sig, title="Demo FFT Spectrum — Bearing Fault Simulation")

    # Load scored data if available
    scored_path = PROC_DIR / "ai4i_health_scored.csv"
    if scored_path.exists():
        df = pd.read_csv(scored_path)
        plot_health_distribution(df)
        plot_fault_distribution(df)
        plot_dashboard_summary(df)

        for feat in ["rotational_speed_rpm", "torque_Nm", "tool_wear_min"]:
            if feat in df.columns:
                plot_feature_trend(df, feat)

        num_cols = [c for c in df.select_dtypes("number").columns
                    if "id" not in c.lower() and "uid" not in c.lower()]
        if len(num_cols) >= 4:
            plot_correlation_heatmap(df, cols=num_cols[:10])

    print(f"\n[OK]   All plots saved to {PLOTS_DIR}")
