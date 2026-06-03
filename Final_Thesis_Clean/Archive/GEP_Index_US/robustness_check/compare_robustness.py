#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_robustness.py

Compares the original GEP index (min-2, from INDEX/data) against three
robustness-check variants (min-1, min-3, min-4) using:
  1. Overlaid monthly time series (all normalized to 100)
  2. Pearson & Spearman correlation table (levels + first differences)
  3. Rolling 24-month correlation with the original
  4. Scatter matrix (original vs each variant)
  5. Summary statistics printed to console

Outputs (saved to robustness_check/):
  GEP_Robustness_Overlay_Monthly.png
  GEP_Robustness_Correlations_Rolling.png
  GEP_Robustness_Scatter_Matrix.png
  GEP_Robustness_Corr_Heatmap.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/GEP_Index_US/robustness_check"
DATA_DIR = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/GEP_Index_US/INDEX/data"

# ── Load monthly data ──────────────────────────────────────────────────────────
def load_monthly(path, col):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    return df[["month", col]].set_index("month")[col]

orig   = load_monthly(os.path.join(DATA_DIR, "GEP_Monthly_Robust_min2.csv"), "GEP_monthly")
upd    = load_monthly(os.path.join(BASE, "GEP_Monthly_Updated.csv"),          "GEP_monthly")
min3   = load_monthly(os.path.join(BASE, "GEP_Monthly_Robust_min3.csv"),      "GEP_monthly")
min4   = load_monthly(os.path.join(BASE, "GEP_Monthly_Robust_min4.csv"),      "GEP_monthly")

# ── Normalize each series to 100 (avg = 100) ──────────────────────────────────
def norm100(s):
    return s / s.mean() * 100

orig_n = norm100(orig).rename("Baseline (min-2)")
upd_n  = norm100(upd).rename("Min-1")
min3_n = norm100(min3).rename("Min-3")
min4_n = norm100(min4).rename("Min-4")

# Align on common dates
panel = pd.concat([orig_n, upd_n, min3_n, min4_n], axis=1).dropna()

COLORS = {
    "Baseline (min-2)": "#152F5F",
    "Min-1":            "#E74C3C",
    "Min-3":            "#27AE60",
    "Min-4":            "#E67E22",
}
STYLES = {
    "Baseline (min-2)": ("-",  2.0),
    "Min-1":            ("--", 1.4),
    "Min-3":            ("-.", 1.4),
    "Min-4":            (":",  1.6),
}

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Overlaid monthly time series
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5))

for col in panel.columns:
    ls, lw = STYLES[col]
    ax.plot(panel.index, panel[col],
            color=COLORS[col], linewidth=lw, linestyle=ls,
            alpha=0.9, label=col)

ax.axhline(100, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
ax.set_title("GEP Monthly Index — Baseline vs. Robustness Checks (avg = 100)", fontsize=13, pad=12)
ax.set_ylabel("GEP Index (avg = 100)", fontsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, framealpha=0.8, loc="upper left")
plt.tight_layout()
out = os.path.join(BASE, "GEP_Robustness_Overlay_Monthly.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Rolling 24-month Pearson correlation with baseline
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 4))

for col in ["Min-1", "Min-3", "Min-4"]:
    roll_corr = panel[col].rolling(24).corr(panel["Baseline (min-2)"])
    ls, lw = STYLES[col]
    ax.plot(panel.index, roll_corr,
            color=COLORS[col], linewidth=lw, linestyle=ls,
            alpha=0.9, label=col)

ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
ax.axhline(0.9, color="#AAAAAA", linewidth=0.5, linestyle=":", alpha=0.6)
ax.set_ylim(0.5, 1.05)
ax.set_title("Rolling 24-month Pearson Correlation with Baseline GEP (min-2)", fontsize=13, pad=12)
ax.set_ylabel("Pearson r", fontsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, framealpha=0.8, loc="lower left")
plt.tight_layout()
out = os.path.join(BASE, "GEP_Robustness_Correlations_Rolling.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Scatter matrix: baseline vs each variant
# ══════════════════════════════════════════════════════════════════════════════
variants = ["Min-1", "Min-3", "Min-4"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, col in zip(axes, variants):
    x = panel["Baseline (min-2)"]
    y = panel[col]
    ax.scatter(x, y, s=10, alpha=0.35, color=COLORS[col], linewidths=0)

    # OLS fit line
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, intercept + slope * x_line,
            color="#333333", linewidth=1.2, linestyle="--")

    ax.set_xlabel("Baseline GEP (min-2, avg=100)", fontsize=9)
    ax.set_ylabel(f"{col} (avg=100)", fontsize=9)
    ax.set_title(f"r = {r:.3f}   slope = {slope:.3f}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

fig.suptitle("Scatter Plots: Baseline vs. Robustness Variants (monthly, normalized to 100)",
             fontsize=12, y=1.01)
plt.tight_layout()
out = os.path.join(BASE, "GEP_Robustness_Scatter_Matrix.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Correlation heatmap (levels + first differences)
# ══════════════════════════════════════════════════════════════════════════════
def corr_table(df):
    cols = df.columns.tolist()
    pearson  = df.corr(method="pearson")
    spearman = df.corr(method="spearman")
    return pearson, spearman

pearson_lv,  spearman_lv  = corr_table(panel)
pearson_fd,  spearman_fd  = corr_table(panel.diff().dropna())

try:
    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    kw = dict(annot=True, fmt=".3f", cmap="Blues", vmin=0.7, vmax=1.0,
              linewidths=0.5, annot_kws={"size": 10})
    sns.heatmap(pearson_lv,  ax=axes[0], **kw)
    axes[0].set_title("Pearson r — Levels", fontsize=12)
    sns.heatmap(pearson_fd,  ax=axes[1], **kw)
    axes[1].set_title("Pearson r — First Differences", fontsize=12)
    plt.suptitle("Correlation Heatmaps: GEP Robustness Variants (monthly)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(BASE, "GEP_Robustness_Corr_Heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
except ImportError:
    print("seaborn not installed — skipping heatmap (pip install seaborn)")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE — Summary statistics + full correlation tables
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY STATISTICS (monthly, normalized to avg=100)")
print("="*70)

stats_rows = []
for col in panel.columns:
    s = panel[col]
    fd = s.diff().dropna()
    stats_rows.append({
        "Index":      col,
        "Mean":       f"{s.mean():.1f}",
        "Std":        f"{s.std():.1f}",
        "CV (%)":     f"{s.std()/s.mean()*100:.1f}",
        "Skewness":   f"{s.skew():.2f}",
        "Kurtosis":   f"{s.kurtosis():.2f}",
        "Max":        f"{s.max():.1f}",
        "Std(Δ)":     f"{fd.std():.1f}",
    })

stats_df = pd.DataFrame(stats_rows).set_index("Index")
print(stats_df.to_string())

print("\n" + "="*70)
print("PEARSON CORRELATIONS — Levels")
print("="*70)
print(pearson_lv.round(4).to_string())

print("\n" + "="*70)
print("SPEARMAN CORRELATIONS — Levels")
print("="*70)
print(spearman_lv.round(4).to_string())

print("\n" + "="*70)
print("PEARSON CORRELATIONS — First Differences")
print("="*70)
print(pearson_fd.round(4).to_string())

print("\n" + "="*70)
print("SPEARMAN CORRELATIONS — First Differences")
print("="*70)
print(spearman_fd.round(4).to_string())

# Pairwise full stats with baseline
print("\n" + "="*70)
print("PAIRWISE STATS vs BASELINE (min-2)")
print("="*70)
for col in variants:
    x = panel["Baseline (min-2)"]
    y = panel[col]
    r_p, p_p   = stats.pearsonr(x, y)
    r_s, p_s   = stats.spearmanr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)
    fd_x = x.diff().dropna()
    fd_y = y.diff().dropna()
    r_fd, _ = stats.pearsonr(fd_x, fd_y)
    print(f"\n  {col}:")
    print(f"    Pearson r (levels)      = {r_p:.4f}  (p={p_p:.2e})")
    print(f"    Spearman r (levels)     = {r_s:.4f}  (p={p_s:.2e})")
    print(f"    Pearson r (Δ monthly)   = {r_fd:.4f}")
    print(f"    OLS slope               = {slope:.4f}  intercept = {intercept:.4f}")
    print(f"    Volatility ratio σ/σ₀   = {y.std()/x.std():.4f}")
