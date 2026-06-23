#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_robustness.py

Robustness checks for the GEP index:
  1. Stacked 3-panel plot: min-1, min-3, min-4 in C&I style with annotations
  2. Comparison overlay: all four variants normalized to 100
  3. Rolling 24-month correlation with baseline (min-2)
  4. GTM v2 index plots (alternative seed words)

DATA layout (relative to this script):
  data/gep/GEP_Monthly_Robust_min2.csv    (baseline)
  data/gep/GEP_Daily_Robust_min2.csv      (baseline)
  data/robustness/GEP_Monthly_min1.csv
  data/robustness/GEP_Daily_min1.csv
  data/robustness/GEP_Monthly_Robust_min3.csv
  data/robustness/GEP_Daily_Robust_min3.csv
  data/robustness/GEP_Monthly_Robust_min4.csv
  data/robustness/GEP_Daily_Robust_min4.csv
  data/robustness/GEP_Monthly_gtm_v2.csv  (GTM v2 — optional)
  data/robustness/GEP_Daily_gtm_v2.csv    (GTM v2 — optional)

Outputs saved to output/robustness/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

matplotlib.rcParams['font.family'] = 'serif'
warnings.filterwarnings("ignore")

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()
REPO   = next((p for p in [HERE, *HERE.parents] if (p / "data" / "gep_us").exists()), HERE.parent)
GEP    = REPO / "data" / "gep_us"
ROBUST = REPO / "data" / "robustness"
OUT    = REPO / "analysis" / "output" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Style constants (matching plot_index.py)
# ─────────────────────────────────────────────────────────────────────────────
COL_GEP = "#2b4c8c"   # dark navy

# ─────────────────────────────────────────────────────────────────────────────
# Annotation dictionaries
# ─────────────────────────────────────────────────────────────────────────────
# Used for monthly stacked panel — same peaks as plot_index.py
ANNOTATIONS = {
    "2001-09": ("9/11",                    (0,   15),  False),
    "2003-03": ("Iraq\nWar",               (30,  10),  True),
    "2008-09": ("GFC",                     (0,  -55),  True),
    "2014-03": ("Crimea\nannexation",      (0,   45),  True),
    "2018-06": ("US–China\ntariffs",       (-35, 30),  True),
    "2019-05": ("Trade war\nescalation",   (35, -25),  True),
    "2020-03": ("COVID-19",                (0,  -40),  True),
    "2022-02": ("Russia invades\nUkraine", (-25, 45),  True),
    "2025-04": ("Liberation Day\ntariffs", (20,  15),  False),
}

# Used for overlay / correlation plots
PEAKS = {
    "1997-07": "Asian Financial Crisis",
    "1998-08": "Russian Ruble Crisis",
    "2001-09": "9/11",
    "2003-03": "Iraq War",
    "2008-09": "GFC",
    "2011-08": "US credit downgrade",
    "2014-03": "Crimea annexation",
    "2018-06": "US–China tariffs",
    "2019-05": "Trade war escalation",
    "2020-03": "COVID-19",
    "2022-02": "Russia invades Ukraine",
    "2022-10": "US chip controls on China",
    "2025-04": "Liberation Day tariffs",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load and normalise a monthly CSV
# ─────────────────────────────────────────────────────────────────────────────
def load_monthly(path, col="GEP_monthly"):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    s = df.set_index("month")[col]
    return df, s / s.mean() * 100   # return full df and normalised series


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Stacked 3-panel: min-1, min-3, min-4  (C&I style)
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Stacked robustness panel (C&I style) ---")

VARIANTS = [
    ("Robust min-1", ROBUST / "GEP_Monthly_min1.csv",         "min-1"),
    ("Robust min-3", ROBUST / "GEP_Monthly_Robust_min3.csv",  "min-3"),
    ("Robust min-4", ROBUST / "GEP_Monthly_Robust_min4.csv",  "min-4"),
]

# Check all files exist before attempting
missing = [str(p) for _, p, _ in VARIANTS if not p.exists()]
if missing:
    print(f"[WARNING] Missing files for stacked panel:\n  " + "\n  ".join(missing))
else:
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    for ax, (label, path, short) in zip(axes, VARIANTS):
        df, s_norm = load_monthly(path)
        df["gep_norm_mo"] = s_norm.values

        # Main line
        ax.plot(df["month"], df["gep_norm_mo"],
                color=COL_GEP, linewidth=1.8, alpha=0.95)

        # Log y-axis matching plot_index.py style
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.set_yticks([50, 100, 200, 400])
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(axis="y", colors=COL_GEP, labelsize=10, direction="out")
        ax.set_ylabel("GEP Index\n(avg = 100)", fontsize=9, color=COL_GEP)

        # Panel label top-left
        ax.text(0.01, 0.95, label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", color="#333333")

        # Annotations — same offsets as plot_index.py
        for month_str, (ann_label, offset, use_arrow) in ANNOTATIONS.items():
            row = df[df["month"].dt.strftime("%Y-%m") == month_str]
            if row.empty:
                continue
            peak_date = row["month"].values[0]
            peak_val  = row["gep_norm_mo"].values[0]
            ax.scatter(peak_date, peak_val, s=22, color=COL_GEP,
                       zorder=5, linewidths=0)
            kwargs = dict(
                text=ann_label, xy=(peak_date, peak_val),
                xytext=offset, textcoords="offset points",
                fontsize=9, ha="center", va="center", color="black"
            )
            if use_arrow:
                ax.annotate(**kwargs,
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            lw=0.8, mutation_scale=8))
            else:
                ax.annotate(**kwargs)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")

    # Shared x-axis
    axes[-1].set_xlim(pd.Timestamp("1996-01-01"), pd.Timestamp("2026-06-01"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].tick_params(axis="x", labelsize=10, direction="out")

    fig.suptitle(
        "GEP Monthly Index — Hit-Rate Threshold Variants (1996–2025)",
        fontsize=13, y=1.002
    )

    plt.tight_layout()
    out_path = OUT / "GEP_Robustness_Stacked_Panel.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path.name}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Overlay: all four variants normalised to 100
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Robustness overlay (all variants) ---")

ALL_VARIANTS = [
    ("Robust min-1", ROBUST / "GEP_Monthly_min1.csv"),
    ("Robust min-2", GEP    / "GEP_Monthly_Robust_min2.csv"),
    ("Robust min-3", ROBUST / "GEP_Monthly_Robust_min3.csv"),
    ("Robust min-4", ROBUST / "GEP_Monthly_Robust_min4.csv"),
]

COLORS = {
    "Robust min-1": "#E74C3C",
    "Robust min-2": "#152F5F",
    "Robust min-3": "#27AE60",
    "Robust min-4": "#E67E22",
}

series = {}
for label, path in ALL_VARIANTS:
    if path.exists():
        _, s = load_monthly(path)
        series[label] = s
    else:
        print(f"[WARNING] Missing: {path.name}")

if series:
    panel = pd.concat(series, axis=1).dropna()

    print("\nCorrelation table (monthly, normalised to 100):")
    print(panel.corr().to_string())

    fig, ax = plt.subplots(figsize=(16, 6))
    for name, s in panel.items():
        ax.plot(s.index, s.values, color=COLORS.get(name, "gray"),
                linewidth=1.4, alpha=0.85, label=name)
    ax.axhline(100, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_title(
        "GEP Robustness Check — All Variants Normalized to 100 (1996–2025)",
        fontsize=13, pad=10)
    ax.set_ylabel("GEP Index (avg = 100)", fontsize=10)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out_path = OUT / "GEP_Robustness_Overlay_Monthly.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path.name}")
    plt.close()

    # ── Rolling 24-month correlation with baseline ──────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    baseline = panel.get("Robust min-2")
    if baseline is not None:
        for name, s in panel.items():
            if name == "Robust min-2":
                continue
            roll = baseline.rolling(24, min_periods=18).corr(s)
            ax.plot(roll.index, roll.values,
                    color=COLORS.get(name, "gray"), linewidth=1.2, label=name)
    ax.axhline(1, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.set_title(
        "GEP Robustness — Rolling 24-month Correlation with Baseline (min-2)",
        fontsize=13, pad=10)
    ax.set_ylabel("Pearson r", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out_path = OUT / "GEP_Robustness_Correlations_Rolling.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path.name}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — GTM v2 (alternative seed words)
# ═════════════════════════════════════════════════════════════════════════════
v2_monthly = ROBUST / "GEP_Monthly_gtm_v2.csv"

if v2_monthly.exists():
    print("\n--- GTM v2 (alternative seeds) ---")
    df_v2, s_v2 = load_monthly(v2_monthly)
    df_v2["gep_norm_mo"] = s_v2.values

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(df_v2["month"], df_v2["gep_norm_mo"],
            color=COL_GEP, linewidth=1.8, alpha=0.95)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_yticks([50, 100, 200, 400, 600])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis="y", colors=COL_GEP, labelsize=11, direction="out")

    ax.set_xlim(pd.Timestamp("1996-01-01"), pd.Timestamp("2026-06-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=11, direction="out")

    for month_str, (ann_label, offset, use_arrow) in ANNOTATIONS.items():
        row = df_v2[df_v2["month"].dt.strftime("%Y-%m") == month_str]
        if row.empty:
            continue
        peak_date = row["month"].values[0]
        peak_val  = row["gep_norm_mo"].values[0]
        ax.scatter(peak_date, peak_val, s=25, color=COL_GEP, zorder=5, linewidths=0)
        kwargs = dict(
            text=ann_label, xy=(peak_date, peak_val),
            xytext=offset, textcoords="offset points",
            fontsize=11, ha="center", va="center", color="black"
        )
        if use_arrow:
            ax.annotate(**kwargs,
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        lw=0.8, mutation_scale=8))
        else:
            ax.annotate(**kwargs)

    ax.set_title(
        "GEP Monthly Index — GTM v2 (Alternative Seeds), normalized to 100 (1996–2025)",
        fontsize=13, pad=12)
    ax.set_ylabel("GEP Index (avg = 100)", fontsize=10, color=COL_GEP)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")

    plt.tight_layout()
    out_path = OUT / "GEP_GTM_v2_Monthly_norm100.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path.name}")
    plt.close()
else:
    print(f"\n[INFO] GTM v2 file not found ({v2_monthly.name}). "
          f"Place it in data/robustness/ to generate this plot.")

print("\n═══ All robustness plots saved to output/robustness/ ═══")