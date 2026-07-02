#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_index.py

GEP Robust min-2 index — main plots + descriptive statistics.

DATA layout (relative to this script):
  data/gep/GEP_Monthly_Robust_min2.csv
  data/gep/GEP_Daily_Robust_min2.csv

Outputs saved to output/index/
  GEP_Monthly_Robust_min2_norm100.png
  GEP_Daily_Robust_min2_norm100.png
  GEP_Daily_2025_Zoom_norm100.png
  gep_summary_rolling_vol.png
  gep_summary_distribution.png
  gep_summary_acf_pacf.png
  gep_summary_annual.png
  gep_summary_heatmap.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import scipy.stats as stats
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
warnings.filterwarnings("ignore")

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()
REPO = next((p for p in [HERE, *HERE.parents] if (p / "data" / "gep_us").exists()), HERE.parent)
DATA = REPO / "data" / "gep_us"
OUT  = REPO / "analysis" / "output" / "index"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
monthly = pd.read_csv(DATA / "GEP_Monthly_Robust_min2.csv")
monthly["month"] = pd.to_datetime(monthly["month"])

daily = pd.read_csv(DATA / "GEP_Daily_Robust_min2.csv")
daily["date"] = pd.to_datetime(daily["date"])

# Normalize to 100 (mean = 100)
daily_obs  = daily[daily["n_articles"] > 0].copy()
daily_mean = daily_obs["GEP_daily"].mean()
daily_obs["gep_norm"]  = daily_obs["GEP_daily"] / daily_mean * 100
monthly["gep_norm"]    = monthly["GEP_monthly"] / daily_mean * 100  # same scale as daily
monthly_mean = monthly["GEP_monthly"].mean()
monthly["gep_norm_mo"] = monthly["GEP_monthly"] / monthly_mean * 100  # monthly-scaled

gep_d = daily_obs["GEP_daily"]
gep_m = monthly["GEP_monthly"]

# ─────────────────────────────────────────────────────────────────────────────
# Event dictionaries (Curated for C&I Academic Style)
# ─────────────────────────────────────────────────────────────────────────────

# Refined list of major, structurally significant global shocks +
# geoeconomic-pressure-specific events (sanctions, export controls,
# financial coercion, trade coercion — mirroring the GTM topic dictionaries)
EVENTS = [
    ("1996-08-05", "Iran-Libya Sanctions Act"),
    ("1997-07-02", "Asian Financial Crisis"),
    ("1998-08-17", "Russian Ruble Crisis"),
    ("2001-09-11", "9/11 Attacks"),
    ("2003-03-20", "Iraq War"),
    ("2006-10-09", "N. Korea Nuclear Test"),
    ("2008-09-15", "Lehman Brothers (GFC)"),
    ("2010-07-01", "Iran Sanctions Act (CISADA)"),
    ("2011-08-05", "US Credit Downgrade"),
    ("2012-03-17", "Iran Banks Cut From SWIFT (Initial)"),
    ("2014-03-18", "Crimea Annexation"),
    ("2016-06-23", "Brexit Referendum"),
    ("2017-08-02", "CAATSA Sanctions Act"),
    ("2018-03-23", "Section 232 Steel/Aluminum Tariffs"),
    ("2018-05-08", "US Exits Iran Nuclear Deal"),
    ("2018-07-06", "US–China Trade War"),
    ("2018-11-05", "Iran Banks Cut From SWIFT (Blanket)"),
    ("2019-05-10", "Trade War Escalation & Huawei Entity List Ban", "10-16/05/2019"),
    ("2020-03-11", "COVID-19 Pandemic"),
    ("2020-08-17", "Huawei Chip Export Rule Tightened"),
    ("2022-02-24", "Russia Invades Ukraine & Cut From SWIFT", "24-26/02/2022"),
    ("2022-06-21", "Xinjiang Forced-Labor Import Ban (UFLPA)"),
    ("2022-10-07", "US Chip Export Controls on China"),
    ("2023-10-17", "AI Chip Export Rules Tightened"),
    ("2024-05-14", "US 100% Tariffs on Chinese EVs"),
    ("2025-02-01", "Canada/Mexico Tariffs Announced"),
    ("2025-04-02", "Liberation Day Tariffs"),
]

TARIFF_2025 = [
    ("2025-01-20", "Trump\ninauguration"),
    ("2025-02-01", "25% tariffs on\nCanada & Mexico"),
    ("2025-04-02", "Liberation Day\ntariffs"),
    ("2025-05-12", "US–China\nGeneva truce"),
    ("2025-06-05", "US/Israel strike Iran\n(oil spike)"),
    ("2025-08-07", "10–41% broad US\ntariffs take effect"),
    ("2025-10-10", "US–China tariff\nescalation"),
]

KEY_EVENTS = {
    "2001-09-11": "9/11",
    "2003-03-20": "Iraq War",
    "2008-09-15": "GFC",
    "2018-07-06": "Trade War",
    "2020-03-11": "COVID-19",
    "2022-02-24": "Ukraine",
    "2025-04-02": "Liberation Day",
}

# Classic C&I dark navy blue
COL_GEP = "#2b4c8c"
COL_V30, COL_V90 = "#E05C2A", "#5A4FCF"


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Monthly index (line plot) normalized to 100
# ═════════════════════════════════════════════════════════════════════════════

ANNOTATIONS = {
    "2001-09": ("9/11",                    (0, 15),   False),
    "2003-03": ("Iraq\nWar",               (30, 10),  True),
    "2008-09": ("GFC",                     (0, -55),  True),
    "2014-03": ("Crimea\nannexation",      (0, 45),   True),
    "2018-06": ("US–China\ntariffs",       (-35, 30), True),
    "2019-05": ("Trade war\nescalation",   (35, -25), True),
    "2020-03": ("COVID-19",                (0, -40),  True),
    "2022-02": ("Russia invades\nUkraine", (-25, 45), True),
    "2025-04": ("Liberation Day\ntariffs", (20, 15),  False),
}

fig, ax = plt.subplots(figsize=(12, 6.5))
ax.plot(monthly["month"], monthly["gep_norm_mo"], color=COL_GEP, linewidth=1.8, alpha=0.95)

# Setup Y-Axis (Log scale, blue labels)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.set_yticks([50, 100, 200, 400, 600])
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.tick_params(axis="y", colors=COL_GEP, labelsize=11, direction="out")

# Setup X-Axis
ax.set_xlim(pd.Timestamp("1996-01-01"), pd.Timestamp("2026-06-01"))
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", labelsize=11, direction="out")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black") 

for month_str, (label, offset, use_arrow) in ANNOTATIONS.items():
    row = monthly[monthly["month"].dt.strftime("%Y-%m") == month_str]
    if row.empty: continue
        
    peak_date, peak_val = row["month"].values[0], row["gep_norm_mo"].values[0]
    ax.scatter(peak_date, peak_val, s=25, color=COL_GEP, zorder=5, linewidths=0)

    text_kwargs = dict(
        text=label, xy=(peak_date, peak_val), xytext=offset,
        textcoords="offset points", fontsize=11, ha="center", va="center", color="black"
    )
    if use_arrow:
        ax.annotate(**text_kwargs, arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, mutation_scale=8))
    else:
        ax.annotate(**text_kwargs)

plt.tight_layout()
plt.savefig(OUT / "GEP_Monthly_Robust_min2_norm100.png", dpi=300, bbox_inches="tight")
print("Saved: GEP_Monthly_Robust_min2_norm100.png")
plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Daily index (horizontal dot plot) C&I Style Refactor
# ═════════════════════════════════════════════════════════════════════════════
def find_nearby_peak(date_str, window_days=15):
    dt = pd.to_datetime(date_str)
    sub = daily_obs[(daily_obs["date"] >= dt - pd.Timedelta(days=window_days)) &
                    (daily_obs["date"] <= dt + pd.Timedelta(days=window_days))]
    if sub.empty: return dt, 0.0
    idx = sub["gep_norm"].idxmax()
    return sub.loc[idx, "date"], sub.loc[idx, "gep_norm"]

raw_peaks = []
for event in EVENTS:
    d, lbl = event[0], event[1]
    date_str = event[2] if len(event) > 2 else pd.to_datetime(d).strftime("%d/%m/%Y")
    peak_date, peak_score = find_nearby_peak(d)
    label_text = f"{date_str}: {lbl}"
    raw_peaks.append((peak_date, peak_score, label_text, pd.to_datetime(d)))

# Collision avoidance: enforce a minimum vertical gap (in days of axis
# space) between labels so dense clusters (e.g. 2018-2020, Feb 2022) fan
# out instead of overlapping. Two-pass forward/backward sweep centers each
# cluster around its own natural dates instead of cascading drift forward
# across the whole timeline. Each label keeps a thin leader line back to
# its true dot position.
MIN_LABEL_GAP_DAYS = 245
raw_peaks_sorted = sorted((p for p in raw_peaks if p[1] != 0.0), key=lambda p: p[3])
placed = [mdates.date2num(p[3]) for p in raw_peaks_sorted]
for i in range(1, len(placed)):
    placed[i] = max(placed[i], placed[i - 1] + MIN_LABEL_GAP_DAYS)
for i in range(len(placed) - 2, -1, -1):
    placed[i] = min(placed[i], placed[i + 1] - MIN_LABEL_GAP_DAYS)
label_placements = [
    (peak_date, peak_score, label_text, mdates.num2date(p))
    for (peak_date, peak_score, label_text, _), p in zip(raw_peaks_sorted, placed)
]

# Much larger figure so 27 events have room without crowding the data line
fig, ax = plt.subplots(figsize=(16, 34))

# Plot Daily GEP as subtle background dots
ax.scatter(daily_obs["gep_norm"], daily_obs["date"],
           s=12, color="#d3d9e8", alpha=0.6, linewidths=0, zorder=2, label="Daily GEP")

# Plot Monthly GEP as the structural line
ax.plot(monthly["gep_norm_mo"], monthly["month"],
        color=COL_GEP, linewidth=1.5, alpha=0.9, zorder=3, label="Monthly GEP")

# Apply Log Scale to X-Axis to match C&I style from Plot 1
ax.set_xscale("log")
ax.set_xticks([50, 100, 200, 400, 800])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.tick_params(axis="x", colors=COL_GEP, labelsize=14, direction="out")

# Y-Axis formatting (Inverted so recent dates are at the top).
# Bottom limit extends a year past the last data point to give the
# collision-avoided label stack (see below) room without touching the axis.
ax.set_ylim(pd.Timestamp("2027-06-01"), pd.Timestamp("1995-06-01"))
ax.yaxis.set_major_locator(mdates.YearLocator(2))
ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="y", labelsize=14, direction="out", colors="black")

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")
ax.set_xlabel("GEP Index (Log Scale)", fontsize=15, labelpad=12)

# Annotate the peaks. Text sits at its collision-avoided placement date
# (data y) offset slightly right of the dot (offset-points x); a thin
# leader line connects text back to the dot's true date/value when the
# two diverge.
for peak_date, peak_score, label_text, label_date in label_placements:
    # Solid dot marker at the true event/peak position
    ax.scatter(peak_score, peak_date, s=36, color=COL_GEP, zorder=5, linewidths=0)
    ax.annotate(
        label_text, xy=(peak_score, peak_date), xycoords="data",
        xytext=(20, mdates.date2num(label_date)), textcoords=("offset points", "data"),
        fontsize=18, ha="left", va="center", color="black", zorder=6,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7, alpha=0.7,
                         shrinkA=2, shrinkB=2)
    )

plt.tight_layout()
plt.savefig(OUT / "GEP_Daily_Robust_min2_norm100.png", dpi=300, bbox_inches="tight")
print("Saved: GEP_Daily_Robust_min2_norm100.png")
plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2b — Same daily dot plot, zoomed to 2017 onward (smaller figure, since
# fewer events need to share the space)
# ═════════════════════════════════════════════════════════════════════════════
EVENTS_2017 = [e for e in EVENTS if pd.to_datetime(e[0]) >= pd.Timestamp("2017-01-01")]

raw_peaks_zoom = []
for event in EVENTS_2017:
    d, lbl = event[0], event[1]
    date_str = event[2] if len(event) > 2 else pd.to_datetime(d).strftime("%d/%m/%Y")
    peak_date, peak_score = find_nearby_peak(d)
    label_text = f"{date_str}: {lbl}"
    raw_peaks_zoom.append((peak_date, peak_score, label_text, pd.to_datetime(d)))

MIN_LABEL_GAP_DAYS_ZOOM = 150
raw_peaks_zoom_sorted = sorted((p for p in raw_peaks_zoom if p[1] != 0.0), key=lambda p: p[3])
placed_zoom = [mdates.date2num(p[3]) for p in raw_peaks_zoom_sorted]
for i in range(1, len(placed_zoom)):
    placed_zoom[i] = max(placed_zoom[i], placed_zoom[i - 1] + MIN_LABEL_GAP_DAYS_ZOOM)
for i in range(len(placed_zoom) - 2, -1, -1):
    placed_zoom[i] = min(placed_zoom[i], placed_zoom[i + 1] - MIN_LABEL_GAP_DAYS_ZOOM)
label_placements_zoom = [
    (peak_date, peak_score, label_text, mdates.num2date(p))
    for (peak_date, peak_score, label_text, _), p in zip(raw_peaks_zoom_sorted, placed_zoom)
]

daily_obs_zoom = daily_obs[(daily_obs["date"] >= "2017-01-01") & (daily_obs["date"] <= "2025-12-31")]
monthly_zoom = monthly[(monthly["month"] >= "2017-01-01") & (monthly["month"] <= "2025-12-31")]

fig, ax = plt.subplots(figsize=(11, 20))

ax.scatter(daily_obs_zoom["gep_norm"], daily_obs_zoom["date"],
           s=12, color="#d3d9e8", alpha=0.6, linewidths=0, zorder=2, label="Daily GEP")
ax.plot(monthly_zoom["gep_norm_mo"], monthly_zoom["month"],
        color=COL_GEP, linewidth=1.5, alpha=0.9, zorder=3, label="Monthly GEP")

ax.set_xscale("log")
ax.set_xticks([50, 100, 200, 400, 800])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.tick_params(axis="x", colors=COL_GEP, labelsize=12, direction="out")

ax.set_ylim(pd.Timestamp("2025-12-31"), pd.Timestamp("2016-09-01"))
ax.yaxis.set_major_locator(mdates.YearLocator(1))
ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="y", labelsize=12, direction="out", colors="black")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")
ax.set_xlabel("GEP Index (Log Scale)", fontsize=13, labelpad=10)

for peak_date, peak_score, label_text, label_date in label_placements_zoom:
    ax.scatter(peak_score, peak_date, s=30, color=COL_GEP, zorder=5, linewidths=0)
    ax.annotate(
        label_text, xy=(peak_score, peak_date), xycoords="data",
        xytext=(16, mdates.date2num(label_date)), textcoords=("offset points", "data"),
        fontsize=15, ha="left", va="center", color="black", zorder=6,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, alpha=0.7,
                         shrinkA=2, shrinkB=2)
    )

plt.tight_layout()
plt.savefig(OUT / "GEP_Daily_Robust_min2_norm100_2017plus.png", dpi=300, bbox_inches="tight")
print("Saved: GEP_Daily_Robust_min2_norm100_2017plus.png")
plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — 2025 zoom (Caldara & Iacoviello style)
# ═════════════════════════════════════════════════════════════════════════════
daily_2025 = daily_obs[(daily_obs["date"] >= "2025-01-01") &
                       (daily_obs["date"] <= "2025-12-31")].copy()

fig, ax = plt.subplots(figsize=(12, 5.5))

if not daily_2025.empty:
    roll = daily_2025.set_index("date")["gep_norm"].rolling("7D").mean()
    ax.plot(roll.index, roll.values, color=COL_GEP, linewidth=1.8, alpha=0.95, zorder=3)

ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.set_yticks([50, 100, 200, 400])
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.tick_params(axis="y", colors=COL_GEP, labelsize=11, direction="out")

ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.tick_params(axis="x", labelsize=11, direction="out")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black") 

y_top, y_bottom = 0.95, 0.05
for i, (date_str, label) in enumerate(TARIFF_2025):
    dt = pd.to_datetime(date_str)
    if dt < pd.Timestamp("2025-01-01") or dt > pd.Timestamp("2025-12-31"): continue
    
    ax.axvline(dt, color="black", linewidth=0.8, linestyle=":", alpha=0.5, zorder=1)
    y_frac = y_top if i % 2 == 0 else y_bottom
    va = "top" if i % 2 == 0 else "bottom"

    ax.text(dt - pd.Timedelta(days=2), y_frac, label, transform=ax.get_xaxis_transform(),
            fontsize=10, ha="right", va=va, color="black", rotation=90)

plt.tight_layout()
plt.savefig(OUT / "GEP_Daily_2025_Zoom_norm100.png", dpi=300, bbox_inches="tight")
print("Saved: GEP_Daily_2025_Zoom_norm100.png")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS & OTHER PLOTS 
# ═════════════════════════════════════════════════════════════════════════════
def summary_stats(s, label):
    adf_stat, adf_p, *_ = adfuller(s.dropna(), autolag="AIC")
    print(f"\n{'═'*55}\n  {label}\n{'─'*55}")
    print(f"  Observations   : {len(s):,}")
    print(f"  Mean           : {s.mean():.6f}")
    print(f"  Median         : {s.median():.6f}")
    print(f"  Std dev        : {s.std():.6f}")
    print(f"  Min / Max      : {s.min():.6f} / {s.max():.6f}")
    print(f"  Skewness       : {s.skew():.4f}")
    print(f"  Excess kurtosis: {s.kurt():.4f}")
    print(f"  ADF stat       : {adf_stat:.4f}   p = {adf_p:.4f} "
          f"{'[stationary]' if adf_p < 0.05 else '[non-stationary]'}")

print("\n╔══════════════════════════════════════════════╗")
print("║     GEP Robust min-2 — Summary Statistics    ║")
print("╚══════════════════════════════════════════════╝")

gep_d_idx = gep_d.copy(); gep_d_idx.index = daily_obs["date"]
summary_stats(gep_d_idx, "Daily GEP (trading days with >0 articles)")

gep_m_idx = gep_m.copy(); gep_m_idx.index = monthly["month"]
summary_stats(gep_m_idx, "Monthly GEP")


daily_obs["roll_vol_30d"]  = gep_d.rolling(30,  min_periods=20).std()
daily_obs["roll_vol_90d"]  = gep_d.rolling(90,  min_periods=60).std()
daily_obs["roll_mean_90d"] = gep_d.rolling(90,  min_periods=60).mean()

annual = (daily_obs.set_index("date")["GEP_daily"]
          .resample("YS").agg(mean="mean", std="std", q90=lambda x: x.quantile(0.90)).dropna())

def add_key_events(ax, dates, y_top, fontsize=6.5):
    for ds, label in KEY_EVENTS.items():
        xd = pd.to_datetime(ds)
        if pd.Timestamp(dates.min()) <= xd <= pd.Timestamp(dates.max()):
            ax.axvline(xd, color="gray", lw=0.6, ls="--", alpha=0.55)
            ax.text(xd, y_top, label, rotation=90, fontsize=fontsize, va="top", color="#555555", ha="right")

fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2, 1.2]})
ax1, ax2, ax3 = axes

ax1.plot(daily_obs["date"], gep_d, color=COL_GEP, lw=0.5, alpha=0.6, label="GEP daily")
ax1.plot(daily_obs["date"], daily_obs["roll_mean_90d"], color="#1A3F7A", lw=1.4, label="90-day rolling mean")
add_key_events(ax1, daily_obs["date"], y_top=gep_d.max() * 0.95)
ax1.set_ylabel("GEP index", fontsize=10)
ax1.legend(fontsize=9, framealpha=0.75, loc="upper left")
ax1.spines[["top", "right"]].set_visible(False)

ax2.fill_between(daily_obs["date"], daily_obs["roll_vol_30d"], alpha=0.35, color=COL_V30)
ax2.plot(daily_obs["date"], daily_obs["roll_vol_30d"], color=COL_V30, lw=0.7, label="30-day rolling σ")
add_key_events(ax2, daily_obs["date"], y_top=daily_obs["roll_vol_30d"].max() * 0.92)
ax2.set_ylabel("σ (30-day)", fontsize=9)
ax2.legend(fontsize=8, framealpha=0.75)
ax2.spines[["top", "right"]].set_visible(False)

ax3.fill_between(daily_obs["date"], daily_obs["roll_vol_90d"], alpha=0.35, color=COL_V90)
ax3.plot(daily_obs["date"], daily_obs["roll_vol_90d"], color=COL_V90, lw=0.9, label="90-day rolling σ")
add_key_events(ax3, daily_obs["date"], y_top=daily_obs["roll_vol_90d"].max() * 0.92)
ax3.set_ylabel("σ (90-day)", fontsize=9)
ax3.legend(fontsize=8, framealpha=0.75)
ax3.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "gep_summary_rolling_vol.png", dpi=150, bbox_inches="tight")
print("\nSaved: gep_summary_rolling_vol.png")
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for (ax_hist, ax_qq), series, label in [
    ((axes[0, 0], axes[0, 1]), gep_d, "Daily GEP"),
    ((axes[1, 0], axes[1, 1]), gep_m, "Monthly GEP"),
]:
    s = series.dropna()
    ax_hist.hist(s, bins=80, density=True, color=COL_GEP, alpha=0.45, edgecolor="white", linewidth=0.3)
    xs = np.linspace(s.min(), s.max(), 400)
    ax_hist.plot(xs, stats.gaussian_kde(s)(xs), color="#1A3F7A", lw=2, label="KDE")
    mu, sigma = s.mean(), s.std()
    ax_hist.plot(xs, stats.norm.pdf(xs, mu, sigma), color=COL_V30, lw=1.5, ls="--", label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})")
    ax_hist.axvline(mu, color="black", lw=0.8, ls="--", alpha=0.7)
    ax_hist.set_xlabel("GEP value", fontsize=9)
    ax_hist.set_ylabel("Density", fontsize=9)
    ax_hist.legend(fontsize=8, framealpha=0.75)
    ax_hist.spines[["top", "right"]].set_visible(False)

    (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
    ax_qq.scatter(osm, osr, s=4, alpha=0.4, color=COL_GEP, label="Quantiles")
    x_line = np.array([osm.min(), osm.max()])
    ax_qq.plot(x_line, slope * x_line + intercept, color=COL_V30, lw=1.5, label=f"Normal fit  r={r:.4f}")
    ax_qq.set_xlabel("Theoretical quantiles", fontsize=9)
    ax_qq.set_ylabel("Sample quantiles", fontsize=9)
    ax_qq.legend(fontsize=8, framealpha=0.75)
    ax_qq.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "gep_summary_distribution.png", dpi=150, bbox_inches="tight")
print("Saved: gep_summary_distribution.png")
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(14, 8))
plot_acf( gep_d.dropna(), lags=60, ax=axes[0, 0], color=COL_GEP)
plot_pacf(gep_d.dropna(), lags=60, ax=axes[0, 1], color=COL_GEP, method="ywm")
plot_acf( gep_m.dropna(), lags=36, ax=axes[1, 0], color=COL_GEP)
plot_pacf(gep_m.dropna(), lags=36, ax=axes[1, 1], color=COL_GEP, method="ywm")
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Lag", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "gep_summary_acf_pacf.png", dpi=150, bbox_inches="tight")
print("Saved: gep_summary_acf_pacf.png")
plt.close()


fig, ax = plt.subplots(figsize=(14, 5))
years = annual.index.year
x = np.arange(len(years))
ax.bar(x, annual["mean"], color=COL_GEP, alpha=0.75, width=0.6, label="Annual mean")
ax.errorbar(x, annual["mean"], yerr=annual["std"], fmt="none", color="#1A3F7A", capsize=3, lw=1.2, label="±1 std")
ax.plot(x, annual["q90"], color=COL_V30, lw=1.5, marker="o", ms=4, label="90th percentile")
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("GEP index", fontsize=10)
ax.legend(fontsize=9, framealpha=0.75)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "gep_summary_annual.png", dpi=150, bbox_inches="tight")
print("Saved: gep_summary_annual.png")
plt.close()


monthly["year"]    = monthly["month"].dt.year
monthly["month_n"] = monthly["month"].dt.month
pivot = monthly.pivot(index="year", columns="month_n", values="GEP_monthly")
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.35)))
im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=np.nanpercentile(pivot.values, 5), vmax=np.nanpercentile(pivot.values, 95))
ax.set_xticks(range(12)); ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index, fontsize=8)
for i in range(len(pivot)):
    for j in range(12):
        val = pivot.values[i, j]
        if not np.isnan(val): ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=5.5, color="black")
cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
cbar.set_label("GEP index", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "gep_summary_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: gep_summary_heatmap.png")
plt.close()

print("\n═══ All plots saved to output/index/ ═══")