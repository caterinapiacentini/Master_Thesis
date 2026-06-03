#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gep_summary_stats.py

Descriptive statistics and volatility analysis for the GEP Robust min-2 index.

Produces:
  1. gep_summary_rolling_vol.png   — index level + rolling volatility (30/90-day)
  2. gep_summary_distribution.png  — histogram + KDE + Q-Q plot (daily & monthly)
  3. gep_summary_acf_pacf.png      — ACF / PACF of daily and monthly series
  4. gep_summary_annual.png        — annual mean ± 1-std bar chart
  5. gep_summary_heatmap.png       — monthly-mean heatmap by year
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/GEP_IndeX_US/INDEX/data"
OUT_DIR  = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/GEP_IndeX_US/INDEX/gep_index"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
daily = pd.read_csv(os.path.join(DATA_DIR, "GEP_Daily_Robust_min2.csv"),
                    parse_dates=["date"])
daily = daily[daily["n_articles"] > 0].copy()
daily = daily.sort_values("date").reset_index(drop=True)

monthly = pd.read_csv(os.path.join(DATA_DIR, "GEP_Monthly_Robust_min2.csv"))
monthly["date"] = pd.to_datetime(monthly["month"].astype(str), format="%Y-%m")
monthly = monthly.sort_values("date").reset_index(drop=True)

gep_d = daily["GEP_daily"]
gep_m = monthly["GEP_monthly"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Summary statistics (printed)
# ─────────────────────────────────────────────────────────────────────────────
def summary_stats(s, label):
    adf_stat, adf_p, *_ = adfuller(s.dropna(), autolag="AIC")
    print(f"\n{'═'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Observations   : {len(s):,}")
    print(f"  Date range     : {s.index[0] if hasattr(s.index, 'min') else ''}")
    print(f"  Mean           : {s.mean():.6f}")
    print(f"  Median         : {s.median():.6f}")
    print(f"  Std dev        : {s.std():.6f}")
    print(f"  Min            : {s.min():.6f}")
    print(f"  Max            : {s.max():.6f}")
    print(f"  Skewness       : {s.skew():.4f}")
    print(f"  Excess kurtosis: {s.kurt():.4f}")
    print(f"  Pct 5 / 25     : {s.quantile(0.05):.6f}  /  {s.quantile(0.25):.6f}")
    print(f"  Pct 75 / 95    : {s.quantile(0.75):.6f}  /  {s.quantile(0.95):.6f}")
    print(f"  ADF stat       : {adf_stat:.4f}   p = {adf_p:.4f} "
          f"{'[stationary]' if adf_p < 0.05 else '[non-stationary]'}")

print("\n╔══════════════════════════════════════════════╗")
print("║     GEP Robust min-2 — Summary Statistics    ║")
print("╚══════════════════════════════════════════════╝")

gep_d_idx = gep_d.copy()
gep_d_idx.index = daily["date"]
summary_stats(gep_d_idx, "Daily GEP  (trading days with >0 articles)")

gep_m_idx = gep_m.copy()
gep_m_idx.index = monthly["date"]
summary_stats(gep_m_idx, "Monthly GEP")

# Sub-period means
periods = [
    ("1996–2001", "1996-01-01", "2001-12-31"),
    ("2002–2009", "2002-01-01", "2009-12-31"),
    ("2010–2019", "2010-01-01", "2019-12-31"),
    ("2020–2025", "2020-01-01", "2025-12-31"),
]
print(f"\n{'─'*55}")
print("  Sub-period statistics (daily)")
print(f"{'─'*55}")
for label, start, end in periods:
    sub = daily[(daily["date"] >= start) & (daily["date"] <= end)]["GEP_daily"]
    if len(sub) > 0:
        print(f"  {label}: mean={sub.mean():.5f}  std={sub.std():.5f}  "
              f"max={sub.max():.5f}  n={len(sub):,}")

# Article coverage
print(f"\n{'─'*55}")
print("  Article coverage (daily)")
print(f"{'─'*55}")
print(f"  Total articles (mean/day) : {daily['n_articles'].mean():.1f}")
print(f"  GEP articles  (mean/day)  : {daily['n_gep_articles'].mean():.1f}")
ratio = (daily["n_gep_articles"] / daily["n_articles"] * 100)
print(f"  GEP share     (mean)      : {ratio.mean():.2f}%")
print(f"  GEP share     (max)       : {ratio.max():.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Volatility measures
# ─────────────────────────────────────────────────────────────────────────────
daily["roll_vol_30d"]  = gep_d.rolling(30,  min_periods=20).std()
daily["roll_vol_90d"]  = gep_d.rolling(90,  min_periods=60).std()
daily["roll_mean_90d"] = gep_d.rolling(90,  min_periods=60).mean()

monthly["roll_vol_12m"] = gep_m.rolling(12, min_periods=9).std()
monthly["roll_mean_12m"] = gep_m.rolling(12, min_periods=9).mean()

# Annual stats
annual = (
    daily.set_index("date")["GEP_daily"]
    .resample("YS")
    .agg(mean="mean", std="std", q90=lambda x: x.quantile(0.90))
    .dropna()
)

# ─────────────────────────────────────────────────────────────────────────────
# Key events
# ─────────────────────────────────────────────────────────────────────────────
key_events = {
    "2001-09-11": "9/11",
    "2003-03-20": "Iraq War",
    "2008-09-15": "GFC",
    "2018-07-06": "Trade War",
    "2020-03-11": "COVID-19",
    "2022-02-24": "Ukraine",
    "2025-04-02": "Liberation Day",
}

def add_events(ax, dates, y_top, fontsize=6.5):
    for ds, label in key_events.items():
        xd = pd.to_datetime(ds)
        if pd.Timestamp(dates.min()) <= xd <= pd.Timestamp(dates.max()):
            ax.axvline(xd, color="gray", lw=0.6, ls="--", alpha=0.55)
            ax.text(xd, y_top, label, rotation=90, fontsize=fontsize,
                    va="top", color="#555555", ha="right")

COL_GEP  = "#378ADD"
COL_V30  = "#E05C2A"
COL_V90  = "#5A4FCF"

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Index level + rolling volatility
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True,
                         gridspec_kw={"height_ratios": [2.5, 1.2, 1.2]})

ax1, ax2, ax3 = axes

# Panel A: daily index
ax1.plot(daily["date"], gep_d, color=COL_GEP, lw=0.5, alpha=0.6, label="GEP daily")
ax1.plot(daily["date"], daily["roll_mean_90d"], color="#1A3F7A", lw=1.4,
         label="90-day rolling mean")
add_events(ax1, daily["date"], y_top=gep_d.max() * 0.95)
ax1.set_ylabel("GEP index", fontsize=10)
ax1.set_title("GEP Robust min-2 — Index Level & Rolling Volatility (1996–2025)",
              fontsize=13, pad=10)
ax1.legend(fontsize=9, framealpha=0.75, loc="upper left")
ax1.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax1.spines[["top", "right"]].set_visible(False)

# Panel B: 30-day rolling vol
ax2.fill_between(daily["date"], daily["roll_vol_30d"], alpha=0.35, color=COL_V30)
ax2.plot(daily["date"], daily["roll_vol_30d"], color=COL_V30, lw=0.7,
         label="30-day rolling σ")
add_events(ax2, daily["date"], y_top=daily["roll_vol_30d"].max() * 0.92)
ax2.set_ylabel("σ (30-day)", fontsize=9)
ax2.legend(fontsize=8, framealpha=0.75)
ax2.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax2.spines[["top", "right"]].set_visible(False)

# Panel C: 90-day rolling vol
ax3.fill_between(daily["date"], daily["roll_vol_90d"], alpha=0.35, color=COL_V90)
ax3.plot(daily["date"], daily["roll_vol_90d"], color=COL_V90, lw=0.9,
         label="90-day rolling σ")
add_events(ax3, daily["date"], y_top=daily["roll_vol_90d"].max() * 0.92)
ax3.set_ylabel("σ (90-day)", fontsize=9)
ax3.legend(fontsize=8, framealpha=0.75)
ax3.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax3.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "gep_summary_rolling_vol.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Distribution: histogram + KDE + Q-Q  (daily & monthly side by side)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for col, (ax_hist, ax_qq), series, label in [
    (0, (axes[0, 0], axes[0, 1]), gep_d, "Daily GEP"),
    (0, (axes[1, 0], axes[1, 1]), gep_m, "Monthly GEP"),
]:
    # Histogram + KDE
    ax = ax_hist
    s = series.dropna()
    ax.hist(s, bins=80, density=True, color=COL_GEP, alpha=0.45,
            edgecolor="white", linewidth=0.3)
    xs = np.linspace(s.min(), s.max(), 400)
    kde = stats.gaussian_kde(s)
    ax.plot(xs, kde(xs), color="#1A3F7A", lw=2, label="KDE")
    mu, sigma = s.mean(), s.std()
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color=COL_V30,
            lw=1.5, ls="--", label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})")
    ax.axvline(mu, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("GEP value", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(f"{label} — Distribution\n"
                 f"skew={s.skew():.3f}  kurtosis={s.kurt():.3f}", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)

    # Q-Q plot
    ax = ax_qq
    (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
    ax.scatter(osm, osr, s=4, alpha=0.4, color=COL_GEP, label="Quantiles")
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, color=COL_V30,
            lw=1.5, label=f"Normal fit  r={r:.4f}")
    ax.set_xlabel("Theoretical quantiles", fontsize=9)
    ax.set_ylabel("Sample quantiles", fontsize=9)
    ax.set_title(f"{label} — Q-Q Plot", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("GEP Robust min-2 — Distribution Analysis", fontsize=13, y=1.01)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "gep_summary_distribution.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — ACF / PACF  (daily and monthly)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf( gep_d.dropna(), lags=60, ax=axes[0, 0], color=COL_GEP,
          title="Daily GEP — ACF (60 lags)")
plot_pacf(gep_d.dropna(), lags=60, ax=axes[0, 1], color=COL_GEP,
          title="Daily GEP — PACF (60 lags)", method="ywm")
plot_acf( gep_m.dropna(), lags=36, ax=axes[1, 0], color="#2CA02C",
          title="Monthly GEP — ACF (36 lags)")
plot_pacf(gep_m.dropna(), lags=36, ax=axes[1, 1], color="#2CA02C",
          title="Monthly GEP — PACF (36 lags)", method="ywm")

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Lag", fontsize=9)

fig.suptitle("GEP Robust min-2 — Autocorrelation Structure", fontsize=13)
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "gep_summary_acf_pacf.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Annual mean ± 1 std
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))

years = annual.index.year
x = np.arange(len(years))

bars = ax.bar(x, annual["mean"], color=COL_GEP, alpha=0.75, width=0.6,
              label="Annual mean")
ax.errorbar(x, annual["mean"], yerr=annual["std"],
            fmt="none", color="#1A3F7A", capsize=3, lw=1.2, label="±1 std")
ax.plot(x, annual["q90"], color=COL_V30, lw=1.5, marker="o", ms=4,
        label="90th percentile")

ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("GEP index", fontsize=10)
ax.set_title("GEP Robust min-2 — Annual Statistics (1996–2025)", fontsize=13, pad=10)
ax.legend(fontsize=9, framealpha=0.75)
ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "gep_summary_annual.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved: {out4}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Monthly heatmap by year
# ══════════════════════════════════════════════════════════════════════════════
monthly["year"]  = monthly["date"].dt.year
monthly["month_n"] = monthly["date"].dt.month

pivot = monthly.pivot(index="year", columns="month_n", values="GEP_monthly")
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.35)))
cmap = plt.get_cmap("YlOrRd")
im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
               vmin=np.nanpercentile(pivot.values, 5),
               vmax=np.nanpercentile(pivot.values, 95))

ax.set_xticks(range(12))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=8)
ax.set_title("GEP Robust min-2 — Monthly Heatmap by Year", fontsize=13, pad=10)

# Cell text
for i in range(len(pivot)):
    for j in range(12):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=5.5, color="black")

cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
cbar.set_label("GEP index", fontsize=9)

plt.tight_layout()
out5 = os.path.join(OUT_DIR, "gep_summary_heatmap.png")
plt.savefig(out5, dpi=150, bbox_inches="tight")
print(f"Saved: {out5}")
plt.close()

print("\n═══ Done ═══")
