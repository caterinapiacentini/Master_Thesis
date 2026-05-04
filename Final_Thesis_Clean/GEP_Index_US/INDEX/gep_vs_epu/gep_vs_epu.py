#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gep_vs_epu.py

Compares the GEP Robust min-2 Daily Index against the EPU index
(Baker, Bloom & Davis) contemporaneously (no lag adjustment).

Monthly comparison uses the official Baker-Bloom-Davis monthly EPU index
(News_Based_Policy_Uncert_Index) from US_Policy_Uncertainty_Data.xlsx,
merged with the monthly mean of GEP.

Produces:
  1. gep_vs_epud_timeseries_daily.png   — Z-score overlay (daily) +
                                          rolling 90-day correlation
  2. gep_vs_epum_timeseries_monthly.png — Monthly Z-score overlay +
                                          rolling 12-month correlation
  3. gep_vs_epud_crosscorr.png          — Cross-correlation ±60 days (daily)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

BASE = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/GEP_IndeX_US/INDEX/data"

EPU_DAILY_CSV   = "/Users/catepiacentini/Desktop/tesi/literature/All_Daily_Policy_Data.csv"
EPU_MONTHLY_XLS = "/Users/catepiacentini/Desktop/tesi/literature/US_Policy_Uncertainty_Data.xlsx"

OUT_DIR = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/GEP_IndeX_US/INDEX/gep_vs_epu"

# ─────────────────────────────────────────────────────────────────────────────
# 0. Helpers
# ─────────────────────────────────────────────────────────────────────────────
def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────────────────────────────────────

# --- Daily EPU ---
epu_daily_raw = pd.read_csv(EPU_DAILY_CSV)
epu_daily_raw["date"] = pd.to_datetime(
    {"year": epu_daily_raw["year"],
     "month": epu_daily_raw["month"],
     "day": epu_daily_raw["day"]}
)
epu_daily_raw = (
    epu_daily_raw[["date", "daily_policy_index"]]
    .rename(columns={"daily_policy_index": "EPU"})
    .dropna(subset=["EPU"])
)
epu_daily_raw = epu_daily_raw[epu_daily_raw["date"] >= "1996-01-01"].reset_index(drop=True)

# --- Monthly EPU (official Baker-Bloom-Davis index) ---
epu_monthly_raw = pd.read_excel(EPU_MONTHLY_XLS)
epu_monthly_raw["Year"] = pd.to_numeric(epu_monthly_raw["Year"], errors="coerce")
epu_monthly_raw = epu_monthly_raw.dropna(subset=["Year", "Month", "News_Based_Policy_Uncert_Index"])
epu_monthly_raw["date"] = pd.to_datetime(
    {"year": epu_monthly_raw["Year"].astype(int),
     "month": epu_monthly_raw["Month"].astype(int),
     "day": 1}
)
epu_monthly_raw = (
    epu_monthly_raw[["date", "News_Based_Policy_Uncert_Index"]]
    .rename(columns={"News_Based_Policy_Uncert_Index": "EPU"})
    .sort_values("date")
    .reset_index(drop=True)
)
epu_monthly_raw = epu_monthly_raw[epu_monthly_raw["date"] >= "1996-01-01"]

# --- GEP Daily ---
gep_raw = pd.read_csv(
    os.path.join(BASE, "GEP_Daily_Robust_min2.csv"), parse_dates=["date"]
)
gep_raw = gep_raw[gep_raw["date"] >= "1996-01-01"]
gep_raw = (
    gep_raw[gep_raw["n_articles"] > 0][["date", "GEP_daily"]]
    .rename(columns={"GEP_daily": "GEP"})
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Merge — contemporaneous daily
# ─────────────────────────────────────────────────────────────────────────────
df = (
    pd.merge(gep_raw, epu_daily_raw, on="date", how="inner")
    .sort_values("date")
    .reset_index(drop=True)
)
df["GEP_z"] = zscore(df["GEP"])
df["EPU_z"] = zscore(df["EPU"])
r, pval = stats.pearsonr(df["GEP_z"], df["EPU_z"])

print(f"Merged dataset: {len(df):,} days  "
      f"({df['date'].min().date()} → {df['date'].max().date()})")
print(f"\nOverall Pearson r (daily, contemporaneous): {r:.4f}  (p = {pval:.2e})")

# Sub-period correlations
periods = [
    ("1996–2001", "1996-01-01", "2001-12-31"),
    ("2002–2009", "2002-01-01", "2009-12-31"),
    ("2010–2019", "2010-01-01", "2019-12-31"),
    ("2020–2025", "2020-01-01", "2025-12-31"),
]
print("\nSub-period correlations:")
for label, start, end in periods:
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(sub) > 10:
        r_sub, _ = stats.pearsonr(sub["GEP_z"], sub["EPU_z"])
        print(f"  {label}: r = {r_sub:.4f}  (n={len(sub):,})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Granger causality (daily)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Granger causality (max lag = 5 days) ---")
gc_data = df[["GEP_z", "EPU_z"]].dropna()

print("\nH0: GEP does NOT Granger-cause EPU")
res_g2e = grangercausalitytests(gc_data[["EPU_z", "GEP_z"]], maxlag=5, verbose=False)
for lag, result in res_g2e.items():
    f, p = result[0]["ssr_ftest"][:2]
    print(f"  lag {lag}: F = {f:.3f}, p = {p:.4f} {'*' if p < 0.05 else ''}")

print("\nH0: EPU does NOT Granger-cause GEP")
res_e2g = grangercausalitytests(gc_data[["GEP_z", "EPU_z"]], maxlag=5, verbose=False)
for lag, result in res_e2g.items():
    f, p = result[0]["ssr_ftest"][:2]
    print(f"  lag {lag}: F = {f:.3f}, p = {p:.4f} {'*' if p < 0.05 else ''}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Monthly aggregation
#    GEP: resample daily to monthly mean
#    EPU: use official monthly index from Baker-Bloom-Davis
# ─────────────────────────────────────────────────────────────────────────────
gep_monthly = (
    df.set_index("date")[["GEP"]]
    .resample("MS")
    .mean()
    .dropna()
)

df_m = (
    gep_monthly.join(epu_monthly_raw.set_index("date")[["EPU"]], how="inner")
    .dropna()
)
df_m["GEP_z"] = zscore(df_m["GEP"])
df_m["EPU_z"] = zscore(df_m["EPU"])
r_m, pval_m = stats.pearsonr(df_m["GEP_z"], df_m["EPU_z"])
df_m["roll_corr_12m"] = (
    df_m["GEP_z"].rolling(12, min_periods=9).corr(df_m["EPU_z"])
)
print(f"\nMonthly Pearson r: {r_m:.4f}  (p = {pval_m:.2e})")

# Rolling 90-day correlation (daily)
df["roll_corr_90d"] = df["GEP_z"].rolling(90, min_periods=60).corr(df["EPU_z"])

# ─────────────────────────────────────────────────────────────────────────────
# Key events annotation
# ─────────────────────────────────────────────────────────────────────────────
key_events = {
    "1997-07-01": "Asian Crisis",
    "1998-08-17": "Russian Crisis",
    "2001-09-11": "9/11",
    "2003-03-20": "Iraq War",
    "2008-09-15": "GFC",
    "2014-03-01": "Crimea",
    "2018-07-06": "US–China Trade War",
    "2020-03-11": "COVID-19",
    "2022-02-24": "Ukraine Invasion",
    "2025-04-02": "Liberation Day",
}

def add_events(ax, df_dates, y_top=5, fontsize=6.5):
    for date_str, label in key_events.items():
        xd = pd.to_datetime(date_str)
        if df_dates.min() <= xd <= df_dates.max():
            ax.axvline(xd, color="gray", linewidth=0.6, linestyle="--", alpha=0.55)
            ax.text(xd, y_top, label, rotation=90, fontsize=fontsize,
                    va="top", color="#555555", ha="right")

COL_GEP  = "#378ADD"
COL_EPU  = "#2CA02C"
COL_ROLL = "#5A4FCF"

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Daily Z-scores + rolling 90-day correlation
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(16, 9), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.2]}
)

ax1.plot(df["date"], df["GEP_z"], color=COL_GEP, lw=0.7, alpha=0.85,
         label="GEP Robust min-2  [this thesis, Reuters online]")
ax1.plot(df["date"], df["EPU_z"], color=COL_EPU, lw=0.7, alpha=0.85,
         label="EPU Daily  [Baker, Bloom & Davis]")
add_events(ax1, df["date"])
ax1.axhline(0, color="black", lw=0.4)
ax1.set_ylabel("Z-score", fontsize=10)
ax1.set_title(
    f"GEP Robust min-2 vs EPU Daily Index  (1996–2025)  |  r = {r:.4f}",
    fontsize=12, pad=10
)
ax1.legend(fontsize=9, framealpha=0.75, loc="upper left")
ax1.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax1.spines[["top", "right"]].set_visible(False)

ax2.plot(df["date"], df["roll_corr_90d"], color=COL_ROLL, lw=0.8)
ax2.axhline(0, color="black", lw=0.4)
ax2.axhline(r, color="gray", lw=0.7, ls="--", alpha=0.75,
            label=f"Overall r = {r:.3f}")
ax2.fill_between(df["date"], df["roll_corr_90d"], 0,
                 where=df["roll_corr_90d"] > 0, alpha=0.15, color=COL_ROLL)
ax2.fill_between(df["date"], df["roll_corr_90d"], 0,
                 where=df["roll_corr_90d"] < 0, alpha=0.15, color="red")
ax2.set_ylabel("90-day rolling\ncorrelation", fontsize=9)
ax2.set_ylim(-1, 1)
ax2.legend(fontsize=8, framealpha=0.75)
ax2.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "gep_vs_epud_timeseries_daily.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Monthly Z-scores + rolling 12-month correlation
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(16, 9), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.2]}
)

ax1.plot(df_m.index, df_m["GEP_z"], color=COL_GEP, lw=1.3, alpha=0.9,
         label="GEP Robust min-2  [this thesis, Reuters online]")
ax1.plot(df_m.index, df_m["EPU_z"], color=COL_EPU, lw=1.3, alpha=0.9,
         label="EPU Monthly  [Baker, Bloom & Davis, News-Based]")
ax1.fill_between(df_m.index, df_m["GEP_z"], 0, alpha=0.07, color=COL_GEP)
ax1.fill_between(df_m.index, df_m["EPU_z"], 0, alpha=0.07, color=COL_EPU)
add_events(ax1, df_m.index, y_top=3.5, fontsize=7)
ax1.axhline(0, color="black", lw=0.4)
ax1.set_ylabel("Z-score (monthly mean)", fontsize=10)
ax1.set_title(
    f"GEP Robust min-2 vs EPU — Monthly Averages  (1996–2025)  |  r = {r_m:.4f}",
    fontsize=12, pad=10
)
ax1.legend(fontsize=9, framealpha=0.75, loc="upper left")
ax1.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax1.spines[["top", "right"]].set_visible(False)

ax2.plot(df_m.index, df_m["roll_corr_12m"], color=COL_ROLL, lw=1.2)
ax2.axhline(0, color="black", lw=0.4)
ax2.axhline(r_m, color="gray", lw=0.7, ls="--", alpha=0.75,
            label=f"Overall r = {r_m:.3f}")
ax2.fill_between(df_m.index, df_m["roll_corr_12m"], 0,
                 where=df_m["roll_corr_12m"] > 0, alpha=0.15, color=COL_ROLL)
ax2.fill_between(df_m.index, df_m["roll_corr_12m"], 0,
                 where=df_m["roll_corr_12m"] < 0, alpha=0.15, color="red")
ax2.set_ylabel("12-month rolling\ncorrelation", fontsize=9)
ax2.set_ylim(-1, 1)
ax2.legend(fontsize=8, framealpha=0.75)
ax2.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "gep_vs_epum_timeseries_monthly.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Cross-correlation ±60 days
# ══════════════════════════════════════════════════════════════════════════════
max_lag = 60
lags = list(range(-max_lag, max_lag + 1))

def cross_corr(s1, s2, lags):
    out = []
    for lag in lags:
        if lag == 0:
            out.append(s1.corr(s2))
        elif lag > 0:
            out.append(s1.iloc[:-lag].corr(s2.iloc[lag:]))
        else:
            out.append(s1.iloc[-lag:].corr(s2.iloc[:lag]))
    return np.array(out)

lags_arr = np.array(lags)
xcorr    = cross_corr(df["GEP_z"], df["EPU_z"], lags)

peak_lag  = lags_arr[np.argmax(xcorr)]
peak_corr = xcorr.max()

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.bar(lags_arr, xcorr, width=0.8,
       color=np.where(lags_arr >= 0, "#378ADD", "#2CA02C"),
       alpha=0.65)
ax.axvline(0,        color="black", lw=0.8)
ax.axvline(peak_lag, color="#E05C2A", lw=1.4, ls="--",
           label=f"Peak: lag={peak_lag}d  r={peak_corr:.4f}")
ax.axhline(0, color="black", lw=0.4)
ax.set_xlabel(
    "Lag (days)  |  Positive = GEP leads EPU  |  Negative = EPU leads GEP",
    fontsize=10
)
ax.set_ylabel("Pearson r", fontsize=10)
ax.set_title(
    "Cross-correlation: GEP Robust min-2 vs EPU Daily (1996–2025)",
    fontsize=12, pad=10
)
ax.legend(fontsize=9, framealpha=0.8)
ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "gep_vs_epud_crosscorr.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n═══ Summary ═══")
print(f"  Daily  Pearson r              : {r:.4f}")
print(f"  Monthly Pearson r             : {r_m:.4f}")
print(f"  Cross-corr peak               : lag={peak_lag}d, r={peak_corr:.4f}")
if peak_lag > 0:
    print(f"  -> GEP leads EPU by {peak_lag} day(s)")
elif peak_lag < 0:
    print(f"  -> EPU leads GEP by {-peak_lag} day(s)")
else:
    print("  -> No lead-lag (contemporaneous peak)")
