#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gep_vs_gpr_robust_min2.py

Compares the GEP Robust min-2 Daily Index against the daily GPR index
(Caldara & Iacoviello 2022), with a 1-day lag adjustment:

    GEP(t)  vs  GPR(t+1)

Rationale: GEP is built from online Reuters wire news (real-time), while
GPR is constructed from printed newspaper articles, which are published
with a ~1-day delay relative to wire news. The adjustment aligns both
indices to the same underlying "event day".

Implementation: GPR is shifted backwards by 1 day (df["GPRD"].shift(-1)),
so that GPR(t+1) is paired with GEP(t) in every row.

Produces:
  1. gep_vs_gprd_timeseries_daily.png   — Z-score overlay (daily, lag-adj) +
                                          rolling 90-day correlation
  2. gep_vs_gprd_timeseries_monthly.png — Monthly-mean Z-score overlay +
                                          rolling 12-month correlation
  3. gep_vs_gprd_crosscorr.png          — Cross-correlation ±60 days
                                          (raw vs lag-adjusted, daily)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

BASE    = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/MIN2"

# Two GPR source files — recent data takes priority; export used as fallback
GPR_XLS_RECENT = "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_daily_recent.xls"
GPR_XLS_EXPORT = "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_export.xls"
GPR_XLS = GPR_XLS_RECENT if os.path.exists(GPR_XLS_RECENT) else GPR_XLS_EXPORT

# ─────────────────────────────────────────────────────────────────────────────
# 0. Helpers
# ─────────────────────────────────────────────────────────────────────────────
def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────────────────────────────────────
gpr_raw = pd.read_excel(GPR_XLS, parse_dates=["date"])
gpr_raw = gpr_raw[["date", "GPRD"]].copy()
gpr_raw["date"] = pd.to_datetime(gpr_raw["date"]).dt.normalize()
gpr_raw = gpr_raw[gpr_raw["date"] >= "1996-01-01"]

gep_raw = pd.read_csv(
    os.path.join(BASE, "data", "GEP_Daily_Robust_min2.csv"), parse_dates=["date"]
)
gep_raw = gep_raw[gep_raw["date"] >= "1996-01-01"]
gep_raw = (
    gep_raw[gep_raw["n_articles"] > 0][["date", "GEP_daily"]]
    .rename(columns={"GEP_daily": "GEP"})
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Merge — RAW (contemporaneous) for cross-corr baseline
# ─────────────────────────────────────────────────────────────────────────────
df_raw = (
    pd.merge(gep_raw, gpr_raw, on="date", how="inner")
    .sort_values("date")
    .reset_index(drop=True)
)
df_raw["GEP_z"]  = zscore(df_raw["GEP"])
df_raw["GPRD_z"] = zscore(df_raw["GPRD"])
r_raw, _ = stats.pearsonr(df_raw["GEP_z"], df_raw["GPRD_z"])

# ─────────────────────────────────────────────────────────────────────────────
# 3. Merge — LAG-ADJUSTED: GEP(t)  vs  GPR(t+1)
#    Shift GPR back by 1 trading day so GPR(t+1) sits in the row dated t.
# ─────────────────────────────────────────────────────────────────────────────
gpr_adj = gpr_raw.copy()
gpr_adj["GPRD"] = gpr_adj["GPRD"].shift(-1)   # GPR(t+1) → row t

df = (
    pd.merge(gep_raw, gpr_adj, on="date", how="inner")
    .sort_values("date")
    .reset_index(drop=True)
    .dropna(subset=["GPRD"])   # last row loses GPR after shift
    .reset_index(drop=True)
)
df["GEP_z"]  = zscore(df["GEP"])
df["GPRD_z"] = zscore(df["GPRD"])
r, pval = stats.pearsonr(df["GEP_z"], df["GPRD_z"])

print(f"Merged dataset (lag-adj): {len(df):,} days  "
      f"({df['date'].min().date()} → {df['date'].max().date()})")
print(f"\nOverall Pearson r (raw, contemporaneous)  : {r_raw:.4f}")
print(f"Overall Pearson r (lag-adj, GEP vs GPR+1) : {r:.4f}  (p = {pval:.2e})")

# Sub-period correlations
periods = [
    ("1996–2001", "1996-01-01", "2001-12-31"),
    ("2002–2009", "2002-01-01", "2009-12-31"),
    ("2010–2019", "2010-01-01", "2019-12-31"),
    ("2020–2025", "2020-01-01", "2025-12-31"),
]
print("\nSub-period correlations (lag-adjusted):")
for label, start, end in periods:
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(sub) > 10:
        r_sub, _ = stats.pearsonr(sub["GEP_z"], sub["GPRD_z"])
        print(f"  {label}: r = {r_sub:.4f}  (n={len(sub):,})")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Granger causality (lag-adjusted daily)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Granger causality on lag-adjusted series (max lag = 5 days) ---")
gc_data = df[["GEP_z", "GPRD_z"]].dropna()

print("\nH0: GEP does NOT Granger-cause GPRD")
res_g2g = grangercausalitytests(gc_data[["GPRD_z", "GEP_z"]], maxlag=5, verbose=False)
for lag, result in res_g2g.items():
    f, p = result[0]["ssr_ftest"][:2]
    print(f"  lag {lag}: F = {f:.3f}, p = {p:.4f} {'*' if p < 0.05 else ''}")

print("\nH0: GPRD does NOT Granger-cause GEP")
res_g2g2 = grangercausalitytests(gc_data[["GEP_z", "GPRD_z"]], maxlag=5, verbose=False)
for lag, result in res_g2g2.items():
    f, p = result[0]["ssr_ftest"][:2]
    print(f"  lag {lag}: F = {f:.3f}, p = {p:.4f} {'*' if p < 0.05 else ''}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Monthly aggregation (lag-adjusted)
# ─────────────────────────────────────────────────────────────────────────────
df_m = (
    df.set_index("date")[["GEP", "GPRD"]]
    .resample("MS")          # month-start frequency
    .mean()
    .dropna()
)
df_m["GEP_z"]  = zscore(df_m["GEP"])
df_m["GPRD_z"] = zscore(df_m["GPRD"])
r_m, pval_m = stats.pearsonr(df_m["GEP_z"], df_m["GPRD_z"])
df_m["roll_corr_12m"] = (
    df_m["GEP_z"].rolling(12, min_periods=9).corr(df_m["GPRD_z"])
)
print(f"\nMonthly Pearson r (lag-adj): {r_m:.4f}  (p = {pval_m:.2e})")

# Rolling 90-day correlation (daily)
df["roll_corr_90d"] = df["GEP_z"].rolling(90, min_periods=60).corr(df["GPRD_z"])

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
COL_GPR  = "#E05C2A"
COL_ROLL = "#5A4FCF"

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Daily Z-scores (lag-adjusted) + rolling 90-day correlation
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(16, 9), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.2]}
)

ax1.plot(df["date"], df["GEP_z"],  color=COL_GEP,  lw=0.7, alpha=0.85,
         label="GEP Robust min-2  [this thesis, Reuters online]")
ax1.plot(df["date"], df["GPRD_z"], color=COL_GPR,  lw=0.7, alpha=0.85,
         label="GPR  [Caldara & Iacoviello, newspapers  t+1]")
add_events(ax1, df["date"])
ax1.axhline(0, color="black", lw=0.4)
ax1.set_ylabel("Z-score", fontsize=10)
ax1.set_title(
    f"GEP Robust min-2 vs GPR Daily Index — lag-adjusted GEP(t) vs GPR(t+1)  "
    f"(1996–2025)  |  r = {r:.4f}",
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
out1 = os.path.join(BASE, "gep_vs_gprd_timeseries_daily.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Monthly Z-scores (lag-adjusted) + rolling 12-month correlation
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(16, 9), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.2]}
)

ax1.plot(df_m.index, df_m["GEP_z"],  color=COL_GEP,  lw=1.3, alpha=0.9,
         label="GEP Robust min-2  [this thesis, Reuters online]")
ax1.plot(df_m.index, df_m["GPRD_z"], color=COL_GPR,  lw=1.3, alpha=0.9,
         label="GPR  [Caldara & Iacoviello, newspapers  t+1]")
ax1.fill_between(df_m.index, df_m["GEP_z"],  0, alpha=0.07, color=COL_GEP)
ax1.fill_between(df_m.index, df_m["GPRD_z"], 0, alpha=0.07, color=COL_GPR)
add_events(ax1, df_m.index, y_top=3.5, fontsize=7)
ax1.axhline(0, color="black", lw=0.4)
ax1.set_ylabel("Z-score (monthly mean)", fontsize=10)
ax1.set_title(
    f"GEP Robust min-2 vs GPR — Monthly Averages, lag-adjusted GEP(t) vs GPR(t+1)  "
    f"(1996–2025)  |  r = {r_m:.4f}",
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
out2 = os.path.join(BASE, "gep_vs_gprd_timeseries_monthly.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Cross-correlation ±60 days (raw vs lag-adjusted overlay)
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

lags_arr   = np.array(lags)
xcorr_raw  = cross_corr(df_raw["GEP_z"], df_raw["GPRD_z"], lags)
xcorr_adj  = cross_corr(df["GEP_z"],     df["GPRD_z"],     lags)

peak_lag_raw  = lags_arr[np.argmax(xcorr_raw)]
peak_corr_raw = xcorr_raw.max()
peak_lag_adj  = lags_arr[np.argmax(xcorr_adj)]
peak_corr_adj = xcorr_adj.max()

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.bar(lags_arr - 0.22, xcorr_raw, width=0.44,
       color=np.where(lags_arr >= 0, "#378ADD", "#E05C2A"),
       alpha=0.55, label="Raw (contemporaneous)")
ax.bar(lags_arr + 0.22, xcorr_adj, width=0.44,
       color=np.where(lags_arr >= 0, "#1A5FAD", "#B03A10"),
       alpha=0.75, label="Lag-adjusted  GEP(t) vs GPR(t+1)")
ax.axvline(0,            color="black", lw=0.8)
ax.axvline(peak_lag_raw, color="#888888", lw=1.2, ls="--",
           label=f"Peak raw:  lag={peak_lag_raw}d  r={peak_corr_raw:.4f}")
ax.axvline(peak_lag_adj, color="#2ECC71", lw=1.2, ls="--",
           label=f"Peak adj:  lag={peak_lag_adj}d  r={peak_corr_adj:.4f}")
ax.axhline(0, color="black", lw=0.4)
ax.set_xlabel(
    "Lag (days)  |  Positive = GEP leads GPR  |  Negative = GPR leads GEP",
    fontsize=10
)
ax.set_ylabel("Pearson r", fontsize=10)
ax.set_title(
    "Cross-correlation: GEP Robust min-2 vs GPRD (1996–2025)\n"
    "Raw (contemporaneous) vs Lag-adjusted  [GEP(t) — GPR(t+1)]",
    fontsize=12, pad=10
)
ax.legend(fontsize=8.5, framealpha=0.8)
ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out3 = os.path.join(BASE, "gep_vs_gprd_crosscorr.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n═══ Summary ═══")
print(f"  Daily  Pearson r — raw        : {r_raw:.4f}")
print(f"  Daily  Pearson r — lag-adj    : {r:.4f}")
print(f"  Monthly Pearson r — lag-adj   : {r_m:.4f}")
print(f"  Cross-corr peak (raw)         : lag={peak_lag_raw}d, r={peak_corr_raw:.4f}")
print(f"  Cross-corr peak (lag-adj)     : lag={peak_lag_adj}d, r={peak_corr_adj:.4f}")
if peak_lag_adj > 0:
    print(f"  -> GEP leads GPR by {peak_lag_adj} day(s) even after 1-day adjustment")
elif peak_lag_adj < 0:
    print(f"  -> GPR still leads GEP by {-peak_lag_adj} day(s) after adjustment")
else:
    print("  -> No remaining lead-lag after adjustment (contemporaneous)")