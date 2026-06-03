#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_comparisons.py

GEP vs VIX, GPR, EPU, and S&P 500 — time-series comparisons + regressions.

DATA layout (relative to this script):
  data/gep/GEP_Monthly_Robust_min2.csv
  data/gep/GEP_Daily_Robust_min2.csv
  data/external/data_gpr_daily_recent.xls    (Caldara & Iacoviello GPR daily)
  data/external/data_gpr_export.xls          (GPR monthly / fallback)
  data/external/All_Daily_Policy_Data.csv    (Baker-Bloom-Davis EPU daily)
  data/external/US_Policy_Uncertainty_Data.xlsx (Baker-Bloom-Davis EPU monthly)

External downloads (via yfinance / pandas_datareader):
  ^VIX, ^GSPC, Fama-French factors

Outputs saved to output/comparisons/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf
import statsmodels.api as sm
import pandas_datareader.data as web
from pathlib import Path
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()
REPO = next((p for p in [HERE, *HERE.parents] if (p / "data" / "gep_us").exists()), HERE.parent)
DATA = REPO / "data"
GEP  = DATA / "gep_us"
EXT  = DATA / "external"
OUT  = REPO / "analysis" / "output" / "comparisons"
OUT.mkdir(parents=True, exist_ok=True)

GPR_RECENT = EXT / "data_gpr_daily_recent.xls"
GPR_EXPORT = EXT / "data_gpr_export.xls"
GPR_XLS    = GPR_RECENT if GPR_RECENT.exists() else GPR_EXPORT

# ─────────────────────────────────────────────────────────────────────────────
# Load GEP
# ─────────────────────────────────────────────────────────────────────────────
monthly_gep = pd.read_csv(GEP / "GEP_Monthly_Robust_min2.csv")
monthly_gep["month"] = pd.to_datetime(monthly_gep["month"])
monthly_gep = monthly_gep.set_index("month").sort_index()

daily_gep = pd.read_csv(GEP / "GEP_Daily_Robust_min2.csv")
daily_gep["date"] = pd.to_datetime(daily_gep["date"])
daily_gep = daily_gep[daily_gep["n_articles"] > 0].set_index("date").sort_index()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def zscore(s): return (s - s.mean()) / s.std()

def cross_corr(s1, s2, lags):
    out = []
    for lag in lags:
        if lag == 0:  out.append(s1.corr(s2))
        elif lag > 0: out.append(s1.iloc[:-lag].corr(s2.iloc[lag:]))
        else:         out.append(s1.iloc[-lag:].corr(s2.iloc[:lag]))
    return np.array(out)

KEY_EVENTS = {
    "1997-07-01": "Asian Crisis", "1998-08-17": "Russian Crisis",
    "2001-09-11": "9/11",        "2003-03-20": "Iraq War",
    "2008-09-15": "GFC",         "2014-03-01": "Crimea",
    "2018-07-06": "US–China Trade War", "2020-03-11": "COVID-19",
    "2022-02-24": "Ukraine Invasion",   "2025-04-02": "Liberation Day",
}

def add_events(ax, df_dates, y_top=5, fontsize=6.5):
    for date_str, label in KEY_EVENTS.items():
        xd = pd.to_datetime(date_str)
        if pd.Timestamp(df_dates.min()) <= xd <= pd.Timestamp(df_dates.max()):
            ax.axvline(xd, color="gray", linewidth=0.6, linestyle="--", alpha=0.55)
            ax.text(xd, y_top, label, rotation=90, fontsize=fontsize,
                    va="top", color="#555555", ha="right")

SUBSAMPLES = {
    "Full sample (1996–2025)"            : (None,        None),
    "Pre-GEP era (1996–2017)"            : ("1996-01",   "2017-12"),
    "Trade war (2018–2019)"              : ("2018-01",   "2019-12"),
    "Russia-Ukraine (2022–2023)"         : ("2022-02",   "2023-12"),
    "High-pressure combined (2018–2025)" : ("2018-01",   "2025-12"),
}

def slice_df(df, start, end):
    d = df.copy()
    if start: d = d[d.index >= start]
    if end:   d = d[d.index <= end]
    return d

def run_ols(y, X_cols, data, label, gep_col, hac_lags):
    clean = data[X_cols + [y]].dropna()
    if len(clean) < 30:
        print(f"  {label:<40}  [skipped — N={len(clean)} too small]"); return
    m = sm.OLS(clean[y], sm.add_constant(clean[X_cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    coef, se, t, p = m.params[gep_col], m.bse[gep_col], m.tvalues[gep_col], m.pvalues[gep_col]
    stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "   "))
    print(f"  {label:<40}  β={coef:+.5f}  SE={se:.5f}  t={t:+.2f}  "
          f"p={p:.3f} {stars}  R²={m.rsquared:.4f}  N={int(m.nobs)}")

COL_GEP, COL_GPR, COL_EPU = "#378ADD", "#E05C2A", "#2CA02C"
COL_VIX, COL_ROLL = "#8E44AD", "#5A4FCF"


# ═════════════════════════════════════════════════════════════════════════════
# GEP vs GPR
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nGEP vs GPR\n" + "="*60)

gpr_raw = pd.read_excel(GPR_XLS, parse_dates=["date"])
gpr_raw = (gpr_raw[["date", "GPRD"]].copy()
           .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
           .query("date >= '1996-01-01'"))

gep_d = daily_gep[["GEP_daily"]].rename(columns={"GEP_daily": "GEP"})
gpr_raw_df = gpr_raw.set_index("date").sort_index()
gpr_raw_df = gpr_raw_df[~gpr_raw_df.index.duplicated(keep="first")]

# Lag-adjusted: GEP(t) vs GPR(t+1)
gpr_adj = gpr_raw_df.copy(); gpr_adj["GPRD"] = gpr_adj["GPRD"].shift(-1)
df_raw_gpr = pd.merge(gep_d.reset_index(), gpr_raw_df.reset_index(),
                      on="date", how="inner").sort_values("date").reset_index(drop=True)
df_gpr = pd.merge(gep_d.reset_index(), gpr_adj.reset_index(),
                  on="date", how="inner").sort_values("date").dropna(subset=["GPRD"]).reset_index(drop=True)

for df in (df_raw_gpr, df_gpr):
    df["GEP_z"]  = zscore(df["GEP"])
    df["GPRD_z"] = zscore(df["GPRD"])

r_raw, _ = stats.pearsonr(df_raw_gpr["GEP_z"], df_raw_gpr["GPRD_z"])
r, pval   = stats.pearsonr(df_gpr["GEP_z"],     df_gpr["GPRD_z"])

df_m_gpr = (df_gpr.set_index("date")[["GEP", "GPRD"]].resample("MS").mean().dropna())
df_m_gpr["GEP_z"]  = zscore(df_m_gpr["GEP"])
df_m_gpr["GPRD_z"] = zscore(df_m_gpr["GPRD"])
r_m, pval_m = stats.pearsonr(df_m_gpr["GEP_z"], df_m_gpr["GPRD_z"])
df_m_gpr["roll_corr_12m"] = df_m_gpr["GEP_z"].rolling(12, min_periods=9).corr(df_m_gpr["GPRD_z"])
df_gpr["roll_corr_90d"] = df_gpr["GEP_z"].rolling(90, min_periods=60).corr(df_gpr["GPRD_z"])

print(f"Daily  r raw: {r_raw:.4f} | lag-adj: {r:.4f} (p={pval:.2e})")
print(f"Monthly r lag-adj: {r_m:.4f} (p={pval_m:.2e})")

for fig_data, df_plot, corr, ylabel, title, fname, roll_col, roll_ylabel in [
    (df_gpr, df_gpr, r, "Z-score",
     f"GEP vs GPR Daily — lag-adj GEP(t) vs GPR(t+1) | r={r:.4f}",
     "gep_vs_gprd_timeseries_daily.png", "roll_corr_90d", "90-day rolling\ncorrelation"),
    (df_m_gpr, df_m_gpr, r_m, "Z-score (monthly mean)",
     f"GEP vs GPR Monthly | r={r_m:.4f}",
     "gep_vs_gprd_timeseries_monthly.png", "roll_corr_12m", "12-month rolling\ncorrelation"),
]:
    date_col = "date" if "date" in df_plot.columns else df_plot.index
    x = df_plot["date"] if "date" in df_plot.columns else df_plot.index
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.2]})
    ax1.plot(x, df_plot["GEP_z"],  color=COL_GEP, lw=0.7 if "date" in df_plot.columns else 1.3,
             alpha=0.85, label="GEP Robust min-2 [Reuters wire]")
    ax1.plot(x, df_plot["GPRD_z"], color=COL_GPR, lw=0.7 if "date" in df_plot.columns else 1.3,
             alpha=0.85, label="GPR [Caldara & Iacoviello, newspapers t+1]")
    if "date" in df_plot.columns:
        add_events(ax1, df_plot["date"])
    else:
        add_events(ax1, df_plot.index, y_top=3.5, fontsize=7)
    ax1.axhline(0, color="black", lw=0.4)
    ax1.set_ylabel(ylabel, fontsize=10)
    ax1.set_title(title, fontsize=12, pad=10)
    ax1.legend(fontsize=9, framealpha=0.75, loc="upper left")
    ax1.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(x, df_plot[roll_col], color=COL_ROLL, lw=0.8 if "date" in df_plot.columns else 1.2)
    ax2.axhline(0, color="black", lw=0.4)
    ax2.axhline(corr, color="gray", lw=0.7, ls="--", alpha=0.75, label=f"Overall r={corr:.3f}")
    ax2.fill_between(x, df_plot[roll_col], 0,
                     where=df_plot[roll_col] > 0, alpha=0.15, color=COL_ROLL)
    ax2.fill_between(x, df_plot[roll_col], 0,
                     where=df_plot[roll_col] < 0, alpha=0.15, color="red")
    ax2.set_ylabel(roll_ylabel, fontsize=9)
    ax2.set_ylim(-1, 1)
    ax2.legend(fontsize=8, framealpha=0.75)
    ax2.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}"); plt.close()

# Cross-correlation plot
lags_range = list(range(-60, 61))
lags_arr   = np.array(lags_range)
xcorr_raw  = cross_corr(df_raw_gpr["GEP_z"], df_raw_gpr["GPRD_z"], lags_range)
xcorr_adj  = cross_corr(df_gpr["GEP_z"],     df_gpr["GPRD_z"],     lags_range)

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.bar(lags_arr - 0.22, xcorr_raw, width=0.44,
       color=np.where(lags_arr >= 0, "#378ADD", "#E05C2A"), alpha=0.55, label="Raw (contemp)")
ax.bar(lags_arr + 0.22, xcorr_adj, width=0.44,
       color=np.where(lags_arr >= 0, "#1A5FAD", "#B03A10"), alpha=0.75, label="Lag-adj GEP(t) vs GPR(t+1)")
ax.axvline(0, color="black", lw=0.8)
ax.axhline(0, color="black", lw=0.4)
ax.set_xlabel("Lag (days)  |  Positive = GEP leads GPR  |  Negative = GPR leads GEP", fontsize=10)
ax.set_ylabel("Pearson r", fontsize=10)
ax.set_title("Cross-correlation: GEP vs GPRD (1996–2025)", fontsize=12, pad=10)
ax.legend(fontsize=8.5, framealpha=0.8)
ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "gep_vs_gprd_crosscorr.png", dpi=150, bbox_inches="tight")
print("Saved: gep_vs_gprd_crosscorr.png"); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# GEP vs EPU
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nGEP vs EPU\n" + "="*60)

epu_d_raw = pd.read_csv(EXT / "All_Daily_Policy_Data.csv")
epu_d_raw["date"] = pd.to_datetime(
    {"year": epu_d_raw["year"], "month": epu_d_raw["month"], "day": epu_d_raw["day"]})
epu_d = (epu_d_raw[["date", "daily_policy_index"]]
         .rename(columns={"daily_policy_index": "EPU"})
         .dropna(subset=["EPU"])
         .query("date >= '1996-01-01'")
         .reset_index(drop=True))

epu_m_raw = pd.read_excel(EXT / "US_Policy_Uncertainty_Data.xlsx")
epu_m_raw["Year"] = pd.to_numeric(epu_m_raw["Year"], errors="coerce")
epu_m_raw = epu_m_raw.dropna(subset=["Year", "Month", "News_Based_Policy_Uncert_Index"])
epu_m_raw["date"] = pd.to_datetime(
    {"year": epu_m_raw["Year"].astype(int),
     "month": epu_m_raw["Month"].astype(int), "day": 1})
epu_m = (epu_m_raw[["date", "News_Based_Policy_Uncert_Index"]]
         .rename(columns={"News_Based_Policy_Uncert_Index": "EPU"})
         .query("date >= '1996-01-01'").sort_values("date").reset_index(drop=True))

df_epu = (pd.merge(gep_d.reset_index(), epu_d, on="date", how="inner")
          .sort_values("date").reset_index(drop=True))
df_epu["GEP_z"] = zscore(df_epu["GEP"])
df_epu["EPU_z"] = zscore(df_epu["EPU"])
r_epu, pval_epu = stats.pearsonr(df_epu["GEP_z"], df_epu["EPU_z"])

gep_monthly_agg = df_epu.set_index("date")[["GEP"]].resample("MS").mean().dropna()
df_m_epu = gep_monthly_agg.join(epu_m.set_index("date")[["EPU"]], how="inner").dropna()
df_m_epu["GEP_z"] = zscore(df_m_epu["GEP"])
df_m_epu["EPU_z"] = zscore(df_m_epu["EPU"])
r_m_epu, pval_m_epu = stats.pearsonr(df_m_epu["GEP_z"], df_m_epu["EPU_z"])
df_m_epu["roll_corr_12m"] = df_m_epu["GEP_z"].rolling(12, min_periods=9).corr(df_m_epu["EPU_z"])
df_epu["roll_corr_90d"] = df_epu["GEP_z"].rolling(90, min_periods=60).corr(df_epu["EPU_z"])

print(f"Daily  r: {r_epu:.4f} (p={pval_epu:.2e})")
print(f"Monthly r: {r_m_epu:.4f} (p={pval_m_epu:.2e})")

for df_plot, corr, ylabel, title, fname, roll_col, roll_ylabel in [
    (df_epu,   r_epu,   "Z-score",
     f"GEP vs EPU Daily | r={r_epu:.4f}",
     "gep_vs_epud_timeseries_daily.png",  "roll_corr_90d", "90-day rolling\ncorrelation"),
    (df_m_epu, r_m_epu, "Z-score (monthly mean)",
     f"GEP vs EPU Monthly | r={r_m_epu:.4f}",
     "gep_vs_epum_timeseries_monthly.png", "roll_corr_12m", "12-month rolling\ncorrelation"),
]:
    x = df_plot["date"] if "date" in df_plot.columns else df_plot.index
    epu_col = "EPU_z"; gep_col = "GEP_z"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.2]})
    ax1.plot(x, df_plot[gep_col], color=COL_GEP, lw=0.7 if "date" in df_plot.columns else 1.3,
             alpha=0.85, label="GEP Robust min-2 [Reuters wire]")
    ax1.plot(x, df_plot[epu_col], color=COL_EPU, lw=0.7 if "date" in df_plot.columns else 1.3,
             alpha=0.85, label="EPU [Baker, Bloom & Davis]")
    if "date" in df_plot.columns:
        add_events(ax1, df_plot["date"])
    else:
        add_events(ax1, df_plot.index, y_top=3.5, fontsize=7)
    ax1.axhline(0, color="black", lw=0.4)
    ax1.set_ylabel(ylabel, fontsize=10); ax1.set_title(title, fontsize=12, pad=10)
    ax1.legend(fontsize=9, framealpha=0.75, loc="upper left")
    ax1.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(x, df_plot[roll_col], color=COL_ROLL, lw=0.8 if "date" in df_plot.columns else 1.2)
    ax2.axhline(0, color="black", lw=0.4)
    ax2.axhline(corr, color="gray", lw=0.7, ls="--", alpha=0.75, label=f"Overall r={corr:.3f}")
    ax2.fill_between(x, df_plot[roll_col], 0,
                     where=df_plot[roll_col] > 0, alpha=0.15, color=COL_ROLL)
    ax2.fill_between(x, df_plot[roll_col], 0,
                     where=df_plot[roll_col] < 0, alpha=0.15, color="red")
    ax2.set_ylabel(roll_ylabel, fontsize=9); ax2.set_ylim(-1, 1)
    ax2.legend(fontsize=8, framealpha=0.75)
    ax2.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}"); plt.close()

# EPU cross-correlation
xcorr_epu  = cross_corr(df_epu["GEP_z"], df_epu["EPU_z"], lags_range)
peak_lag   = lags_arr[np.argmax(xcorr_epu)]
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.bar(lags_arr, xcorr_epu, width=0.8,
       color=np.where(lags_arr >= 0, "#378ADD", "#2CA02C"), alpha=0.65)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(peak_lag, color="#E05C2A", lw=1.4, ls="--",
           label=f"Peak: lag={peak_lag}d  r={xcorr_epu.max():.4f}")
ax.axhline(0, color="black", lw=0.4)
ax.set_xlabel("Lag (days)  |  Positive = GEP leads EPU  |  Negative = EPU leads GEP", fontsize=10)
ax.set_ylabel("Pearson r", fontsize=10)
ax.set_title("Cross-correlation: GEP vs EPU Daily (1996–2025)", fontsize=12, pad=10)
ax.legend(fontsize=9, framealpha=0.8)
ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "gep_vs_epud_crosscorr.png", dpi=150, bbox_inches="tight")
print("Saved: gep_vs_epud_crosscorr.png"); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# GEP vs VIX
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nGEP vs VIX\n" + "="*60)

print("Downloading VIX...")
vix_mo = yf.download("^VIX", start="1996-01-01", end="2025-12-31", interval="1mo",
                     auto_adjust=False, progress=False).reset_index()
if isinstance(vix_mo.columns, pd.MultiIndex):
    vix_mo.columns = vix_mo.columns.get_level_values(0)
vix_mo = vix_mo[["Date", "Close"]].dropna().copy()
vix_mo.rename(columns={"Close": "VIX"}, inplace=True)
vix_mo["Date"] = pd.to_datetime(vix_mo["Date"]).dt.to_period("M").dt.to_timestamp()

vix_d = yf.download("^VIX", start="1996-01-01", end="2025-12-31", interval="1d",
                    auto_adjust=False, progress=False).reset_index()
if isinstance(vix_d.columns, pd.MultiIndex):
    vix_d.columns = vix_d.columns.get_level_values(0)
vix_d = vix_d[["Date", "Close"]].dropna().rename(columns={"Close": "VIX"})
vix_d["Date"] = pd.to_datetime(vix_d["Date"])

print("Downloading FF3 monthly...")
ff3_mo = web.DataReader("F-F_Research_Data_Factors", "famafrench",
                        start="1996-01-01", end="2025-12-31")[0].reset_index()
ff3_mo["Date"] = ff3_mo["Date"].dt.to_timestamp()

print("Downloading FF3 daily...")
ff3_d = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                       start="1996-01-01", end="2025-12-31")[0].reset_index()
ff3_d.rename(columns={"Date": "Date"}, inplace=True)
ff3_d["Date"] = pd.to_datetime(ff3_d["Date"])

gpr_mo_gep = (gpr_raw_df["GPRD"].resample("MS").mean().rename("GPR")
              .reset_index().rename(columns={"date": "Date"}))
gpr_d_df   = gpr_raw_df.rename_axis("Date").rename(columns={"GPRD": "GPR"}).reset_index()

# Monthly VIX setup
reg_mo = pd.merge(monthly_gep[["GEP_monthly"]].reset_index().rename(columns={"month": "Date"}),
                  vix_mo, on="Date", how="inner")
reg_mo = pd.merge(reg_mo, ff3_mo[["Date", "Mkt-RF", "SMB", "HML"]], on="Date", how="inner")
reg_mo = pd.merge(reg_mo, gpr_mo_gep, on="Date", how="left").set_index("Date").sort_index()
gep_mean_mo = reg_mo["GEP_monthly"][(reg_mo.index.year >= 1996) & (reg_mo.index.year <= 2025)].mean()
vix_mean_mo = reg_mo["VIX"][(reg_mo.index.year >= 1996) & (reg_mo.index.year <= 2025)].mean()
reg_mo["GEP_Norm"] = reg_mo["GEP_monthly"] / gep_mean_mo * 100
reg_mo["VIX_Norm"] = reg_mo["VIX"] / vix_mean_mo * 100

# Plot monthly VIX
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(reg_mo.index, reg_mo["VIX_Norm"], color=COL_VIX, linewidth=1.5, alpha=0.85, label="VIX")
ax.plot(reg_mo.index, reg_mo["GEP_Norm"], color=COL_GEP, linewidth=1.5, alpha=0.85, label="GEP (Robust min-2)")
ax.axhline(100, color="gray", linewidth=1, linestyle="--", alpha=0.7)
ax.set_title("GEP vs VIX — Normalized to 100 (1996–2025)", fontsize=14, pad=12)
ax.set_ylabel("Index (avg = 100)", fontsize=11)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(pd.Timestamp("1996-01-01"), pd.Timestamp("2025-12-31"))
plt.tight_layout()
plt.savefig(OUT / "Comparison_GEP_vs_VIX.png", dpi=150, bbox_inches="tight")
print("Saved: Comparison_GEP_vs_VIX.png"); plt.close()

# Monthly VIX regressions
for col in ["GEP_Norm", "Mkt-RF", "SMB", "HML", "VIX_Norm", "GPR"]:
    reg_mo[f"{col}_lag1"] = reg_mo[col].shift(1)

SUBS_MO = {k: (s, e) for k, (s, e) in SUBSAMPLES.items()}

print("\nMONTHLY REGRESSIONS: VIX (HAC, 4 lags)")
print("[M1] Contemp  VIX_t ~ GEP_t + FF3_t + GPR_t")
for name, (s, e) in SUBS_MO.items():
    run_ols("VIX_Norm", ["GEP_Norm", "Mkt-RF", "SMB", "HML", "GPR"],
            slice_df(reg_mo, s, e), name, "GEP_Norm", hac_lags=4)
print("[M2] Predictive  VIX_t ~ GEP_{t-1} + FF3_{t-1} + VIX_{t-1} + GPR_{t-1}")
for name, (s, e) in SUBS_MO.items():
    run_ols("VIX_Norm",
            ["GEP_Norm_lag1", "Mkt-RF_lag1", "SMB_lag1", "HML_lag1", "VIX_Norm_lag1", "GPR_lag1"],
            slice_df(reg_mo, s, e), name, "GEP_Norm_lag1", hac_lags=4)


# ═════════════════════════════════════════════════════════════════════════════
# GEP vs S&P 500
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nGEP vs S&P 500\n" + "="*60)

print("Downloading S&P 500...")
sp500_mo = yf.download("^GSPC", start="1995-12-01", end="2025-12-31",
                       interval="1mo", auto_adjust=True, progress=False)[["Close"]].copy()
sp500_mo.index = sp500_mo.index.to_period("M").to_timestamp()
sp500_mo.columns = ["sp500"]
sp500_mo = sp500_mo.sort_index()
sp500_mo["log_ret"] = np.log(sp500_mo["sp500"] / sp500_mo["sp500"].shift(1))
sp500_mo = sp500_mo.dropna()

sp500_d = yf.download("^GSPC", start="1995-12-01", end="2025-12-31",
                      interval="1d", auto_adjust=True, progress=False)[["Close"]].copy()
sp500_d.index = pd.to_datetime(sp500_d.index)
sp500_d.columns = ["sp500"]
sp500_d["log_ret"] = np.log(sp500_d["sp500"] / sp500_d["sp500"].shift(1))
sp500_d = sp500_d.dropna()

ff3_mo_raw = web.DataReader("F-F_Research_Data_Factors", "famafrench",
                            start="1995-12-01", end="2025-12-31")[0]
ff3_mo_raw.index = ff3_mo_raw.index.to_timestamp()
ff3_d_raw  = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                            start="1995-12-01", end="2025-12-31")[0]

gpr_mo_sp = gpr_raw_df["GPRD"].resample("MS").mean().rename("GPR")

df_mo_sp = (monthly_gep[["GEP_monthly"]]
            .join(sp500_mo[["log_ret"]], how="inner")
            .join(ff3_mo_raw[["Mkt-RF", "SMB", "HML"]], how="inner")
            .join(gpr_mo_sp, how="left"))
df_mo_sp["gep_pct"]     = df_mo_sp["GEP_monthly"] * 100
df_mo_sp["cum_log_ret"] = df_mo_sp["log_ret"].cumsum()

df_d_sp = (daily_gep[["GEP_daily"]]
           .join(sp500_d[["log_ret"]], how="inner")
           .join(ff3_d_raw[["Mkt-RF", "SMB", "HML"]], how="inner")
           .join(gpr_raw_df["GPRD"].rename("GPR"), how="left"))
df_d_sp["gep_pct"]     = df_d_sp["GEP_daily"] * 100
df_d_sp["cum_log_ret"] = df_d_sp["log_ret"].cumsum()

events_mo = {"1997-07": "Asian Crisis", "2001-09": "9/11", "2008-09": "GFC",
             "2020-03": "COVID-19", "2022-02": "Ukraine war", "2025-04": "Liberation Day"}

# Plot 1: monthly Z-score overlay
df_z = df_mo_sp.copy()
df_z["gep_z"]   = zscore(df_z["GEP_monthly"])
df_z["sp500_z"] = zscore(sp500_mo.reindex(df_z.index)["sp500"].dropna())

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df_z.index, df_z["gep_z"],   color=COL_GEP, linewidth=1.1, label="GEP (Z-score)", zorder=3)
ax.fill_between(df_z.index, df_z["gep_z"], alpha=0.12, color=COL_GEP)
ax.plot(df_z.index, df_z["sp500_z"], color="#E74C3C", linewidth=1.1, label="S&P 500 (Z-score)", zorder=3)
ax.fill_between(df_z.index, df_z["sp500_z"], alpha=0.08, color="#E74C3C")
ax.axhline(0, color="#555555", linewidth=0.6, linestyle="--")
for month_str, label in events_mo.items():
    ts = pd.Timestamp(month_str)
    if ts in df_z.index:
        y_val = df_z.loc[ts, "gep_z"]
        ax.annotate(label, xy=(ts, y_val), xytext=(0, 14), textcoords="offset points",
                    fontsize=7.5, ha="center", color="#333333",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
ax.set_title("GEP vs S&P 500 — Monthly Z-scores (1996–2025)", fontsize=13, pad=12)
ax.set_ylabel("Standard deviations from mean", fontsize=10)
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(OUT / "gep_vs_sp500_zscore.png", dpi=150, bbox_inches="tight")
print("Saved: gep_vs_sp500_zscore.png"); plt.close()

# Plot 2: monthly GEP level + S&P log returns
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                               gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08})
ax1.fill_between(df_mo_sp.index, df_mo_sp["gep_pct"], alpha=0.30, color=COL_GEP)
ax1.plot(df_mo_sp.index, df_mo_sp["gep_pct"], color="#1A5FA8", linewidth=1.1,
         label="GEP Robust min-2 (monthly, %)")
ax1.set_ylabel("Share of articles (%)", fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax1.spines[["top", "right", "bottom"]].set_visible(False)
ax1.legend(fontsize=10, framealpha=0.7, loc="upper left")
ax1.set_title("GEP vs S&P 500 Monthly Log Returns (1996–2025)", fontsize=13, pad=10)
colors_mo = ["#C0392B" if r < 0 else "#27AE60" for r in df_mo_sp["log_ret"]]
ax2.bar(df_mo_sp.index, df_mo_sp["log_ret"], color=colors_mo, width=20, alpha=0.75)
ax2.axhline(0, color="#333333", linewidth=0.7)
ax2.set_ylabel("Monthly log return", fontsize=10)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax2.spines[["top", "right"]].set_visible(False)
ax2r = ax2.twinx()
ax2r.plot(df_mo_sp.index, df_mo_sp["cum_log_ret"], color="#555555", lw=1.4, alpha=0.6,
          label="Cumul. log return", zorder=4)
ax2r.set_ylabel("Cumulative log return", fontsize=10, color="#555555")
ax2r.tick_params(axis="y", labelcolor="#555555")
ax2r.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2r.spines["top"].set_visible(False)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(OUT / "gep_vs_sp500_logret.png", dpi=150, bbox_inches="tight")
print("Saved: gep_vs_sp500_logret.png"); plt.close()

# S&P regressions
for df in (df_mo_sp, df_d_sp):
    for col in ["gep_pct", "Mkt-RF", "SMB", "HML", "GPR"]:
        df[f"{col}_lag1"] = df[col].shift(1)

SUBS_D = {k: (s.replace("-01", "-01-01").replace("-12", "-12-31") if s else s,
              e.replace("-01", "-01-01").replace("-12", "-12-31") if e else e)
          for k, (s, e) in SUBSAMPLES.items()}

print("\nMONTHLY SP500 REGRESSIONS (HAC, 4 lags)")
print("[A1] Contemp  log_ret_t ~ GEP_t + GPR_t")
for name, (s, e) in SUBS_MO.items():
    run_ols("log_ret", ["gep_pct", "GPR"], slice_df(df_mo_sp, s, e), name, "gep_pct", hac_lags=4)
print("[A2] Predictive  log_ret_t ~ GEP_{t-1} + FF3_{t-1} + GPR_{t-1}")
for name, (s, e) in SUBS_MO.items():
    run_ols("log_ret", ["gep_pct_lag1", "Mkt-RF_lag1", "SMB_lag1", "HML_lag1", "GPR_lag1"],
            slice_df(df_mo_sp, s, e), name, "gep_pct_lag1", hac_lags=4)

print("\nDAILY SP500 REGRESSIONS (HAC, 10 lags)")
print("[B1] Contemp  log_ret_t ~ GEP_t + GPR_t")
for name, (s, e) in SUBS_D.items():
    run_ols("log_ret", ["gep_pct", "GPR"], slice_df(df_d_sp, s, e), name, "gep_pct", hac_lags=10)
print("[B2] Predictive  log_ret_t ~ GEP_{t-1} + FF3_{t-1} + GPR_{t-1}")
for name, (s, e) in SUBS_D.items():
    run_ols("log_ret", ["gep_pct_lag1", "Mkt-RF_lag1", "SMB_lag1", "HML_lag1", "GPR_lag1"],
            slice_df(df_d_sp, s, e), name, "gep_pct_lag1", hac_lags=10)

print("\n═══ All comparison plots saved to output/comparisons/ ═══")
