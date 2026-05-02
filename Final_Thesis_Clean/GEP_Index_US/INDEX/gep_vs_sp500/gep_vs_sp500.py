#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gep_vs_sp500_robust_min2.py
GEP Robust min-2 index vs S&P 500 — three comparison plots + regressions
with FF3 and GPR controls, full sample and subsamples.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf
import statsmodels.api as sm
import pandas_datareader.data as web

# ── Hardcoded Paths ────────────────────────────────────────────────────────────
BASE     = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/GEP_Index_US/INDEX"
GPR_D_PATH  = "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_daily_recent.xls"
GPR_MO_PATH = "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_export.xls"

# ── Load GEP data ──────────────────────────────────────────────────────────────
monthly = pd.read_csv(os.path.join(BASE, "data", "GEP_Monthly_Robust_min2.csv"))
monthly["month"] = pd.to_datetime(monthly["month"])
monthly = monthly.set_index("month").sort_index()

daily = pd.read_csv(os.path.join(BASE, "data", "GEP_Daily_Robust_min2.csv"))
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.set_index("date").sort_index()
daily = daily[daily["n_articles"] > 0]

# ── Download S&P 500 ───────────────────────────────────────────────────────────
print("Downloading S&P 500 data...")
sp500_mo_raw = yf.download("^GSPC", start="1995-12-01", end="2025-12-31",
                           interval="1mo", auto_adjust=True, progress=False)
sp500_mo = sp500_mo_raw[["Close"]].copy()
sp500_mo.index = sp500_mo.index.to_period("M").to_timestamp()
sp500_mo.columns = ["sp500"]
sp500_mo = sp500_mo.sort_index()
sp500_mo["log_ret"] = np.log(sp500_mo["sp500"] / sp500_mo["sp500"].shift(1))
sp500_mo = sp500_mo.dropna()

sp500_d_raw = yf.download("^GSPC", start="1995-12-01", end="2025-12-31",
                          interval="1d", auto_adjust=True, progress=False)
sp500_d = sp500_d_raw[["Close"]].copy()
sp500_d.index = pd.to_datetime(sp500_d.index)
sp500_d.columns = ["sp500"]
sp500_d = sp500_d.sort_index()
sp500_d["log_ret"] = np.log(sp500_d["sp500"] / sp500_d["sp500"].shift(1))
sp500_d = sp500_d.dropna()

# ── Download Fama-French 3 Factors ─────────────────────────────────────────────
print("Downloading Fama-French 3 Factors...")
ff3_mo_raw = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
                            start='1995-12-01', end='2025-12-31')[0]
ff3_mo_raw.index = ff3_mo_raw.index.to_timestamp()

ff3_d_raw = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench',
                           start='1995-12-01', end='2025-12-31')[0]
ff3_d_raw.index = pd.to_datetime(ff3_d_raw.index)

# ── Load GPR ───────────────────────────────────────────────────────────────────
print("Loading GPR data...")
gpr_d = pd.read_excel(GPR_D_PATH)
gpr_d = gpr_d[['date', 'GPRD']].copy()
gpr_d['date'] = pd.to_datetime(gpr_d['date'])
gpr_d = gpr_d.set_index('date').sort_index()
gpr_d = gpr_d[~gpr_d.index.duplicated(keep='first')]

gpr_mo = gpr_d['GPRD'].resample('MS').mean().rename('GPR')

# ── Merge ──────────────────────────────────────────────────────────────────────
df_mo = monthly[["GEP_monthly"]].join(sp500_mo[["sp500", "log_ret"]], how="inner")
df_mo = df_mo.join(ff3_mo_raw[['Mkt-RF', 'SMB', 'HML']], how="inner")
df_mo = df_mo.join(gpr_mo, how="left")
df_mo["gep_pct"]     = df_mo["GEP_monthly"] * 100
df_mo["cum_log_ret"] = df_mo["log_ret"].cumsum()

df_d = daily[["GEP_daily"]].join(sp500_d[["log_ret"]], how="inner")
df_d = df_d.join(ff3_d_raw[['Mkt-RF', 'SMB', 'HML']], how="inner")
df_d = df_d.join(gpr_d[['GPRD']].rename(columns={'GPRD': 'GPR'}), how="left")
df_d["gep_pct"]     = df_d["GEP_daily"] * 100
df_d["cum_log_ret"] = df_d["log_ret"].cumsum()

# ── Events ─────────────────────────────────────────────────────────────────────
events_mo = {
    "1997-07": "Asian Crisis",
    "2001-09": "9/11",
    "2008-09": "GFC",
    "2020-03": "COVID-19",
    "2022-02": "Ukraine war",
    "2025-04": "Liberation Day",
}
events_d = {
    "1997-07-02": "Asian\nCrisis",
    "2001-09-11": "9/11",
    "2008-09-15": "GFC",
    "2020-03-16": "COVID-19",
    "2022-02-24": "Ukraine\nWar",
    "2025-04-02": "Liberation\nDay",
}

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Monthly Z-score overlay
# ══════════════════════════════════════════════════════════════════════════════
print("Generating plots...")
df_z = df_mo.copy()
df_z["gep_z"]   = (df_z["GEP_monthly"] - df_z["GEP_monthly"].mean()) / df_z["GEP_monthly"].std()
df_z["sp500_z"] = (df_z["sp500"]       - df_z["sp500"].mean())       / df_z["sp500"].std()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df_z.index, df_z["gep_z"],   color="#378ADD", linewidth=1.1,
        label="GEP Robust min-2 (Z-score)", zorder=3)
ax.fill_between(df_z.index, df_z["gep_z"], alpha=0.12, color="#378ADD")
ax.plot(df_z.index, df_z["sp500_z"], color="#E74C3C", linewidth=1.1,
        label="S&P 500 (Z-score)", zorder=3)
ax.fill_between(df_z.index, df_z["sp500_z"], alpha=0.08, color="#E74C3C")
ax.axhline(0, color="#555555", linewidth=0.6, linestyle="--")
for month_str, label in events_mo.items():
    ts = pd.Timestamp(month_str)
    if ts in df_z.index:
        y_val = df_z.loc[ts, "gep_z"]
        ax.annotate(label, xy=(ts, y_val),
                    xytext=(0, 14), textcoords="offset points",
                    fontsize=7.5, ha="center", color="#333333",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
ax.set_title("GEP Robust min-2 vs S&P 500 — Monthly Z-scores (1996–2025)",
             fontsize=13, pad=12)
ax.set_ylabel("Standard deviations from mean", fontsize=10)
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(os.path.join(BASE, "gep_vs_sp500_zscore.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Two-panel: GEP monthly level + S&P monthly log returns
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                               gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08})
ax1.fill_between(df_mo.index, df_mo["gep_pct"], alpha=0.30, color="#378ADD")
ax1.plot(df_mo.index, df_mo["gep_pct"], color="#1A5FA8", linewidth=1.1,
         label="GEP Robust min-2 (monthly, %)")
ax1.set_ylabel("Share of articles (%)", fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.legend(fontsize=10, framealpha=0.7, loc="upper left")
ax1.set_title("GEP Robust min-2 vs S&P 500 Monthly Log Returns (1996–2025)",
              fontsize=13, pad=10)

colors = ["#C0392B" if r < 0 else "#27AE60" for r in df_mo["log_ret"]]
ax2.bar(df_mo.index, df_mo["log_ret"], color=colors, width=20, alpha=0.75,
        label="S&P 500 log return")
ax2.axhline(0, color="#333333", linewidth=0.7)
ax2.set_ylabel("Monthly log return", fontsize=10)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2r = ax2.twinx()
ax2r.plot(df_mo.index, df_mo["cum_log_ret"], color="#555555", linewidth=1.4,
          linestyle="-", alpha=0.6, label="Cumulative log return", zorder=4)
ax2r.set_ylabel("Cumulative log return", fontsize=10, color="#555555")
ax2r.tick_params(axis="y", labelcolor="#555555")
ax2r.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2r.spines["top"].set_visible(False)

handles1, labels1 = ax2.get_legend_handles_labels()
handles2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(handles1 + handles2, labels1 + labels2,
           fontsize=10, framealpha=0.7, loc="upper left")
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

for month_str, label in events_mo.items():
    ts = pd.Timestamp(month_str)
    if ts in df_mo.index:
        for ax in (ax1, ax2):
            ax.axvline(ts, color="#888888", linewidth=0.8, linestyle="--", zorder=2)
        ax1.text(ts, ax1.get_ylim()[1] * 0.97, label,
                 fontsize=7, ha="center", va="top", color="#444444")

plt.tight_layout()
plt.savefig(os.path.join(BASE, "gep_vs_sp500_logret.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Two-panel: GEP daily level + S&P daily log returns
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 7), sharex=True,
                               gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08})
ax1.fill_between(df_d.index, df_d["gep_pct"], alpha=0.30, color="#378ADD")
ax1.plot(df_d.index, df_d["gep_pct"], color="#1A5FA8", linewidth=0.7,
         label="GEP Robust min-2 (daily, %)")
ax1.set_ylabel("Share of articles (%)", fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.legend(fontsize=10, framealpha=0.7, loc="upper left")
ax1.set_title("GEP Robust min-2 vs S&P 500 Daily Log Returns (1996–2025)",
              fontsize=13, pad=10)

colors = ["#C0392B" if r < 0 else "#27AE60" for r in df_d["log_ret"]]
ax2.bar(df_d.index, df_d["log_ret"], color=colors, width=1, alpha=0.75,
        label="S&P 500 log return")
ax2.axhline(0, color="#333333", linewidth=0.7)
ax2.set_ylabel("Daily log return", fontsize=10)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2r = ax2.twinx()
ax2r.plot(df_d.index, df_d["cum_log_ret"], color="#555555", linewidth=1.4,
          linestyle="-", alpha=0.6, label="Cumulative log return", zorder=4)
ax2r.set_ylabel("Cumulative log return", fontsize=10, color="#555555")
ax2r.tick_params(axis="y", labelcolor="#555555")
ax2r.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2r.spines["top"].set_visible(False)

handles1, labels1 = ax2.get_legend_handles_labels()
handles2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(handles1 + handles2, labels1 + labels2,
           fontsize=10, framealpha=0.7, loc="upper left")
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

for date_str, label in events_d.items():
    ts = pd.Timestamp(date_str)
    if ts not in df_d.index:
        idx = df_d.index.searchsorted(ts)
        if idx < len(df_d.index):
            ts = df_d.index[idx]
        else:
            continue
    for ax in (ax1, ax2):
        ax.axvline(ts, color="#888888", linewidth=0.8, linestyle="--", zorder=2)
    ax1.text(ts, ax1.get_ylim()[1] * 0.97, label,
             fontsize=7, ha="center", va="top", color="#444444", linespacing=1.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE, "gep_vs_sp500_logret_daily.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# ── Lag variables ──────────────────────────────────────────────────────────────
for df in (df_mo, df_d):
    for col in ['gep_pct', 'Mkt-RF', 'SMB', 'HML', 'GPR']:
        df[f'{col}_lag1'] = df[col].shift(1)

# ── Subsamples ─────────────────────────────────────────────────────────────────
SUBSAMPLES_MO = {
    "Full sample (1996–2025)"            : (None,      None),
    "Pre-GEP era (1996–2017)"            : ("1996-01", "2017-12"),
    "Trade war (2018–2019)"              : ("2018-01", "2019-12"),
    "Russia-Ukraine (2022–2023)"         : ("2022-02", "2023-12"),
    "High-pressure combined (2018–2025)" : ("2018-01", "2025-12"),
}
SUBSAMPLES_D = {
    "Full sample (1996–2025)"            : (None,           None),
    "Pre-GEP era (1996–2017)"            : ("1996-01-01",   "2017-12-31"),
    "Trade war (2018–2019)"              : ("2018-01-01",   "2019-12-31"),
    "Russia-Ukraine (2022–2023)"         : ("2022-02-24",   "2023-12-31"),
    "High-pressure combined (2018–2025)" : ("2018-01-01",   "2025-12-31"),
}

def slice_df(df, start, end):
    d = df.copy()
    if start: d = d[d.index >= start]
    if end:   d = d[d.index <= end]
    return d

def run_ols(y, X_cols, data, label, gep_col, hac_lags):
    clean = data[X_cols + [y]].dropna()
    if len(clean) < 30:
        print(f"  {label:<40}  [skipped — N={len(clean)} too small]")
        return
    X = sm.add_constant(clean[X_cols])
    model = sm.OLS(clean[y], X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    coef  = model.params[gep_col]
    se    = model.bse[gep_col]
    tstat = model.tvalues[gep_col]
    pval  = model.pvalues[gep_col]
    stars = "***" if pval<0.01 else ("**" if pval<0.05 else ("*" if pval<0.10 else "   "))
    print(f"  {label:<40}  β={coef:+.5f}  SE={se:.5f}  t={tstat:+.2f}  "
          f"p={pval:.3f} {stars}  R²={model.rsquared:.4f}  N={int(model.nobs)}")

# ── MONTHLY ────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("MONTHLY REGRESSIONS  (HAC Newey-West, 4 lags)")
print("="*80)

print("""
  [A1] Contemporaneous  log_ret_t ~ GEP_t + GPR_t
       (bivariate with GPR — FF3 excluded to avoid near-tautology with Mkt-RF)""")
for name, (s, e) in SUBSAMPLES_MO.items():
    run_ols('log_ret', ['gep_pct', 'GPR'],
            slice_df(df_mo, s, e), name, 'gep_pct', hac_lags=4)

print("""
  [A2] Predictive  log_ret_t ~ GEP_{t-1} + FF3_{t-1} + GPR_{t-1}
       (all regressors lagged — no collinearity issue)""")
for name, (s, e) in SUBSAMPLES_MO.items():
    run_ols('log_ret',
            ['gep_pct_lag1', 'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1', 'GPR_lag1'],
            slice_df(df_mo, s, e), name, 'gep_pct_lag1', hac_lags=4)

# ── DAILY ──────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("DAILY REGRESSIONS  (HAC Newey-West, 10 lags)")
print("="*80)

print("""
  [B1] Contemporaneous  log_ret_t ~ GEP_t + GPR_t
       (bivariate with GPR — FF3 excluded to avoid near-tautology with Mkt-RF)""")
for name, (s, e) in SUBSAMPLES_D.items():
    run_ols('log_ret', ['gep_pct', 'GPR'],
            slice_df(df_d, s, e), name, 'gep_pct', hac_lags=10)

print("""
  [B2] Predictive  log_ret_t ~ GEP_{t-1} + FF3_{t-1} + GPR_{t-1}
       (all regressors lagged — no collinearity issue)""")
for name, (s, e) in SUBSAMPLES_D.items():
    run_ols('log_ret',
            ['gep_pct_lag1', 'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1', 'GPR_lag1'],
            slice_df(df_d, s, e), name, 'gep_pct_lag1', hac_lags=10)

print("\n" + "="*80)