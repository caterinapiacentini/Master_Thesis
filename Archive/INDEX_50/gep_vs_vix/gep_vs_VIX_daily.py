#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_gep_daily_vs_vix.py
GEP Daily vs VIX daily — plot + regressions with FF3 and GPR controls.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import statsmodels.api as sm
import pandas_datareader.data as web

# ── File Paths ─────────────────────────────────────────────────────────────────
GEP_PATH   = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/INDEX_50/data/GEP_Daily_Robust_min2.csv"
OUT_PATH   = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/INDEX_50/Comparison_Daily_GEP_vs_VIX.png"
GPR_D_PATH = "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_daily_recent.xls"

# ── 1. Load GEP daily ──────────────────────────────────────────────────────────
try:
    gep_df = pd.read_csv(GEP_PATH, parse_dates=["date"])
    gep_df.rename(columns={"date": "Date"}, inplace=True)
except FileNotFoundError:
    print(f"ERROR: Cannot find GEP file at: {GEP_PATH}")
    exit(1)

# ── 2. Download VIX daily ──────────────────────────────────────────────────────
print("Downloading daily VIX data...")
vix_df = yf.download("^VIX", start="1996-01-01", end="2025-12-31", interval="1d")
vix_df = vix_df.reset_index()
if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = vix_df.columns.get_level_values(0)
vix_df = vix_df[["Date", "Close"]].dropna().copy()
vix_df.rename(columns={"Close": "VIX_Close"}, inplace=True)
vix_df["Date"] = pd.to_datetime(vix_df["Date"])

# ── 3. Download FF3 daily ──────────────────────────────────────────────────────
print("Downloading Fama-French 3 Factors (daily)...")
ff3_dict = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench',
                          start='1996-01-01', end='2025-12-31')
ff3_df = ff3_dict[0].reset_index()
ff3_df.rename(columns={'Date': 'Date_FF'}, inplace=True)
ff3_df['Date'] = pd.to_datetime(ff3_df['Date_FF'])

# ── 4. Load GPR daily ──────────────────────────────────────────────────────────
print("Loading GPR data...")
gpr_d = pd.read_excel(GPR_D_PATH)
gpr_d = gpr_d[['date', 'GPRD']].copy()
gpr_d['date'] = pd.to_datetime(gpr_d['date'])
gpr_d = gpr_d.rename(columns={'date': 'Date', 'GPRD': 'GPR'})
gpr_d = gpr_d.set_index('Date').sort_index()
gpr_d = gpr_d[~gpr_d.index.duplicated(keep='first')]

# ── 5. Merge and normalize ─────────────────────────────────────────────────────
df = pd.merge(gep_df[['Date', 'GEP_daily']], vix_df, on='Date', how='inner')
df = pd.merge(df, ff3_df[['Date', 'Mkt-RF', 'SMB', 'HML']], on='Date', how='inner')
df = df.set_index('Date').sort_index()
df = df.join(gpr_d[['GPR']], how='left')

gep_mean = df["GEP_daily"].mean()
vix_mean = df["VIX_Close"].mean()
df["GEP_Norm"] = (df["GEP_daily"] / gep_mean) * 100
df["VIX_Norm"] = (df["VIX_Close"] / vix_mean) * 100
df["GEP_Norm_MA30"] = df["GEP_Norm"].rolling(window=30, min_periods=1).mean()
df["VIX_Norm_MA30"] = df["VIX_Norm"].rolling(window=30, min_periods=1).mean()

# ── 6. Plot ────────────────────────────────────────────────────────────────────
print("Generating chart...")
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(df.index, df["VIX_Norm"], color="#8E44AD", linewidth=0.5, alpha=0.15)
ax.plot(df.index, df["GEP_Norm"], color="#378ADD", linewidth=0.5, alpha=0.15)
ax.plot(df.index, df["VIX_Norm_MA30"], color="#8E44AD", linewidth=2, alpha=0.95, label="VIX Index (30-Day Trend)")
ax.plot(df.index, df["GEP_Norm_MA30"], color="#378ADD", linewidth=2, alpha=0.95, label="Daily GEP (30-Day Trend)")
ax.axhline(100, color="gray", linewidth=1, linestyle="--", alpha=0.7)
ax.set_title("Daily GEP vs VIX Volatility Index (Normalized to 100 over 1996–2025)", fontsize=14, pad=12)
ax.set_ylabel("Index (Average = 100)", fontsize=11)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(pd.Timestamp("1996-01-01"), pd.Timestamp("2025-12-31"))
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Chart saved to: {OUT_PATH}")
plt.close()

# ── 7. Regression setup ────────────────────────────────────────────────────────
for col in ['GEP_Norm', 'Mkt-RF', 'SMB', 'HML', 'VIX_Norm', 'GPR']:
    df[f'{col}_lag1'] = df[col].shift(1)

# ── Subsamples ─────────────────────────────────────────────────────────────────
SUBSAMPLES = {
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

def run_ols_vix(y, X_cols, data, label, gep_col, hac_lags=10):
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
    print(f"  {label:<40}  β={coef:+.4f}  SE={se:.4f}  t={tstat:+.2f}  "
          f"p={pval:.3f} {stars}  R²={model.rsquared:.4f}  N={int(model.nobs)}")

# ── Regressions ────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("REGRESSION ANALYSIS: GEP vs VIX — DAILY (HAC Newey-West, 10 lags)")
print("="*75)

print("""
  [M1] Contemporaneous  VIX_t ~ GEP_t + FF3_t + GPR_t
       (VIX not mechanically related to Mkt-RF, so contemporaneous FF3 valid)""")
for name, (s, e) in SUBSAMPLES.items():
    run_ols_vix('VIX_Norm', ['GEP_Norm', 'Mkt-RF', 'SMB', 'HML', 'GPR'],
                slice_df(df, s, e), name, 'GEP_Norm')

print("""
  [M2] Predictive  VIX_t ~ GEP_{t-1} + FF3_{t-1} + VIX_{t-1} + GPR_{t-1}
       (VIX lag controls for volatility persistence)""")
for name, (s, e) in SUBSAMPLES.items():
    run_ols_vix('VIX_Norm',
                ['GEP_Norm_lag1', 'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1',
                 'VIX_Norm_lag1', 'GPR_lag1'],
                slice_df(df, s, e), name, 'GEP_Norm_lag1')

print("\n" + "="*75)