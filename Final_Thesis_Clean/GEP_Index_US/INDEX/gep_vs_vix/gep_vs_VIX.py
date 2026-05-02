#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_gep_vs_vix.py
GEP Monthly vs VIX monthly — plot + regressions with FF3 and GPR controls.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import statsmodels.api as sm
import pandas_datareader.data as web

# ── File Paths ─────────────────────────────────────────────────────────────────
GEP_PATH    = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/GEP_Index_US/INDEX/data/GEP_Monthly_Robust_min2.csv"
OUT_PATH    = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/GEP_Index_US/INDEX/Comparison_GEP_vs_VIX.png"
GPR_D_PATH  = "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_daily_recent.xls"



# ── 1. Load GEP ────────────────────────────────────────────────────────────────
try:
    gep_df = pd.read_csv(GEP_PATH, parse_dates=["month"])
    gep_df["Date"] = pd.to_datetime(gep_df["month"])
except FileNotFoundError:
    print(f"ERROR: Cannot find GEP file at: {GEP_PATH}")
    exit(1)

# ── 2. Download VIX monthly ────────────────────────────────────────────────────
print("Downloading VIX data...")
vix_df = yf.download("^VIX", start="1996-01-01", end="2025-12-31", interval="1mo")
vix_df = vix_df.reset_index()
if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = vix_df.columns.get_level_values(0)
vix_df = vix_df[["Date", "Close"]].dropna().copy()
vix_df.rename(columns={"Close": "VIX_Close"}, inplace=True)
vix_df["Date"] = pd.to_datetime(vix_df["Date"]).dt.to_period('M').dt.to_timestamp()

# ── 3. Load GPR (daily → resample to monthly mean) ────────────────────────────
print("Loading GPR data...")
gpr_d = pd.read_excel(GPR_D_PATH)
gpr_d = gpr_d[['date', 'GPRD']].copy()
gpr_d['date'] = pd.to_datetime(gpr_d['date'])
gpr_d = gpr_d.set_index('date').sort_index()
gpr_d = gpr_d[~gpr_d.index.duplicated(keep='first')]
gpr_mo = gpr_d['GPRD'].resample('MS').mean().rename('GPR').reset_index()
gpr_mo.rename(columns={'date': 'Date'}, inplace=True)

# ── 4. Normalize (base 100 over 1996–2025) ────────────────────────────────────
vix_mean = vix_df[(vix_df["Date"].dt.year >= 1996) & (vix_df["Date"].dt.year <= 2025)]["VIX_Close"].mean()
gep_mean = gep_df[(gep_df["Date"].dt.year >= 1996) & (gep_df["Date"].dt.year <= 2025)]["GEP_monthly"].mean()

vix_df["VIX_Norm"] = (vix_df["VIX_Close"] / vix_mean) * 100
gep_df["GEP_Norm"] = (gep_df["GEP_monthly"] / gep_mean) * 100

# ── 5. Plot ────────────────────────────────────────────────────────────────────
print("Generating chart...")
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(vix_df["Date"], vix_df["VIX_Norm"], color="#8E44AD", linewidth=1.5, alpha=0.85, label="VIX Index (Close)")
ax.plot(gep_df["Date"], gep_df["GEP_Norm"], color="#378ADD", linewidth=1.5, alpha=0.85, label="GEP Monthly (Robust min-2)")
ax.axhline(100, color="gray", linewidth=1, linestyle="--", alpha=0.7)
ax.set_title("GEP vs VIX Volatility Index (Normalized to 100 over 1996–2025)", fontsize=14, pad=12)
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

# ── 6. Regression setup ────────────────────────────────────────────────────────
print("Downloading Fama-French 3 Factors (monthly)...")
ff3_dict = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
                          start='1996-01-01', end='2025-12-31')
ff3_df = ff3_dict[0].reset_index()
ff3_df['Date'] = ff3_df['Date'].dt.to_timestamp()

reg_df = pd.merge(gep_df[['Date', 'GEP_Norm']], vix_df[['Date', 'VIX_Norm']], on='Date', how='inner')
reg_df = pd.merge(reg_df, ff3_df[['Date', 'Mkt-RF', 'SMB', 'HML']], on='Date', how='inner')
reg_df = pd.merge(reg_df, gpr_mo[['Date', 'GPR']], on='Date', how='left')
reg_df = reg_df.set_index('Date').sort_index()

# Lag variables
for col in ['GEP_Norm', 'Mkt-RF', 'SMB', 'HML', 'VIX_Norm', 'GPR']:
    reg_df[f'{col}_lag1'] = reg_df[col].shift(1)

# ── Subsamples ─────────────────────────────────────────────────────────────────
SUBSAMPLES = {
    "Full sample (1996–2025)"            : (None,      None),
    "Pre-GEP era (1996–2017)"            : ("1996-01", "2017-12"),
    "Trade war (2018–2019)"              : ("2018-01", "2019-12"),
    "Russia-Ukraine (2022–2023)"         : ("2022-02", "2023-12"),
    "High-pressure combined (2018–2025)" : ("2018-01", "2025-12"),
}

def slice_df(df, start, end):
    d = df.copy()
    if start: d = d[d.index >= start]
    if end:   d = d[d.index <= end]
    return d

def run_ols_vix(y, X_cols, data, label, gep_col, hac_lags=4):
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
print("REGRESSION ANALYSIS: GEP vs VIX — MONTHLY (HAC Newey-West, 4 lags)")
print("="*75)

print("""
  [M1] Contemporaneous  VIX_t ~ GEP_t + FF3_t + GPR_t
       (VIX not mechanically related to Mkt-RF, so contemporaneous FF3 valid)""")
for name, (s, e) in SUBSAMPLES.items():
    run_ols_vix('VIX_Norm', ['GEP_Norm', 'Mkt-RF', 'SMB', 'HML', 'GPR'],
                slice_df(reg_df, s, e), name, 'GEP_Norm')

print("""
  [M2] Predictive  VIX_t ~ GEP_{t-1} + FF3_{t-1} + VIX_{t-1} + GPR_{t-1}
       (VIX lag controls for volatility persistence)""")
for name, (s, e) in SUBSAMPLES.items():
    run_ols_vix('VIX_Norm',
                ['GEP_Norm_lag1', 'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1',
                 'VIX_Norm_lag1', 'GPR_lag1'],
                slice_df(reg_df, s, e), name, 'GEP_Norm_lag1')

print("\n" + "="*75)