#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_gep_daily_vs_vix.py

Downloads VIX daily data, loads the local GEP Daily index, and fetches 
daily Fama-French 3 Factors. It normalizes both series to 100, plots them 
(with a 30-day moving average for readability), and runs OLS regressions 
at the daily level.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import statsmodels.api as sm
import pandas_datareader.data as web

# ── File Paths ─────────────────────────────────────────────────────────────────
# Make sure this points to your DAILY GEP csv
GEP_PATH = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/MIN2/GEP_Daily_Robust_min2.csv"
OUT_PATH = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/MIN2/Comparison_Daily_GEP_vs_VIX.png"

# ── 1. Load and Prepare GEP Daily Data ─────────────────────────────────────────
try:
    gep_df = pd.read_csv(GEP_PATH, parse_dates=["date"])
    gep_df.rename(columns={"date": "Date"}, inplace=True)
except FileNotFoundError:
    print(f"ERROR: Cannot find GEP file at: {GEP_PATH}")
    exit(1)

# ── 2. Download and Prepare VIX Daily Data ─────────────────────────────────────
print("Downloading Daily VIX data from Yahoo Finance...")
vix_df = yf.download("^VIX", start="1996-01-01", end="2025-12-31", interval="1d")
vix_df = vix_df.reset_index()

if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = vix_df.columns.get_level_values(0)

vix_df = vix_df[["Date", "Close"]].dropna().copy()
vix_df.rename(columns={"Close": "VIX_Close"}, inplace=True)
vix_df["Date"] = pd.to_datetime(vix_df["Date"])

# ── 3. Download Fama-French Daily Data ─────────────────────────────────────────
print("Downloading Fama-French 3 Factors (Daily) from Kenneth French Library...")
ff3_dict = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1996-01-01', end='2025-12-31')
ff3_df = ff3_dict[0].reset_index()
ff3_df.rename(columns={'Date': 'Date_FF'}, inplace=True)
ff3_df['Date'] = pd.to_datetime(ff3_df['Date_FF'])

# ── 4. Merge and Normalize ─────────────────────────────────────────────────────
# Inner merge aligns all datasets to valid TRADING DAYS only (drops weekends/holidays)
df = pd.merge(gep_df[['Date', 'GEP_daily']], vix_df, on='Date', how='inner')
df = pd.merge(df, ff3_df[['Date', 'Mkt-RF', 'SMB', 'HML']], on='Date', how='inner')

# Calculate mean over the overlapping trading days
gep_mean = df["GEP_daily"].mean()
vix_mean = df["VIX_Close"].mean()

# Normalize to 100
df["GEP_Norm"] = (df["GEP_daily"] / gep_mean) * 100
df["VIX_Norm"] = (df["VIX_Close"] / vix_mean) * 100

# Calculate 30-day Moving Averages for the plot (smooths out the daily noise)
df["GEP_Norm_MA30"] = df["GEP_Norm"].rolling(window=30, min_periods=1).mean()
df["VIX_Norm_MA30"] = df["VIX_Norm"].rolling(window=30, min_periods=1).mean()

# ── 5. Generate Plot ───────────────────────────────────────────────────────────
print("Generating chart...")
fig, ax = plt.subplots(figsize=(16, 7))

# Plot raw daily data as highly transparent, thin lines/dots in the background
ax.plot(df["Date"], df["VIX_Norm"], color="#8E44AD", linewidth=0.5, alpha=0.15)
ax.plot(df["Date"], df["GEP_Norm"], color="#378ADD", linewidth=0.5, alpha=0.15)

# Plot 30-day Moving Averages as bold, clear lines
ax.plot(df["Date"], df["VIX_Norm_MA30"], color="#8E44AD", linewidth=2, alpha=0.95, label="VIX Index (30-Day Trend)")
ax.plot(df["Date"], df["GEP_Norm_MA30"], color="#378ADD", linewidth=2, alpha=0.95, label="Daily GEP (30-Day Trend)")

# Add a baseline at 100
ax.axhline(100, color="gray", linewidth=1, linestyle="--", alpha=0.7)

ax.set_title("Daily GEP vs VIX Volatility Index (Normalized to 100 over 1996-2025)", fontsize=14, pad=12)
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
print(f"Chart successfully saved to: {OUT_PATH}")
plt.close()

# ── 6. Regression Analysis (Daily Level) ───────────────────────────────────────
print("\n" + "="*60)
print("DAILY REGRESSION ANALYSIS: GEP vs VIX (Controlling for FF3)")
print("="*60)

# ---- Model 1: Contemporaneous (Same Day) ----
print("\n--- Model 1: Contemporaneous (Today's GEP & FF3 predicting Today's VIX) ---")
X1 = df[['GEP_Norm', 'Mkt-RF', 'SMB', 'HML']]
X1 = sm.add_constant(X1)
y1 = df['VIX_Norm']

model1 = sm.OLS(y1, X1).fit()
print(model1.summary().tables[1])
print(f"R-squared: {model1.rsquared:.4f} | p-value (GEP_Norm): {model1.pvalues['GEP_Norm']:.4f}")

# ---- Model 2: Predictive (Lagged 1 Trading Day) ----
print("\n--- Model 2: Predictive (Yesterday's GEP & FF3 predicting Today's VIX) ---")
# Shift by 1 row (which equals 1 trading day because we dropped weekends)
df['GEP_Norm_Lag1'] = df['GEP_Norm'].shift(1)
df['Mkt-RF_Lag1'] = df['Mkt-RF'].shift(1)
df['SMB_Lag1'] = df['SMB'].shift(1)
df['HML_Lag1'] = df['HML'].shift(1)

df_clean = df.dropna()

X2 = df_clean[['GEP_Norm_Lag1', 'Mkt-RF_Lag1', 'SMB_Lag1', 'HML_Lag1']]
X2 = sm.add_constant(X2)
y2 = df_clean['VIX_Norm']

model2 = sm.OLS(y2, X2).fit()
print(model2.summary().tables[1])
print(f"R-squared: {model2.rsquared:.4f} | p-value (GEP_Norm_Lag1): {model2.pvalues['GEP_Norm_Lag1']:.4f}")
print("="*60 + "\n")