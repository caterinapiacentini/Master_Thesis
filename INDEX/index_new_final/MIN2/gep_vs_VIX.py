#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_gep_vs_vix.py

Downloads VIX (CBOE Volatility Index) monthly data via Yahoo Finance,
loads the local GEP Monthly index, normalizes both to an average of 100 
over the 1996-2025 period, and plots them overlaid for direct comparison.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import statsmodels.api as sm
import pandas_datareader.data as web

# ── File Paths ─────────────────────────────────────────────────────────────────
GEP_PATH = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/MIN2/GEP_Monthly_Robust_min2.csv"
OUT_PATH = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/MIN2/Comparison_GEP_vs_VIX.png"

# ── 1. Load and Prepare GEP Data ───────────────────────────────────────────────
try:
    gep_df = pd.read_csv(GEP_PATH, parse_dates=["month"])
    gep_df["Date"] = pd.to_datetime(gep_df["month"])
except FileNotFoundError:
    print(f"ERROR: Cannot find GEP file at: {GEP_PATH}")
    exit(1)

# ── 2. Download and Prepare VIX Data ───────────────────────────────────────────
print("Downloading VIX data from Yahoo Finance...")
# Download monthly VIX data from Jan 1996 onwards
vix_df = yf.download("^VIX", start="1996-01-01", end="2025-12-31", interval="1mo")

# Clean up the downloaded dataframe
vix_df = vix_df.reset_index()

# Handle potential MultiIndex columns from newer versions of yfinance
if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = vix_df.columns.get_level_values(0)

# Extract only the Date and Close price
vix_df = vix_df[["Date", "Close"]].dropna().copy()
vix_df.rename(columns={"Close": "VIX_Close"}, inplace=True)

# Normalize the VIX dates to the first day of the month to match GEP data format
vix_df["Date"] = pd.to_datetime(vix_df["Date"]).dt.to_period('M').dt.to_timestamp()

# ── 3. Normalization (Base 100 over 1996-2025) ─────────────────────────────────
# Filter data for the period of interest to calculate the mean
vix_base_period = vix_df[(vix_df["Date"].dt.year >= 1996) & (vix_df["Date"].dt.year <= 2025)]
gep_base_period = gep_df[(gep_df["Date"].dt.year >= 1996) & (gep_df["Date"].dt.year <= 2025)]

vix_mean = vix_base_period["VIX_Close"].mean()
gep_mean = gep_base_period["GEP_monthly"].mean()

# Apply normalization to the entire dataset
vix_df["VIX_Norm"] = (vix_df["VIX_Close"] / vix_mean) * 100
gep_df["GEP_Norm"] = (gep_df["GEP_monthly"] / gep_mean) * 100

# ── 4. Generate Plot ───────────────────────────────────────────────────────────
print("Generating chart...")
fig, ax = plt.subplots(figsize=(16, 7))

# Plot both series on the same axis
ax.plot(vix_df["Date"], vix_df["VIX_Norm"], color="#8E44AD", linewidth=1.5, alpha=0.85, label="VIX Index (Close)")
ax.plot(gep_df["Date"], gep_df["GEP_Norm"], color="#378ADD", linewidth=1.5, alpha=0.85, label="GEP Monthly (Robust min-2)")

# Add a baseline at 100
ax.axhline(100, color="gray", linewidth=1, linestyle="--", alpha=0.7)

# Chart styling and labels
ax.set_title("GEP vs VIX Volatility Index (Normalized to 100 over 1996-2025)", fontsize=14, pad=12)
ax.set_ylabel("Index (Average = 100)", fontsize=11)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", framealpha=0.9, fontsize=11)

# X-axis formatting (Years)
ax.xaxis.set_major_locator(mdates.YearLocator(2)) # Labels every 2 years
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Force x-axis limits from 1996 to 2025
ax.set_xlim(pd.Timestamp("1996-01-01"), pd.Timestamp("2025-12-31"))

plt.tight_layout()

# Save the chart
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Chart successfully saved to: {OUT_PATH}")

# plt.show()
plt.close()

# ── 5. Regression Analysis with Fama-French 3 Factors ──────────────────────────
print("\n" + "="*60)
print("REGRESSION ANALYSIS: GEP vs VIX (Controlling for FF3)")
print("="*60)



print("Downloading Fama-French 3 Factors from Kenneth French Data Library...")
# Download FF3 monthly data ('F-F_Research_Data_Factors' is the standard dataset)
ff3_dict = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start='1996-01-01', end='2025-12-31')
ff3_df = ff3_dict[0] # The first item in the returned dictionary contains the factors
ff3_df = ff3_df.reset_index()

# Rename and format the date column to match GEP/VIX (First day of the month)
ff3_df.rename(columns={'Date': 'Date_FF'}, inplace=True)
ff3_df['Date'] = ff3_df['Date_FF'].dt.to_timestamp()

# Merge all three datasets perfectly on the Date column
reg_df = pd.merge(gep_df[['Date', 'GEP_Norm']], vix_df[['Date', 'VIX_Norm']], on='Date', how='inner')
reg_df = pd.merge(reg_df, ff3_df[['Date', 'Mkt-RF', 'SMB', 'HML']], on='Date', how='inner')

# ---- Model 1: Contemporaneous ----
print("\n--- Model 1: Contemporaneous (GEP & FF3 predicting VIX) ---")
# Independent variables: GEP + Market Premium + Size + Value
X1 = reg_df[['GEP_Norm', 'Mkt-RF', 'SMB', 'HML']]
X1 = sm.add_constant(X1) # Intercept
y1 = reg_df['VIX_Norm']

model1 = sm.OLS(y1, X1).fit()
# Print a condensed, clean summary
print(model1.summary().tables[1])
print(f"R-squared: {model1.rsquared:.4f} | p-value (GEP_Norm): {model1.pvalues['GEP_Norm']:.4f}")

# ---- Model 2: Predictive (Lagged 1 Month) ----
print("\n--- Model 2: Predictive (Lagged GEP & FF3 predicting this month's VIX) ---")
# Create lagged versions of all predictors to avoid look-ahead bias
reg_df['GEP_Norm_Lag1'] = reg_df['GEP_Norm'].shift(1)
reg_df['Mkt-RF_Lag1'] = reg_df['Mkt-RF'].shift(1)
reg_df['SMB_Lag1'] = reg_df['SMB'].shift(1)
reg_df['HML_Lag1'] = reg_df['HML'].shift(1)

# Drop the first row which is now NaN due to the shift
reg_df_clean = reg_df.dropna() 

X2 = reg_df_clean[['GEP_Norm_Lag1', 'Mkt-RF_Lag1', 'SMB_Lag1', 'HML_Lag1']]
X2 = sm.add_constant(X2)
y2 = reg_df_clean['VIX_Norm']

model2 = sm.OLS(y2, X2).fit()
print(model2.summary().tables[1])
print(f"R-squared: {model2.rsquared:.4f} | p-value (GEP_Norm_Lag1): {model2.pvalues['GEP_Norm_Lag1']:.4f}")
print("="*60 + "\n")

