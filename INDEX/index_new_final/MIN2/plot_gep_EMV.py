#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_gep_vs_emv.py

Loads the Overall EMV Tracker from a local Excel file and the local GEP Monthly.
Normalizes both series to 100 by calculating the mean over the 1985-2025 period,
then plots them overlaid on a single chart for direct comparison.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── File Paths ─────────────────────────────────────────────────────────────────
EMV_PATH = "/Users/catepiacentini/Desktop/EMV_Data.xlsx"
GEP_PATH = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/MIN2/GEP_Monthly_Robust_min2.csv"
OUT_PATH = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/MIN2/Comparison_GEP_vs_EMV.png"

# ── 1. Load and Prepare EMV Data ───────────────────────────────────────────────
# Read only the first 3 columns (0, 1, 2) and rename them for convenience
try:
    emv_df = pd.read_excel(EMV_PATH, usecols=[0, 1, 2], names=["Year", "Month", "EMV_Tracker"])
except FileNotFoundError:
    print(f"ERROR: Cannot find EMV file at: {EMV_PATH}")
    exit(1)

# Clean data: Force numeric conversion to handle footer text (e.g., "Source: ...")
emv_df["Year"] = pd.to_numeric(emv_df["Year"], errors="coerce")
emv_df["Month"] = pd.to_numeric(emv_df["Month"], errors="coerce")
emv_df["EMV_Tracker"] = pd.to_numeric(emv_df["EMV_Tracker"], errors="coerce")

# Drop rows that became NaN (the text rows at the bottom of the Excel file)
emv_df = emv_df.dropna(subset=["Year", "Month", "EMV_Tracker"]).copy()

# Ensure Year and Month are integers before creating dates
emv_df["Year"] = emv_df["Year"].astype(int)
emv_df["Month"] = emv_df["Month"].astype(int)

# Create a date column by combining Year and Month (setting day to 1)
emv_df["Date"] = pd.to_datetime(emv_df[["Year", "Month"]].assign(DAY=1))

# ── 2. Load and Prepare GEP Data ───────────────────────────────────────────────
try:
    gep_df = pd.read_csv(GEP_PATH, parse_dates=["month"])
    gep_df["Date"] = pd.to_datetime(gep_df["month"])
except FileNotFoundError:
    print(f"ERROR: Cannot find GEP file at: {GEP_PATH}")
    exit(1)

# ── 3. Normalization (Base 100 over 1996-2025) ─────────────────────────────────
# Filter data for the period of interest to calculate the mean
emv_base_period = emv_df[(emv_df["Date"].dt.year >= 1996) & (emv_df["Date"].dt.year <= 2025)]
gep_base_period = gep_df[(gep_df["Date"].dt.year >= 1996) & (gep_df["Date"].dt.year <= 2025)]

emv_mean = emv_base_period["EMV_Tracker"].mean()
gep_mean = gep_base_period["GEP_monthly"].mean()

# Apply normalization to the entire dataset
emv_df["EMV_Norm"] = (emv_df["EMV_Tracker"] / emv_mean) * 100
gep_df["GEP_Norm"] = (gep_df["GEP_monthly"] / gep_mean) * 100

# ── 4. Generate Plot ───────────────────────────────────────────────────────────
# Create a single plot for overlaid series
fig, ax = plt.subplots(figsize=(16, 7))

# Plot both series on the same axis
ax.plot(emv_df["Date"], emv_df["EMV_Norm"], color="#C0392B", linewidth=1.5, alpha=0.85, label="Overall EMV Tracker")
ax.plot(gep_df["Date"], gep_df["GEP_Norm"], color="#378ADD", linewidth=1.5, alpha=0.85, label="GEP Monthly (Robust min-2)")

# Add a baseline at 100
ax.axhline(100, color="gray", linewidth=1, linestyle="--", alpha=0.7)

# Chart styling and labels
ax.set_title("GEP vs Overall EMV Tracker (Normalized to 100 over 1985-2025)", fontsize=14, pad=12)
ax.set_ylabel("Index (Average = 100)", fontsize=11)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", framealpha=0.9, fontsize=11)

# X-axis formatting (Years)
ax.xaxis.set_major_locator(mdates.YearLocator(5)) # Labels every 5 years for clean look
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Force x-axis limits from 1985 to 2025
ax.set_xlim(pd.Timestamp("1996-01-01"), pd.Timestamp("2025-12-31"))

plt.tight_layout()

# Save the chart
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Chart successfully saved to: {OUT_PATH}")

# Uncomment the line below to display the plot interactively
# plt.show()
plt.close()