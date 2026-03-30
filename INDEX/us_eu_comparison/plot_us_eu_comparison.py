#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_us_eu_comparison.py

Plots the US and EU GEP Monthly Index on the same axis using EPU-style
normalization (GEP_norm, long-run mean = 100) so the two series are
directly comparable in terms of relative swings.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import os

BASE_US  = "/home/h12429576/Master_Thesis/INDEX/index_8"
BASE_EU  = "/home/h12429576/Master_Thesis/INDEX/index_8_europe"
OUT_DIR  = "/home/h12429576/Master_Thesis/INDEX/us_eu_comparison"

# ── Load data ──────────────────────────────────────────────────────────────────
us = pd.read_csv(f"{BASE_US}/GEP_Monthly_Index.csv", parse_dates=['month'])
eu = pd.read_csv(f"{BASE_EU}/GEP_Monthly_Index.csv", parse_dates=['month'])

# Restrict to overlapping sample
start = max(us['month'].min(), eu['month'].min())
end   = min(us['month'].max(), eu['month'].max())
us = us[(us['month'] >= start) & (us['month'] <= end)].reset_index(drop=True)
eu = eu[(eu['month'] >= start) & (eu['month'] <= end)].reset_index(drop=True)

# ── Key events ─────────────────────────────────────────────────────────────────
events = {
    '1997-07': 'Asian Financial Crisis',
    '1998-08': 'Russian Ruble Crisis',
    '2001-09': '9/11',
    '2003-03': 'Iraq War',
    '2008-09': 'GFC',
    '2010-05': 'Eurozone crisis',
    '2014-03': 'Crimea annexation',
    '2018-06': 'US–China tariffs',
    '2020-03': 'COVID-19',
    '2022-02': 'Russia invades Ukraine',
    '2022-10': 'US chip controls on China',
    '2025-04': 'Liberation Day tariffs',
}

COLOR_US = '#2166AC'   # blue
COLOR_EU = '#D6604D'   # red-orange

# ══════════════════════════════════════════════════════════════════════════════
# PLOT — US vs EU GEP_norm (EPU-style, mean = 100)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(us['month'], us['GEP_norm'], color=COLOR_US, linewidth=1.1,
        alpha=0.95, label='US GEP Index (INDEX_8)', zorder=3)
ax.fill_between(us['month'], us['GEP_norm'], alpha=0.08, color=COLOR_US)

ax.plot(eu['month'], eu['GEP_norm'], color=COLOR_EU, linewidth=1.1,
        alpha=0.95, label='EU GEP Index (INDEX_8_europe)', zorder=3)
ax.fill_between(eu['month'], eu['GEP_norm'], alpha=0.08, color=COLOR_EU)

# Reference line at 100
ax.axhline(100, color='#888888', linewidth=0.7, linestyle='--', alpha=0.6, zorder=1)

# ── Event annotations (use max of US/EU at that month as anchor) ──────────────
y_max = ax.get_ylim()[1]
for month_str, label in events.items():
    us_row = us[us['month'] == month_str]
    eu_row = eu[eu['month'] == month_str]
    if us_row.empty and eu_row.empty:
        continue
    x = pd.to_datetime(month_str)
    y_vals = []
    if not us_row.empty:
        y_vals.append(us_row['GEP_norm'].values[0])
    if not eu_row.empty:
        y_vals.append(eu_row['GEP_norm'].values[0])
    y = max(y_vals)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(0, 10),
        textcoords='offset points',
        fontsize=7.5,
        ha='center',
        va='bottom',
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.6),
        color='#333333',
    )

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_title('GEP Monthly Index — US vs Europe (EPU-normalized, mean = 100)',
             fontsize=13, pad=12)
ax.set_ylabel('GEP Index (long-run avg = 100)', fontsize=10)
ax.set_xlabel('')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=10, framealpha=0.7, loc='upper left')

plt.tight_layout()
out = os.path.join(OUT_DIR, "GEP_Monthly_US_vs_EU.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
