#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEP Index (Monthly, Revised) vs S&P 500 — Z-score comparison
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load GEP monthly index ─────────────────────────────────────────────────────
monthly = pd.read_csv(f"{BASE}/GEP_Monthly_Index.csv")
monthly['month'] = pd.to_datetime(monthly['month'])
monthly = monthly.set_index('month').sort_index()

# ── Download S&P 500 monthly close ─────────────────────────────────────────────
sp500_raw = yf.download('^GSPC', start='1996-01-01', end='2025-12-31',
                        interval='1mo', auto_adjust=True, progress=False)
sp500 = sp500_raw[['Close']].copy()
sp500.index = sp500.index.to_period('M').to_timestamp()   # align to month-start
sp500.columns = ['sp500']
sp500 = sp500.sort_index()

# ── Merge on common months ─────────────────────────────────────────────────────
df = monthly[['GEP_monthly']].join(sp500, how='inner')

# ── Z-score normalisation ──────────────────────────────────────────────────────
df['gep_z']   = (df['GEP_monthly'] - df['GEP_monthly'].mean()) / df['GEP_monthly'].std()
df['sp500_z'] = (df['sp500']       - df['sp500'].mean())       / df['sp500'].std()

# ── Key event annotations ──────────────────────────────────────────────────────
events = {
    '1997-07': 'Asian Crisis',
    '2001-09': '9/11',
    '2008-09': 'GFC',
    '2020-03': 'COVID-19',
    '2022-02': 'Ukraine war',
    '2025-04': 'Liberation Day',
}

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — dual Z-score time series (monthly)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(df.index, df['gep_z'],   color='#378ADD', linewidth=1.1,
        label='GEP Index (Z-score)', zorder=3)
ax.fill_between(df.index, df['gep_z'], alpha=0.12, color='#378ADD')

ax.plot(df.index, df['sp500_z'], color='#E74C3C', linewidth=1.1,
        label='S&P 500 (Z-score)', zorder=3)
ax.fill_between(df.index, df['sp500_z'], alpha=0.08, color='#E74C3C')

ax.axhline(0, color='#555555', linewidth=0.6, linestyle='--')

# annotate events
for month_str, label in events.items():
    ts = pd.Timestamp(month_str)
    if ts in df.index:
        y_val = df.loc[ts, 'gep_z']
        ax.annotate(label, xy=(ts, y_val),
                    xytext=(0, 14), textcoords='offset points',
                    fontsize=7.5, ha='center', color='#333333',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.6))

ax.set_title('GEP Index vs S&P 500 — Monthly Z-scores (1996–2025)',
             fontsize=13, pad=12)
ax.set_xlabel('')
ax.set_ylabel('Standard deviations from mean', fontsize=10)
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
out1 = f"{BASE}/gep_vs_sp500_zscore.png"
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Saved: {out1}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — dual-axis (original scales) for reference
# ══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(16, 5))
ax2 = ax1.twinx()

ax1.plot(df.index, df['GEP_monthly'] * 10_000, color='#378ADD',
         linewidth=1.0, label='GEP Index (×10⁻⁴)')
ax1.fill_between(df.index, df['GEP_monthly'] * 10_000,
                 alpha=0.12, color='#378ADD')
ax1.set_ylabel('GEP score (×10⁻⁴)', color='#378ADD', fontsize=10)
ax1.tick_params(axis='y', labelcolor='#378ADD')

ax2.plot(df.index, df['sp500'], color='#E74C3C',
         linewidth=1.0, label='S&P 500 (USD)')
ax2.set_ylabel('S&P 500 Close (USD)', color='#E74C3C', fontsize=10)
ax2.tick_params(axis='y', labelcolor='#E74C3C')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, framealpha=0.7)

ax1.set_title('GEP Index vs S&P 500 — Dual-axis (1996–2025)',
              fontsize=13, pad=12)
ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
out2 = f"{BASE}/gep_vs_sp500_dualaxis.png"
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close()
