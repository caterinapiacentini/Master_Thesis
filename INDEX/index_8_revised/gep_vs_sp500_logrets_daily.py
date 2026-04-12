#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEP Index (Daily) vs S&P 500 Log Returns — two-panel comparison
No z-scoring: GEP shown at natural scale, S&P shown as log returns.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf

BASE = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_8_revised"

# ── Load GEP daily index ───────────────────────────────────────────────────────
daily = pd.read_csv(f"{BASE}/GEP_Daily_Index.csv")
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.set_index('date').sort_index()

# ── Download S&P 500 daily close ───────────────────────────────────────────────
sp500_raw = yf.download('^GSPC', start='1995-12-01', end='2025-12-31',
                        interval='1d', auto_adjust=True, progress=False)
sp500 = sp500_raw[['Close']].copy()
sp500.index = pd.to_datetime(sp500.index)
sp500.columns = ['sp500']
sp500 = sp500.sort_index()

# ── Compute daily log returns ──────────────────────────────────────────────────
sp500['log_ret'] = np.log(sp500['sp500'] / sp500['sp500'].shift(1))
sp500 = sp500.dropna()

# ── Merge on common trading days ───────────────────────────────────────────────
df = daily[['score']].join(sp500[['log_ret']], how='inner')
df['gep_scaled']  = df['score'] * 10_000   # ×10⁻⁴ for readability
df['cum_log_ret'] = df['log_ret'].cumsum()  # cumulative log return

# ── Key events ────────────────────────────────────────────────────────────────
events = {
    '1997-07-02': 'Asian\nCrisis',
    '2001-09-11': '9/11',
    '2008-09-15': 'GFC',
    '2020-03-16': 'COVID-19',
    '2022-02-24': 'Ukraine\nWar',
    '2025-04-02': 'Liberation\nDay',
}

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 7),
                                sharex=True,
                                gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.08})

# — Top panel: GEP index level ─────────────────────────────────────────────────
ax1.fill_between(df.index, df['gep_scaled'], alpha=0.30, color='#378ADD')
ax1.plot(df.index, df['gep_scaled'], color='#1A5FA8', linewidth=0.7,
         label='GEP Index (daily)')
ax1.set_ylabel('GEP score (×10⁻⁴)', fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.legend(fontsize=10, framealpha=0.7, loc='upper left')
ax1.set_title('GEP Index vs S&P 500 Daily Log Returns (1996–2025)',
              fontsize=13, pad=10)

# — Bottom panel: S&P 500 daily log returns + cumulative ───────────────────────
colors = ['#C0392B' if r < 0 else '#27AE60' for r in df['log_ret']]
ax2.bar(df.index, df['log_ret'], color=colors, width=1, alpha=0.75,
        label='S&P 500 log return')
ax2.axhline(0, color='#333333', linewidth=0.7)
ax2.set_ylabel('Daily log return', fontsize=10)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# cumulative log return on right axis
ax2r = ax2.twinx()
ax2r.plot(df.index, df['cum_log_ret'], color='#555555', linewidth=1.4,
          linestyle='-', alpha=0.6, label='Cumulative log return', zorder=4)
ax2r.set_ylabel('Cumulative log return', fontsize=10, color='#555555')
ax2r.tick_params(axis='y', labelcolor='#555555')
ax2r.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2r.spines['top'].set_visible(False)

# combined legend
handles1, labels1 = ax2.get_legend_handles_labels()
handles2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(handles1 + handles2, labels1 + labels2,
           fontsize=10, framealpha=0.7, loc='upper left')

# — Shared x-axis ──────────────────────────────────────────────────────────────
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')

# — Event vertical lines across both panels ────────────────────────────────────
for date_str, label in events.items():
    ts = pd.Timestamp(date_str)
    # snap to nearest available trading day if exact date missing
    if ts not in df.index:
        idx = df.index.searchsorted(ts)
        if idx < len(df.index):
            ts = df.index[idx]
        else:
            continue
    for ax in (ax1, ax2):
        ax.axvline(ts, color='#888888', linewidth=0.8, linestyle='--', zorder=2)
    ax1.text(ts, ax1.get_ylim()[1] * 0.97, label,
             fontsize=7, ha='center', va='top',
             color='#444444', linespacing=1.3)

plt.tight_layout()
out = f"{BASE}/gep_vs_sp500_logret_daily.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()