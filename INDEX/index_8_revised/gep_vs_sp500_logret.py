#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEP Index (Monthly, Revised) vs S&P 500 Log Returns — two-panel comparison
No z-scoring: GEP shown at natural scale, S&P shown as log returns.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load GEP monthly index ─────────────────────────────────────────────────────
monthly = pd.read_csv(f"{BASE}/GEP_Monthly_Index.csv")
monthly['month'] = pd.to_datetime(monthly['month'])
monthly = monthly.set_index('month').sort_index()

# ── Download S&P 500 monthly close ─────────────────────────────────────────────
sp500_raw = yf.download('^GSPC', start='1995-12-01', end='2025-12-31',
                        interval='1mo', auto_adjust=True, progress=False)
sp500 = sp500_raw[['Close']].copy()
sp500.index = sp500.index.to_period('M').to_timestamp()
sp500.columns = ['sp500']
sp500 = sp500.sort_index()

# ── Compute monthly log returns ────────────────────────────────────────────────
sp500['log_ret'] = np.log(sp500['sp500'] / sp500['sp500'].shift(1))
sp500 = sp500.dropna()

# ── Merge on common months ─────────────────────────────────────────────────────
df = monthly[['GEP_monthly']].join(sp500[['log_ret']], how='inner')
df['gep_scaled']  = df['GEP_monthly'] * 10_000   # ×10⁻⁴ for readability
df['cum_log_ret'] = df['log_ret'].cumsum()        # cumulative log return

# ── Key events ────────────────────────────────────────────────────────────────
events = {
    '1997-07': 'Asian\nCrisis',
    '2001-09': '9/11',
    '2008-09': 'GFC',
    '2020-03': 'COVID-19',
    '2022-02': 'Ukraine\nWar',
    '2025-04': 'Liberation\nDay',
}

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7),
                                sharex=True,
                                gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.08})

# — Top panel: GEP index level ─────────────────────────────────────────────────
ax1.fill_between(df.index, df['gep_scaled'], alpha=0.30, color='#378ADD', step=None)
ax1.plot(df.index, df['gep_scaled'], color='#1A5FA8', linewidth=1.1,
         label='GEP Index (monthly)')
ax1.set_ylabel('GEP score (×10⁻⁴)', fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.legend(fontsize=10, framealpha=0.7, loc='upper left')
ax1.set_title('GEP Index vs S&P 500 Monthly Log Returns (1996–2025)',
              fontsize=13, pad=10)

# — Bottom panel: S&P 500 log returns + cumulative log return ─────────────────
colors = ['#C0392B' if r < 0 else '#27AE60' for r in df['log_ret']]
ax2.bar(df.index, df['log_ret'], color=colors, width=20, alpha=0.75,
        label='S&P 500 log return')
ax2.axhline(0, color='#333333', linewidth=0.7)
ax2.set_ylabel('Monthly log return', fontsize=10)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# cumulative log return on a separate right axis
ax2r = ax2.twinx()
ax2r.plot(df.index, df['cum_log_ret'], color='#555555', linewidth=1.4,
          linestyle='-', alpha=0.6, label='Cumulative log return', zorder=4)
ax2r.set_ylabel('Cumulative log return', fontsize=10, color='#555555')
ax2r.tick_params(axis='y', labelcolor='#555555')
ax2r.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2r.spines['top'].set_visible(False)

# combined legend for both axes
handles1, labels1 = ax2.get_legend_handles_labels()
handles2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(handles1 + handles2, labels1 + labels2,
           fontsize=10, framealpha=0.7, loc='upper left')

# — Shared x-axis ──────────────────────────────────────────────────────────────
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')

# — Event vertical lines across both panels ────────────────────────────────────
for month_str, label in events.items():
    ts = pd.Timestamp(month_str)
    if ts in df.index:
        for ax in (ax1, ax2):
            ax.axvline(ts, color='#888888', linewidth=0.8,
                       linestyle='--', zorder=2)
        # label at top of upper panel
        ax1.text(ts, ax1.get_ylim()[1] * 0.97, label,
                 fontsize=7, ha='center', va='top',
                 color='#444444', linespacing=1.3)

plt.tight_layout()
out = f"{BASE}/gep_vs_sp500_logret.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
