#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

BASE = "/home/h12429576/Master_Thesis/INDEX/index_8_europe"

# ── Load data ──────────────────────────────────────────────────────────────────
monthly = pd.read_csv(f"{BASE}/GEP_Monthly_Index.csv")
monthly['month'] = pd.to_datetime(monthly['month'])
monthly['GEP_monthly_scaled'] = monthly['GEP_monthly'] * 10_000

# ══════════════════════════════════════════════════════════════════════════════
# PLOT — Monthly GEP Index Europe (horizontal, annotated peaks)
# ══════════════════════════════════════════════════════════════════════════════

peaks = {
    '1997-07': 'Asian Financial Crisis',
    '1998-08': 'Russian Ruble Crisis',
    '2001-09': '9/11',
    '2003-03': 'Iraq War',
    '2008-09': 'GFC',
    '2011-08': 'US credit downgrade',
    '2014-03': 'Crimea annexation',
    '2018-06': 'US–China tariffs',
    '2019-05': 'Trade war escalation',
    '2020-03': 'COVID-19',
    '2022-02': 'Russia invades Ukraine',
    '2022-10': 'US chip controls on China',
    '2025-04': 'Liberation Day tariffs',
}

fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(monthly['month'], monthly['GEP_monthly_scaled'],
        color='#378ADD', linewidth=0.9, alpha=0.9)
ax.fill_between(monthly['month'], monthly['GEP_monthly_scaled'],
                alpha=0.15, color='#378ADD')

for month_str, label in peaks.items():
    row = monthly[monthly['month'] == month_str]
    if not row.empty:
        x = row['month'].values[0]
        y = row['GEP_monthly_scaled'].values[0]
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, 12),
            textcoords='offset points',
            fontsize=7.5,
            ha='center',
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.7),
            color='#333333',
        )

ax.set_title('GEP Monthly Index — Europe (1996–2025)', fontsize=13, pad=12)
ax.set_xlabel('')
ax.set_ylabel('GEP score (×10⁻⁴)', fontsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/GEP_Monthly_Index_Europe.png", dpi=150, bbox_inches='tight')
print("Saved: GEP_Monthly_Index_Europe.png")
plt.close()
