#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

BASE = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_8"

# ── Load data ──────────────────────────────────────────────────────────────────
daily = pd.read_csv(f"{BASE}/GEP_Daily_Index.csv", parse_dates=['date'])
daily_obs = daily[daily['n_articles'] > 0].copy()
daily_obs['score_scaled'] = daily_obs['score'] * 10_000

# ── Peaks dictionary ───────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# PLOT — Daily GEP Index
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(daily_obs['date'], daily_obs['score_scaled'],
        color='#378ADD', linewidth=0.5, alpha=0.7)
ax.fill_between(daily_obs['date'], daily_obs['score_scaled'],
                alpha=0.12, color='#378ADD')

# Annotate macro-events on the peak day within each month
for month_str, label in peaks.items():
    mask = daily_obs['date'].dt.to_period('M') == pd.Period(month_str, 'M')
    subset = daily_obs[mask]
    if subset.empty:
        continue
    peak_row = subset.loc[subset['score_scaled'].idxmax()]
    x = peak_row['date']
    y = peak_row['score_scaled']
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

ax.set_title('GEP Daily Index (1996–2025)', fontsize=13, pad=12)
ax.set_xlabel('')
ax.set_ylabel('GEP score (×10⁻⁴)', fontsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/GEP_Daily_Index.png", dpi=150, bbox_inches='tight')
print("Saved: GEP_Daily_Index.png")
plt.close()