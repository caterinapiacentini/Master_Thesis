#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_check_GEP.py

Compares two versions of the monthly GEP index:

  1. GEP_monthly          — article-weighted average: each article in the month
                            contributes equally regardless of publication day.
                            (Σ score_t * n_articles_t) / Σ n_articles_t

  2. GEP_monthly_daily_avg — day-weighted average: each trading day contributes
                            equally regardless of how many articles were published.
                            (1/T_m) * Σ score_t

The robustness version is computed on-the-fly from GEP_Daily_Index.csv if
GEP_Monthly_Robustness.csv does not exist yet (requires compute_monthly_robustness.py
to have been run first), or it can be derived directly here for convenience.

Saves: robustness_check_GEP.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = "~/Desktop/tesi/Master_Thesis/INDEX/index_8"

# ----------------------------------------------------------------
# 1. Load original monthly index
# ----------------------------------------------------------------
monthly = pd.read_csv(f"{BASE}/GEP_Monthly_Index.csv")
monthly['month'] = pd.to_datetime(monthly['month'])

# ----------------------------------------------------------------
# 2. Build robustness monthly index from the daily index
#    (simple average of daily scores over trading days in each month)
# ----------------------------------------------------------------
daily = pd.read_csv(f"{BASE}/GEP_Daily_Index.csv", parse_dates=['date'])
daily_obs = daily[daily['n_articles'] > 0].copy()
daily_obs['month'] = daily_obs['date'].dt.to_period('M')

rob = (
    daily_obs
    .groupby('month')['score']
    .mean()
    .reset_index()
    .rename(columns={'score': 'GEP_monthly_daily_avg'})
)
rob['month'] = rob['month'].dt.to_timestamp()

# ----------------------------------------------------------------
# 3. Merge
# ----------------------------------------------------------------
df = monthly[['month', 'GEP_monthly']].merge(rob, on='month', how='inner')
df['GEP_monthly_scaled']     = df['GEP_monthly']          * 10_000
df['GEP_daily_avg_scaled']   = df['GEP_monthly_daily_avg'] * 10_000

# ----------------------------------------------------------------
# 4. Plot
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(df['month'], df['GEP_monthly_scaled'],
        color='#378ADD', linewidth=0.9, alpha=0.9,
        label='Article-weighted avg (baseline)')

ax.plot(df['month'], df['GEP_daily_avg_scaled'],
        color='#E05C2A', linewidth=0.9, alpha=0.85, linestyle='--',
        label='Day-weighted avg (robustness)')

# Correlation annotation
corr = np.corrcoef(df['GEP_monthly_scaled'], df['GEP_daily_avg_scaled'])[0, 1]
ax.text(0.01, 0.97, f'Correlation: {corr:.4f}',
        transform=ax.transAxes, fontsize=9, va='top',
        color='#333333')

ax.set_title('GEP Monthly Index — Robustness Check (1996–2025)', fontsize=13, pad=12)
ax.set_xlabel('')
ax.set_ylabel('GEP score (×10⁻⁴)', fontsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.legend(fontsize=9, framealpha=0.7)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/robustness_check_GEP.png", dpi=150, bbox_inches='tight')
print("Saved: robustness_check_GEP.png")
