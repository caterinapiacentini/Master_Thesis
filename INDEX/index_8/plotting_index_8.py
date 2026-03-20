#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv("GEP_Monthly_Index.csv")
df['month'] = pd.to_datetime(df['month'])
df['GEP_monthly_scaled'] = df['GEP_monthly'] * 10_000

fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(df['month'], df['GEP_monthly_scaled'], color='#378ADD', linewidth=0.9, alpha=0.9)
ax.fill_between(df['month'], df['GEP_monthly_scaled'], alpha=0.15, color='#378ADD')

# Annotate key peaks
peaks = {
    '1997-07': 'Asian Financial Crisis',
    '1998-08': 'Russian Ruble Crisis',
    '2001-09': '9/11 attacks',
    '2003-11': 'Iraq War',
    '2008-01': 'GFC',
    '2011-08': 'Eurozone crisis',
    '2014-03': 'Crimea annexation',
    '2018-07': 'US-China trade war',
    '2020-03': 'COVID-19',
    '2022-02': 'Russia invades Ukraine',
    '2025-04': 'Trade war peak',
}
for month_str, label in peaks.items():
    row = df[df['month'] == month_str]
    if not row.empty:
        x = row['month'].values[0]
        y = row['GEP_monthly_scaled'].values[0]
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, 10),
            textcoords='offset points',
            fontsize=8,
            ha='center',
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.8),
            color='#333333',
        )

ax.set_title('GEP Monthly Index (1996–2025)', fontsize=13, pad=12)
ax.set_xlabel('')
ax.set_ylabel('GEP score (×10⁻⁴)', fontsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("GEP_Monthly_Index.png", dpi=150, bbox_inches='tight')
print("Saved: GEP_Monthly_Index.png")

