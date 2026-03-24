#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import numpy as np

BASE = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_8"

# ── Load data ──────────────────────────────────────────────────────────────
monthly = pd.read_csv(f"{BASE}/GEP_Monthly_Index.csv")
monthly['month'] = pd.to_datetime(monthly['month'])
monthly['GEP_monthly_scaled'] = monthly['GEP_monthly'] * 10_000

daily = pd.read_csv(f"{BASE}/GEP_Daily_Index.csv", parse_dates=['date'])
daily_obs = daily[daily['n_articles'] > 0].copy()   # exclude gap-filled days
daily_obs['score_scaled'] = daily_obs['score'] * 10_000


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Monthly GEP Index with annotated peaks
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

ax.set_title('GEP Monthly Index (1996–2025)', fontsize=13, pad=12)
ax.set_xlabel('')
ax.set_ylabel('GEP score (×10⁻⁴)', fontsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/GEP_Monthly_Index.png", dpi=150, bbox_inches='tight')
print("Saved: GEP_Monthly_Index.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Daily dots + Monthly line + Comprehensive geoeconomic events
#           Inspired by Caldara & Iacoviello (2022), Table 2 / Figure 1
# ══════════════════════════════════════════════════════════════════════════════

# Comprehensive geoeconomic events with precise dates.
# Format: 'YYYY-MM-DD': ('Label', level)
# level 0-3 → stagger label heights to avoid overlap between nearby events
events = [
    # ── 1990s ──────────────────────────────────────────────────────────────
    ('1997-07-02', 'Thai baht devaluation\n(Asian crisis begins)',      0),
    ('1998-08-17', 'Russia ruble default',                              1),
    ('1999-11-30', 'WTO Seattle ministerial\ncollapse',                 2),
    # ── 2000s ──────────────────────────────────────────────────────────────
    ('2001-09-11', '9/11 attacks',                                      0),
    ('2001-12-11', 'China joins WTO',                                   2),
    ('2002-03-05', 'Bush 201 steel tariffs (30%)',                      1),
    ('2003-03-20', 'US-led invasion of Iraq',                           0),
    ('2006-10-09', 'N. Korea nuclear test',                             3),
    ('2007-08-09', 'BNP Paribas freezes\nsubprime funds (GFC onset)',  1),
    ('2008-09-15', 'Lehman Brothers collapse',                          0),
    ('2009-04-02', 'G20 London Summit\n(GFC response)',                 2),
    # ── 2010s ──────────────────────────────────────────────────────────────
    ('2010-05-02', 'Greek €110bn bailout\n(Eurozone crisis)',           0),
    ('2011-08-05', 'S&P downgrades US\n(Black Monday)',                 1),
    ('2012-07-26', 'Draghi "whatever\nit takes"',                       3),
    ('2014-03-18', 'Russia annexes Crimea',                             0),
    ('2014-07-31', 'US/EU expand\nRussia sanctions',                    2),
    ('2015-07-14', 'Iran nuclear\ndeal (JCPOA)',                        1),
    ('2016-06-23', 'Brexit vote',                                       3),
    ('2016-11-08', 'Trump wins\nUS election',                           0),
    ('2018-01-22', 'US solar panel\ntariffs',                           2),
    ('2018-03-22', 'Trump §301 tariffs\non China ($60bn)',              0),
    ('2018-06-15', 'US-China tariffs\neffective ($34bn)',               1),
    ('2018-08-10', 'US doubles tariffs\non Turkey',                     3),
    ('2018-12-01', 'Trump-Xi G20\n90-day truce',                       2),
    ('2019-05-10', 'US raises China\ntariffs to 25%',                  0),
    ('2019-08-23', 'China retaliates\n($75bn tariffs)',                 1),
    # ── 2020s ──────────────────────────────────────────────────────────────
    ('2020-01-15', 'US-China Phase 1\ntrade deal signed',              3),
    ('2020-03-11', 'WHO: COVID-19\npandemic declared',                 0),
    ('2021-03-23', 'Suez Canal\nblocked (Ever Given)',                  2),
    ('2022-02-24', 'Russia full-scale\ninvasion of Ukraine',            0),
    ('2022-03-26', 'Russia: gas payments\nin rubles',                   1),
    ('2022-10-07', 'US chip export\ncontrols on China',                 3),
    ('2023-08-09', 'Biden: US outbound\ninvestment curbs on China',    2),
    ('2024-05-14', 'Biden doubles\nChina EV tariffs (100%)',            0),
    ('2025-01-20', 'Trump inauguration\n(tariff threats)',              1),
    ('2025-04-02', '"Liberation Day"\nsweeping tariffs',                0),
]

# ── Heights for the 4 stagger levels (in axes-fraction coords above the plot) ──
# The more levels, the more vertical space needed in the top margin.
LABEL_HEIGHTS  = [1.03, 1.16, 1.29, 1.42]   # axes-fraction y coords
LINE_COLORS    = ['#888888']                  # single gray for all event lines

fig, ax = plt.subplots(figsize=(22, 10))

# Extra top margin to accommodate the staggered labels
fig.subplots_adjust(top=0.62)    # leaves ~38% of figure height above axes

# Daily dots (only true trading days)
ax.scatter(daily_obs['date'], daily_obs['score_scaled'],
           s=1.5, color='#AACCEE', alpha=0.35, linewidths=0, zorder=2,
           label='Daily GEP score')

# Monthly line
ax.plot(monthly['month'], monthly['GEP_monthly_scaled'],
        color='#1A5FA8', linewidth=1.6, alpha=0.95, zorder=3,
        label='Monthly GEP (article-weighted avg)')

# ── Event annotations ───────────────────────────────────────────────────────
xmin = mdates.date2num(daily_obs['date'].min())
xmax = mdates.date2num(daily_obs['date'].max())

for date_str, label, level in events:
    xd = pd.to_datetime(date_str)
    if not (daily_obs['date'].min() <= xd <= daily_obs['date'].max()):
        continue

    # Vertical dashed line spanning full axes height
    ax.axvline(xd, color='#BBBBBB', linewidth=0.55, linestyle='--',
               zorder=1, alpha=0.85)

    # Staggered label: x in data coordinates, y in axes fraction
    ax.annotate(
        label,
        xy=(xd, 1.0),                          # foot of label (top of axes)
        xycoords=('data', 'axes fraction'),
        xytext=(xd, LABEL_HEIGHTS[level]),      # label position
        textcoords=('data', 'axes fraction'),
        fontsize=6.0,
        ha='center',
        va='bottom',
        color='#222222',
        arrowprops=dict(arrowstyle='-', color='#BBBBBB', lw=0.5),
        annotation_clip=False,
    )

# ── Axis formatting ─────────────────────────────────────────────────────────
ax.set_xlim(daily_obs['date'].min(), daily_obs['date'].max())
ax.set_ylim(bottom=0)
ax.set_title('GEP Index — Daily Scores and Monthly Average (1996–2025)',
             fontsize=13, pad=12)
ax.set_xlabel('')
ax.set_ylabel('GEP score (×10⁻⁴)', fontsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.get_xticklabels(), fontsize=9)
ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9, framealpha=0.7, loc='upper left')

plt.savefig(f"{BASE}/GEP_Daily_Events.png", dpi=180, bbox_inches='tight')
print("Saved: GEP_Daily_Events.png")
plt.close()
