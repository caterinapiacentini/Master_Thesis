#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

BASE = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_8"

# ── Load data ──────────────────────────────────────────────────────────────────
monthly = pd.read_csv(f"{BASE}/GEP_Monthly_Index.csv")
monthly['month'] = pd.to_datetime(monthly['month'])
monthly['GEP_monthly_scaled'] = monthly['GEP_monthly'] * 10_000

daily = pd.read_csv(f"{BASE}/GEP_Daily_Index.csv", parse_dates=['date'])
daily_obs = daily[daily['n_articles'] > 0].copy()
daily_obs['score_scaled'] = daily_obs['score'] * 10_000


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Monthly GEP Index (horizontal, annotated peaks)
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
# PLOT 2 — Vertical: Daily dots + Monthly line + Geoeconomic events
#           Layout matches Caldara & Iacoviello (2022), Figure 2:
#           • Y-axis = time (1996 at top → 2025 at bottom)
#           • X-axis = GEP score
#           • Cyan dots = daily GEP scores
#           • Dark blue line = monthly GEP average
#           • Red dots + arrows = annotated geoeconomic events
# ══════════════════════════════════════════════════════════════════════════════

# (date_str, label)   — dates to double-check
EVENTS = [
    ('1997-07-02', '1997/07/02: Thai baht devaluation — Asian crisis begins'),
    ('1998-08-17', '1998/08/17: Russia ruble default and debt moratorium'),
    ('1999-11-30', '1999/11/30: WTO Seattle ministerial collapse'),
    ('2001-09-11', '2001/09/11: 9/11 terrorist attacks'),
    ('2001-12-11', '2001/12/11: China joins the WTO'),
    ('2002-03-05', '2002/03/05: Bush imposes 30% steel tariffs (Section 201)'),
    ('2003-03-20', '2003/03/20: US-led invasion of Iraq'),
    ('2006-10-09', '2006/10/09: North Korea first nuclear test'),
    ('2007-08-09', '2007/08/09: BNP Paribas freezes subprime funds — GFC onset'),
    ('2008-09-15', '2008/09/15: Lehman Brothers collapse'),
    ('2009-04-02', '2009/04/02: G20 London Summit — coordinated GFC response'),
    ('2010-05-02', '2010/05/02: Greece €110bn bailout — Eurozone crisis'),
    ('2011-08-05', '2011/08/05: S&P downgrades US credit rating'),
    ('2012-07-26', '2012/07/26: Draghi "whatever it takes" speech'),
    ('2014-03-18', '2014/03/18: Russia annexes Crimea'),
    ('2014-07-31', '2014/07/31: US and EU expand Russia sanctions'),
    ('2015-07-14', '2015/07/14: Iran nuclear deal signed (JCPOA)'),
    ('2016-06-23', '2016/06/23: Brexit referendum — UK votes to leave EU'),
    ('2016-11-08', '2016/11/08: Trump wins US presidential election'),
    ('2018-01-22', '2018/01/22: US imposes tariffs on solar panels and washers'),
    ('2018-03-22', '2018/03/22: Trump announces Section 301 tariffs on China ($60bn)'),
    ('2018-06-15', '2018/06/15: US–China tariffs effective — $34bn tranche'),
    ('2018-08-10', '2018/08/10: US doubles steel and aluminum tariffs on Turkey'),
    ('2018-12-01', '2018/12/01: Trump–Xi G20 Buenos Aires 90-day truce'),
    ('2019-05-10', '2019/05/10: US raises tariffs on $200bn of Chinese goods to 25%'),
    ('2019-08-23', '2019/08/23: China announces $75bn in retaliatory tariffs'),
    ('2020-01-15', '2020/01/15: US–China Phase 1 trade deal signed'),
    ('2020-03-11', '2020/03/11: WHO declares COVID-19 pandemic'),
    ('2021-03-23', '2021/03/23: Suez Canal blocked — Ever Given runs aground'),
    ('2022-02-24', '2022/02/24: Russia launches full-scale invasion of Ukraine'),
    ('2022-03-26', '2022/03/26: Russia demands gas payments in rubles'),
    ('2022-10-07', '2022/10/07: US imposes advanced chip export controls on China'),
    ('2023-08-09', '2023/08/09: Biden signs executive order on outbound China investment'),
    ('2024-05-14', '2024/05/14: Biden raises tariffs on Chinese EVs to 100%'),
    ('2025-01-20', '2025/01/20: Trump returns to office — tariff agenda begins'),
    ('2025-04-02', '2025/04/02: "Liberation Day" — Trump sweeping global tariffs'),
]


def find_nearby_peak(date_str, window_days=10):
    """Return (peak_date, peak_score_scaled) within ±window_days of date_str."""
    dt = pd.to_datetime(date_str)
    lo, hi = dt - pd.Timedelta(days=window_days), dt + pd.Timedelta(days=window_days)
    sub = daily_obs[(daily_obs['date'] >= lo) & (daily_obs['date'] <= hi)]
    if sub.empty:
        return dt, 0.0
    idx = sub['score_scaled'].idxmax()
    return sub.loc[idx, 'date'], sub.loc[idx, 'score_scaled']


def spread_label_dates(event_peaks, min_gap_days=38):
    """
    Greedy vertical spreading: if two event labels would be within min_gap_days
    of each other on the y-axis, push the lower one down.
    Returns list of (peak_date, peak_score, label, label_date).
    """
    # Sort by peak date (top → bottom = chronological)
    items = sorted(event_peaks, key=lambda r: r[0])
    placed = []   # (label_date, ...)
    result = []

    for peak_date, peak_score, label in items:
        label_date = peak_date
        # Keep pushing down until there is no clash
        changed = True
        while changed:
            changed = False
            for placed_date, *_ in placed:
                gap = (label_date - placed_date).days
                if 0 <= gap < min_gap_days:
                    label_date = placed_date + pd.Timedelta(days=min_gap_days)
                    changed = True

        placed.append((label_date, peak_score, label))
        result.append((peak_date, peak_score, label, label_date))

    return result


# ── Build event list with peaks ───────────────────────────────────────────────
raw_peaks = []
for date_str, label in EVENTS:
    peak_date, peak_score = find_nearby_peak(date_str)
    raw_peaks.append((peak_date, peak_score, label))

spread = spread_label_dates(raw_peaks, min_gap_days=38)

# ── Figure ────────────────────────────────────────────────────────────────────
X_DATA_MAX = daily_obs['score_scaled'].quantile(0.999)   # robust max (~35)
X_LABEL    = X_DATA_MAX * 1.25                           # x-start for all labels
X_MAX      = X_DATA_MAX * 3.8                            # total x-axis width (room for text)

fig, ax = plt.subplots(figsize=(16, 28))

# Daily dots
ax.scatter(daily_obs['score_scaled'], daily_obs['date'],
           s=28, color='#27AE60', alpha=0.30, linewidths=0, zorder=2,
           label='Daily GEP score')

# Monthly line
ax.plot(monthly['GEP_monthly_scaled'], monthly['month'],
        color='#152F5F', linewidth=1.6, alpha=0.95, zorder=3,
        label='Monthly GEP (article-weighted avg)')

# Event annotations
for peak_date, peak_score, label, label_date in spread:
    # Red dot at the daily peak
    ax.scatter(peak_score, peak_date,
               s=25, color='#C0392B', zorder=5, linewidths=0)

    # Arrow from label position → peak dot
    ax.annotate(
        label,
        xy       =(peak_score,  peak_date),    # arrowhead at peak
        xytext   =(X_LABEL,     label_date),   # text anchor
        fontsize =13,
        ha       ='left',
        va       ='center',
        color    ='#1A1A1A',
        arrowprops=dict(
            arrowstyle='->', color='#777777', lw=0.65,
            connectionstyle='arc3,rad=0.0'
        ),
        annotation_clip=False,
    )

# ── Axis formatting ───────────────────────────────────────────────────────────
# Y-axis: time, 1996 at top → 2025 at bottom
ax.set_ylim(pd.Timestamp('2026-03-01'), pd.Timestamp('1995-10-01'))
ax.yaxis.set_major_locator(mdates.YearLocator(1))
ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(axis='y', labelsize=9)

# X-axis: GEP score
ax.set_xlim(0, X_MAX)
ax.set_xlabel('GEP score (×10⁻⁴)', fontsize=11)
ax.tick_params(axis='x', labelsize=9)

ax.set_title('Daily GEP Index (1996–2025)', fontsize=13, pad=10)
ax.grid(axis='x', linestyle='--', linewidth=0.4, alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9, framealpha=0.7, loc='lower right')

plt.tight_layout()
plt.savefig(f"{BASE}/GEP_Daily_Events.png", dpi=160, bbox_inches='tight')
print("Saved: GEP_Daily_Events.png")
plt.close()
