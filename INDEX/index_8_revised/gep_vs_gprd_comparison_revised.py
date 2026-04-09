#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gep_vs_gprd_comparison_revised.py

Compares the revised GEP Daily Index against the daily GPR index
(Caldara & Iacoviello 2022) from 1996 onwards.

Usage:
    python gep_vs_gprd_comparison_revised.py \
        --gep_path  /path/to/GEP_Daily_Index.csv \
        --gpr_path  /path/to/data_gpr_daily_recent.xls \
        --output_dir /path/to/output/

The GPR file can be downloaded from: https://www.matteoiacoviello.com/gpr.htm
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gep_path",   type=str, required=True,
                        help="Path to GEP_Daily_Index.csv")
    parser.add_argument("--gpr_path",   type=str, required=True,
                        help="Path to data_gpr_daily_recent.xls")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output PNGs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load GPR
    # ----------------------------------------------------------------
    gpr = pd.read_excel(args.gpr_path, parse_dates=['date'])
    gpr = gpr[['date', 'GPRD']].copy()
    gpr = gpr[gpr['date'] >= '1996-01-01']
    gpr['date'] = pd.to_datetime(gpr['date']).dt.normalize()

    # ----------------------------------------------------------------
    # 2. Load GEP — true trading days only
    # ----------------------------------------------------------------
    gep = pd.read_csv(args.gep_path, parse_dates=['date'])
    gep = gep[gep['date'] >= '1996-01-01']
    gep = gep[gep['n_articles'] > 0][['date', 'score']].rename(columns={'score': 'GEP'})

    # ----------------------------------------------------------------
    # 3. Merge
    # ----------------------------------------------------------------
    df = pd.merge(gep, gpr, on='date', how='inner').sort_values('date').reset_index(drop=True)
    print(f"Merged dataset: {len(df):,} days  ({df['date'].min().date()} → {df['date'].max().date()})")

    # ----------------------------------------------------------------
    # 4. Z-score
    # ----------------------------------------------------------------
    df['GEP_z']  = (df['GEP']  - df['GEP'].mean())  / df['GEP'].std()
    df['GPRD_z'] = (df['GPRD'] - df['GPRD'].mean()) / df['GPRD'].std()

    # ----------------------------------------------------------------
    # 5. Pearson correlation (overall + sub-periods)
    # ----------------------------------------------------------------
    r, pval = stats.pearsonr(df['GEP_z'], df['GPRD_z'])
    print(f"\nOverall Pearson correlation  : r = {r:.4f}  (p = {pval:.2e})")

    periods = [
        ('1996-2001', '1996-01-01', '2001-12-31'),
        ('2002-2009', '2002-01-01', '2009-12-31'),
        ('2010-2019', '2010-01-01', '2019-12-31'),
        ('2020-2025', '2020-01-01', '2025-12-31'),
    ]
    print("\nSub-period correlations:")
    for label, start, end in periods:
        sub = df[(df['date'] >= start) & (df['date'] <= end)]
        if len(sub) > 10:
            r_sub, _ = stats.pearsonr(sub['GEP_z'], sub['GPRD_z'])
            print(f"  {label}: r = {r_sub:.4f}  (n={len(sub):,})")

    # ----------------------------------------------------------------
    # 6. Granger causality
    # ----------------------------------------------------------------
    print("\n--- Granger causality (max lag = 5 days) ---")
    gc_data = df[['GEP_z', 'GPRD_z']].dropna()

    print("\nH0: GEP does NOT Granger-cause GPRD")
    res_gep2gpr = grangercausalitytests(gc_data[['GPRD_z', 'GEP_z']], maxlag=5, verbose=False)
    for lag, result in res_gep2gpr.items():
        f_stat = result[0]['ssr_ftest'][0]
        p      = result[0]['ssr_ftest'][1]
        print(f"  lag {lag}: F = {f_stat:.3f}, p = {p:.4f} {'*' if p < 0.05 else ''}")

    print("\nH0: GPRD does NOT Granger-cause GEP")
    res_gpr2gep = grangercausalitytests(gc_data[['GEP_z', 'GPRD_z']], maxlag=5, verbose=False)
    for lag, result in res_gpr2gep.items():
        f_stat = result[0]['ssr_ftest'][0]
        p      = result[0]['ssr_ftest'][1]
        print(f"  lag {lag}: F = {f_stat:.3f}, p = {p:.4f} {'*' if p < 0.05 else ''}")

    # ----------------------------------------------------------------
    # 7. Rolling 90-day correlation
    # ----------------------------------------------------------------
    df['roll_corr'] = df['GEP_z'].rolling(90, min_periods=60).corr(df['GPRD_z'])

    # ----------------------------------------------------------------
    # 8. Plot 1: Time series + rolling correlation
    # ----------------------------------------------------------------
    key_events = {
        '1997-07-01': 'Asian Crisis',
        '1998-08-17': 'Russian Crisis',
        '2001-09-11': '9/11',
        '2003-03-20': 'Iraq War',
        '2008-09-15': 'GFC',
        '2014-03-01': 'Crimea',
        '2018-07-06': 'US-China trade war',
        '2020-03-11': 'COVID-19',
        '2022-02-24': 'Ukraine invasion',
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1.2]})

    ax1.plot(df['date'], df['GEP_z'],  color='#378ADD', linewidth=0.7,
             alpha=0.85, label='GEP Revised (this thesis)')
    ax1.plot(df['date'], df['GPRD_z'], color='#E05C2A', linewidth=0.7,
             alpha=0.85, label='GPR (Caldara & Iacoviello)')

    for date_str, label in key_events.items():
        xd = pd.to_datetime(date_str)
        if df['date'].min() <= xd <= df['date'].max():
            ax1.axvline(xd, color='gray', linewidth=0.6, linestyle='--', alpha=0.6)
            ax1.text(xd, 5, label, rotation=90, fontsize=6.5,
                     va='top', color='#555555', ha='right')

    ax1.axhline(0, color='black', linewidth=0.4)
    ax1.set_ylabel('Z-score', fontsize=10)
    ax1.set_title(f'GEP Revised vs GPR Daily Index (1996–2025)  |  r = {r:.4f}',
                  fontsize=13, pad=10)
    ax1.legend(fontsize=9, framealpha=0.7, loc='upper left')
    ax1.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(df['date'], df['roll_corr'], color='#5A4FCF', linewidth=0.8)
    ax2.axhline(0, color='black', linewidth=0.4)
    ax2.axhline(r, color='gray', linewidth=0.6, linestyle='--', alpha=0.7,
                label=f'Overall r = {r:.3f}')
    ax2.fill_between(df['date'], df['roll_corr'], 0,
                     where=df['roll_corr'] > 0, alpha=0.15, color='#5A4FCF')
    ax2.fill_between(df['date'], df['roll_corr'], 0,
                     where=df['roll_corr'] < 0, alpha=0.15, color='red')
    ax2.set_ylabel('90-day rolling\ncorrelation', fontsize=9)
    ax2.set_ylim(-1, 1)
    ax2.legend(fontsize=8, framealpha=0.7)
    ax2.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out1 = os.path.join(args.output_dir, "gep_vs_gprd_timeseries_revised.png")
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out1}")
    plt.close()

    # ----------------------------------------------------------------
    # 9. Plot 2: Cross-correlation ±60 days
    # ----------------------------------------------------------------
    max_lag = 60
    lags    = range(-max_lag, max_lag + 1)
    xcorrs  = []
    for lag in lags:
        if lag == 0:
            xcorrs.append(r)
        elif lag > 0:
            xcorrs.append(df['GEP_z'].iloc[:-lag].corr(df['GPRD_z'].iloc[lag:]))
        else:
            xcorrs.append(df['GEP_z'].iloc[-lag:].corr(df['GPRD_z'].iloc[:lag]))

    lags_arr   = np.array(list(lags))
    xcorrs_arr = np.array(xcorrs)
    peak_lag   = lags_arr[np.argmax(xcorrs_arr)]
    peak_corr  = xcorrs_arr.max()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(lags_arr, xcorrs_arr, width=1.0,
           color=np.where(lags_arr >= 0, '#378ADD', '#E05C2A'), alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(peak_lag, color='green', linewidth=1.2, linestyle='--',
               label=f'Peak: lag = {peak_lag} days (r = {peak_corr:.4f})')
    ax.axhline(0, color='black', linewidth=0.4)
    ax.set_xlabel('Lag (days)  |  Blue: GEP leads → GPR  |  Red: GPR leads → GEP', fontsize=10)
    ax.set_ylabel('Pearson r', fontsize=10)
    ax.set_title('Cross-correlation: GEP Revised vs GPRD (1996–2025)', fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out2 = os.path.join(args.output_dir, "gep_vs_gprd_crosscorr_revised.png")
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")
    print(f"\nPeak cross-correlation: lag = {peak_lag} days, r = {peak_corr:.4f}")
    if peak_lag > 0:
        print(f"  → GEP leads GPR by {peak_lag} day(s)")
    elif peak_lag < 0:
        print(f"  → GPR leads GEP by {-peak_lag} day(s)")
    else:
        print("  → No lead-lag (contemporaneous)")
    plt.close()


if __name__ == "__main__":
    main()
