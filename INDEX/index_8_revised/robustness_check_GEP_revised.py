#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_check_GEP_revised.py

Robustness check for the GTM-8 Revised GEP index:
compares article-weighted vs day-weighted monthly aggregation.

Usage:
    python robustness_check_GEP_revised.py
        --monthly_path <path/to/GEP_Monthly_Index.csv>
        --robust_path  <path/to/GEP_Monthly_Robustness.csv>
        --output_path  <path/to/robustness_check_GEP_revised.png>
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly_path", type=str, required=True)
    parser.add_argument("--robust_path",  type=str, required=True)
    parser.add_argument("--output_path",  type=str, required=True)
    args = parser.parse_args()

    monthly = pd.read_csv(args.monthly_path)
    monthly['month'] = pd.to_datetime(monthly['month'])

    rob = pd.read_csv(args.robust_path)
    rob['month'] = pd.to_datetime(rob['month'])

    df = monthly[['month', 'GEP_monthly']].merge(
        rob[['month', 'GEP_monthly_daily_avg']], on='month', how='inner'
    )
    df['GEP_monthly_scaled']   = df['GEP_monthly']          * 10_000
    df['GEP_daily_avg_scaled'] = df['GEP_monthly_daily_avg'] * 10_000

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(df['month'], df['GEP_monthly_scaled'],
            color='#378ADD', linewidth=0.9, alpha=0.9,
            label='Article-weighted avg (baseline)')
    ax.plot(df['month'], df['GEP_daily_avg_scaled'],
            color='#E05C2A', linewidth=0.9, alpha=0.85, linestyle='--',
            label='Day-weighted avg (robustness)')

    corr = np.corrcoef(df['GEP_monthly_scaled'], df['GEP_daily_avg_scaled'])[0, 1]
    ax.text(0.01, 0.97, f'Correlation: {corr:.4f}',
            transform=ax.transAxes, fontsize=9, va='top', color='#333333')

    ax.set_title('GEP Monthly Index — Robustness Check, Revised (1996–2025)', fontsize=13, pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('GEP score (×10⁻⁴)', fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output_path}")
    print(f"Correlation between the two series: {corr:.4f}")


if __name__ == "__main__":
    main()
