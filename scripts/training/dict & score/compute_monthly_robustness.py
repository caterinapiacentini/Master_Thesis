#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_monthly_robustness.py

Computes an alternative (robustness) version of the monthly GEP index
from the existing GEP_Daily_Index.csv.

The original GEP_monthly is a weighted average of daily scores by article
count — equivalent to averaging over all articles in the month:
    GEP_monthly = Σ_t (score_t * n_articles_t) / Σ_t n_articles_t

The robustness version weights each trading day equally (simple average
of daily GEP scores):
    GEP_monthly_daily_avg = (1/T_m) * Σ_t score_t   for t in month m

Saves GEP_Monthly_Robustness.csv alongside the original monthly index.

Usage:
    python compute_monthly_robustness.py \
        --daily_path  /path/to/GEP_Daily_Index.csv \
        --output_dir  /path/to/output/
"""

import os
import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compute robustness monthly GEP index (simple average of daily scores)."
    )
    parser.add_argument("--daily_path",  type=str, required=True,
                        help="Path to GEP_Daily_Index.csv")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="Directory to save GEP_Monthly_Robustness.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load daily index — keep only days with actual articles
    #    (gap-filled days have n_articles == 0 and forward-filled scores;
    #     we exclude them so the average is over true trading days only)
    # ----------------------------------------------------------------
    daily = pd.read_csv(args.daily_path, parse_dates=['date'])
    daily_obs = daily[daily['n_articles'] > 0].copy()
    print(f"[INFO] Loaded {len(daily):,} calendar days, "
          f"{len(daily_obs):,} trading days with articles.")

    daily_obs['month'] = daily_obs['date'].dt.to_period('M')

    # ----------------------------------------------------------------
    # 2. Robustness monthly index: simple average of daily scores
    #    (each trading day weighted equally)
    # ----------------------------------------------------------------
    monthly_rob = daily_obs.groupby('month').apply(
        lambda g: pd.Series({
            'GEP_monthly_daily_avg': g['score'].mean(),
            'n_articles':            g['n_articles'].sum(),
            'n_gep_articles':        g['n_gep_articles'].sum(),
            'n_trading_days':        len(g),
        })
    ).reset_index()

    monthly_rob['month'] = monthly_rob['month'].astype(str)

    out_path = os.path.join(args.output_dir, "GEP_Monthly_Robustness.csv")
    monthly_rob.to_csv(out_path, index=False)

    print(f"[OK] Saved: {out_path}")
    print(f"     Months covered : {len(monthly_rob)}")
    print(f"     Mean / std     : {monthly_rob['GEP_monthly_daily_avg'].mean():.6f} "
          f"/ {monthly_rob['GEP_monthly_daily_avg'].std():.6f}")


if __name__ == "__main__":
    main()
