#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adds an EPU-style GEP_norm column (long-run mean = 100) to an existing
GEP_Monthly_Index.csv, for the US and EU indices. Does not recompute the index.
"""

import os
import csv
import argparse
import statistics


def normalize(rows, value_col='GEP_monthly'):
    vals = [r[value_col] for r in rows]
    mean = statistics.mean(vals)
    for r in rows:
        r['GEP_norm']       = r[value_col] / mean * 100.0
        r['GEP_norm_mean']  = mean          # stored for transparency / reproducibility
    return rows, mean


def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    # Cast numeric columns
    numeric = {'GEP_monthly', 'n_articles', 'n_gep_articles', 'n_trading_days'}
    for row in rows:
        for col in numeric:
            if col in row:
                row[col] = float(row[col])
    return rows


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def describe(rows, label, col='GEP_norm'):
    vals = sorted(r[col] for r in rows)
    n    = len(vals)
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals)
    cv   = std / mean
    p25  = vals[int(n * 0.25)]
    p50  = vals[int(n * 0.50)]
    p75  = vals[int(n * 0.75)]
    print(f"  {label}: mean={mean:.2f}  std={std:.2f}  CV={cv:.3f}"
          f"  min={vals[0]:.2f}  25%={p25:.2f}  50%={p50:.2f}"
          f"  75%={p75:.2f}  max={vals[-1]:.2f}  (n={n})")


def main():
    parser = argparse.ArgumentParser(description="Add EPU-style GEP_norm column to monthly index CSVs.")
    parser.add_argument('--us_path', type=str, required=True,
                        help='Path to US GEP_Monthly_Index.csv')
    parser.add_argument('--eu_path', type=str, required=True,
                        help='Path to EU GEP_Monthly_Index.csv')
    args = parser.parse_args()

    for label, path in [('US', args.us_path), ('EU', args.eu_path)]:
        if not os.path.isfile(path):
            print(f'[ERROR] File not found: {path}')
            continue

        rows = load_csv(path)
        rows, mean = normalize(rows)
        save_csv(path, rows)

        print(f'[OK] {label} ({path})')
        print(f'     Normalization mean (GEP_monthly): {mean:.8f}')
        describe(rows, label)

    print()
    print('GEP_norm interpretation: 100 = long-run average month')
    print('Both series normalized independently — levels not comparable,')
    print('but relative swings and timing are.')


if __name__ == '__main__':
    main()
