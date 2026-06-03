#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import argparse
import gzip
import json
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Daily GEP Index from news corpus and GTM dictionary."
    )
    parser.add_argument("--dict_path",   type=str, required=True, help="Path to geoeconomic_dictionary.csv")
    parser.add_argument("--text_dir",    type=str, required=True, help="Path to DATA1 (text .txt.gz files)")
    parser.add_argument("--meta_dir",    type=str, required=True, help="Path to INFO_DATA1 (meta .jsonl.gz files)")
    parser.add_argument("--output_dir",  type=str, required=True, help="Path to save final index")
    parser.add_argument("--freq_cap",    type=int, default=4,
                        help="Max per-article count for any single dictionary word (default: 4)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load dictionary
    # ----------------------------------------------------------------
    dict_df       = pd.read_csv(args.dict_path)
    T_dict        = dict(zip(dict_df['word'], dict_df['weight']))
    relevant_words = set(T_dict.keys())
    print(f"[INFO] Dictionary loaded: {len(T_dict)} words")
    print(f"[INFO] Frequency cap    : {args.freq_cap}")

    # ----------------------------------------------------------------
    # 2. Identify paired text / meta files
    # ----------------------------------------------------------------
    text_files = sorted([f for f in os.listdir(args.text_dir) if f.endswith('.txt.gz')])
    if not text_files:
        print(f"[ERROR] No .txt.gz files found in {args.text_dir}")
        return

    all_daily_results = []

    for txt_file in text_files:
        # Derive year and corresponding meta filename
        # Expected format: rtrs_YYYY_clean.txt.gz  or  rtrs_YYYY_region_...txt.gz
        parts    = txt_file.split('_')
        year     = parts[1]
        txt_path = os.path.join(args.text_dir, txt_file)

        # Find the meta file for this year: handles both rtrs_YYYY_meta.jsonl.gz
        # and rtrs_YYYY_world_meta.jsonl.gz (or any other infix).
        matches = glob.glob(os.path.join(args.meta_dir, f"rtrs_{year}_*meta.jsonl.gz"))
        if not matches:
            print(f"[SKIP] Meta file not found for {txt_file} — expected rtrs_{year}_*meta.jsonl.gz in {args.meta_dir}")
            continue
        meta_path = matches[0]

        print(f"\n[INFO] Processing year: {year}")
        daily_data = []

        try:
            with gzip.open(txt_path,  'rt', encoding='utf-8') as f_txt, \
                 gzip.open(meta_path, 'rt', encoding='utf-8') as f_meta:

                for txt_line, meta_line in tqdm(zip(f_txt, f_meta), desc=year):
                    meta_obj = json.loads(meta_line)

                    # Robust date extraction: use versionCreated (the date this specific
                    # version was published) rather than firstCreated (the original article
                    # creation date). firstCreated is identical across all subsequent
                    # updates of the same article, so using it causes all update versions
                    # to pile up on the original publication date, inflating article counts
                    # on a few days by hundreds of thousands. Fall back to firstCreated
                    # only if versionCreated is absent.
                    raw_date = (
                        meta_obj.get('versionCreated') or
                        meta_obj.get('firstCreated') or
                        ''
                    )[:10]
                    try:
                        date_str = pd.to_datetime(raw_date).strftime('%Y-%m-%d')
                    except Exception:
                        continue   # skip articles with unparseable dates

                    text        = txt_line.strip()
                    words       = text.split()
                    total_words = len(words)

                    if total_words == 0:
                        continue

                    # Per-article scoring with frequency capping
                    # score_i = Σ_w [ min(count_w, freq_cap) × weight_w ] / N_words
                    # Aligned with Dangl & Salbrechter (2023): capping prevents
                    # a single high-frequency keyword from dominating the signal;
                    # length normalization ensures comparability across article sizes.
                    counts = {}
                    for w in words:
                        if w in relevant_words:
                            counts[w] = min(counts.get(w, 0) + 1, args.freq_cap)

                    weighted_sum = sum(counts[w] * T_dict[w] for w in counts)
                    score        = weighted_sum / total_words

                    daily_data.append({'date': date_str, 'score': score})

        except Exception as e:
            print(f"[ERROR] Failed on year {year}: {e}")
            continue

        if not daily_data:
            print(f"[WARN] No valid articles found for year {year}")
            continue

        df_year = pd.DataFrame(daily_data)
        df_year['date'] = pd.to_datetime(df_year['date'])

        # Per-day aggregation
        # score          : mean article score (intensity — how geoeconomic each article is)
        # score_volume   : sum of article scores (total daily GEP signal mass)
        # n_articles     : total articles published that day (needed to interpret score_volume)
        # n_gep_articles : articles with score > 0 (actually matched the dictionary)
        agg = df_year.groupby('date').agg(
            score          = ('score', 'mean'),
            score_volume   = ('score', 'sum'),
            n_articles     = ('score', 'count'),
            n_gep_articles = ('score', lambda x: (x > 0).sum()),
        ).reset_index()

        all_daily_results.append(agg)
        print(f"[OK] Year {year}: {len(df_year):,} articles → {len(agg)} trading days")

    if not all_daily_results:
        print("[ERROR] No results to consolidate.")
        return

    # ----------------------------------------------------------------
    # 3. Consolidate across years
    # FIX: re-aggregate after concat to eliminate duplicate dates that
    # arise when articles in one yearly file have dates spilling into
    # an adjacent year (e.g. a 1996 file containing a few 1997 dates).
    # Recompute mean score as a proper weighted average by n_articles
    # so the daily mean is never distorted by duplicate rows.
    # ----------------------------------------------------------------
    combined = (
        pd.concat(all_daily_results)
        .sort_values('date')
        .reset_index(drop=True)
    )

    # Weighted-average score across duplicate date rows
    combined['score_x_n'] = combined['score'] * combined['n_articles']

    final_index = combined.groupby('date').agg(
        score_x_n      = ('score_x_n',     'sum'),
        score_volume   = ('score_volume',  'sum'),
        n_articles     = ('n_articles',    'sum'),
        n_gep_articles = ('n_gep_articles','sum'),
    ).reset_index()

    final_index['score'] = (
        final_index['score_x_n'] / final_index['n_articles'].replace(0, np.nan)
    )
    final_index = final_index.drop(columns='score_x_n')
    final_index = final_index.sort_values('date').reset_index(drop=True)

    # Keep a pre-gap-fill copy for monthly aggregation (needs true
    # article counts, not the zeros inserted for gap days)
    all_daily_results_combined = final_index.copy()

    # ----------------------------------------------------------------
    # 4. Fill date gaps (weekends, holidays, coverage gaps)
    # Forward-fill score and intensity; set volume/count to 0 on gap days
    # so downstream models have a complete daily series.
    # ----------------------------------------------------------------
    full_range  = pd.date_range(final_index['date'].min(), final_index['date'].max(), freq='D')
    final_index = (
        final_index
        .set_index('date')
        .reindex(full_range)
    )
    final_index.index.name = 'date'

    # Gap days: volume and count are genuinely 0 (no news published)
    final_index['n_articles']     = final_index['n_articles'].fillna(0).astype(int)
    final_index['n_gep_articles'] = final_index['n_gep_articles'].fillna(0).astype(int)
    final_index['score_volume']   = final_index['score_volume'].fillna(0.0)

    # Intensity on gap days: forward-fill is the standard convention
    # for text-based indices (no news = carry last signal forward)
    final_index['score'] = final_index['score'].ffill()

    final_index = final_index.reset_index()

    # ----------------------------------------------------------------
    # 5. Save daily index
    # ----------------------------------------------------------------
    daily_path = os.path.join(args.output_dir, "GEP_Daily_Index.csv")
    final_index.to_csv(daily_path, index=False)

    # ----------------------------------------------------------------
    # 6. Monthly GEP  (GEP_m^M = 1/|A_m| * Σ Score_d over month m)
    # Computed directly from raw article scores, NOT from daily averages,
    # to match the slide formula exactly: average over all articles in
    # the month, not average of daily averages (which would weight each
    # day equally regardless of how many articles it contained).
    # ----------------------------------------------------------------
    all_articles = all_daily_results_combined[['date', 'score', 'n_articles', 'n_gep_articles']].copy()

    # Reconstruct article-level approximation: we have daily means and counts,
    # so we can recover the exact monthly mean as a weighted average of daily means
    # weighted by n_articles — equivalent to averaging over all articles in the month.
    # GEP_m^M = Σ_t (score_t * n_articles_t) / Σ_t n_articles_t   for t in month m
    all_articles['date'] = pd.to_datetime(all_articles['date'])
    all_articles['month'] = all_articles['date'].dt.to_period('M')

    monthly_index = all_articles.groupby('month').apply(
        lambda g: pd.Series({
            'GEP_monthly':       np.average(g['score'], weights=g['n_articles']),
            'n_articles':        g['n_articles'].sum(),
            'n_gep_articles':    g['n_gep_articles'].sum(),
            'n_trading_days':    len(g),
        })
    ).reset_index()

    monthly_index['month'] = monthly_index['month'].astype(str)

    monthly_path = os.path.join(args.output_dir, "GEP_Monthly_Index.csv")
    monthly_index.to_csv(monthly_path, index=False)

    # ----------------------------------------------------------------
    # 7. Summary
    # ----------------------------------------------------------------
    n_days      = len(final_index)
    n_gap_days  = (final_index['n_articles'] == 0).sum()
    date_min    = final_index['date'].min()
    date_max    = final_index['date'].max()

    print(f"\n{'='*55}")
    print(f"  GEP Daily Index   : {daily_path}")
    print(f"  GEP Monthly Index : {monthly_path}")
    print(f"  Date range        : {date_min} → {date_max}")
    print(f"  Calendar days     : {n_days:,}  (gap days: {n_gap_days:,}, {100*n_gap_days/n_days:.1f}%)")
    print(f"  Months covered    : {len(monthly_index)}")
    print(f"  Daily  score mean / std : {final_index['score'].mean():.6f} / {final_index['score'].std():.6f}")
    print(f"  Monthly GEP mean / std  : {monthly_index['GEP_monthly'].mean():.6f} / {monthly_index['GEP_monthly'].std():.6f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
