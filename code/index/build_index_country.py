#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
new_index_country.py

Builds a country-specific GEP index.  An article is a hit only if:
  1. It contains words from >= min_topics distinct GTM topic categories (default 2)
  2. It mentions at least one of the country's name terms in the text

Supported --country values: japan, us, uk, china, iran, russia, germany
"""

import os
import glob
import argparse
import gzip
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Country name terms (all lowercase — corpus is already lowercased)
# Substring match on raw text, so multi-word terms like "united states" work.
# ---------------------------------------------------------------------------

COUNTRY_TERMS = {
    "japan":   {"japan", "japanese", "tokyo"},
    "us":      {"united states", "us", "american", "americans"},
    "uk":      {"united kingdom", "britain", "british", "uk"},
    "china":   {"china", "chinese", "beijing"},
    "iran":    {"iran", "iranian", "tehran"},
    "russia":  {"russia", "russian", "moscow"},
    "germany": {"germany", "german", "deutschland"},
}


# ---------------------------------------------------------------------------
# 1.  Build per-topic word sets from GTM topic CSVs
# ---------------------------------------------------------------------------

def build_topic_word_sets(gtm_results_dir: str) -> dict:
    csv_files = sorted(glob.glob(os.path.join(gtm_results_dir, "topic_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No topic_*.csv files found in {gtm_results_dir}")

    topic_sets = {}
    union = set()
    for path in csv_files:
        topic_name = os.path.basename(path).replace("topic_", "").replace(".csv", "")
        df = pd.read_csv(path, index_col=0)
        words = frozenset(str(w).strip() for w in df.index if str(w).strip())
        topic_sets[topic_name] = words
        union.update(words)
        print(f"  [dict] {topic_name:<25} {len(words):>4} words  "
              f"(running union: {len(union)})")

    print(f"\n[INFO] Combined dictionary : {len(union)} unique words")
    print(f"[INFO] Number of topics    : {len(topic_sets)}\n")
    return topic_sets


# ---------------------------------------------------------------------------
# 2.  Process corpus
# ---------------------------------------------------------------------------

def process_corpus(
    topic_sets: dict,
    country_terms: set,
    text_dir: str,
    meta_dir: str,
    min_topics: int = 2,
) -> pd.DataFrame:
    """
    For every article: hit=1 iff it contains >= min_topics distinct GTM topic
    categories AND at least one country term appears in the article text.
    """
    union_set = set()
    for words in topic_sets.values():
        union_set.update(words)

    text_files = sorted(
        [f for f in os.listdir(text_dir) if f.endswith(".txt.gz")]
    )
    if not text_files:
        raise FileNotFoundError(f"No .txt.gz files found in {text_dir}")

    records = []

    for txt_file in text_files:
        year = txt_file.split("_")[1]
        meta_matches = glob.glob(
            os.path.join(meta_dir, f"rtrs_{year}_*meta.jsonl.gz")
        )
        if not meta_matches:
            print(f"[SKIP] No meta file for year {year}")
            continue
        meta_path = meta_matches[0]
        txt_path  = os.path.join(text_dir, txt_file)

        print(f"[INFO] Processing year: {year}")
        year_records = []

        try:
            with gzip.open(txt_path,  "rt", encoding="utf-8") as f_txt, \
                 gzip.open(meta_path, "rt", encoding="utf-8") as f_meta:

                for txt_line, meta_line in tqdm(
                    zip(f_txt, f_meta), desc=year, unit="art"
                ):
                    meta_obj = json.loads(meta_line)

                    raw_date = (
                        meta_obj.get("versionCreated") or
                        meta_obj.get("firstCreated") or
                        ""
                    )[:10]
                    try:
                        date_str = pd.to_datetime(raw_date).strftime("%Y-%m-%d")
                    except Exception:
                        continue

                    text = txt_line.strip()
                    if not text:
                        continue

                    # Condition 1: article must mention the country
                    country_present = any(term in text for term in country_terms)
                    if not country_present:
                        year_records.append({"date": date_str, "hit": 0})
                        continue

                    # Condition 2: >= min_topics distinct GTM topic categories
                    word_set_article = set(text.split())
                    topics_hit = sum(
                        1 for topic_words in topic_sets.values()
                        if word_set_article & topic_words
                    )
                    hit = int(topics_hit >= min_topics)

                    year_records.append({"date": date_str, "hit": hit})

        except Exception as exc:
            print(f"[ERROR] Year {year}: {exc}")
            continue

        if year_records:
            n_arts = len(year_records)
            n_hits = sum(r["hit"] for r in year_records)
            print(f"  → {n_arts:>10,} articles | {n_hits:>8,} hits "
                  f"({100 * n_hits / n_arts:.2f}%)")
            records.extend(year_records)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3.  Aggregate to daily and monthly indices
# ---------------------------------------------------------------------------

def build_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])

    agg = df.groupby("date").agg(
        n_articles     = ("hit", "count"),
        n_gep_articles = ("hit", "sum"),
    ).reset_index()

    agg["GEP_daily"] = agg["n_gep_articles"] / agg["n_articles"].replace(0, np.nan)

    full_range = pd.date_range(agg["date"].min(), agg["date"].max(), freq="D")
    agg = agg.set_index("date").reindex(full_range)
    agg.index.name = "date"
    agg["n_articles"]     = agg["n_articles"].fillna(0).astype(int)
    agg["n_gep_articles"] = agg["n_gep_articles"].fillna(0).astype(int)
    agg["GEP_daily"]      = agg["GEP_daily"].ffill()

    return agg.reset_index()


def build_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    monthly = df.groupby("month").agg(
        n_articles     = ("hit", "count"),
        n_gep_articles = ("hit", "sum"),
    ).reset_index()

    monthly["GEP_monthly"] = (
        monthly["n_gep_articles"] / monthly["n_articles"].replace(0, np.nan)
    )
    monthly["month"] = monthly["month"].astype(str)

    return monthly


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a country-specific GEP index: articles must mention the country "
            "AND contain words from >= min_topics distinct GTM topic categories."
        )
    )
    parser.add_argument("--gtm_dir",    type=str, required=True)
    parser.add_argument("--text_dir",   type=str, required=True)
    parser.add_argument("--meta_dir",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--country", type=str, required=True,
        choices=list(COUNTRY_TERMS.keys()),
        help="Country to filter on. Supported: " + ", ".join(COUNTRY_TERMS.keys())
    )
    parser.add_argument(
        "--min_topics", type=int, default=2,
        help="Min distinct GTM topic categories per article (default: 2)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    country_terms = COUNTRY_TERMS[args.country]
    country_label = args.country.upper()

    daily_fname   = f"GEP_Daily_{country_label}_min{args.min_topics}.csv"
    monthly_fname = f"GEP_Monthly_{country_label}_min{args.min_topics}.csv"

    print("=" * 60)
    print(f"  Country filter        : {country_label}")
    print(f"  Country terms         : {sorted(country_terms)}")
    print(f"  Min topics threshold  : {args.min_topics}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("  Step 1: Building word sets from GTM topic CSVs")
    print("=" * 60)
    topic_sets = build_topic_word_sets(args.gtm_dir)

    print("=" * 60)
    print("  Step 2: Scanning corpus for dictionary + country hits")
    print("=" * 60)
    df_all = process_corpus(
        topic_sets, country_terms,
        args.text_dir, args.meta_dir,
        min_topics=args.min_topics,
    )

    if df_all.empty:
        print("[ERROR] No article records found. Check paths.")
        return

    total_arts = len(df_all)
    total_hits = df_all["hit"].sum()
    print(f"\n[INFO] Total articles : {total_arts:,}")
    print(f"[INFO] Total GEP hits : {total_hits:,}  "
          f"({100 * total_hits / total_arts:.2f}% overall share)")

    print("\n" + "=" * 60)
    print("  Step 3: Building daily index")
    print("=" * 60)
    daily = build_daily_index(df_all.copy())
    daily_path = os.path.join(args.output_dir, daily_fname)
    daily.to_csv(daily_path, index=False)
    print(f"[OK] {daily_fname}  → {daily_path}")

    print("\n" + "=" * 60)
    print("  Step 4: Building monthly index")
    print("=" * 60)
    monthly = build_monthly_index(df_all.copy())
    monthly_path = os.path.join(args.output_dir, monthly_fname)
    monthly.to_csv(monthly_path, index=False)
    print(f"[OK] {monthly_fname} → {monthly_path}")

    n_topics = len(topic_sets)
    union_sz = len(set().union(*topic_sets.values()))
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Country               : {country_label}")
    print(f"  Topics / union words  : {n_topics} topics / {union_sz:,} unique words")
    print(f"  Min topics threshold  : {args.min_topics}")
    print(f"  Date range            : {daily['date'].min().date()} → "
          f"{daily['date'].max().date()}")
    print(f"  Calendar days         : {len(daily):,}")
    print(f"  Months covered        : {len(monthly)}")
    gep_days = daily[daily["n_articles"] > 0]
    print(f"  Daily GEP mean / std  : "
          f"{gep_days['GEP_daily'].mean():.4f} / {gep_days['GEP_daily'].std():.4f}")
    print(f"  Monthly GEP mean / std: "
          f"{monthly['GEP_monthly'].mean():.4f} / {monthly['GEP_monthly'].std():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
