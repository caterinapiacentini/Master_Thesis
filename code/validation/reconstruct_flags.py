#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconstruct_flags.py

Reconstructs, at the article level, the official GEP flag definition
(>= --min_topics distinct GTM topic categories matched, baseline = 2,
see Master_Thesis/README.md "GEP_*_Robust_min2.csv") over the cleaned
US corpus (clean_txt/DATA_US + clean_txt/INFO_DATA_US), and records which
dictionary terms matched, for downstream human-annotation validation.

Processes ONE year per invocation (designed to run as a SLURM array job,
one task per year — see slurm/validation/reconstruct_flags.slurm) and
writes a single parquet file per year with one row per article:

    article_id, date, year, month, headline,
    matched_terms (";"-joined, sorted), n_matched_terms,
    topics_hit (";"-joined topic names), n_topics_hit, gep_flag

This intentionally does NOT store article body text — only headline and
match metadata — to keep per-year files small over the ~89M-article corpus.
Readable text for the small human-annotation sample is fetched separately,
on demand, from raw_data by build_sample.py.
"""

import argparse
import gzip
import json
import os

import pandas as pd
from tqdm import tqdm


def load_topic_sets(dict_csv: str) -> dict:
    df = pd.read_csv(dict_csv)
    topic_sets = {}
    for topic, group in df.groupby("topic"):
        topic_sets[topic] = frozenset(group["term"].astype(str))
    return topic_sets


def process_year(
    year: str,
    topic_sets: dict,
    text_dir: str,
    meta_dir: str,
    min_topics: int,
) -> pd.DataFrame:
    union_set = set()
    for words in topic_sets.values():
        union_set.update(words)

    text_path = os.path.join(text_dir, f"rtrs_{year}_us_nodiary_noboiler.txt.gz")
    meta_matches = [
        f for f in os.listdir(meta_dir)
        if f.startswith(f"rtrs_{year}_") and f.endswith("meta.jsonl.gz")
    ]
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Missing text file: {text_path}")
    if not meta_matches:
        raise FileNotFoundError(f"No meta file for year {year} in {meta_dir}")
    meta_path = os.path.join(meta_dir, meta_matches[0])

    records = []
    n_skipped_no_date = 0

    with gzip.open(text_path, "rt", encoding="utf-8") as f_txt, \
         gzip.open(meta_path, "rt", encoding="utf-8") as f_meta:

        for txt_line, meta_line in tqdm(zip(f_txt, f_meta), desc=year, unit="art"):
            meta_obj = json.loads(meta_line)

            raw_date = (
                meta_obj.get("versionCreated") or
                meta_obj.get("firstCreated") or
                ""
            )[:10]
            try:
                date_str = pd.to_datetime(raw_date).strftime("%Y-%m-%d")
            except Exception:
                n_skipped_no_date += 1
                continue

            article_id = meta_obj.get("guid") or meta_obj.get("id")
            headline = (meta_obj.get("headline_raw") or "").strip()

            text = txt_line.strip()
            word_set = set(text.split())

            matched = sorted(union_set & word_set)
            topics_hit = sorted(
                topic for topic, wset in topic_sets.items() if wset & word_set
            )

            records.append({
                "article_id": article_id,
                "date": date_str,
                "year": int(year),
                "month": date_str[:7],
                "headline": headline,
                "matched_terms": ";".join(matched),
                "n_matched_terms": len(matched),
                "topics_hit": ";".join(topics_hit),
                "n_topics_hit": len(topics_hit),
                "gep_flag": len(topics_hit) >= min_topics,
            })

    if n_skipped_no_date:
        print(f"[WARN] {year}: skipped {n_skipped_no_date} articles with unparsable dates")

    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct per-article GEP flag + matched terms for one year."
    )
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument(
        "--dict_csv", type=str,
        default=os.path.expanduser("~/Master_Thesis/data/validation/gep_dictionary.csv"),
    )
    parser.add_argument(
        "--text_dir", type=str,
        default=os.path.expanduser("~/Final_Thesis_Clean/clean_txt/DATA_US"),
    )
    parser.add_argument(
        "--meta_dir", type=str,
        default=os.path.expanduser("~/Final_Thesis_Clean/clean_txt/INFO_DATA_US"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.expanduser("~/Final_Thesis_Clean/output/GEP_Validation/flags"),
    )
    parser.add_argument(
        "--min_topics", type=int, default=2,
        help="Distinct-topic threshold for gep_flag (2 = official baseline)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Year        : {args.year}")
    print(f"[INFO] Dictionary  : {args.dict_csv}")
    print(f"[INFO] min_topics  : {args.min_topics}")

    topic_sets = load_topic_sets(args.dict_csv)
    n_words = sum(len(v) for v in topic_sets.values())
    print(f"[INFO] Loaded {len(topic_sets)} topics, {n_words} (term,topic) rows")

    df = process_year(args.year, topic_sets, args.text_dir, args.meta_dir, args.min_topics)

    out_path = os.path.join(args.output_dir, f"flags_{args.year}.parquet")
    df.to_parquet(out_path, index=False)

    n_articles = len(df)
    n_flagged = int(df["gep_flag"].sum()) if n_articles else 0
    print("-" * 60)
    print(f"[OK] {args.year}: {n_articles:,} articles, {n_flagged:,} flagged "
          f"({100 * n_flagged / n_articles:.2f}%) -> {out_path}")


if __name__ == "__main__":
    main()
