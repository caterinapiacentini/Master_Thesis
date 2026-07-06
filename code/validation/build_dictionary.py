#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dictionary.py

Flattens the official GTM_6 US topic word lists (data/gtm/topic_*.csv, i.e.
GTM_6_results_US — the baseline dictionary, NOT the gtm_v2 robustness variant)
into a single "CSV dictionary of GEP terms" with columns:

    term, topic, weight

A term that appears under more than one topic gets one row per topic. This
file is the single source of truth for the validation pipeline's dictionary
matching (build_dictionary.py -> reconstruct_flags.py -> build_sample.py).
"""

import argparse
import glob
import os

import pandas as pd


def build_dictionary(gtm_dir: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(gtm_dir, "topic_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No topic_*.csv files found in {gtm_dir}")

    rows = []
    for path in csv_files:
        topic_name = os.path.basename(path).replace("topic_", "").replace(".csv", "")
        df = pd.read_csv(path, index_col=0)
        df.index.name = "term"
        df = df.reset_index()
        df["term"] = df["term"].astype(str).str.strip()
        df = df[df["term"] != ""]
        df["topic"] = topic_name
        rows.append(df[["term", "topic", "weight"]])
        print(f"  [dict] {topic_name:<25} {len(df):>4} terms")

    dictionary = pd.concat(rows, ignore_index=True)
    dictionary = dictionary.drop_duplicates(subset=["term", "topic"])
    dictionary = dictionary.sort_values(["topic", "term"]).reset_index(drop=True)
    return dictionary


def main():
    parser = argparse.ArgumentParser(
        description="Build flat GEP term dictionary CSV from GTM_6_results_US topic CSVs."
    )
    parser.add_argument(
        "--gtm_dir", type=str,
        default=os.path.expanduser("~/Final_Thesis_Clean/output/GTM_6_results_US"),
        help="Directory containing topic_*.csv (baseline GTM_6 US topics, not v2)",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.expanduser("~/Master_Thesis/data/validation/gep_dictionary.csv"),
        help="Output path for the flat term dictionary CSV",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Building flat GEP term dictionary")
    print(f"  Source: {args.gtm_dir}")
    print("=" * 60)

    dictionary = build_dictionary(args.gtm_dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dictionary.to_csv(args.output, index=False)

    n_topics = dictionary["topic"].nunique()
    n_unique_terms = dictionary["term"].nunique()
    n_multi_topic = (
        dictionary.groupby("term")["topic"].nunique().gt(1).sum()
    )

    print("-" * 60)
    print(f"[OK] {len(dictionary):,} (term, topic) rows written -> {args.output}")
    print(f"[INFO] Topics             : {n_topics}")
    print(f"[INFO] Unique terms       : {n_unique_terms:,}")
    print(f"[INFO] Terms in >1 topic  : {n_multi_topic:,}")


if __name__ == "__main__":
    main()
