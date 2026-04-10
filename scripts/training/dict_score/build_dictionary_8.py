#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### daje roma daje

import os
import pandas as pd
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate GTM topic CSVs into a global geoeconomic pressure dictionary."
    )
    parser.add_argument("--input_dir",   type=str, required=True,  help="Path to GTM results directory")
    parser.add_argument("--output_dir",  type=str, required=True,  help="Path to save final dictionary")
    parser.add_argument("--num_topics",  type=int, required=True,  help="Total number of sub-topics Q (used in mean aggregation)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Identify all topic CSVs (e.g., topic_Sanctions.csv, topic_Trade_War.csv)
    csv_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.startswith("topic_") and f.endswith(".csv")
    ])

    if not csv_files:
        print(f"[ERROR] No topic CSVs found in {args.input_dir}")
        return

    print(f"[INFO] Found {len(csv_files)} topic CSVs:")
    for f in csv_files:
        print(f"       {f}")

    if len(csv_files) != args.num_topics:
        print(
            f"[WARNING] {len(csv_files)} CSVs found but --num_topics={args.num_topics}. "
            f"Global weights will be divided by {args.num_topics} as specified. "
            f"Verify this is intentional."
        )

    # Per-word accumulators
    # word_weight_sums : summed normalized local weights across topics
    # word_topic_count : number of topics in which the word appears
    # word_topics      : list of topic names where the word appears (for provenance)
    word_weight_sums  = defaultdict(float)
    word_topic_count  = defaultdict(int)
    word_topics       = defaultdict(list)

    for file_name in csv_files:
        # Derive a clean topic label from the filename (e.g. "topic_Trade_War.csv" → "Trade War")
        topic_label = file_name.replace("topic_", "").replace(".csv", "").replace("_", " ")
        file_path   = os.path.join(args.input_dir, file_name)

        df = pd.read_csv(file_path, index_col=0)

        if 'weight' not in df.columns:
            print(f"[WARNING] No 'weight' column in {file_name} — skipping.")
            continue

        # Local normalization: scale each topic's weights to [0, 1]
        # Ensures no single sub-topic dominates due to differences in absolute scale
        # (Dangl & Salbrechter 2023, Section 2.1)
        max_val = df['weight'].max()
        if max_val <= 0:
            print(f"[WARNING] Max weight <= 0 in {file_name} — skipping.")
            continue

        df['weight'] = df['weight'] / max_val

        for word, row in df.iterrows():
            clean_word = str(word).strip()
            w          = float(row['weight'])
            word_weight_sums[clean_word] += w
            word_topic_count[clean_word] += 1
            word_topics[clean_word].append(topic_label)

        print(f"[OK] {topic_label:<30} {len(df)} words loaded, max_val={max_val:.4f}")

    # Global aggregation: mean normalized weight across all Q sub-dimensions
    # Global_weight(w) = (1/Q) * sum_q [ normalized_weight_q(w) ]
    # Words appearing in multiple topics receive proportionally higher global weight,
    # reflecting their cross-dimensional relevance to geoeconomic pressure.
    # (Dangl & Salbrechter 2023, Equation 10)
    Q = args.num_topics

    final_list = []
    for word, total_weight in word_weight_sums.items():
        final_list.append({
            'word':         word,
            'weight':       total_weight / Q,
            'topic_count':  word_topic_count[word],
            'topics':       "; ".join(word_topics[word]),
        })

    df_final = (
        pd.DataFrame(final_list)
        .sort_values(by='weight', ascending=False)
        .reset_index(drop=True)
    )

    output_path = os.path.join(args.output_dir, "geoeconomic_dictionary.csv")
    df_final.to_csv(output_path, index=False)

    # Summary statistics
    n_total    = len(df_final)
    n_multi    = (df_final['topic_count'] > 1).sum()
    top10      = df_final.head(10)[['word', 'weight', 'topic_count']].to_string(index=False)

    print(f"\n{'='*55}")
    print(f"  Dictionary saved to: {output_path}")
    print(f"  Total unique words : {n_total}")
    print(f"  Words in >1 topic  : {n_multi}  ({100*n_multi/n_total:.1f}%)")
    print(f"\n  Top 10 words by global weight:")
    print(f"  {top10}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
