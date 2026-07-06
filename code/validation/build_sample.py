#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_sample.py

Builds the stratified human-annotation sample from the per-article flag
files produced by reconstruct_flags.py, and exports an annotation CSV.

Sample design (balanced across decade x high/low-GEP-month strata):
    - 500 flagged articles       (gep_flag == True)
    - 500 unflagged articles     (gep_flag == False), drawn from the exact
                                  same calendar months as the flagged sample
    - 100 hard negatives         (gep_flag == False, exactly 1 matched
                                  dictionary term — a near-miss on the
                                  official >=2-topic rule)

"High" vs "low" GEP months = above/below the median monthly GEP share
(computed across all eligible months). Decades = floor(year/10)*10.

KNOWN DATA CAVEAT: clean_world.py has an unfixed bug that double-counts
Jan-Aug 2023 articles in DATA_US (old- and new-format source files both
processed). Those 8 months are excluded from sampling by default via
--exclude_months so the sample and month classification aren't distorted
by the duplication.

For each sampled article, readable text (headline + body snippet) is
fetched on demand from raw_data (the flags files only carry cleaned/
tokenized text, which isn't fit for human reading).
"""

import argparse
import glob
import gzip
import json
import os
import random
from collections import Counter

import pandas as pd
from tqdm import tqdm

DEFAULT_EXCLUDE_MONTHS = [f"2023-{m:02d}" for m in range(1, 9)]


# ---------------------------------------------------------------------------
# 1. Monthly stats + decade / high-low classification
# ---------------------------------------------------------------------------

def load_monthly_stats(flags_dir: str, years: list) -> pd.DataFrame:
    rows = []
    for year in years:
        path = os.path.join(flags_dir, f"flags_{year}.parquet")
        if not os.path.exists(path):
            print(f"[WARN] Missing flags file for {year}, skipping: {path}")
            continue
        df = pd.read_parquet(path, columns=["month", "gep_flag", "n_matched_terms"])
        g = df.groupby("month").agg(
            n_articles=("gep_flag", "count"),
            n_flagged=("gep_flag", "sum"),
            n_hard_neg_candidates=("n_matched_terms", lambda s: (s == 1).sum()),
        ).reset_index()
        rows.append(g)
    monthly = pd.concat(rows, ignore_index=True)
    monthly["year"] = monthly["month"].str[:4].astype(int)
    monthly["decade"] = (monthly["year"] // 10) * 10
    monthly["gep_share"] = monthly["n_flagged"] / monthly["n_articles"]
    return monthly


def classify_months(monthly: pd.DataFrame, exclude_months: list) -> pd.DataFrame:
    eligible = monthly[~monthly["month"].isin(exclude_months)].copy()
    median_share = eligible["gep_share"].median()
    eligible["month_type"] = eligible["gep_share"].apply(
        lambda s: "high" if s >= median_share else "low"
    )
    print(f"[INFO] Median monthly GEP share (eligible months): {median_share:.4f}")
    print(f"[INFO] Eligible months: {len(eligible)} "
          f"(excluded {len(monthly) - len(eligible)}: {exclude_months})")
    return eligible


# ---------------------------------------------------------------------------
# 2. Stratum target allocation (largest-remainder, as-equal-as-possible)
# ---------------------------------------------------------------------------

def allocate_equal(keys: list, total: int) -> dict:
    n = len(keys)
    base = total // n
    remainder = total - base * n
    keys_sorted = sorted(keys)
    targets = {k: base for k in keys_sorted}
    for k in keys_sorted[:remainder]:
        targets[k] += 1
    return targets


# ---------------------------------------------------------------------------
# 3. Streaming reservoir sampling (Algorithm R) per stratum key
# ---------------------------------------------------------------------------

def reservoir_sample_by_key(
    flags_dir: str,
    years: list,
    row_filter,
    key_fn,
    targets: dict,
    rng: random.Random,
) -> dict:
    reservoirs = {k: [] for k in targets}
    seen = {k: 0 for k in targets}

    for year in years:
        path = os.path.join(flags_dir, f"flags_{year}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        mask = row_filter(df)
        sub = df[mask]
        for row in sub.itertuples(index=False):
            k = key_fn(row)
            if k not in targets or targets[k] <= 0:
                continue
            seen[k] += 1
            res = reservoirs[k]
            cap = targets[k]
            if len(res) < cap:
                res.append(row._asdict())
            else:
                j = rng.randint(0, seen[k] - 1)
                if j < cap:
                    res[j] = row._asdict()

    for k, cap in targets.items():
        if seen[k] < cap:
            print(f"[WARN] Stratum {k}: only {seen[k]} candidates available, "
                  f"wanted {cap}")

    return reservoirs


# ---------------------------------------------------------------------------
# 4. Raw-text lookup for the final sample (readable short_text)
# ---------------------------------------------------------------------------

def iter_rtrs_items(gz_path: str):
    """Minimal RTRS Items-array parser (mirrors code/cleaning/Cleaning_All_World.py)."""
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and '"Items"' in s and "[" in s:
                break
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("]}") or s in ("]}", "]"):
                break
            if s.endswith(","):
                s = s[:-1].rstrip()
            if s.endswith("]}"):
                s = s[:-2].rstrip()
            if not s.startswith("{"):
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def fetch_short_texts(sample_df: pd.DataFrame, raw_data_dir: str, snippet_len: int) -> dict:
    needed = {}
    for _, r in sample_df.iterrows():
        year, month = r["date"][:4], r["date"][5:7]
        needed.setdefault((year, month), set()).add(r["article_id"])

    short_texts = {}
    for (year, month), ids in tqdm(needed.items(), desc="raw lookup"):
        matches = glob.glob(
            os.path.join(raw_data_dir, year, f"STORY.RTRS.{year}-{month}.REC.JSON.txt.gz")
        )
        if not matches:
            print(f"[WARN] No raw_data file for {year}-{month}")
            continue
        remaining = set(ids)
        for item in iter_rtrs_items(matches[0]):
            guid = item.get("guid")
            if guid not in remaining:
                continue
            data = item.get("data") or {}
            headline = (data.get("headline") or "").strip()
            body = (data.get("body") or "").strip().replace("\n", " ")
            body = " ".join(body.split())
            text = f"{headline} — {body}"[:snippet_len]
            short_texts[guid] = text
            remaining.discard(guid)
            if not remaining:
                break
        if remaining:
            print(f"[WARN] {len(remaining)} article(s) not found in {matches[0]}")

    return short_texts


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build stratified GEP annotation sample.")
    parser.add_argument(
        "--flags_dir", type=str,
        default=os.path.expanduser("~/Final_Thesis_Clean/output/GEP_Validation/flags"),
    )
    parser.add_argument(
        "--raw_data_dir", type=str,
        default=os.path.expanduser("~/Final_Thesis_Clean/raw_data"),
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.expanduser("~/Master_Thesis/data/validation/annotation_sample.csv"),
    )
    parser.add_argument("--years", type=int, nargs=2, default=[1996, 2025])
    parser.add_argument("--n_flagged", type=int, default=500)
    parser.add_argument("--n_unflagged", type=int, default=500)
    parser.add_argument("--n_hard_neg", type=int, default=100)
    parser.add_argument("--snippet_len", type=int, default=600)
    parser.add_argument("--exclude_months", type=str, nargs="*", default=DEFAULT_EXCLUDE_MONTHS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    years = list(range(args.years[0], args.years[1] + 1))
    rng = random.Random(args.seed)

    print("=" * 60)
    print("  Step 1: Monthly stats + decade/high-low classification")
    print("=" * 60)
    monthly = load_monthly_stats(args.flags_dir, years)
    eligible = classify_months(monthly, args.exclude_months)
    month_type_map = dict(zip(eligible["month"], eligible["month_type"]))
    eligible_months = set(eligible["month"])

    def decade_of(year):
        return (year // 10) * 10

    strata = [(decade_of(y), mt) for y in years for mt in ("high", "low")]
    strata = sorted(set(strata))

    print(f"[INFO] Strata (decade, month_type): {strata}")

    # ---- Step 2: flagged sample -------------------------------------------------
    print("=" * 60)
    print("  Step 2: Sampling 500 flagged articles")
    print("=" * 60)
    flagged_targets = allocate_equal(strata, args.n_flagged)
    print(f"[INFO] Targets per stratum: {flagged_targets}")

    def flagged_filter(df):
        return df["gep_flag"] & df["month"].isin(eligible_months)

    def strata_key(row):
        return (decade_of(row.year), month_type_map.get(row.month))

    flagged_res = reservoir_sample_by_key(
        args.flags_dir, years, flagged_filter, strata_key, flagged_targets, rng
    )
    flagged_rows = [r for res in flagged_res.values() for r in res]
    for r in flagged_rows:
        r["sample_group"] = "flagged"
    print(f"[OK] Sampled {len(flagged_rows)} flagged articles")

    # ---- Step 3: hard negatives (exactly 1 matched term) -------------------------
    print("=" * 60)
    print("  Step 3: Sampling 100 hard negatives (exactly 1 matched term)")
    print("=" * 60)
    hard_neg_targets = allocate_equal(strata, args.n_hard_neg)

    def hard_neg_filter(df):
        return (~df["gep_flag"]) & (df["n_matched_terms"] == 1) & df["month"].isin(eligible_months)

    hard_neg_res = reservoir_sample_by_key(
        args.flags_dir, years, hard_neg_filter, strata_key, hard_neg_targets, rng
    )
    hard_neg_rows = [r for res in hard_neg_res.values() for r in res]
    hard_neg_ids = {r["article_id"] for r in hard_neg_rows}
    for r in hard_neg_rows:
        r["sample_group"] = "hard_negative"
    print(f"[OK] Sampled {len(hard_neg_rows)} hard negatives")

    # ---- Step 4: unflagged sample, matched to the flagged sample's months -------
    print("=" * 60)
    print("  Step 4: Sampling 500 unflagged articles from the same months")
    print("=" * 60)
    month_targets = dict(Counter(r["month"] for r in flagged_rows))
    # scale/round so total == n_unflagged even if flagged sample size != n_flagged exactly
    scale = args.n_unflagged / max(sum(month_targets.values()), 1)
    scaled = {m: round(c * scale) for m, c in month_targets.items()}
    diff = args.n_unflagged - sum(scaled.values())
    for m in list(scaled)[: abs(diff)]:
        scaled[m] += 1 if diff > 0 else -1
    month_targets = {m: max(c, 0) for m, c in scaled.items()}

    def unflagged_filter(df):
        return (
            (~df["gep_flag"])
            & df["month"].isin(month_targets.keys())
            & (~df["article_id"].isin(hard_neg_ids))
        )

    def month_key(row):
        return row.month

    unflagged_res = reservoir_sample_by_key(
        args.flags_dir, years, unflagged_filter, month_key, month_targets, rng
    )
    unflagged_rows = [r for res in unflagged_res.values() for r in res]
    for r in unflagged_rows:
        r["sample_group"] = "unflagged_same_month"
    print(f"[OK] Sampled {len(unflagged_rows)} unflagged articles")

    # ---- Step 5: assemble sample table ------------------------------------------
    sample = pd.DataFrame(flagged_rows + hard_neg_rows + unflagged_rows)
    sample["decade"] = sample["year"].apply(decade_of)
    sample["month_type"] = sample["month"].map(month_type_map)

    n_dupes = sample.duplicated("article_id").sum()
    if n_dupes:
        print(f"[WARN] {n_dupes} duplicate article_id(s) across groups — dropping duplicates")
        sample = sample.drop_duplicates("article_id")

    print(f"[INFO] Total sample size: {len(sample)}")

    # ---- Step 6: fetch readable short_text from raw_data ------------------------
    print("=" * 60)
    print("  Step 5: Fetching readable text from raw_data")
    print("=" * 60)
    short_texts = fetch_short_texts(sample, args.raw_data_dir, args.snippet_len)
    sample["short_text"] = sample["article_id"].map(short_texts)
    n_missing = sample["short_text"].isna().sum()
    if n_missing:
        print(f"[WARN] {n_missing} article(s) missing raw text")
        sample["short_text"] = sample["short_text"].fillna("[RAW TEXT UNAVAILABLE]")

    # ---- Step 7: export annotation CSV ------------------------------------------
    sample["human_label"] = ""
    sample["notes"] = ""
    sample = sample.sort_values(["sample_group", "date"]).reset_index(drop=True)

    out_cols = [
        "article_id", "date", "headline", "short_text", "gep_flag", "matched_terms",
        "human_label", "notes",
        "sample_group", "decade", "month_type", "n_matched_terms", "topics_hit",
    ]
    sample = sample[out_cols]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sample.to_csv(args.output, index=False)

    print("=" * 60)
    print(f"[OK] Annotation sample written -> {args.output}")
    print(sample["sample_group"].value_counts())
    print("=" * 60)


if __name__ == "__main__":
    main()
