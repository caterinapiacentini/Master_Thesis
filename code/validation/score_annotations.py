#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_annotations.py

Scores the human-annotated GEP validation sample (data/validation/annotation_sample.csv,
with the `human_label` column filled in) against the reconstructed `gep_flag`.

Main confusion matrix / precision / recall / F1 (+ bootstrap 95% CIs) are computed
on the "flagged" + "unflagged_same_month" rows only — the representative, decade x
high/low-month-balanced core sample. The 100 "hard_negative" rows (deliberately
oversampled near-misses: exactly 1 matched dictionary term) are NOT part of this
denominator, since they are not a random draw and would bias FN counts upward;
they're reported separately as a boundary diagnostic (how often a 1-topic near-miss
is actually GEP-relevant).

Outputs:
    - results/metrics.json           point estimates + 95% CIs + hard-negative rate
    - results/metrics_table.tex       LaTeX (booktabs) summary table
    - results/false_positives.csv     gep_flag=True, human_label=False (main sample)
    - results/false_negatives.csv     gep_flag=False, human_label=True (main sample
                                       + hard-negative probes, flagged by `source`)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

TRUE_STRINGS = {"1", "true", "yes", "y", "t"}
FALSE_STRINGS = {"0", "false", "no", "n", "f"}


def normalize_label(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and not pd.isna(x):
        return bool(x)
    s = str(x).strip().lower()
    if s in TRUE_STRINGS:
        return True
    if s in FALSE_STRINGS:
        return False
    raise ValueError(f"Unrecognized human_label value: {x!r}")


def confusion_counts(gep_flag: pd.Series, human_label: pd.Series) -> dict:
    tp = int(( gep_flag &  human_label).sum())
    fp = int(( gep_flag & ~human_label).sum())
    fn = int((~gep_flag &  human_label).sum())
    tn = int((~gep_flag & ~human_label).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def precision_recall_f1(counts: dict) -> dict:
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
        else float("nan")
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def bootstrap_ci(gep_flag: np.ndarray, human_label: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(gep_flag)
    boot_p, boot_r, boot_f1 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c = confusion_counts(pd.Series(gep_flag[idx]), pd.Series(human_label[idx]))
        m = precision_recall_f1(c)
        boot_p.append(m["precision"])
        boot_r.append(m["recall"])
        boot_f1.append(m["f1"])

    def ci(vals):
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) == 0:
            return (float("nan"), float("nan"))
        return tuple(np.percentile(vals, [2.5, 97.5]))

    return {
        "precision_ci95": ci(boot_p),
        "recall_ci95": ci(boot_r),
        "f1_ci95": ci(boot_f1),
    }


def to_latex_table(metrics: dict, counts: dict, hard_neg: dict, path: str):
    def fmt_ci(point, ci):
        return f"{point:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}]"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Validation of the reconstructed GEP flag against human annotation}",
        r"\label{tab:gep_validation}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Estimate & 95\% CI \\",
        r"\midrule",
        f"Precision & {fmt_ci(metrics['precision'], metrics['precision_ci95'])} \\\\",
        f"Recall & {fmt_ci(metrics['recall'], metrics['recall_ci95'])} \\\\",
        f"F1 & {fmt_ci(metrics['f1'], metrics['f1_ci95'])} \\\\",
        r"\midrule",
        f"TP & {counts['TP']} & \\\\",
        f"FP & {counts['FP']} & \\\\",
        f"FN & {counts['FN']} & \\\\",
        f"TN & {counts['TN']} & \\\\",
        r"\midrule",
        (
            f"Hard-negative near-miss rate & {hard_neg['rate']:.3f} & "
            f"[{hard_neg['ci95'][0]:.3f}, {hard_neg['ci95'][1]:.3f}] \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Score human-annotated GEP validation sample.")
    parser.add_argument(
        "--annotated_csv", type=str,
        default=os.path.expanduser("~/Master_Thesis/data/validation/annotation_sample.csv"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.expanduser("~/Master_Thesis/data/validation/results"),
    )
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.annotated_csv, dtype={"article_id": str})
    n_missing = df["human_label"].isna().sum() + (df["human_label"].astype(str).str.strip() == "").sum()
    if n_missing:
        print(f"[WARN] {n_missing} row(s) missing human_label — dropping from scoring")
        df = df[df["human_label"].notna() & (df["human_label"].astype(str).str.strip() != "")]

    df["gep_flag"] = df["gep_flag"].apply(
        lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true"
    )
    df["human_label_bool"] = df["human_label"].apply(normalize_label)

    main_df = df[df["sample_group"].isin(["flagged", "unflagged_same_month"])].copy()
    hard_neg_df = df[df["sample_group"] == "hard_negative"].copy()

    print(f"[INFO] Main evaluation sample : {len(main_df)} rows")
    print(f"[INFO] Hard-negative probes   : {len(hard_neg_df)} rows")

    # ---- main confusion matrix / precision / recall / F1 -----------------------
    counts = confusion_counts(main_df["gep_flag"], main_df["human_label_bool"])
    point = precision_recall_f1(counts)
    ci = bootstrap_ci(
        main_df["gep_flag"].to_numpy(),
        main_df["human_label_bool"].to_numpy(),
        args.n_bootstrap,
        args.seed,
    )
    metrics = {**counts, **point, **ci}

    # ---- hard-negative near-miss diagnostic -------------------------------------
    if len(hard_neg_df) > 0:
        hn_labels = hard_neg_df["human_label_bool"].to_numpy()
        hn_rate = float(hn_labels.mean())
        rng = np.random.default_rng(args.seed)
        boot_rates = [
            rng.choice(hn_labels, size=len(hn_labels), replace=True).mean()
            for _ in range(args.n_bootstrap)
        ]
        hn_ci = tuple(np.percentile(boot_rates, [2.5, 97.5]))
    else:
        hn_rate, hn_ci = float("nan"), (float("nan"), float("nan"))
    hard_neg_summary = {"n": len(hard_neg_df), "rate": hn_rate, "ci95": hn_ci}

    # ---- print summary ------------------------------------------------------
    print("=" * 60)
    print("  Confusion matrix (main sample)")
    print("=" * 60)
    print(counts)
    print(f"Precision: {point['precision']:.4f}  CI95: {ci['precision_ci95']}")
    print(f"Recall   : {point['recall']:.4f}  CI95: {ci['recall_ci95']}")
    print(f"F1       : {point['f1']:.4f}  CI95: {ci['f1_ci95']}")
    print("-" * 60)
    print(f"Hard-negative near-miss rate (share of exactly-1-topic articles a human "
          f"still calls GEP-relevant): {hn_rate:.4f}  CI95: {hn_ci}  (n={len(hard_neg_df)})")

    # ---- export metrics.json -------------------------------------------------
    out_json = {
        "counts": counts,
        "precision": point["precision"],
        "recall": point["recall"],
        "f1": point["f1"],
        "precision_ci95": list(ci["precision_ci95"]),
        "recall_ci95": list(ci["recall_ci95"]),
        "f1_ci95": list(ci["f1_ci95"]),
        "n_bootstrap": args.n_bootstrap,
        "hard_negative": {
            "n": hard_neg_summary["n"],
            "near_miss_rate": hard_neg_summary["rate"],
            "ci95": list(hard_neg_summary["ci95"]),
        },
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    # ---- export LaTeX table --------------------------------------------------
    to_latex_table(
        {**point, **ci}, counts, hard_neg_summary,
        os.path.join(args.output_dir, "metrics_table.tex"),
    )

    # ---- export FP / FN CSVs -------------------------------------------------
    fp = main_df[main_df["gep_flag"] & ~main_df["human_label_bool"]].copy()
    fp.to_csv(os.path.join(args.output_dir, "false_positives.csv"), index=False)

    fn_main = main_df[~main_df["gep_flag"] & main_df["human_label_bool"]].copy()
    fn_main["source"] = "main_sample"
    fn_hard = hard_neg_df[hard_neg_df["human_label_bool"]].copy()
    fn_hard["source"] = "hard_negative_probe"
    fn = pd.concat([fn_main, fn_hard], ignore_index=True)
    fn.to_csv(os.path.join(args.output_dir, "false_negatives.csv"), index=False)

    print("=" * 60)
    print(f"[OK] Wrote metrics.json, metrics_table.tex, "
          f"false_positives.csv ({len(fp)} rows), false_negatives.csv ({len(fn)} rows)")
    print(f"     -> {args.output_dir}")


if __name__ == "__main__":
    main()
