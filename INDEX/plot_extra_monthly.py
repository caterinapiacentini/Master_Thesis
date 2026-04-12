#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_extra_monthly.py

Plots monthly GEP index for:
  1. Robust version (min2)  → index_new_final/GEP_Monthly_Robust_min2.png
  2. Old-def new-GTM        → index_old_def_new_gtm/GEP_Monthly_Index.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = os.path.dirname(os.path.abspath(__file__))

PEAKS = {
    "1997-07": "Asian Financial Crisis",
    "1998-08": "Russian Ruble Crisis",
    "2001-09": "9/11",
    "2003-03": "Iraq War",
    "2008-09": "GFC",
    "2011-08": "US credit downgrade",
    "2014-03": "Crimea annexation",
    "2018-06": "US–China tariffs",
    "2019-05": "Trade war escalation",
    "2020-03": "COVID-19",
    "2022-02": "Russia invades Ukraine",
    "2022-10": "US chip controls on China",
    "2025-04": "Liberation Day tariffs",
}


def plot_monthly(csv_path, out_path, title):
    df = pd.read_csv(csv_path)
    df["month"] = pd.to_datetime(df["month"])
    df["gep_pct"] = df["GEP_monthly"] * 100

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(df["month"], df["gep_pct"],
            color="#378ADD", linewidth=0.9, alpha=0.9)
    ax.fill_between(df["month"], df["gep_pct"],
                    alpha=0.15, color="#378ADD")

    for month_str, label in PEAKS.items():
        row = df[df["month"].dt.strftime("%Y-%m") == month_str]
        if not row.empty:
            x = row["month"].values[0]
            y = row["gep_pct"].values[0]
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(0, 14),
                textcoords="offset points",
                fontsize=7.5,
                ha="center",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.7),
                color="#333333",
            )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Share of articles mentioning GEP (%)", fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ── 1. Robust (min2) ──────────────────────────────────────────────────────────
plot_monthly(
    csv_path=os.path.join(BASE, "index_new_final", "MIN2", "GEP_Monthly_Robust_min2.csv"),
    out_path=os.path.join(BASE, "index_new_final", "MIN2", "GEP_Monthly_Robust_min2.png"),
    title="GEP Monthly Index — Robust min-2 (1996–2025)",
)

# ── 2. Old-def new-GTM ────────────────────────────────────────────────────────
plot_monthly(
    csv_path=os.path.join(BASE, "index_old_def_new_gtm", "GEP_Monthly_Index.csv"),
    out_path=os.path.join(BASE, "index_old_def_new_gtm", "GEP_Monthly_Index.png"),
    title="GEP Monthly Index — Old Definition, New GTM (1996–2025)",
)
