#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_robustness.py

Robustness checks for the GEP index:
  1. Min-1, Min-2 (baseline), Min-3, Min-4 normalized plots (monthly + daily + 2025 zoom)
  2. Comparison overlay: all four variants + correlation analysis
  3. GTM v2 index plots (alternative seed words)

DATA layout (relative to this script):
  data/gep/GEP_Monthly_Robust_min2.csv    (baseline)
  data/gep/GEP_Daily_Robust_min2.csv      (baseline)
  data/robustness/GEP_Monthly_Updated.csv  (min-1)
  data/robustness/GEP_Daily_Updated.csv    (min-1)
  data/robustness/GEP_Monthly_Robust_min3.csv
  data/robustness/GEP_Daily_Robust_min3.csv
  data/robustness/GEP_Monthly_Robust_min4.csv
  data/robustness/GEP_Daily_Robust_min4.csv
  data/robustness/GEP_Monthly_Robust_min2_v2.csv  (GTM v2 — optional)
  data/robustness/GEP_Daily_Robust_min2_v2.csv    (GTM v2 — optional)

Outputs saved to output/robustness/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()
REPO   = next((p for p in [HERE, *HERE.parents] if (p / "data" / "gep_us").exists()), HERE.parent)
GEP    = REPO / "data" / "gep_us"
ROBUST = REPO / "data" / "robustness"
OUT    = REPO / "analysis" / "output" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

PEAKS = {
    "1997-07": "Asian Financial Crisis",  "1998-08": "Russian Ruble Crisis",
    "2001-09": "9/11",                    "2003-03": "Iraq War",
    "2008-09": "GFC",                     "2011-08": "US credit downgrade",
    "2014-03": "Crimea annexation",       "2018-06": "US–China tariffs",
    "2019-05": "Trade war escalation",    "2020-03": "COVID-19",
    "2022-02": "Russia invades Ukraine",  "2022-10": "US chip controls on China",
    "2025-04": "Liberation Day tariffs",
}

EVENTS = [
    ("1997-07-02", "1997/07/02: Thai baht devaluation — Asian crisis begins"),
    ("1998-08-17", "1998/08/17: Russia ruble default and debt moratorium"),
    ("1999-11-30", "1999/11/30: WTO Seattle ministerial collapse"),
    ("2001-09-11", "2001/09/11: 9/11 terrorist attacks"),
    ("2001-12-11", "2001/12/11: China joins the WTO"),
    ("2002-03-05", "2002/03/05: Bush imposes 30% steel tariffs (Section 201)"),
    ("2003-03-20", "2003/03/20: US-led invasion of Iraq"),
    ("2006-10-09", "2006/10/09: North Korea first nuclear test"),
    ("2007-08-09", "2007/08/09: BNP Paribas freezes subprime funds — GFC onset"),
    ("2008-09-15", "2008/09/15: Lehman Brothers collapse"),
    ("2009-04-02", "2009/04/02: G20 London Summit — coordinated GFC response"),
    ("2010-05-02", "2010/05/02: Greece €110bn bailout — Eurozone crisis"),
    ("2011-08-05", "2011/08/05: S&P downgrades US credit rating"),
    ("2012-07-26", "2012/07/26: Draghi 'whatever it takes' speech"),
    ("2014-03-18", "2014/03/18: Russia annexes Crimea"),
    ("2014-07-31", "2014/07/31: US and EU expand Russia sanctions"),
    ("2015-07-14", "2015/07/14: Iran nuclear deal signed (JCPOA)"),
    ("2016-06-23", "2016/06/23: Brexit referendum"),
    ("2016-11-08", "2016/11/08: Trump wins US presidential election"),
    ("2018-01-22", "2018/01/22: US tariffs on solar panels and washers"),
    ("2018-03-22", "2018/03/22: Section 301 tariffs on China announced"),
    ("2018-06-15", "2018/06/15: US–China tariffs take effect ($34bn tranche)"),
    ("2018-08-10", "2018/08/10: US doubles tariffs on Turkey"),
    ("2019-05-10", "2019/05/10: US raises tariffs on $200bn of Chinese goods to 25%"),
    ("2019-05-16", "2019/05/16: Huawei added to US entity list"),
    ("2019-08-23", "2019/08/23: China announces $75bn in retaliatory tariffs"),
    ("2020-01-15", "2020/01/15: US–China Phase 1 trade deal signed"),
    ("2020-03-11", "2020/03/11: WHO declares COVID-19 pandemic"),
    ("2021-03-23", "2021/03/23: Suez Canal blocked — Ever Given"),
    ("2022-02-24", "2022/02/24: Russia launches full-scale invasion of Ukraine"),
    ("2022-03-26", "2022/03/26: Russia demands gas payments in rubles"),
    ("2022-10-07", "2022/10/07: US advanced chip export controls on China"),
    ("2023-08-09", "2023/08/09: Biden EO on outbound China investment"),
    ("2024-05-14", "2024/05/14: Biden raises tariffs on Chinese EVs to 100%"),
    ("2025-01-20", "2025/01/20: Trump returns to office — tariff agenda begins"),
    ("2025-04-02", "2025/04/02: 'Liberation Day' — Trump sweeping global tariffs"),
]

TARIFF_2025 = [
    ("2025-01-19", "US threatens tariffs on EU over Greenland"),
    ("2025-01-20", "Trump inauguration"),
    ("2025-02-01", "25% tariffs on Canada & Mexico announced"),
    ("2025-02-10", "Section 232 steel & aluminum reinstated"),
    ("2025-03-04", "Canada/Mexico tariffs take effect"),
    ("2025-04-02", "Liberation Day tariffs"),
    ("2025-04-09", "90-day pause; China raised to 145%"),
    ("2025-05-12", "US-China Geneva truce; tariffs cut to 30%"),
    ("2025-05-23", "Trump threatens 50% tariffs on EU"),
    ("2025-06-05", "US/Israel strike Iran nuclear facilities (Oil spike)"),
    ("2025-07-09", "Liberation Day pause expires"),
    ("2025-08-07", "10-41% broad US tariffs take effect globally"),
    ("2025-09-25", "US introduces new pharma tariffs"),
    ("2025-10-09", "China export controls: rare earths, lithium, graphite"),
    ("2025-10-10", "Trump retaliatory tariffs on China"),
    ("2025-10-23", "US sanctions Rosneft & Lukoil"),
    ("2025-10-25", "US–China de-escalation talks, Malaysia"),
    ("2025-11-03", "Supreme Court upholds Trump tariffs"),
    ("2025-12-12", "US–UK zero-tariff pharma deal"),
]


def find_nearby_peak(daily_obs, date_str, window_days=10):
    dt = pd.to_datetime(date_str)
    sub = daily_obs[(daily_obs["date"] >= dt - pd.Timedelta(days=window_days)) &
                    (daily_obs["date"] <= dt + pd.Timedelta(days=window_days))]
    if sub.empty: return dt, 0.0
    idx = sub["gep_norm"].idxmax()
    return sub.loc[idx, "date"], sub.loc[idx, "gep_norm"]


def spread_label_dates(event_peaks, min_gap_days=38):
    items = sorted(event_peaks, key=lambda r: r[0])
    placed, result = [], []
    for peak_date, peak_score, label in items:
        label_date, changed = peak_date, True
        while changed:
            changed = False
            for placed_date, *_ in placed:
                if 0 <= (label_date - placed_date).days < min_gap_days:
                    label_date = placed_date + pd.Timedelta(days=min_gap_days)
                    changed = True
        placed.append((label_date, peak_score, label))
        result.append((peak_date, peak_score, label, label_date))
    return result


def plot_version(label, monthly_path, daily_path, out_prefix):
    """Three standard plots for one robustness variant."""
    monthly = pd.read_csv(monthly_path)
    monthly["month"] = pd.to_datetime(monthly["month"])

    daily = pd.read_csv(daily_path)
    daily["date"] = pd.to_datetime(daily["date"])

    daily_obs = daily[daily["n_articles"] > 0].copy()
    daily_mean = daily_obs["GEP_daily"].mean()
    monthly_mean = monthly["GEP_monthly"].mean()
    daily_obs["gep_norm"] = daily_obs["GEP_daily"] / daily_mean * 100
    monthly["gep_norm"]   = monthly["GEP_monthly"] / monthly_mean * 100

    # Monthly plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(monthly["month"], monthly["gep_norm"], color="#378ADD", linewidth=0.9, alpha=0.9)
    ax.fill_between(monthly["month"], monthly["gep_norm"], alpha=0.15, color="#378ADD")
    ax.axhline(100, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    for month_str, lbl in PEAKS.items():
        row = monthly[monthly["month"].dt.strftime("%Y-%m") == month_str]
        if not row.empty:
            ax.annotate(lbl, xy=(row["month"].values[0], row["gep_norm"].values[0]),
                        xytext=(0, 14), textcoords="offset points", fontsize=7.5, ha="center",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.7), color="#333333")
    ax.set_title(f"GEP Monthly Index — {label}, normalized to 100 (1996–2025)", fontsize=13, pad=12)
    ax.set_ylabel("GEP Index (avg = 100)", fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / f"{out_prefix}_Monthly_norm100.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_prefix}_Monthly_norm100.png"); plt.close()

    # Daily horizontal dot plot
    raw_peaks = [(find_nearby_peak(daily_obs, d)[0], find_nearby_peak(daily_obs, d)[1], lbl)
                 for d, lbl in EVENTS]
    spread    = spread_label_dates(raw_peaks)
    X_DATA_MAX = daily_obs["gep_norm"].quantile(0.999)
    X_LABEL    = X_DATA_MAX * 1.25; X_MAX = X_DATA_MAX * 3.8

    fig, ax = plt.subplots(figsize=(16, 28))
    ax.scatter(daily_obs["gep_norm"], daily_obs["date"],
               s=28, color="#27AE60", alpha=0.30, linewidths=0, zorder=2)
    ax.plot(monthly["gep_norm"], monthly["month"],
            color="#152F5F", linewidth=1.6, alpha=0.95, zorder=3)
    ax.axvline(100, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    for peak_date, peak_score, lbl, label_date in spread:
        ax.scatter(peak_score, peak_date, s=25, color="#C0392B", zorder=5, linewidths=0)
        ax.annotate(lbl, xy=(peak_score, peak_date), xytext=(X_LABEL, label_date),
                    fontsize=13, ha="left", va="center", color="#1A1A1A",
                    arrowprops=dict(arrowstyle="->", color="#777777", lw=0.65,
                                    connectionstyle="arc3,rad=0.0"), annotation_clip=False)
    ax.set_ylim(pd.Timestamp("2026-03-01"), pd.Timestamp("1995-10-01"))
    ax.yaxis.set_major_locator(mdates.YearLocator(1))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(0, X_MAX)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlabel("GEP Index (avg = 100)", fontsize=11)
    ax.set_title(f"Daily GEP Index — {label}, normalized to 100 (1996–2025)", fontsize=13, pad=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / f"{out_prefix}_Daily_norm100.png", dpi=160, bbox_inches="tight")
    print(f"Saved: {out_prefix}_Daily_norm100.png"); plt.close()

    # 2025 zoom
    daily_2025 = daily_obs[(daily_obs["date"] >= "2025-01-01") &
                           (daily_obs["date"] <= "2025-12-31")].copy()
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.scatter(daily_2025["date"], daily_2025["gep_norm"],
               s=22, color="#27AE60", alpha=0.45, linewidths=0, zorder=2)
    if not daily_2025.empty:
        roll = daily_2025.set_index("date")["gep_norm"].rolling("7D").mean()
        ax.plot(roll.index, roll.values, color="#1A6B3C", linewidth=1.8, alpha=0.85, zorder=3,
                label="7-day rolling avg")
    ax.axhline(100, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    y_fracs = [0.97, 0.88] * 10
    for i, (date_str, lbl) in enumerate(TARIFF_2025):
        dt = pd.to_datetime(date_str)
        ax.axvline(dt, color="#C0392B", linewidth=0.9, linestyle="--", alpha=0.5, zorder=1)
        ax.text(dt, y_fracs[i % len(y_fracs)], lbl,
                transform=ax.get_xaxis_transform(), fontsize=7.5, ha="center", va="top",
                color="#8B0000", rotation=90,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("GEP Index (avg = 100)", fontsize=10)
    ax.set_title(f"Daily GEP Index — {label} — 2025 Zoom", fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.75, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / f"{out_prefix}_2025_Zoom_norm100.png", dpi=160, bbox_inches="tight")
    print(f"Saved: {out_prefix}_2025_Zoom_norm100.png"); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Run all variant plots
# ═════════════════════════════════════════════════════════════════════════════
VERSIONS = [
    ("Robust min-1", ROBUST / "GEP_Monthly_min1.csv",          ROBUST / "GEP_Daily_min1.csv",          "GEP_min1"),
    ("Robust min-2", GEP    / "GEP_Monthly_Robust_min2.csv",  GEP    / "GEP_Daily_Robust_min2.csv",  "GEP_min2"),
    ("Robust min-3", ROBUST / "GEP_Monthly_Robust_min3.csv",  ROBUST / "GEP_Daily_Robust_min3.csv",  "GEP_min3"),
    ("Robust min-4", ROBUST / "GEP_Monthly_Robust_min4.csv",  ROBUST / "GEP_Daily_Robust_min4.csv",  "GEP_min4"),
]

for label, m_path, d_path, prefix in VERSIONS:
    if m_path.exists() and d_path.exists():
        print(f"\n--- {label} ---")
        plot_version(label, m_path, d_path, prefix)
    else:
        print(f"[WARNING] Files missing for {label}: {m_path.name}, {d_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Comparison overlay: all variants normalized to 100
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Robustness comparison overlay ---")

def load_monthly(path, col="GEP_monthly"):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    return df[["month", col]].set_index("month")[col]

series = {}
for label, m_path, d_path, prefix in VERSIONS:
    if m_path.exists():
        s = load_monthly(m_path)
        series[label] = s / s.mean() * 100

if series:
    COLORS = {
        "Robust min-2": "#152F5F",
        "Robust min-1": "#E74C3C",
        "Robust min-3": "#27AE60",
        "Robust min-4": "#E67E22",
    }
    panel = pd.concat(series, axis=1).dropna()

    # Correlation table
    print("\nCorrelation table (monthly, normalized):")
    print(panel.corr().to_string())

    # Overlay plot
    fig, ax = plt.subplots(figsize=(16, 6))
    for name, s in panel.items():
        ax.plot(s.index, s.values, color=COLORS.get(name, "gray"), linewidth=1.4,
                alpha=0.85, label=name)
    ax.axhline(100, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_title("GEP Robustness Check — All Variants Normalized to 100 (1996–2025)", fontsize=13, pad=10)
    ax.set_ylabel("GEP Index (avg = 100)", fontsize=10)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "GEP_Robustness_Overlay_Monthly.png", dpi=150, bbox_inches="tight")
    print("Saved: GEP_Robustness_Overlay_Monthly.png"); plt.close()

    # Rolling 24-month correlation with baseline
    fig, ax = plt.subplots(figsize=(16, 5))
    baseline = panel.get("Robust min-2")
    if baseline is not None:
        for name, s in panel.items():
            if name == "Robust min-2": continue
            roll = baseline.rolling(24, min_periods=18).corr(s)
            ax.plot(roll.index, roll.values, color=COLORS.get(name, "gray"),
                    linewidth=1.2, label=name)
        ax.axhline(1, color="gray", lw=0.6, ls="--", alpha=0.6)
        ax.set_title("GEP Robustness — Rolling 24-month Correlation with Baseline (min-2)",
                     fontsize=13, pad=10)
        ax.set_ylabel("Pearson r", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10, framealpha=0.8)
        ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(OUT / "GEP_Robustness_Correlations_Rolling.png", dpi=150, bbox_inches="tight")
        print("Saved: GEP_Robustness_Correlations_Rolling.png"); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# GTM v2 plots (alternative seed words — robustness check)
# ═════════════════════════════════════════════════════════════════════════════
v2_monthly = ROBUST / "GEP_Monthly_gtm_v2.csv"
v2_daily   = ROBUST / "GEP_Daily_gtm_v2.csv"

if v2_monthly.exists() and v2_daily.exists():
    print("\n--- GTM v2 (alternative seeds) ---")
    plot_version("GTM v2 (alt. seeds, min-2)", v2_monthly, v2_daily, "GEP_GTM_v2")
else:
    print(f"\n[INFO] GTM v2 files not found ({v2_monthly.name}). Place them in data/robustness/ to generate these plots.")

print("\n═══ All robustness plots saved to output/robustness/ ═══")
