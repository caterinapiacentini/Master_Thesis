#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_countries.py

GEP country indices — 3x2 panel (Japan, UK, Germany, Russia, Iran, China)
+ individual country plots.

DATA layout (relative to this script):
  data/countries/GEP_Monthly_JAPAN_min2.csv
  data/countries/GEP_Monthly_UK_min2.csv
  data/countries/GEP_Monthly_GERMANY_min2.csv
  data/countries/GEP_Monthly_RUSSIA_min2.csv
  data/countries/GEP_Monthly_IRAN_min2.csv
  data/countries/GEP_Monthly_CHINA_min2.csv

Outputs saved to output/countries/
  GEP_6Countries_1996_2025.png
  GEP_Japan_1996_2025.png
  GEP_UK_1996_2025.png
  GEP_Germany_1996_2025.png
  GEP_Russia_1996_2025.png
  GEP_Iran_1996_2025.png
  GEP_China_1996_2025.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pathlib import Path

HERE = Path(__file__).parent
CTRY = HERE / "data" / "countries"
OUT  = HERE / "output" / "countries"
OUT.mkdir(parents=True, exist_ok=True)

START = "1996-01-01"
END   = "2025-12-31"

COUNTRIES = [
    {
        "title":  "GEP Japan",
        "file":   "GEP_Monthly_JAPAN_min2.csv",
        "events": [
            ("Asian Crisis",      "1997-07-01", 0.08),
            ("9/11",              "2001-09-01", 0.08),
            ("Iraq War",          "2003-03-01", 0.08),
            ("Senkaku\nTension",  "2010-09-01", 0.10),
            ("N.Korea\nNuclears", "2013-02-01", 0.08),
            ("N.Korea\nMissiles", "2017-08-01", 0.10),
            ("US-China\nTrade",   "2018-07-01", 0.08),
            ("COVID-19",          "2020-03-01", 0.08),
            ("Ukraine\nInvasion", "2022-02-01", 0.10),
            ("Export\nControls",  "2023-07-01", 0.08),
        ],
    },
    {
        "title":  "GEP United Kingdom",
        "file":   "GEP_Monthly_UK_min2.csv",
        "events": [
            ("Asian Crisis",      "1997-07-01", 0.08),
            ("9/11",              "2001-09-01", 0.10),
            ("Iraq War",          "2003-03-01", 0.08),
            ("GFC",               "2008-09-01", 0.08),
            ("Brexit\nVote",      "2016-06-01", 0.08),
            ("US-China\nTrade",   "2018-07-01", 0.08),
            ("COVID-19",          "2020-03-01", 0.08),
            ("Ukraine\nInvasion", "2022-02-01", 0.10),
        ],
    },
    {
        "title":  "GEP Germany",
        "file":   "GEP_Monthly_GERMANY_min2.csv",
        "events": [
            ("Asian Crisis",      "1997-07-01", 0.08),
            ("9/11",              "2001-09-01", 0.10),
            ("Iraq War",          "2003-03-01", 0.08),
            ("GFC",               "2008-09-01", 0.08),
            ("Eurozone\nCrisis",  "2010-05-01", 0.10),
            ("Brexit\nVote",      "2016-06-01", 0.08),
            ("US-China\nTrade",   "2018-07-01", 0.08),
            ("COVID-19",          "2020-03-01", 0.08),
            ("Ukraine\nInvasion", "2022-02-01", 0.10),
        ],
    },
    {
        "title":  "GEP Russia",
        "file":   "GEP_Monthly_RUSSIA_min2.csv",
        "events": [
            ("Kosovo\nNATO",       "1999-03-01", 0.08),
            ("9/11",               "2001-09-01", 0.08),
            ("Iraq War",           "2003-03-01", 0.08),
            ("Georgia\nWar",       "2008-08-01", 0.10),
            ("Crimea\nAnnexation", "2014-03-01", 0.10),
            ("Syria\nIntervention","2015-09-01", 0.08),
            ("US-China\nTrade",    "2018-07-01", 0.08),
            ("COVID-19",           "2020-03-01", 0.08),
            ("Ukraine\nInvasion",  "2022-02-01", 0.12),
        ],
    },
    {
        "title":  "GEP Iran",
        "file":   "GEP_Monthly_IRAN_min2.csv",
        "events": [
            ("9/11",              "2001-09-01", 0.08),
            ("Iraq War",          "2003-03-01", 0.08),
            ("Nuclear\nCrisis",   "2006-01-01", 0.08),
            ("JCPOA\nSigning",    "2015-07-01", 0.08),
            ("JCPOA\nWithdrawal", "2018-05-01", 0.10),
            ("Soleimani",         "2020-01-01", 0.10),
            ("COVID-19",          "2020-03-01", 0.08),
            ("Ukraine\nInvasion", "2022-02-01", 0.08),
        ],
    },
    {
        "title":  "GEP China",
        "file":   "GEP_Monthly_CHINA_min2.csv",
        "events": [
            ("Asian Crisis",      "1997-07-01", 0.08),
            ("9/11",              "2001-09-01", 0.08),
            ("Iraq War",          "2003-03-01", 0.08),
            ("GFC",               "2008-09-01", 0.08),
            ("S.China Sea",       "2012-07-01", 0.08),
            ("US-China\nTrade",   "2018-07-01", 0.10),
            ("COVID-19",          "2020-03-01", 0.08),
            ("Ukraine\nInvasion", "2022-02-01", 0.08),
            ("Export\nControls",  "2023-07-01", 0.08),
        ],
    },
]


def load_and_normalize(file_path):
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["month"], format="%Y-%m")
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    df["GEP_norm"] = df["GEP_monthly"] / df["GEP_monthly"].mean() * 100
    return df


def annotate_events(ax, df, events, y_max):
    for label, date_str, y_frac in events:
        event_date = pd.Timestamp(date_str)
        if event_date < df["date"].min() or event_date > df["date"].max():
            continue
        idx   = (df["date"] - event_date).abs().idxmin()
        y_val = df.loc[idx, "GEP_norm"]
        x_val = df.loc[idx, "date"]
        y_text = y_val + y_frac * y_max
        ax.annotate(
            label, xy=(x_val, y_val), xytext=(x_val, y_text),
            fontsize=5.5, ha="center", va="bottom", color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
            annotation_clip=False,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — 3×2 panel: all 6 countries
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=150)

for ax, country in zip(axes.flatten(), COUNTRIES):
    path = CTRY / country["file"]
    if not path.exists():
        print(f"[WARNING] Missing: {path.name}"); ax.set_visible(False); continue

    df    = load_and_normalize(path)
    y_max = df["GEP_norm"].max()
    print(f"{country['title']}: max={y_max:.1f}, mean={df['GEP_norm'].mean():.1f}")

    ax.plot(df["date"], df["GEP_norm"], color="#2b7bba", linewidth=2.1)
    annotate_events(ax, df, country["events"], y_max)

    ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))
    ax.set_ylim(0, y_max * 1.45)
    ax.set_title(country["title"], fontsize=11, fontweight="normal", pad=8)
    ax.set_ylabel("GEP Index (mean = 100)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", labelsize=7, width=0.7)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    ax.axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig(OUT / "GEP_6Countries_1996_2025.png", dpi=150, bbox_inches="tight")
print("Saved: GEP_6Countries_1996_2025.png")
plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Individual country plots (larger, more readable)
# ═════════════════════════════════════════════════════════════════════════════
for country in COUNTRIES:
    path = CTRY / country["file"]
    if not path.exists():
        print(f"[WARNING] Missing: {path.name}"); continue

    df    = load_and_normalize(path)
    y_max = df["GEP_norm"].max()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["GEP_norm"], color="#2b7bba", linewidth=2.0)
    ax.fill_between(df["date"], df["GEP_norm"], alpha=0.15, color="#2b7bba")
    ax.axhline(100, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    annotate_events(ax, df, country["events"], y_max)

    ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))
    ax.set_title(f"{country['title']} — Monthly GEP (normalized to 100, 1996–2025)",
                 fontsize=13, pad=10)
    ax.set_ylabel("GEP Index (mean = 100)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    short_name = country["file"].split("_")[2]
    plt.tight_layout()
    plt.savefig(OUT / f"GEP_{short_name}_1996_2025.png", dpi=150, bbox_inches="tight")
    print(f"Saved: GEP_{short_name}_1996_2025.png")
    plt.close()

print("\n═══ All country plots saved to output/countries/ ═══")
