"""
Plot GEP Monthly Index – Japan, US, UK (1996–2025)
3 panels side by side, normalized to 100 (mean = 100).
Style inspired by Caldara & Iacoviello (2022), Figure 6.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR   = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/GEP_Index_Country"
OUTPUT_PNG = os.path.join(BASE_DIR, "GEP_Japan_US_UK_1996_2025.png")

START = "1996-01-01"
END   = "2025-12-31"

# ── COUNTRY CONFIG ───────────────────────────────────────────────────────────
# (title, subfolder, filename, events)
COUNTRIES = [
    {
        "title":  "GEP Japan",
        "folder": "Japan",
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
        "title":  "GEP United States",
        "folder": "US",
        "file":   "GEP_Monthly_US_min2.csv",
        "events": [
            ("Asian Crisis",     "1997-07-01", 0.08),
            ("9/11",             "2001-09-01", 0.10),
            ("Iraq War",         "2003-03-01", 0.08),
            ("GFC",              "2008-09-01", 0.08),
            ("N.Korea\nNuclears","2013-02-01", 0.08),
            ("US-China\nTrade",  "2018-07-01", 0.08),
            ("COVID-19",         "2020-03-01", 0.08),
            ("Ukraine\nInvasion","2022-02-01", 0.10),
            ("Export\nControls", "2023-07-01", 0.08),
        ],
    },
    {
        "title":  "GEP United Kingdom",
        "folder": "UK",
        "file":   "GEP_Monthly_UK_min2.csv",
        "events": [
            ("Asian Crisis",     "1997-07-01", 0.08),
            ("9/11",             "2001-09-01", 0.10),
            ("Iraq War",         "2003-03-01", 0.08),
            ("GFC",              "2008-09-01", 0.08),
            ("Brexit\nVote",     "2016-06-01", 0.08),
            ("US-China\nTrade",  "2018-07-01", 0.08),
            ("COVID-19",         "2020-03-01", 0.08),
            ("Ukraine\nInvasion","2022-02-01", 0.10),
        ],
    },
]

# ── PLOT ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4), dpi=150)

for ax, country in zip(axes, COUNTRIES):
    file_path = os.path.join(BASE_DIR, country["folder"], country["file"])
    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["month"], format="%Y-%m")
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    df["GEP_norm"] = df["GEP_monthly"] / df["GEP_monthly"].mean() * 100

    print(f"{country['title']}: max={df['GEP_norm'].max():.1f}, mean={df['GEP_norm'].mean():.1f}")

    ax.plot(df["date"], df["GEP_norm"], color="#2b7bba", linewidth=2.1)

    y_max = df["GEP_norm"].max()

    for label, date_str, y_frac in country["events"]:
        event_date = pd.Timestamp(date_str)
        if event_date < df["date"].min() or event_date > df["date"].max():
            continue

        idx   = (df["date"] - event_date).abs().idxmin()
        y_val = df.loc[idx, "GEP_norm"]
        x_val = df.loc[idx, "date"]
        y_text = y_val + y_frac * y_max

        ax.annotate(
            label,
            xy=(x_val, y_val),
            xytext=(x_val, y_text),
            fontsize=5.5,
            ha="center",
            va="bottom",
            color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
            annotation_clip=False,
        )

    ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))
    ax.set_ylim(0, y_max * 1.45)
    ax.set_title(country["title"], fontsize=11, fontweight="normal", pad=8)
    ax.set_ylabel("GEP Index (mean = 100)", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", labelsize=7, width=0.7)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    ax.axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nPNG saved to: {OUTPUT_PNG}")
plt.show()