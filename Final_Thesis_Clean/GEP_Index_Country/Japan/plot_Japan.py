"""
Plot GEP Monthly Index – Japan (1996–2025)
Normalized to 100 (mean = 100), line only.
Style inspired by Caldara & Iacoviello (2022), Figure 6.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
JAPAN_DIR = (
    "/Users/catepiacentini/Desktop/tesi/Master_Thesis/Final_Thesis_Clean/"
    "GEP_Index_Country/Japan/"
)

FILE_PATH  = os.path.join(JAPAN_DIR, "GEP_Monthly_JAPAN_min2.csv")
OUTPUT_PNG = os.path.join(JAPAN_DIR, "GEP_Japan_1996_2025.png")

START = "1996-01-01"
END   = "2025-12-31"

# ── EVENTS TO ANNOTATE ──────────────────────────────────────────────────────
EVENTS = [
    ("Asian Crisis",      "1997-07-01",  0.08),
    ("9/11",              "2001-09-01",  0.08),
    ("Iraq War",          "2003-03-01",  0.08),
    ("Senkaku\nTension",  "2010-09-01",  0.10),
    ("N.Korea\nNuclears", "2013-02-01",  0.08),
    ("N.Korea\nMissiles", "2017-08-01",  0.10),
    ("US-China\nTrade",   "2018-07-01",  0.08),
    ("COVID-19",          "2020-03-01",  0.08),
    ("Ukraine\nInvasion", "2022-02-01",  0.10),
    ("Export\nControls",  "2023-07-01",  0.08),
]

# ── LOAD & PREPARE DATA ──────────────────────────────────────────────────────
df = pd.read_csv(FILE_PATH)

df["date"] = pd.to_datetime(df["month"], format="%Y-%m")
df = df.sort_values("date").reset_index(drop=True)
df = df[(df["date"] >= START) & (df["date"] <= END)].copy()

# Normalize to 100 (mean = 100)
df["GEP_norm"] = df["GEP_monthly"] / df["GEP_monthly"].mean() * 100

print(f"Date range: {df['date'].min().date()} – {df['date'].max().date()}")
print(f"Rows: {len(df)}  |  Mean: {df['GEP_norm'].mean():.1f}  |  Max: {df['GEP_norm'].max():.1f}")

# ── PLOT ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

ax.plot(df["date"], df["GEP_norm"], color="#1a3a6b", linewidth=0.9)

y_max = df["GEP_norm"].max()

for label, date_str, y_frac in EVENTS:
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
        fontsize=6.5,
        ha="center",
        va="bottom",
        color="black",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
        annotation_clip=False,
    )

ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))
ax.set_ylim(0, y_max * 1.45)
ax.set_ylabel("GEP Index (mean = 100)", fontsize=9)
ax.set_title("GEP Japan", fontsize=12, fontweight="normal", pad=8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)
ax.tick_params(axis="both", labelsize=8, width=0.7)

ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
ax.axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"PNG saved to: {OUTPUT_PNG}")
plt.show()