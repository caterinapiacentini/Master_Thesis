#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gep_vs_sp500_robust_min2.py

GEP Robust min-2 index vs S&P 500 — three comparison plots:
  1. gep_vs_sp500_zscore.png         — monthly Z-score overlay
  2. gep_vs_sp500_logret.png         — two-panel: GEP monthly + S&P log returns
  3. gep_vs_sp500_logret_daily.png   — two-panel: GEP daily  + S&P log returns
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load GEP data ──────────────────────────────────────────────────────────────
monthly = pd.read_csv(os.path.join(BASE, "GEP_Monthly_Robust_min2.csv"))
monthly["month"] = pd.to_datetime(monthly["month"])
monthly = monthly.set_index("month").sort_index()

daily = pd.read_csv(os.path.join(BASE, "GEP_Daily_Robust_min2.csv"))
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.set_index("date").sort_index()
daily = daily[daily["n_articles"] > 0]   # trading days only

# ── Download S&P 500 ───────────────────────────────────────────────────────────
sp500_mo_raw = yf.download("^GSPC", start="1995-12-01", end="2025-12-31",
                            interval="1mo", auto_adjust=True, progress=False)
sp500_mo = sp500_mo_raw[["Close"]].copy()
sp500_mo.index = sp500_mo.index.to_period("M").to_timestamp()
sp500_mo.columns = ["sp500"]
sp500_mo = sp500_mo.sort_index()
sp500_mo["log_ret"] = np.log(sp500_mo["sp500"] / sp500_mo["sp500"].shift(1))
sp500_mo = sp500_mo.dropna()

sp500_d_raw = yf.download("^GSPC", start="1995-12-01", end="2025-12-31",
                           interval="1d", auto_adjust=True, progress=False)
sp500_d = sp500_d_raw[["Close"]].copy()
sp500_d.index = pd.to_datetime(sp500_d.index)
sp500_d.columns = ["sp500"]
sp500_d = sp500_d.sort_index()
sp500_d["log_ret"] = np.log(sp500_d["sp500"] / sp500_d["sp500"].shift(1))
sp500_d = sp500_d.dropna()

# ── Merge ──────────────────────────────────────────────────────────────────────
df_mo = monthly[["GEP_monthly"]].join(sp500_mo[["sp500", "log_ret"]], how="inner")
df_mo["gep_pct"]     = df_mo["GEP_monthly"] * 100
df_mo["cum_log_ret"] = df_mo["log_ret"].cumsum()

df_d = daily[["GEP_daily"]].join(sp500_d[["log_ret"]], how="inner")
df_d["gep_pct"]     = df_d["GEP_daily"] * 100
df_d["cum_log_ret"] = df_d["log_ret"].cumsum()

# ── Events ─────────────────────────────────────────────────────────────────────
events_mo = {
    "1997-07": "Asian Crisis",
    "2001-09": "9/11",
    "2008-09": "GFC",
    "2020-03": "COVID-19",
    "2022-02": "Ukraine war",
    "2025-04": "Liberation Day",
}
events_d = {
    "1997-07-02": "Asian\nCrisis",
    "2001-09-11": "9/11",
    "2008-09-15": "GFC",
    "2020-03-16": "COVID-19",
    "2022-02-24": "Ukraine\nWar",
    "2025-04-02": "Liberation\nDay",
}


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Monthly Z-score overlay
# ══════════════════════════════════════════════════════════════════════════════
df_z = df_mo.copy()
df_z["gep_z"]   = (df_z["GEP_monthly"] - df_z["GEP_monthly"].mean()) / df_z["GEP_monthly"].std()
df_z["sp500_z"] = (df_z["sp500"]       - df_z["sp500"].mean())       / df_z["sp500"].std()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df_z.index, df_z["gep_z"],   color="#378ADD", linewidth=1.1,
        label="GEP Robust min-2 (Z-score)", zorder=3)
ax.fill_between(df_z.index, df_z["gep_z"], alpha=0.12, color="#378ADD")
ax.plot(df_z.index, df_z["sp500_z"], color="#E74C3C", linewidth=1.1,
        label="S&P 500 (Z-score)", zorder=3)
ax.fill_between(df_z.index, df_z["sp500_z"], alpha=0.08, color="#E74C3C")
ax.axhline(0, color="#555555", linewidth=0.6, linestyle="--")

for month_str, label in events_mo.items():
    ts = pd.Timestamp(month_str)
    if ts in df_z.index:
        y_val = df_z.loc[ts, "gep_z"]
        ax.annotate(label, xy=(ts, y_val),
                    xytext=(0, 14), textcoords="offset points",
                    fontsize=7.5, ha="center", color="#333333",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))

ax.set_title("GEP Robust min-2 vs S&P 500 — Monthly Z-scores (1996–2025)",
             fontsize=13, pad=12)
ax.set_ylabel("Standard deviations from mean", fontsize=10)
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
out = os.path.join(BASE, "gep_vs_sp500_zscore.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Two-panel: GEP monthly level + S&P monthly log returns
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08})

ax1.fill_between(df_mo.index, df_mo["gep_pct"], alpha=0.30, color="#378ADD")
ax1.plot(df_mo.index, df_mo["gep_pct"], color="#1A5FA8", linewidth=1.1,
         label="GEP Robust min-2 (monthly, %)")
ax1.set_ylabel("Share of articles (%)", fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.legend(fontsize=10, framealpha=0.7, loc="upper left")
ax1.set_title("GEP Robust min-2 vs S&P 500 Monthly Log Returns (1996–2025)",
              fontsize=13, pad=10)

colors = ["#C0392B" if r < 0 else "#27AE60" for r in df_mo["log_ret"]]
ax2.bar(df_mo.index, df_mo["log_ret"], color=colors, width=20, alpha=0.75,
        label="S&P 500 log return")
ax2.axhline(0, color="#333333", linewidth=0.7)
ax2.set_ylabel("Monthly log return", fontsize=10)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2r = ax2.twinx()
ax2r.plot(df_mo.index, df_mo["cum_log_ret"], color="#555555", linewidth=1.4,
          linestyle="-", alpha=0.6, label="Cumulative log return", zorder=4)
ax2r.set_ylabel("Cumulative log return", fontsize=10, color="#555555")
ax2r.tick_params(axis="y", labelcolor="#555555")
ax2r.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2r.spines["top"].set_visible(False)

handles1, labels1 = ax2.get_legend_handles_labels()
handles2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(handles1 + handles2, labels1 + labels2,
           fontsize=10, framealpha=0.7, loc="upper left")

ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

for month_str, label in events_mo.items():
    ts = pd.Timestamp(month_str)
    if ts in df_mo.index:
        for ax in (ax1, ax2):
            ax.axvline(ts, color="#888888", linewidth=0.8, linestyle="--", zorder=2)
        ax1.text(ts, ax1.get_ylim()[1] * 0.97, label,
                 fontsize=7, ha="center", va="top", color="#444444")

plt.tight_layout()
out = os.path.join(BASE, "gep_vs_sp500_logret.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Two-panel: GEP daily level + S&P daily log returns
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 7), sharex=True,
                                gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08})

ax1.fill_between(df_d.index, df_d["gep_pct"], alpha=0.30, color="#378ADD")
ax1.plot(df_d.index, df_d["gep_pct"], color="#1A5FA8", linewidth=0.7,
         label="GEP Robust min-2 (daily, %)")
ax1.set_ylabel("Share of articles (%)", fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.legend(fontsize=10, framealpha=0.7, loc="upper left")
ax1.set_title("GEP Robust min-2 vs S&P 500 Daily Log Returns (1996–2025)",
              fontsize=13, pad=10)

colors = ["#C0392B" if r < 0 else "#27AE60" for r in df_d["log_ret"]]
ax2.bar(df_d.index, df_d["log_ret"], color=colors, width=1, alpha=0.75,
        label="S&P 500 log return")
ax2.axhline(0, color="#333333", linewidth=0.7)
ax2.set_ylabel("Daily log return", fontsize=10)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2r = ax2.twinx()
ax2r.plot(df_d.index, df_d["cum_log_ret"], color="#555555", linewidth=1.4,
          linestyle="-", alpha=0.6, label="Cumulative log return", zorder=4)
ax2r.set_ylabel("Cumulative log return", fontsize=10, color="#555555")
ax2r.tick_params(axis="y", labelcolor="#555555")
ax2r.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2r.spines["top"].set_visible(False)

handles1, labels1 = ax2.get_legend_handles_labels()
handles2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(handles1 + handles2, labels1 + labels2,
           fontsize=10, framealpha=0.7, loc="upper left")

ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

for date_str, label in events_d.items():
    ts = pd.Timestamp(date_str)
    if ts not in df_d.index:
        idx = df_d.index.searchsorted(ts)
        if idx < len(df_d.index):
            ts = df_d.index[idx]
        else:
            continue
    for ax in (ax1, ax2):
        ax.axvline(ts, color="#888888", linewidth=0.8, linestyle="--", zorder=2)
    ax1.text(ts, ax1.get_ylim()[1] * 0.97, label,
             fontsize=7, ha="center", va="top", color="#444444", linespacing=1.3)

plt.tight_layout()
out = os.path.join(BASE, "gep_vs_sp500_logret_daily.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
