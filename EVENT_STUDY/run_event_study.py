#!/usr/bin/env python3
"""
run_event_study.py
==================
Main runner for the GEP event study pipeline.

Steps:
  1. Download S&P500 daily log-returns (yfinance)
  2. Download Fama-French 3 daily factors (pandas_datareader)
  3. For each event: check Reuters headlines (corpus validation)
  4. Run MacKinlay FF3 event study
  5. Aggregate + test (MacKinlay t-stat, BMP test)
  6. Save CSVs + plots

Usage:
    python run_event_study.py [--no-articles] [--events EVENT1 EVENT2 ...]

    --no-articles   skip article checking (faster)
    --events        run only specified event keys (default: all)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

# local imports
sys.path.insert(0, os.path.dirname(__file__))
import config
from article_checker import get_articles_around_date, print_articles
from event_study import EventStudy


# =============================================================================
# 1. MARKET DATA
# =============================================================================

def load_sp500(start: str = "1993-01-01", end: str = "2025-12-31") -> pd.Series:
    """Download S&P500 adjusted close, compute daily log-returns."""
    print("[INFO] Downloading S&P500 data...")
    raw   = yf.download("^GSPC", start=start, end=end,
                        auto_adjust=True, progress=False)
    close = raw["Close"].squeeze()
    lret  = np.log(close / close.shift(1)).dropna()
    lret.name = "R_sp500"
    print(f"       {len(lret)} trading days  ({lret.index[0].date()} → "
          f"{lret.index[-1].date()})")
    return lret


def load_ff3(start: str = "1993-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Download Fama-French 3 daily factors via pandas_datareader."""
    print("[INFO] Downloading Fama-French 3 daily factors...")
    ff3_raw = web.DataReader(
        "F-F_Research_Data_Factors_daily", "famafrench",
        start=start, end=end
    )[0]
    # factors come in %; convert to decimals
    ff3 = ff3_raw.rename(columns={"Mkt-RF": "MktRF"}) / 100.0
    ff3.index = pd.to_datetime(ff3.index)
    print(f"       {len(ff3)} days  ({ff3.index[0].date()} → "
          f"{ff3.index[-1].date()})")
    return ff3


# =============================================================================
# 2. ARTICLE VALIDATION
# =============================================================================

def check_articles(events: dict, meta_dir: str,
                   window_days: int, skip: bool) -> None:
    if skip:
        print("\n[INFO] Article checking skipped (--no-articles)")
        return

    print("\n" + "=" * 70)
    print("ARTICLE VALIDATION — Reuters corpus headlines near event dates")
    print("=" * 70)

    for name, info in events.items():
        articles = get_articles_around_date(
            event_date  = info["date"],
            meta_dir    = meta_dir,
            window_days = window_days,
        )
        print_articles(name, info["date"], articles)

        # save to file
        out_dir  = os.path.join(config.OUT_DIR, "articles")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}_articles.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Event : {name}  ({info['date']})\n")
            f.write(f"Desc  : {info['description']}\n")
            f.write(f"Found : {len(articles)} articles\n")
            f.write("─" * 70 + "\n")
            for a in articles:
                f.write(f"[{a['date']}]  {a['headline']}\n")


# =============================================================================
# 3. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GEP Event Study — MacKinlay FF3")
    parser.add_argument("--no-articles", action="store_true",
                        help="Skip Reuters article checking")
    parser.add_argument("--events", nargs="+", default=None,
                        help="Run only these event keys (default: all)")
    args = parser.parse_args()

    os.makedirs(config.OUT_DIR, exist_ok=True)

    # ── select events ─────────────────────────────────────────────────
    events = config.EVENTS
    if args.events:
        events = {k: v for k, v in events.items() if k in args.events}
        if not events:
            print(f"[ERROR] None of the specified events found in config.")
            sys.exit(1)

    print(f"[INFO] Running event study for {len(events)} events")
    for k, v in events.items():
        print(f"       {k:<40} {v['date']}  {v['category']}")

    # ── article check ─────────────────────────────────────────────────
    check_articles(
        events      = events,
        meta_dir    = config.META_DIR,
        window_days = config.ARTICLE_WINDOW_DAYS,
        skip        = args.no_articles,
    )

    # ── market data ───────────────────────────────────────────────────
    sp500 = load_sp500()
    ff3   = load_ff3()

    # ── event study ───────────────────────────────────────────────────
    params = {
        "ESTIMATION_START": config.ESTIMATION_START,
        "ESTIMATION_END":   config.ESTIMATION_END,
        "EVENT_START":      config.EVENT_START,
        "EVENT_END":        config.EVENT_END,
        "MIN_EST_OBS":      config.MIN_EST_OBS,
        "CAR_WINDOWS":      config.CAR_WINDOWS,
        "OUT_DIR":          config.OUT_DIR,
    }

    es = EventStudy(sp500=sp500, ff3=ff3, events=events, params=params)
    es.run_all()

    # ── aggregate + test ──────────────────────────────────────────────
    agg_results = es.aggregate_and_print()

    # ── save outputs ──────────────────────────────────────────────────
    es.save_csv(config.OUT_DIR)
    es.plot(agg_results, config.OUT_DIR)

    print("\n" + "=" * 70)
    print(f"[DONE] All outputs saved to: {config.OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
