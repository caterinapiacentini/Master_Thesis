"""
article_checker.py
==================
For each event date, scan the Reuters corpus and return the top headlines
published in a ±N day window. Used to validate event dates and confirm
that the corpus was covering the event.
"""

import gzip
import json
import os
from datetime import datetime, timedelta


def _meta_file_for_year(meta_dir: str, year: int) -> str | None:
    path = os.path.join(meta_dir, f"rtrs_{year}_meta.jsonl.gz")
    return path if os.path.exists(path) else None


def get_articles_around_date(
    event_date: str,
    meta_dir: str,
    window_days: int = 3,
    max_results: int = 20,
) -> list[dict]:
    """
    Return up to max_results article headlines published within
    ±window_days calendar days of event_date.

    Parameters
    ----------
    event_date  : 'YYYY-MM-DD'
    meta_dir    : path to INFO_DATA1 folder
    window_days : half-width of the article search window
    max_results : cap on returned articles

    Returns
    -------
    List of dicts with keys: date, headline, guid
    """
    center = datetime.strptime(event_date, "%Y-%m-%d")
    lo     = center - timedelta(days=window_days)
    hi     = center + timedelta(days=window_days)

    # may span two calendar years (e.g. Dec 31 → Jan 1)
    years_needed = {lo.year, hi.year}

    articles = []

    for year in sorted(years_needed):
        meta_path = _meta_file_for_year(meta_dir, year)
        if meta_path is None:
            continue

        with gzip.open(meta_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_ts = rec.get("versionCreated") or rec.get("firstCreated", "")
                if not raw_ts:
                    continue

                try:
                    art_date = datetime.strptime(raw_ts[:10], "%Y-%m-%d")
                except ValueError:
                    continue

                if lo <= art_date <= hi:
                    articles.append({
                        "date":     art_date.strftime("%Y-%m-%d"),
                        "headline": rec.get("headline_raw", "(no headline)"),
                        "guid":     rec.get("guid", ""),
                    })

    # sort by date, trim
    articles.sort(key=lambda x: x["date"])
    return articles[:max_results]


def print_articles(event_name: str, event_date: str, articles: list[dict]) -> None:
    print(f"\n{'─'*70}")
    print(f"  Event : {event_name}  ({event_date})")
    print(f"  Found : {len(articles)} articles in ±3-day window")
    print(f"{'─'*70}")
    for a in articles:
        print(f"  [{a['date']}]  {a['headline']}")
    if not articles:
        print("  (no articles found — check date or corpus coverage)")
