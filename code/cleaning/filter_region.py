#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filters an already-cleaned world corpus down to a region, by matching the
country codes recorded in each article's metadata.

Reads:
  DATA_WORLD/rtrs_YYYY_world_nodiary_noboiler.txt.gz   (token lines, from clean_world.py)
  INFO_DATA_WORLD/rtrs_YYYY_world_meta.jsonl.gz        (metadata, aligned by line)

Writes:
  <out_dir>/rtrs_YYYY_<region_tag>_nodiary_noboiler.txt.gz
"""

import argparse
import gzip
import json
from pathlib import Path


EUROPE_CODES = {
    # EU member states
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
    "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK",
    # EEA / EFTA
    "IS", "LI", "NO", "CH",
    # United Kingdom
    "UK", "GB",
    # Candidate / associate countries commonly covered by European news desks
    "AL", "BA", "ME", "MK", "RS", "XK", "MD", "UA", "BY",
}

US_CODES = {"US"}

REGION_PRESETS = {
    "europe": EUROPE_CODES,
    "us":     US_CODES,
}


def main():
    ap = argparse.ArgumentParser(description="Filter world corpus to a regional subset.")
    ap.add_argument("--year",        required=True, type=int,
                    help="Year to process (e.g. 2005)")
    ap.add_argument("--tokens_dir",  required=True,
                    help="Directory containing rtrs_YYYY_world_nodiary_noboiler.txt.gz")
    ap.add_argument("--meta_dir",    required=True,
                    help="Directory containing rtrs_YYYY_world_meta.jsonl.gz")
    ap.add_argument("--out_dir",      required=True,
                    help="Output directory for filtered token files")
    ap.add_argument("--out_meta_dir", default=None,
                    help="Optional output directory for filtered meta files (aligned with token output)")
    ap.add_argument("--region_tag",  default="europe",
                    help="Short tag used in the output filename (default: europe)")
    ap.add_argument("--region_codes", nargs="*", default=None,
                    help="Override the built-in code list with explicit country codes "
                         "(e.g. --region_codes DE FR UK). Falls back to --region_tag preset.")
    args = ap.parse_args()

    year = args.year
    region_tag = args.region_tag.lower()

    # Resolve country code set
    if args.region_codes:
        codes = set(c.upper() for c in args.region_codes)
    elif region_tag in REGION_PRESETS:
        codes = REGION_PRESETS[region_tag]
    else:
        ap.error(f"Unknown region_tag '{region_tag}' and no --region_codes provided.")

    tokens_dir = Path(args.tokens_dir).expanduser().resolve()
    meta_dir   = Path(args.meta_dir).expanduser().resolve()
    out_dir    = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_meta_dir = None
    if args.out_meta_dir:
        out_meta_dir = Path(args.out_meta_dir).expanduser().resolve()
        out_meta_dir.mkdir(parents=True, exist_ok=True)

    tok_path  = tokens_dir / f"rtrs_{year}_world_nodiary_noboiler.txt.gz"
    meta_path = meta_dir   / f"rtrs_{year}_world_meta.jsonl.gz"
    out_path  = out_dir    / f"rtrs_{year}_{region_tag}_nodiary_noboiler.txt.gz"
    out_meta_path = out_meta_dir / f"rtrs_{year}_{region_tag}_meta.jsonl.gz" if out_meta_dir else None

    if not tok_path.exists():
        raise FileNotFoundError(f"Token file not found: {tok_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    print(f"[INFO] Year {year}: filtering to region '{region_tag}' ({len(codes)} codes)")
    print(f"[INFO]   tokens : {tok_path}")
    print(f"[INFO]   meta   : {meta_path}")
    print(f"[INFO]   output : {out_path}")
    if out_meta_path:
        print(f"[INFO]   meta out: {out_meta_path}")

    kept = 0
    total = 0

    def _run(f_tok, f_meta, f_out, f_meta_out):
        nonlocal kept, total
        for tok_line, meta_line in zip(f_tok, f_meta):
            total += 1
            try:
                meta = json.loads(meta_line)
            except json.JSONDecodeError:
                continue
            doc_countries = set(meta.get("countries", []))
            if doc_countries & codes:
                f_out.write(tok_line)
                if f_meta_out is not None:
                    f_meta_out.write(meta_line)
                kept += 1

    if out_meta_path:
        with (gzip.open(tok_path,      "rt", encoding="utf-8", errors="replace") as f_tok,
              gzip.open(meta_path,     "rt", encoding="utf-8", errors="replace") as f_meta,
              gzip.open(out_path,      "wt", encoding="utf-8") as f_out,
              gzip.open(out_meta_path, "wt", encoding="utf-8") as f_meta_out):
            _run(f_tok, f_meta, f_out, f_meta_out)
    else:
        with (gzip.open(tok_path,  "rt", encoding="utf-8", errors="replace") as f_tok,
              gzip.open(meta_path, "rt", encoding="utf-8", errors="replace") as f_meta,
              gzip.open(out_path,  "wt", encoding="utf-8") as f_out):
            _run(f_tok, f_meta, f_out, None)

    print(f"[OK] Year {year}: kept {kept:,} / {total:,} docs "
          f"({100*kept/total:.1f}% of world corpus) -> {out_path}")


if __name__ == "__main__":
    main()
