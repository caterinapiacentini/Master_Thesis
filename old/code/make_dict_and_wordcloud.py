#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

# Optional dependency; if missing, we still save CSV.
try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False

# Word cloud
try:
    from wordcloud import WordCloud
    _HAS_WORDCLOUD = True
except Exception:
    _HAS_WORDCLOUD = False


RE_HAS_DIGIT = re.compile(r"\d")
RE_POSSESSIVE = re.compile(r".*'s$")


# C) Generic “glue/verb” stoplist (extendable)
DEFAULT_GENERIC_TERMS = {
    "say", "says", "said", "tell", "told",
    "add", "added", "adds",
    "make", "made", "makes",
    "take", "took", "taken", "takes",
    "give", "gave", "given", "gives",
    "come", "came", "comes",
    "go", "went", "gone", "goes",
    "get", "got", "gets",
    "set", "sets", "setting",
    "call", "called", "calls",
    "meet", "met", "meets", "meeting",
    "plan", "plans", "planned", "planning",
    "agree", "agreed", "agrees", "agreeing",
    "aim", "aims", "aimed",
    "seek", "seeks", "sought",

    "impose", "imposed", "imposes", "imposing",
    "apply", "applied", "applies", "applying",
    "impact", "deadline", "deal",
}

# C) Extra “context” terms that often contaminate dictionaries
DEFAULT_CONTEXT_TERMS = {
    "country", "countries", "government", "governments",
    "official", "officials", "administration",
    "ministry", "minister", "president", "congress",
    "parliament", "commission", "agency",
    "washington", "beijing", "moscow", "brussels",
}

# A) Seed/variant exclusion patterns (extendable)
SEED_VARIANT_PATTERNS = [
    r"^tariff(s)?$",
    r"^sanction(s|ed|ing)?$",
]


def load_wordlist(path: str) -> Set[str]:
    s: Set[str] = set()
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            s.add(w.lower())
    return s


def is_ok_token(tok: str, min_len: int) -> bool:
    if not isinstance(tok, str):
        return False
    if len(tok) < min_len:
        return False
    if RE_HAS_DIGIT.search(tok):
        return False
    if RE_POSSESSIVE.match(tok.lower()):
        return False
    return True


def should_exclude_seed_variant(tok: str) -> bool:
    t = tok.lower()
    for pat in SEED_VARIANT_PATTERNS:
        if re.fullmatch(pat, t):
            return True
    return False


def resolve_seed(wv: KeyedVectors, seed: str) -> str:
    """
    Seed hygiene: map missing seeds to common variants (pluralization).
    """
    if seed in wv:
        return seed

    cands = []
    # plural/singular
    if seed.endswith("s"):
        cands.append(seed[:-1])
    else:
        cands.append(seed + "s")

    # common plural with "es"
    if not seed.endswith("es"):
        cands.append(seed + "es")

    for c in cands:
        if c in wv:
            print(f"[INFO] Resolved seed '{seed}' -> '{c}'")
            return c
    return ""


def filter_seeds_in_vocab(wv: KeyedVectors, seeds: List[str]) -> List[str]:
    ok = []
    missing = []
    for s in seeds:
        r = resolve_seed(wv, s)
        if r:
            ok.append(r)
        else:
            missing.append(s)
    if missing:
        print(f"[WARN] Missing seeds (ignored): {missing}")

    # unique preserve order
    seen = set()
    out = []
    for s in ok:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec if n == 0 else (vec / n)


def centroid(wv: KeyedVectors, seeds: List[str]) -> np.ndarray:
    mats = [wv[s] for s in seeds]
    c = np.mean(np.vstack(mats), axis=0)
    return unit(c)


def cos_to_vec(wv: KeyedVectors, word: str, vec_unit: np.ndarray) -> float:
    v = wv[word]
    return float(np.dot(unit(v), vec_unit))


def crude_stem(tok: str) -> str:
    """
    Very light stemming for redundancy control.
    """
    t = tok.lower()
    for suf in ["ing", "ed", "es", "s"]:
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break
    return t


def cap_redundancy(df: pd.DataFrame, max_per_stem: int) -> pd.DataFrame:
    kept_rows = []
    counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        st = crude_stem(str(row["term"]))
        c = counts.get(st, 0)
        if c >= max_per_stem:
            continue
        kept_rows.append(row)
        counts[st] = c + 1
    return pd.DataFrame(kept_rows)


def build_dictionary_v4(
    wv: KeyedVectors,
    trade_seeds: List[str],
    sanc_seeds: List[str],
    ret_seeds: List[str],
    topn: int,
    min_sim: float,
    max_terms: int,
    min_token_len: int,
    generic_stop: Set[str],
    context_stop: Set[str],
    max_per_stem: int,
) -> pd.DataFrame:
    # Dangl-style: theory-driven multi-seed concept with centroid expansion
    trade_seeds = filter_seeds_in_vocab(wv, trade_seeds)
    sanc_seeds = filter_seeds_in_vocab(wv, sanc_seeds)
    ret_seeds = filter_seeds_in_vocab(wv, ret_seeds)

    if len(trade_seeds) < 1 or len(sanc_seeds) < 1 or len(ret_seeds) < 1:
        raise RuntimeError(
            f"Not enough seeds in vocab. trade={len(trade_seeds)} sanc={len(sanc_seeds)} ret={len(ret_seeds)}"
        )

    v_trade = centroid(wv, trade_seeds)
    v_sanc = centroid(wv, sanc_seeds)
    v_ret = centroid(wv, ret_seeds)
    v_concept = unit((v_trade + v_sanc + v_ret) / 3.0)

    # Candidate pool: union of top-N neighbors of ALL seeds (robust and cheap)
    candidates: Set[str] = set()
    for s in (trade_seeds + sanc_seeds + ret_seeds):
        for w, _ in wv.most_similar(s, topn=topn):
            candidates.add(w)

    # remove seeds from candidates
    for s in (trade_seeds + sanc_seeds + ret_seeds):
        candidates.discard(s)

    rows: List[Dict] = []
    for term in candidates:
        if not is_ok_token(term, min_token_len):
            continue
        if should_exclude_seed_variant(term):
            continue

        tl = term.lower()
        if tl in generic_stop:
            continue
        if tl in context_stop:
            continue

        sim_trade = cos_to_vec(wv, term, v_trade)
        sim_sanc = cos_to_vec(wv, term, v_sanc)
        sim_ret = cos_to_vec(wv, term, v_ret)
        sim_concept = cos_to_vec(wv, term, v_concept)

        # B) Threshold both concept and at least one pillar
        if sim_concept < min_sim:
            continue
        if max(sim_trade, sim_sanc, sim_ret) < min_sim:
            continue

        weight = max(0.0, sim_concept)

        rows.append({
            "term": term,
            "weight": weight,
            "sim_concept": sim_concept,
            "sim_trade": sim_trade,
            "sim_sanctions": sim_sanc,
            "sim_retaliation": sim_ret,
        })

    if not rows:
        raise RuntimeError("No terms survived filtering. Lower --min_sim or increase --topn / relax stoplists.")

    df = pd.DataFrame(rows).sort_values(
        ["weight", "sim_concept", "sim_trade", "sim_sanctions", "sim_retaliation"],
        ascending=False,
    )

    # C) Redundancy cap before truncation
    if max_per_stem > 0:
        df = cap_redundancy(df, max_per_stem=max_per_stem)

    # enforce exact dictionary size (or as close as possible if over-filtering)
    df = df.head(max_terms).copy()

    if len(df) < max_terms:
        print(f"[WARN] Only {len(df)} terms available after filters; target was {max_terms}. "
              f"Consider lowering --min_sim or increasing --topn or relaxing stoplists.")

    return df


def save_outputs(df: pd.DataFrame, out_parquet: Path, out_csv: Path) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)

    if _HAS_PYARROW:
        df.to_parquet(out_parquet, index=False)
    else:
        print("[WARN] pyarrow not available -> skipping parquet output (CSV still written).")


def make_wordcloud(df: pd.DataFrame, out_png: Path, width: int = 2000, height: int = 1200) -> None:
    if not _HAS_WORDCLOUD:
        raise RuntimeError("wordcloud package not installed in this environment.")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    freqs = dict(zip(df["term"].astype(str), df["weight"].astype(float)))

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        prefer_horizontal=0.9,
        collocations=False,
    ).generate_from_frequencies(freqs)

    wc.to_file(str(out_png))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", required=True, help="Path to KeyedVectors .kv")

    ap.add_argument("--trade_seeds", type=str,
                    default="tariff,duties,duty,levy,quota,trade_barrier,import_ban,export_control",
                    help="Comma-separated seeds for TRADE restrictions")
    ap.add_argument("--sanction_seeds", type=str,
                    default="sanction,embargo,blacklist,asset_freeze,travel_ban,freeze,designation",
                    help="Comma-separated seeds for FINANCIAL sanctions")
    ap.add_argument("--retaliation_seeds", type=str,
                    default="retaliation,countermeasure,retaliate,reprisal,escalation,restriction,restrictions",
                    help="Comma-separated seeds for RETALIATION/countermeasures")

    ap.add_argument("--topn", type=int, default=3000)
    ap.add_argument("--min_sim", type=float, default=0.35)
    ap.add_argument("--max_terms", type=int, default=100)
    ap.add_argument("--min_token_len", type=int, default=3)
    ap.add_argument("--max_per_stem", type=int, default=2,
                    help="Max tokens per crude stem group (redundancy control). 0 disables.")

    # Optional: extra exclusions
    ap.add_argument("--stoplist", type=str, default="",
                    help="Optional newline-separated terms to exclude (added to generic stoplist).")
    ap.add_argument("--context_stoplist", type=str, default="",
                    help="Optional newline-separated context terms to exclude (added to context stoplist).")

    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", required=True)

    args = ap.parse_args()

    kv_path = Path(args.kv).expanduser().resolve()
    print(f"[INFO] Loading vectors: {kv_path}")
    wv = KeyedVectors.load(str(kv_path), mmap="r")

    generic_stop = set(DEFAULT_GENERIC_TERMS)
    context_stop = set(DEFAULT_CONTEXT_TERMS)

    if args.stoplist:
        generic_stop |= load_wordlist(args.stoplist)
    if args.context_stoplist:
        context_stop |= load_wordlist(args.context_stoplist)

    trade_seeds = [s.strip() for s in args.trade_seeds.split(",") if s.strip()]
    sanc_seeds = [s.strip() for s in args.sanction_seeds.split(",") if s.strip()]
    ret_seeds = [s.strip() for s in args.retaliation_seeds.split(",") if s.strip()]

    df = build_dictionary_v4(
        wv=wv,
        trade_seeds=trade_seeds,
        sanc_seeds=sanc_seeds,
        ret_seeds=ret_seeds,
        topn=args.topn,
        min_sim=args.min_sim,
        max_terms=args.max_terms,
        min_token_len=args.min_token_len,
        generic_stop={x.lower() for x in generic_stop},
        context_stop={x.lower() for x in context_stop},
        max_per_stem=args.max_per_stem,
    )

    out_parquet = Path(args.out_parquet).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_png = Path(args.out_png).expanduser().resolve()

    print(f"[INFO] Writing dictionary: {len(df):,} terms (target={args.max_terms})")
    save_outputs(df, out_parquet=out_parquet, out_csv=out_csv)

    print(f"[INFO] Creating word cloud -> {out_png.name}")
    make_wordcloud(df, out_png=out_png)

    print("[OK] Done.")
    print("[INFO] Top 15 terms by weight:")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
