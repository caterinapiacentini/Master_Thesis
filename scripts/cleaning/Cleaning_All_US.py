#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple


# ----------------------------
# Parsing RTRS gz format (robust)
# ----------------------------
def iter_rtrs_items_from_gz(gz_path: Path) -> Iterator[Dict]:
    """
    File structure:
      {"RIC":"MRN_STORY", ... "Items":[
      {...},
      {...},
      ]}
    We skip the header, then parse one JSON object per line until we hit ']}'.
    """
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and '"Items"' in s and "[" in s:
                break

        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("]}") or s == "]}" or s == "]":
                break

            if s.endswith(","):
                s = s[:-1].rstrip()
            if s.endswith("]}"):
                s = s[:-2].rstrip()

            if not s.startswith("{"):
                continue

            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


# ----------------------------
# Filters & extraction
# ----------------------------
def is_english(item: Dict) -> bool:
    data = item.get("data")
    if not isinstance(data, dict):
        return False
    lang = data.get("language")
    if isinstance(lang, str):
        l = lang.strip().lower()
        return l == "en" or l.startswith("en")
    return True 

def is_diary_item(item: Dict) -> bool:
    data = item.get("data")
    if not isinstance(data, dict):
        return False
    subs = data.get("subjects", [])
    if isinstance(subs, list) and any(s == "N2:DIARY" for s in subs):
        return True
    h = data.get("headline")
    return isinstance(h, str) and h.strip().lower().startswith("diary")

def extract_doc_text(item: Dict, include_headline: bool = True) -> Optional[str]:
    data = item.get("data")
    if not isinstance(data, dict):
        return None

    parts = []
    if include_headline:
        h = data.get("headline")
        if isinstance(h, str) and h.strip():
            parts.append(h.strip())
    b = data.get("body")
    if isinstance(b, str) and b.strip():
        parts.append(b.strip())

    return "\n".join(parts) if parts else None

SUBJ_GEO = re.compile(r"^N2:([A-Z]{2,3})$")
EXCLUDE_N2 = {
    "RTRS", "CEN", "DIARY", "LEN", "D+", "D", "ECO", "COM", "POL",
    "LDE", "LFR", "LGN", "LSP", "LIT"
} 

def extract_countries_from_subjects(subjects: List[str]) -> List[str]:
    countries: List[str] = []
    for s in subjects:
        if not isinstance(s, str):
            continue
        m = SUBJ_GEO.match(s)
        if m:
            code = m.group(1)
            if code not in EXCLUDE_N2:
                countries.append(code)
    return sorted(set(countries))

def extract_meta(item: Dict, gz_path: Path) -> Dict:
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    timestamps = item.get("timestamps") if isinstance(item.get("timestamps"), list) else []

    subs = data.get("subjects", [])
    auds = data.get("audiences", [])
    subjects = subs if isinstance(subs, list) else []
    audiences = auds if isinstance(auds, list) else []
    countries = extract_countries_from_subjects(subjects)

    recorded_ts = None
    if timestamps:
        for t in timestamps:
            if isinstance(t, dict) and t.get("name") == "recorded" and isinstance(t.get("timestamp"), str):
                recorded_ts = t["timestamp"]
                break
        if recorded_ts is None:
            for t in timestamps:
                if isinstance(t, dict) and isinstance(t.get("timestamp"), str):
                    recorded_ts = t["timestamp"]
                    break

    meta = {
        "guid": item.get("guid"),
        "id": data.get("id"),
        "altId": data.get("altId"),
        "firstCreated": data.get("firstCreated"),
        "versionCreated": data.get("versionCreated"),
        "recorded_ts": recorded_ts,
        "headline_raw": data.get("headline"),
        "takeSequence": data.get("takeSequence"),
        "urgency": data.get("urgency"),
        "language": data.get("language"),
        "subjects": subjects,
        "audiences": audiences,
        "countries": countries,
        "file": str(gz_path),
    }
    return meta


# ----------------------------
# Cleaning & tokenization
# ----------------------------
DEFAULT_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "can","could","did","do","does","doing","down","during","each","few","for","from","further",
    "had","has","have","having","he","her","here","hers","herself","him","himself","his","how",
    "i","if","in","into","is","it","its","itself","just","me","more","most","my","myself",
    "no","nor","not","now","of","off","on","once","only","or","other","our","ours","ourselves",
    "out","over","own","s","same","she","should","so","some","such","than","that","the","their",
    "theirs","them","themselves","then","there","these","they","this","those","through","to","too",
    "under","until","up","very","was","we","were","what","when","where","which","while","who",
    "whom","why","will","with","you","your","yours","yourself","yourselves"
}

RE_HTML = re.compile(r"<[^>]+>") 
RE_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
RE_PHONE = re.compile(r"\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}")

RE_MULTISPACE = re.compile(r"\s+") 

RE_DATALINE = re.compile(
    r"^\s*[A-Z][A-Z .'\-]{2,40},\s+[A-Z][a-z]{2}\s+\d{1,2}\s*\(Reuter[s]?\)\s*-\s*",
    flags=re.MULTILINE
) 
RE_CREDITS = re.compile(
    r"\b(Reporting by|Writing by|Editing by|Additional reporting by)\b.*$",
    flags=re.IGNORECASE | re.MULTILINE
) 

RE_BRACKETS = re.compile(r"\[[^\]]+\]") 
RE_DOUBLE_CLICK_LINE = re.compile(r"(?im)^[^\n]*\bdouble[- ]click\b[^\n]*$") 
RE_TOLD_REUTERS = re.compile(r"(?i)\btold reuters\b")
RE_REUTER_TERMINAL = re.compile(r"(?i)\breuter terminal\b")

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?") 

def normalize_text(raw: str) -> str: 
    x = raw.replace("\u00a0", " ") 
    x = RE_HTML.sub(" ", x)
    x = RE_URL.sub(" ", x)
    
    x = RE_EMAIL.sub(" ", x)
    x = RE_PHONE.sub(" ", x)

    x = RE_DATALINE.sub("", x)
    x = RE_CREDITS.sub("", x)
    x = RE_BRACKETS.sub(" ", x)
    x = RE_DOUBLE_CLICK_LINE.sub(" ", x)
    x = RE_TOLD_REUTERS.sub(" ", x)
    x = RE_REUTER_TERMINAL.sub(" ", x)

    x = RE_MULTISPACE.sub(" ", x).strip()
    return x

def tokenize(text: str, remove_stopwords: bool, min_token_len: int) -> List[str]:
    toks = TOKEN_RE.findall(text.lower()) 
    
    # --- METHODOLOGY CHANGE ---
    # The authors train a CBOW Word2Vec model with a window size of 18.
    # Removing stop words compresses the text and ruins the natural context window.
    # Therefore, we bypass this step.
    # if remove_stopwords:
    #     toks = [t for t in toks if t not in DEFAULT_STOPWORDS] 
    # --------------------------

    if min_token_len > 1:
        toks = [t for t in toks if len(t) >= min_token_len] 
    return toks


# ----------------------------
# Corpus iterators (generic year + chunkable)
# ----------------------------
def list_year_gz_files(year_dir: Path, pattern: str = "*.txt.gz") -> List[Path]:
    return sorted(year_dir.glob(pattern))

def iter_docs_with_meta(
    year_dir: Path, 
    remove_stopwords: bool,
    include_headline: bool,
    min_doc_tokens: int,
    min_token_len: int,
    drop_diary: bool,
    file_offset: int = 0,
    n_files: int = 0,
    file_glob: str = "*.txt.gz",
) -> Iterator[Tuple[List[str], Dict]]:
    
    gz_files = list_year_gz_files(year_dir, pattern=file_glob)
    if not gz_files:
        raise FileNotFoundError(f"No gz files found in: {year_dir} (pattern={file_glob})")

    if file_offset < 0 or n_files < 0:
        raise ValueError("Offsets must be >= 0")

    if n_files > 0:
        gz_files = gz_files[file_offset : file_offset + n_files]
    else:
        gz_files = gz_files[file_offset :]

    if not gz_files:
        return

    for gz_path in gz_files:
        try:
            for item in iter_rtrs_items_from_gz(gz_path):
                if not is_english(item):
                    continue
                if drop_diary and is_diary_item(item):
                    continue

                raw = extract_doc_text(item, include_headline=include_headline)
                if not raw:
                    continue

                cleaned = normalize_text(raw)
                if not cleaned:
                    continue

                toks = tokenize(cleaned, remove_stopwords=remove_stopwords, min_token_len=min_token_len)
                if len(toks) < min_doc_tokens:
                    continue

                meta = extract_meta(item, gz_path)
                
                if "US" not in meta.get("countries", []):
                    continue
                
                yield toks, meta 
        except (EOFError, OSError) as e:
            print(f"[WARN] Skipping unreadable gzip: {gz_path} ({type(e).__name__}: {e})")
            continue

def iter_docs_for_phrase_learning(
    year_dir: Path,
    remove_stopwords: bool,
    include_headline: bool,
    min_doc_tokens: int,
    min_token_len: int,
    drop_diary: bool,
    phrase_ignore_terms: Set[str],
    file_offset: int = 0,
    n_files: int = 0,
    file_glob: str = "*.txt.gz",
) -> Iterator[List[str]]:
    for toks, _meta in iter_docs_with_meta( 
        year_dir=year_dir,
        remove_stopwords=remove_stopwords,
        include_headline=include_headline,
        min_doc_tokens=min_doc_tokens,
        min_token_len=min_token_len,
        drop_diary=drop_diary,
        file_offset=file_offset,
        n_files=n_files,
        file_glob=file_glob,
    ):
        if phrase_ignore_terms:
            toks2 = [t for t in toks if t not in phrase_ignore_terms]
            if len(toks2) >= min_doc_tokens: 
                yield toks2
        else:
            yield toks


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year_dir", required=True, type=str, help=".../RTRS/<YEAR>")
    ap.add_argument("--out_tokens", required=True, type=str, help="output tokens gz path")
    ap.add_argument("--out_meta", required=True, type=str, help="output metadata jsonl.gz path")

    ap.add_argument("--include_headline", action="store_true")
    
    # --- METHODOLOGY CHANGE ---
    # Disabled this argument to prevent accidentally destroying the Word2Vec context window.
    # ap.add_argument("--remove_stopwords", action="store_true")
    # --------------------------
    
    ap.add_argument("--drop_diary", action="store_true")

    ap.add_argument("--min_doc_tokens", type=int, default=10) 
    ap.add_argument("--min_token_len", type=int, default=2) 

    ap.add_argument("--phrase_min_count", type=int, default=20) 
    ap.add_argument("--phrase_threshold", type=float, default=10.0) 

    ap.add_argument("--file_offset", type=int, default=0)
    ap.add_argument("--n_files", type=int, default=0)
    ap.add_argument("--file_glob", type=str, default="*.txt.gz")

    args = ap.parse_args()

    from gensim.models.phrases import Phrases, Phraser

    year_dir = Path(args.year_dir).expanduser().resolve()
    out_tokens = Path(args.out_tokens).expanduser().resolve()
    out_meta = Path(args.out_meta).expanduser().resolve()
    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    phrase_ignore_terms = {
        "reuters", "reuter", "said", "says", "mr", "mrs",
        "pct", "percent", "mln", "million", "billion",
        "told", "click", "double"
    }

    print("[INFO] Pass 1/2: learning bigrams...")
    phrases = Phrases( 
        sentences=iter_docs_for_phrase_learning(
            year_dir=year_dir,
            remove_stopwords=False, # Forced to False per methodology
            include_headline=args.include_headline,
            min_doc_tokens=args.min_doc_tokens,
            min_token_len=args.min_token_len,
            drop_diary=args.drop_diary,
            phrase_ignore_terms=phrase_ignore_terms,
            file_offset=args.file_offset,
            n_files=args.n_files,
            file_glob=args.file_glob,
        ),
        min_count=args.phrase_min_count,
        threshold=args.phrase_threshold,
        delimiter="_"   
    )
    phraser = Phraser(phrases)

    print("[INFO] Pass 2/2: writing corpus + metadata...")
    n_docs = 0
    with gzip.open(out_tokens, "wt", encoding="utf-8") as f_tok, gzip.open(out_meta, "wt", encoding="utf-8") as f_meta:
        for toks, meta in iter_docs_with_meta(
            year_dir=year_dir,
            remove_stopwords=False, # Forced to False per methodology
            include_headline=args.include_headline,
            min_doc_tokens=args.min_doc_tokens,
            min_token_len=args.min_token_len,
            drop_diary=args.drop_diary,
            file_offset=args.file_offset,
            n_files=args.n_files,
            file_glob=args.file_glob,
        ):
            toks2 = phraser[toks]
            f_tok.write(" ".join(toks2) + "\n")

            meta_out = {"doc_id": n_docs}
            meta_out.update(meta)
            f_meta.write(json.dumps(meta_out, ensure_ascii=False) + "\n")

            n_docs += 1

    print(f"[OK] wrote {n_docs:,} docs -> {out_tokens}")
    print(f"[OK] wrote {n_docs:,} meta -> {out_meta}")


if __name__ == "__main__":
    main()