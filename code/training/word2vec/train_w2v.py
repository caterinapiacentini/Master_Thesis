#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trains Word2Vec on the cleaned US corpus (original source: the external
RTRS newswire archive, via clean_world.py + filter_region.py).

Reads:
  --data_dir  rtrs_*_us_nodiary_noboiler.txt.gz (not in this repo — path is
              set in slurm/training/train_word2vec.slurm)

Writes:
  --out_model  gensim Word2Vec model
  --out_pkl    {word: vector} dict, pickled — what gtm.py loads
"""

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Iterator, List
from gensim.models import Word2Vec


class GzCorpus:
    def __init__(self, paths: List[Path], min_len: int = 3):
        self.paths = paths
        self.min_len = min_len

    def __iter__(self) -> Iterator[List[str]]:
        for p in self.paths:
            print(f"[INFO] Reading {p.name}")
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) >= self.min_len:
                        yield tokens


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir",    required=True)
    ap.add_argument("--out_model",   required=True)
    ap.add_argument("--out_pkl",     required=True)

    # --- HYPERPARAMETERS (per Dangl methodology) ---
    ap.add_argument("--vector_size", type=int,   default=64)
    ap.add_argument("--window",      type=int,   default=18)
    ap.add_argument("--min_count",   type=int,   default=20)
    ap.add_argument("--negative",    type=int,   default=10)
    ap.add_argument("--sample",      type=float, default=1e-5)
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--sg",          type=int,   default=0)
    ap.add_argument("--workers",     type=int,   default=8)
    ap.add_argument("--seed",        type=int,   default=42)

    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    files = sorted(data_dir.glob("rtrs_*_us_nodiary_noboiler.txt.gz"))

    if not files:
        raise FileNotFoundError(f"No US corpus files found in {data_dir}")

    print(f"[INFO] Found {len(files)} files")

    sentences = GzCorpus(files)

    model = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        negative=args.negative,
        sample=args.sample,
        sg=args.sg,
        workers=args.workers,
        seed=args.seed,
    )

    print("[INFO] Building vocabulary...")
    model.build_vocab(sentences)
    print(f"[INFO] Vocabulary size: {len(model.wv):,}")

    print("[INFO] Training...")
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=args.epochs,
    )

    out_model = Path(args.out_model).expanduser().resolve()
    out_pkl   = Path(args.out_pkl).expanduser().resolve()

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving model -> {out_model}")
    model.save(str(out_model))

    print(f"[INFO] Extracting vectors to dictionary and saving -> {out_pkl}")
    embeddings_dict = {word: model.wv[word] for word in model.wv.index_to_key}
    with open(out_pkl, "wb") as f:
        pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[OK] Done.")


if __name__ == "__main__":
    main()
