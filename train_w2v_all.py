#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
from pathlib import Path
from typing import Iterator, List
from gensim.models import Word2Vec


class GzCorpus: # to avoid dowloading everything into the RAM
    def __init__(self, paths: List[Path], min_len: int = 3): # short lines are dropped
        self.paths = paths
        self.min_len = min_len # stores paths and min length

    def __iter__(self) -> Iterator[List[str]]:
        for p in self.paths:
            print(f"[INFO] Reading {p.name}")
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    tokens = line.strip().split() # removes newline/leading/trailing whitespace and splits the sentences by whitespaces
                    if len(tokens) >= self.min_len:
                        yield tokens


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--out_vectors", required=True)

    ap.add_argument("--vector_size", type=int, default=300) # embedding dimension 300
    ap.add_argument("--window", type=int, default=10) # context window size for gensim
    ap.add_argument("--min_count", type=int, default=20) # ignore tokens that appear less than 20 times
    ap.add_argument("--negative", type=int, default=10) # number of negatuve sample
    ap.add_argument("--sample", type=float, default=1e-5) # subsampling threshold for frequent words
    ap.add_argument("--epochs", type=int, default=5) #training passes over corpus 
    ap.add_argument("--sg", type=int, default=1) # default to skip grams
    ap.add_argument("--workers", type=int, default=8) # number of threads
    ap.add_argument("--seed", type=int, default=42) # seeds for reproducibility 

    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    files = sorted(data_dir.glob("rtrs_*_bigrams_nodiary_noboiler.txt.gz"))

    if not files:
        raise FileNotFoundError(f"No corpus files found in {data_dir}")

    print(f"[INFO] Found {len(files)} files")

    sentences = GzCorpus(files)
# initialise untrained word2vec model
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
    model.build_vocab(sentences) # iterates through corpus to count word frequencies, apply min_count and construct internal vocab and negative sampling table
    print(f"[INFO] Vocabulary size: {len(model.wv):,}")

    print("[INFO] Training...")
    model.train(
        sentences,
        total_examples=model.corpus_count, # how many sentences gensim can expect
        epochs=args.epochs # repeats training epochs times
    )

    out_model = Path(args.out_model).expanduser().resolve()
    out_vectors = Path(args.out_vectors).expanduser().resolve()

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_vectors.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving model -> {out_model}") # save all word2vec model --> training configuration, weights, vocabulary, etc.
    model.save(str(out_model))

    print(f"[INFO] Saving vectors -> {out_vectors}") # saves just word vectors + vocab keys, less computationally heavy.
    model.wv.save(str(out_vectors))

    print("[OK] Done.")


if __name__ == "__main__":
    main()
