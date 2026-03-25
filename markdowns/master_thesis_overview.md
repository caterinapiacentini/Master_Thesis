# Master Thesis: The Geoeconomic Impact on Stock Market Indices
**Author:** Caterina Piacentini — WU Vienna
**Methodology reference:** Dangl & Salbrechter (2023)

---

## Overview

The thesis constructs a **Geoeconomic Pressure (GEP) Index** from 30 years of Reuters newswire data (1996–2025) using NLP and text analysis. The pipeline covers five main stages: corpus cleaning, Word2Vec training, Guided Topic Modeling (GTM), dictionary construction, index scoring, and robustness comparison.

---

## 1. Data Cleaning

**Script:** `master_thesis/scripts/Cleaning_All_World.py`

Reuters RTRS data is stored as gzip-compressed JSON files. The cleaning pipeline runs in two passes: the first learns multi-word expressions (bigrams), the second applies them and outputs tokenized documents + metadata.

**Key operations:**
- Language filter (English only), diary/scheduled news removal
- HTML stripping, URL/email/phone removal, Reuters dateline & boilerplate removal
- Tokenization with regex `[a-z]+(?:'[a-z]+)?` — **no stopword removal** (preserves context windows for Word2Vec)
- Bigram detection (e.g., `trade_war`, `import_tariffs`) via Gensim `Phrases`
- Metadata extraction: article ID, `versionCreated` timestamp, countries from subject codes

```python
# Two-pass pipeline: learn phrases → apply to clean corpus

RE_HTML     = re.compile(r"<[^>]+>")
RE_CREDITS  = re.compile(r"\b(Reporting by|Writing by|Editing by).*$",
                         flags=re.IGNORECASE | re.MULTILINE)
RE_DATALINE = re.compile(r"^\s*[A-Z][A-Z .'\-]{2,40},\s+[A-Z][a-z]{2}\s+\d{1,2}\s*\(Reuter[s]?\)\s*-\s*",
                         flags=re.MULTILINE)
TOKEN_RE    = re.compile(r"[a-z]+(?:'[a-z]+)?")

def normalize_text(raw: str) -> str:
    x = raw.replace("\u00a0", " ")
    x = RE_HTML.sub(" ", x)
    x = RE_URL.sub(" ", x);   x = RE_EMAIL.sub(" ", x)
    x = RE_DATALINE.sub("", x)
    x = RE_CREDITS.sub("", x)
    x = RE_BRACKETS.sub(" ", x)
    return RE_MULTISPACE.sub(" ", x).strip()

# Pass 1 — learn bigrams
phrases = Phrases(
    sentences=iter_docs_for_phrase_learning(...),
    min_count=20, threshold=10.0, delimiter="_"
)
phraser = Phraser(phrases)

# Pass 2 — clean, tokenize, apply bigrams, write gzip output
with gzip.open(out_tokens, "wt") as f_tok, gzip.open(out_meta, "wt") as f_meta:
    for toks, meta in iter_docs_with_meta(...):
        toks2 = phraser[toks]                          # apply bigrams
        f_tok.write(" ".join(toks2) + "\n")
        f_meta.write(json.dumps({"doc_id": n_docs, **meta}) + "\n")
        n_docs += 1
```

**Output:** ~2.5M+ articles per year in `rtrs_YYYY_world_nodiary_noboiler.txt.gz` + matching metadata JSONL.

---

## 2. Word2Vec Training

**Script:** `master_thesis/scripts/train_w2v_world.py`

A CBOW Word2Vec model is trained on the full cleaned corpus. Documents are streamed from disk to avoid loading the entire corpus into RAM.

**Hyperparameters (per Dangl & Salbrechter):**

| Parameter | Value | Rationale |
|---|---|---|
| `vector_size` | 64 | Embedding dimension |
| `window` | 18 | Large context for geoeconomic phrasing |
| `min_count` | 20 | Prune rare tokens |
| `negative` | 10 | Negative sampling |
| `epochs` | 50 | Full corpus passes |
| `sg` | 0 | CBOW architecture |

```python
class GzCorpus:
    """Memory-efficient streamed corpus — never loads full data into RAM."""
    def __init__(self, paths: List[Path], min_len: int = 3):
        self.paths, self.min_len = paths, min_len

    def __iter__(self) -> Iterator[List[str]]:
        for p in self.paths:
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) >= self.min_len:
                        yield tokens

sentences = GzCorpus(sorted(data_dir.glob("rtrs_*_world_nodiary_noboiler.txt.gz")))

model = Word2Vec(vector_size=64, window=18, min_count=20,
                 negative=10, sample=1e-5, sg=0, epochs=50, workers=8, seed=42)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=50)

# Save as pickle dict (word → 64-dim numpy array) for GTM compatibility
embeddings_dict = {word: model.wv[word] for word in model.wv.index_to_key}
with open(out_pkl, "wb") as f:
    pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
```

**Output:** A `.pkl` dictionary mapping ~100k+ vocabulary words to 64-dimensional vectors.

---

## 3. Guided Topic Modeling (GTM)

**Script:** `master_thesis/scripts/GTM_8.py`

GTM expands geoeconomic subtopics iteratively from seed words in the embedding space. Eight subtopics are identified: *Sanctions, Trade War, Export Controls, Tariffs, Financial Coercion, Retaliation, Protectionism, Embargo*.

The algorithm finds words most aligned with the growing topic subspace using angular distance minimization, with FAISS enabling efficient k-NN search across the full vocabulary.

```python
class GTM:
    def __init__(self, model_path, embd_dim=64, nlist=50, nprobe=8):
        with open(model_path, 'rb') as f:
            pca_embds = pickle.load(f)
        # Build FAISS index for fast similarity search
        self.xb       = np.array(list(pca_embds.values())).astype(np.float32)
        quantizer     = faiss.IndexFlatL2(embd_dim)
        self.index    = faiss.IndexIVFFlat(quantizer, embd_dim, nlist)
        self.index.train(self.xb)
        self.index.add(self.xb)
        self.index.nprobe = nprobe

    def run(self, params, pos_seed, neg_seed, output_dir, topic_name=None):
        # --- Similarity search: candidate pool of k-nearest words ---
        xq         = self.embeddings_dict[proj_subspace + neg_seed_words].astype(np.float32)
        _, sim_idx = self.index.search(xq, params['k-similar'])   # k=5000
        V_buckets  = pd.DataFrame(index=self.vocab_series[np.unique(sim_idx.flatten())],
                                  data={'vector': list(self.xb[...])})

        # --- Iterative subspace expansion ---
        X, C = A.copy(), A.copy()   # X = topic subspace matrix
        while run:
            # Project candidates onto current subspace (OLS)
            B      = np.linalg.inv(X.T @ X) @ X.T @ V
            B_adj  = np.diag(pos_weights) @ B
            # Compute residual (orthogonal) component
            V_orth    = V[:, sel_coeff] - (X @ B[:, sel_coeff])
            norm_proj = norm(X @ B_adj[:, sel_coeff], axis=0)
            norm_orth = norm(V_orth, axis=0)
            # Find word with minimum angular deviation from subspace
            alpha    = np.arctan(norm_orth / norm_proj)
            new_word = V_buckets.index[alpha.argmin()]
            self.topic.append(new_word)
            # Update subspace with conjugate gradient optimization
            result = optimize.minimize(self.func, [0]*X.shape[1],
                                       method="CG", args=(W_orth, I, X, C, weights, params))
            X       = self.UnitColumns(self.X_new)
            weights = weights * (1 + gravity)   # increase gravity each step
            if alpha_min > params['alpha_max'] or len(topic) >= params['cluster_size']:
                run = False

    def GenWordCloud(self, X, output_dir):
        """Score each topic word by projection norm onto the topic subspace."""
        topics_dict = {}
        for w in self.topic:
            v     = self.embeddings_dict[w]
            b     = np.linalg.inv(X.T @ X) @ (X.T @ v)
            v_hat = X @ b                           # projection onto subspace
            topics_dict[w] = np.linalg.norm(v_hat)  # importance = projection norm
        ...
```

**Parameters:** `cluster_size=8`, `gravity=1.5`, `alpha_max=2.0 rad`, `k-similar=5000`

---

## 4. Dictionary Construction

**Script:** `master_thesis/scripts/build_dictionary.py`

GTM outputs one CSV per subtopic with word importance weights. These are aggregated into a single geoeconomic dictionary using local normalization per topic followed by global averaging across Q=8 topics (Dangl & Salbrechter, Eq. 10).

```python
word_weight_sums = defaultdict(float)

for file_name in csv_files:                        # one CSV per subtopic
    df      = pd.read_csv(file_path, index_col=0)
    max_val = df['weight'].max()
    if max_val > 0:
        df['weight'] = df['weight'] / max_val      # local normalization → [0, 1]
    for word, row in df.iterrows():
        word_weight_sums[str(word).strip()] += float(row['weight'])

# Global weight = (1/Q) * Σ local_normalized_weights  (Eq. 10)
final_list = [
    {'word': word, 'weight': total / args.num_topics}
    for word, total in word_weight_sums.items()
]
df_final = pd.DataFrame(final_list).sort_values('weight', ascending=False)
df_final.to_csv("geoeconomic_dictionary.csv", index=False)
```

**Result:** 493-word dictionary. Top entries: `protectionist` (0.409), `import_tariffs` (0.365), `impose_tariffs` (0.340), `retaliatory_tariffs` (0.335), `trade_war` (0.299).

---

## 5. GEP Index Construction

**Script:** `master_thesis/scripts/score_daily_index_new.py`

Each article is scored using frequency-capped, length-normalized dictionary matching. Daily scores are aggregated to a monthly index as an article-weighted average.

**Per-article scoring:**
$$\text{score}_i = \frac{\sum_w \min(\text{count}_w,\ 4) \times \text{weight}_w}{N_{\text{words}}}$$

**Monthly GEP (article-weighted, Dangl & Salbrechter):**
$$\text{GEP}_m = \frac{\sum_{t \in m} \text{score}_t \cdot n_t}{\sum_{t \in m} n_t}$$

```python
# Per-article scoring with frequency cap
counts = {}
for w in words:
    if w in relevant_words:
        counts[w] = min(counts.get(w, 0) + 1, args.freq_cap)   # cap at 4
weighted_sum = sum(counts[w] * T_dict[w] for w in counts)
score = weighted_sum / total_words                              # length normalize

# Daily aggregation
agg = df_year.groupby('date').agg(
    score          = ('score', 'mean'),
    score_volume   = ('score', 'sum'),
    n_articles     = ('score', 'count'),
    n_gep_articles = ('score', lambda x: (x > 0).sum()),
).reset_index()

# Gap filling: forward-fill score on weekends/holidays; set counts to 0
final_index['score'] = final_index['score'].ffill()

# Monthly index: weighted average by article count (NOT simple daily average)
monthly_index = all_articles.groupby('month').apply(
    lambda g: pd.Series({
        'GEP_monthly': np.average(g['score'], weights=g['n_articles']),
        'n_articles':  g['n_articles'].sum(),
    })
)
```

**Output:** `GEP_Daily_Index.csv` (10,897 days, 1996–2025) and `GEP_Monthly_Index.csv` (357 months).

---

## 6. Robustness Check vs. GPR

**Script:** `Master_Thesis/INDEX/index_8/robustness_check_GEP.py`

The GEP index is validated against two benchmarks:

1. **Internal robustness:** Article-weighted monthly GEP vs. day-weighted monthly GEP (each trading day contributes equally). The two series should correlate highly, confirming methodology stability.

2. **External benchmark (GPR):** The GEP index is compared against the Caldara & Iacoviello (2022) Geopolitical Risk Index — a widely-used external benchmark — to validate that the constructed index captures genuinely distinct geoeconomic pressure signals.

```python
# Load the two monthly GEP variants and compute correlation
df = monthly[['month', 'GEP_monthly']].merge(
    rob[['month', 'GEP_monthly_daily_avg']], on='month', how='inner'
)
df['GEP_monthly_scaled']   = df['GEP_monthly']          * 10_000
df['GEP_daily_avg_scaled'] = df['GEP_monthly_daily_avg'] * 10_000

corr = np.corrcoef(df['GEP_monthly_scaled'], df['GEP_daily_avg_scaled'])[0, 1]
# Typical result: r > 0.95 → confirms weighting choice does not drive results

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df['month'], df['GEP_monthly_scaled'],
        color='#378ADD', linewidth=0.9, label='Article-weighted avg (baseline)')
ax.plot(df['month'], df['GEP_daily_avg_scaled'],
        color='#E05C2A', linewidth=0.9, linestyle='--', label='Day-weighted avg (robustness)')
ax.text(0.01, 0.97, f'Correlation: {corr:.4f}', transform=ax.transAxes)
```

**Result:** Correlation between the two GEP variants is typically `r > 0.95`, confirming the article-weighted aggregation is not the driving force behind the index shape.

---

## Pipeline Summary

```
Reuters RTRS (.gz JSON)
        │
        ▼
[1. Cleaning] ──────────► rtrs_YYYY_world_nodiary_noboiler.txt.gz
  • Bigram learning                         + rtrs_YYYY_meta.jsonl.gz
  • Normalization & tokenization
        │
        ▼
[2. Word2Vec] ──────────► embeddings.pkl  (~100k words × 64 dims)
  • CBOW, window=18, 50 epochs
        │
        ▼
[3. GTM × 8 topics] ────► topic_Sanctions.csv, topic_Tariffs.csv, ...
  • FAISS k-NN + subspace expansion
        │
        ▼
[4. Dictionary] ─────────► geoeconomic_dictionary.csv  (493 words)
  • Local normalize + global average (Eq. 10)
        │
        ▼
[5. Index Scoring] ──────► GEP_Daily_Index.csv  (10,897 days)
  • Freq-capped article scoring            GEP_Monthly_Index.csv (357 months)
  • Article-weighted monthly aggregation
        │
        ▼
[6. Robustness / GPR] ───► robustness_check_GEP.png  (r > 0.95)
  • Day-weighted vs article-weighted
  • Comparison with Caldara & Iacoviello GPR index
```
