# Master Thesis: The Geoeconomic Impact on Stock Market Indices
**Author:** Caterina Piacentini — WU Vienna
**Methodology reference:** Dangl & Salbrechter (2023)

---

## Overview

The thesis constructs a **Geoeconomic Pressure (GEP) Index** from 30 years of Reuters newswire data (1996–2025) using NLP and text analysis. The final pipeline delivers a **GTM-6 topic model** and a **MIN2 threshold-based index** — a bounded, stationary series measuring the monthly share of news coverage devoted to geoeconomic coercion. The index is then used to study return predictability on the S&P 500.

---

## 1. Data Cleaning

**Script:** `scripts/cleaning/Cleaning_All_US.py`

Reuters RTRS data is stored as gzip-compressed JSON files. The cleaning pipeline runs in two passes: the first learns multi-word expressions (bigrams), the second applies them and outputs tokenized documents + metadata.

**Key operations:**
- Language filter (English only), diary/scheduled news removal
- HTML stripping, URL/email/phone removal, Reuters dateline & boilerplate removal
- Tokenization with regex `[a-z]+(?:'[a-z]+)?` — **no stopword removal** (preserves context windows for Word2Vec)
- Bigram detection (e.g., `trade_war`, `retaliatory_tariffs`) via Gensim `Phrases`
- Metadata extraction: article ID, `versionCreated` timestamp, countries from subject codes

```python
# Two-pass pipeline: learn phrases → apply to clean corpus
RE_HTML     = re.compile(r"<[^>]+>")
RE_CREDITS  = re.compile(r"\b(Reporting by|Writing by|Editing by).*$",
                         flags=re.IGNORECASE | re.MULTILINE)
TOKEN_RE    = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Pass 1 — learn bigrams
phrases = Phrases(sentences=iter_docs(...), min_count=20, threshold=10.0, delimiter="_")
phraser = Phraser(phrases)

# Pass 2 — clean, tokenize, apply bigrams, write gzip output
with gzip.open(out_tokens, "wt") as f_tok, gzip.open(out_meta, "wt") as f_meta:
    for toks, meta in iter_docs_with_meta(...):
        toks2 = phraser[toks]
        f_tok.write(" ".join(toks2) + "\n")
        f_meta.write(json.dumps({"doc_id": n_docs, **meta}) + "\n")
```

**Output:** ~2.5M+ articles per year in `rtrs_YYYY_clean.txt.gz` + matching metadata JSONL.

---

## 2. Word2Vec Training

**Script:** `scripts/training/word2vec & GTM/train_w2v_all.py`

A CBOW Word2Vec model is trained on the full cleaned corpus. Documents are streamed from disk to avoid loading the entire corpus into RAM.

**Hyperparameters:**

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
    """Memory-efficient streamed corpus."""
    def __iter__(self):
        for p in self.paths:
            with gzip.open(p, "rt") as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) >= self.min_len:
                        yield tokens

model = Word2Vec(vector_size=64, window=18, min_count=20,
                 negative=10, sample=1e-5, sg=0, epochs=50, workers=8, seed=42)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=50)
```

**Output:** `.pkl` dictionary mapping ~100k+ vocabulary words to 64-dimensional vectors.

---

## 3. Guided Topic Modeling — GTM-6 (New Final)

**Script:** `scripts/training/word2vec & GTM/GTM_8.py`
**Output:** `GTM_different_versions/GTM_new_final_results/`

GTM expands six geoeconomic sub-topics iteratively from seed words in the embedding space. This is the **final model**, consolidating the earlier 8-topic structure. The key change: Trade War, Tariffs, and Protectionism were found to cluster tightly in the embedding space and have been merged into a single **Trade Coercion** topic, eliminating fragmentation and improving semantic coverage.

### The 6 sub-topics (new final)

| # | Topic | Positive seeds | Negative seeds |
|---|---|---|---|
| 1 | **Sanctions** | `economic_sanctions`, `targeted_sanctions` | `sanctions_relief`, `sanctions_waiver` |
| 2 | **Trade Coercion** | `trade_war`, `retaliatory_tariffs` | `trade_deal`, `trade_pact` |
| 3 | **Export Controls** | `export_ban`, `entity_list` | `export_license`, `export_licenses` |
| 4 | **Financial Coercion** | `asset_freeze`, `secondary_sanctions` | `debt_relief` |
| 5 | **Retaliation** | `retaliation`, `countermeasures` | `concessions`, `goodwill_gesture` |
| 6 | **Embargo** | `trade_embargo`, `oil_embargo` | `lift_embargo`, `lifting_sanctions` |

**Design rationale per topic:**
- **Trade Coercion**: `retaliatory_tariffs` as second seed (instead of `trade_conflict`) — directly anchors the tariff-as-weapon dimension alongside the broader trade war framing.
- **Export Controls**: `entity_list` replaces the generic `export_restriction` seed — captures the US BIS Entity List and semiconductor technology controls that dominate post-2018 coverage.
- **Financial Coercion**: `secondary_sanctions` as second seed — captures the extraterritorial financial weapon, sovereign-level asset freezes and dollar weaponisation.
- **Embargo**: negative seed `lifting_sanctions` added — pushes away from sanctions-relief language that bled into this topic in earlier versions.

### GTM algorithm

```python
class GTM:
    def run(self, params, pos_seed, neg_seed, ...):
        # --- Candidate pool via FAISS k-NN ---
        _, sim_idx = self.index.search(xq, params['k-similar'])   # k=5000

        # --- Iterative subspace expansion ---
        X, C = A.copy(), A.copy()   # topic subspace matrix
        while run:
            B      = np.linalg.inv(X.T @ X) @ X.T @ V
            alpha  = np.arctan(norm_orth / norm_proj)   # angular distance
            new_word = V_buckets.index[alpha.argmin()]   # closest word to subspace
            self.topic.append(new_word)
            # Update subspace with conjugate gradient optimization
            result  = optimize.minimize(self.func, [0]*X.shape[1], method="CG", ...)
            X       = self.UnitColumns(self.X_new)
            weights = weights * (1 + gravity)   # gravity down-weights seeds over time
```

**Parameters:** `cluster_size=100`, `gravity=1.5`, `alpha_max=2.0 rad`, `k-similar=5000`

Each topic produces 100 words. With 6 topics and cross-topic overlap removed, the final dictionary covers the geoeconomic pressure space across all sub-dimensions.

---

## 4. Dictionary Construction

**Script:** `scripts/training/dict & score/build_dictionary_8.py`

GTM outputs one CSV per sub-topic with word importance weights (projection norms onto the topic subspace). These are consolidated using local normalization per topic and global averaging across Q=6 topics (Dangl & Salbrechter 2023, Eq. 10):

```python
word_weight_sums = defaultdict(float)

for file_name in csv_files:                     # one CSV per sub-topic
    df      = pd.read_csv(file_path, index_col=0)
    max_val = df['weight'].max()
    df['weight'] = df['weight'] / max_val       # local normalize → [0, 1]
    for word, row in df.iterrows():
        word_weight_sums[str(word).strip()] += float(row['weight'])

# Global weight = (1/Q) × Σ local_normalized_weights
final_list = [
    {'word': word, 'weight': total / args.num_topics}
    for word, total in word_weight_sums.items()
]
```

Words appearing across multiple sub-topics receive proportionally higher global weights.

**Output:** `DICTIONARY/geoeconomic_dictionary_new.csv`

---

## 5. GEP Index — MIN-K Threshold Construction (New Final)

**Script:** `scripts/training/dict_score/score_daily_index_new.py`
**Final index:** `INDEX/index_new_final/MIN2/`

### Construction philosophy

The final index moves away from continuous article scoring toward a **threshold-based article classification**. Rather than averaging weighted keyword frequencies, the index measures the **share of daily articles that are meaningfully about geoeconomic pressure**:

$$\text{GEP}_m = \frac{n_{\text{GEP articles},\, m}}{n_{\text{articles},\, m}}$$

An article is classified as a **GEP article** if it matches at least $K$ distinct dictionary keywords:

$$\mathbf{1}[\text{GEP article}_i] = \mathbf{1}\!\left[\left|\{w \in \text{article}_i : w \in \mathcal{D}\}\right| \geq K\right]$$

The threshold $K$ filters out articles that happen to mention a single geoeconomic keyword in passing (e.g., the word "tariff" in an unrelated story). Requiring co-occurrence of at least $K=2$ keywords substantially reduces noise.

### Why this construction is better

| Property | Score-based (earlier versions) | MIN-K threshold (final) |
|---|---|---|
| Scale | Weighted freq (~10⁻⁴) | **Proportion [0,1]** — interpretable as % of news |
| Stationarity | Non-stationary in levels; needed first-differencing | **Stationary in levels** (bounded ratio — ADF confirmed) |
| Regression | Required ΔGEP_z, discarding level information | Uses **GEP_z directly in levels** |
| Noise sensitivity | Single high-freq keyword inflates score | Co-occurrence requirement filters incidental mentions |
| Economic meaning | Intensity of GEP language per article | Share of news agenda devoted to geoeconomic pressure |

### Robustness across thresholds (MIN1, MIN2, MIN3, MIN4)

| Variant | Threshold K | Location |
|---|---|---|
| MIN1 | ≥ 1 keyword | `INDEX/index_new_final/robustness/MIN1/` |
| **MIN2** | **≥ 2 keywords** | **`INDEX/index_new_final/MIN2/` (FINAL)** |
| MIN3 | ≥ 3 keywords | `INDEX/index_new_final/robustness/MIN3/` |
| MIN4 | ≥ 4 keywords | `INDEX/index_new_final/robustness/MIN4/` |

The MIN2 index is chosen as the final specification: tight enough to filter noise, broad enough to capture all relevant geoeconomic episodes.

### Key index properties (MIN2, monthly)

The monthly MIN2 series (`GEP_Monthly_Robust_min2.csv`) has columns:
- `month` — first day of month
- `n_articles` — total articles in month
- `n_gep_articles` — articles matching ≥ 2 dictionary keywords
- `GEP_monthly` = `n_gep_articles / n_articles`

Notable values: 9/11 (2001-09): 0.129, Iraq War (2003-02): 0.122, US–China tariffs (2018-05): 0.107, Russia invades Ukraine (2022-03): 0.139, Liberation Day (2025-04): 0.197.

---

## 6. Robustness Check vs. GPR

**Script:** `INDEX/index_new_final/MIN2/gep_vs_gpr_robust_min2.py`

The MIN2 GEP index is validated against the Caldara & Iacoviello (2022) GPR index. Both series are z-scored before comparison.

**Finding:** Low but positive correlation confirms GEP and GPR measure related but distinct phenomena. GEP captures geoeconomic coercion (trade pressure, financial weaponisation, export controls); GPR captures military threat and geopolitical violence. They co-move most during extreme joint escalation events (Ukraine invasion, Liberation Day tariffs).

---

## 7. Return Predictability Regressions

**Script:** `INDEX/index_new_final/MIN2/return_predictability_min2.R`

### Setup

Monthly and quarterly OLS regressions of S&P 500 log returns on MIN2 GEP_z (z-scored levels). Four specifications with Newey-West HAC standard errors:

| Spec | Formula | Purpose |
|---|---|---|
| m1 | `log_ret ~ GEP_z` | Univariate baseline |
| m2 | `log_ret ~ GEP_z + GPR_z` | + GPR control |
| m3 | `log_ret ~ GEP_z + GPR_z + MktRF + SMB + HML` | + Fama-French 3 factors |
| m4 | `log_ret ~ GEP_z + GEP_z_lag1 + GPR_z + GPR_z_lag1 + MktRF + SMB + HML` | + Lag structure |
| m_rob | `log_ret ~ dGEP_z + GPR_z` | Robustness: first differences |

Run at h=0 (contemporaneous) and h=1 (next-period). Subperiods: Full sample (1996–2025), Pre-GFC (1996–2007), GFC & aftermath (2008–2011), Post-GFC (2012–2021), Russia–Ukraine war (2022–2023), 2025 Liberation Day shock.

### Stationarity

ADF tests confirm GEP_z (MIN2 levels) is stationary — a direct consequence of the bounded [0,1] ratio construction. This validates using GEP_z in levels rather than differences.

### Results

**Full sample:** GEP_z not significant in m1/m2 at monthly frequency. After controlling for FF3 (m3), GEP_z is marginally positive — the effect is present but regime-dependent.

**Pre-GFC (1996–2007) — strongest signal:**
- Monthly m1: GEP_z = −0.0094\* (p=0.025). Higher geoeconomic pressure → lower contemporaneous returns.
- Monthly m_rob (dGEP_z): −0.017\* (p=0.021). Levels and differences agree.
- Quarterly m1: GEP_z = −0.028\* (p=0.022). Quarterly aggregation sharpens the signal.
- **Interpretation:** In the pre-crisis era, geoeconomic news drove immediate fear-based selloffs. Once FF3 is added, the effect disappears — it operated through broad market repricing.

**Post-GFC (2012–2021) — sign reversal:**
- Monthly m3: GEP_z = +0.0008\*\* (p=0.004). Positive and significant *after* controlling for market factors.
- GPR_z in m2: −0.013\* (p=0.039). Higher general geopolitical risk reduces returns in this period.
- **Interpretation:** In the low-volatility post-crisis era, markets priced in geoeconomic risk as a compensated factor. High-GEP months earned excess returns, consistent with a risk premium story. GEP and GPR capture distinct channels (positive vs. negative) in this regime.

**Russia–Ukraine war (2022–2023):**
- Monthly m2: GEP_z = −0.047\* (p=0.021). Strong negative contemporaneous effect. The fear-driven regime returns.
- Note: N=23, so directional rather than definitive.

**Predictability (h=1):** No specification produces significant results at h=1 in the full sample or any major subsample. **GEP is a coincident indicator, not a leading one** — it measures the market's real-time response to geoeconomic events, not future returns.

**Quarterly results:** Broadly confirm monthly. Full sample quarterly m4: GEP_z_lag1 = +0.0024\* (p=0.044) — a weak quarterly predictability signal in the lagged structure.

### Summary table of significant GEP effects

| Period | Freq | h | Spec | Estimate | Significance | Story |
|---|---|---|---|---|---|---|
| Pre-GFC | Monthly | 0 | m1 | −0.009 | \* | Fear-driven selloff |
| Pre-GFC | Monthly | 0 | m_rob | −0.017 | \* | Confirms levels result |
| Pre-GFC | Quarterly | 0 | m1 | −0.028 | \* | Stronger at quarterly |
| Post-GFC | Monthly | 0 | m3 | +0.0008 | \*\* | Geopolitical risk premium |
| Russia–Ukraine | Monthly | 0 | m2 | −0.047 | \* | Fear regime returns |
| Full sample | Quarterly | 0 | m4 (lag) | +0.0024 | \* | Weak quarterly lag |

---

## Pipeline Summary

```
Reuters RTRS (.gz JSON)
        │
        ▼
[1. Cleaning] ──────────► rtrs_YYYY_clean.txt.gz  +  rtrs_YYYY_meta.jsonl.gz
  • Bigram learning
  • Normalization & tokenization
        │
        ▼
[2. Word2Vec] ──────────► embeddings.pkl  (~100k words × 64 dims)
  • CBOW, window=18, 50 epochs
        │
        ▼
[3. GTM × 6 topics] ────► topic_Sanctions.csv, topic_Trade_Coercion.csv, ...
  • FINAL model: 6 topics (Trade Coercion consolidates Trade War + Tariffs + Protectionism)
  • FAISS k-NN + iterative subspace expansion
        │
        ▼
[4. Dictionary] ─────────► geoeconomic_dictionary_new.csv
  • Local normalize per topic + global average (Q=6, Eq. 10)
        │
        ▼
[5. MIN-K Index] ─────────► GEP_Daily_Robust_min2.csv
  • Article classified as GEP if ≥K dictionary keywords matched   GEP_Monthly_Robust_min2.csv
  • GEP_monthly = n_gep_articles / n_articles
  • Stationary bounded ratio — no differencing required
  • Robustness: MIN1, MIN3, MIN4
        │
        ▼
[6. Regressions] ─────────► return_predictability_min2.R
  • Monthly + quarterly, h=0 and h=1
  • Full sample + 5 subperiods
  • Key: negative Pre-GFC, positive Post-GFC (risk premium), no h=1 predictability
```
