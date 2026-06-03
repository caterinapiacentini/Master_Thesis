# Master Thesis — Caterina Piacentini
## The Geoeconomic Impact on Stock Market Indices
### WU Vienna University of Economics and Business

---

## Overview

This thesis constructs a novel **Geoeconomic Pressure (GEP) Index** from Reuters newswire text (1996–2025) using a fully data-driven NLP pipeline. The index quantifies the monthly share of news coverage devoted to geoeconomic stress — trade coercion, sanctions, export controls, financial coercion, retaliation, and embargoes — and is used to study its impact on financial markets.

The methodology follows and extends Dangl & Salbrechter (2023), combining **Word2Vec embeddings**, **Guided Topic Modeling (GTM)**, and a **threshold-based article classification** approach.

---

## Repository Structure

```
Master_Thesis/
│
├── code/                          # Cluster pipeline scripts
│   ├── cleaning/
│   │   ├── clean_world.py         # World corpus cleaning (Reuters → tokenised .txt.gz)
│   │   └── filter_region.py       # Filter world corpus → US (or other region)
│   ├── training/
│   │   ├── word2vec/
│   │   │   └── train_w2v.py       # Word2Vec CBOW training on cleaned corpus
│   │   └── gtm/
│   │       └── gtm.py             # Guided Topic Model (6 topics)
│   └── index/
│       ├── build_index.py         # GEP index construction (MIN-K threshold)
│       ├── build_index_country.py # Country-level GEP index
│       └── normalize_index.py     # EPU-style normalisation (mean = 100)
│
├── slurm/                         # SLURM job scripts (WU cluster)
│   ├── cleaning/
│   │   ├── clean_world.slurm
│   │   └── filter_us.slurm
│   ├── training/
│   │   ├── train_word2vec.slurm
│   │   ├── run_gtm.slurm          # Main GTM (baseline seeds)
│   │   └── run_gtm_v2.slurm       # GTM robustness (alternative seeds)
│   └── index/
│       ├── build_index_us.slurm
│       └── build_index_countries.slurm
│
├── data/                          # GEP index output (CSV)
│   ├── gep_us/
│   │   ├── GEP_Daily_Robust_min2.csv    # Official daily index (baseline, MIN-2)
│   │   └── GEP_Monthly_Robust_min2.csv  # Official monthly index (baseline, MIN-2)
│   ├── robustness/
│   │   ├── GEP_Daily_min1.csv           # MIN-1 threshold (broadest)
│   │   ├── GEP_Monthly_min1.csv
│   │   ├── GEP_Daily_Robust_min3.csv    # MIN-3 threshold (strict)
│   │   ├── GEP_Monthly_Robust_min3.csv
│   │   ├── GEP_Daily_Robust_min4.csv    # MIN-4 threshold (most conservative)
│   │   ├── GEP_Monthly_Robust_min4.csv
│   │   ├── GEP_Daily_gtm_v2.csv         # GTM v2 — alternative seed words
│   │   └── GEP_Monthly_gtm_v2.csv
│   └── countries/
│       ├── GEP_Monthly_JAPAN_min2.csv
│       ├── GEP_Monthly_UK_min2.csv
│       ├── GEP_Monthly_GERMANY_min2.csv
│       ├── GEP_Monthly_RUSSIA_min2.csv
│       ├── GEP_Monthly_IRAN_min2.csv
│       └── GEP_Monthly_CHINA_min2.csv
│
├── analysis/                      # Local analysis scripts (run on Mac)
│   ├── plot_index.py              # Main index plots + descriptive statistics
│   ├── plot_comparisons.py        # GEP vs GPR / EPU / VIX / S&P 500 + regressions
│   ├── plot_industries.py         # FF49 industry regressions + JKP factor regressions
│   ├── plot_robustness.py         # Robustness checks (min-1/3/4 + GTM v2)
│   ├── plot_countries.py          # Country-level GEP plots (6-panel + individual)
│   └── output/                    # Generated plots saved here (gitignored)
│
├── Archive/                       # Superseded scripts and old index versions
└── Final_Thesis_Clean/Archive/    # Previous analysis scripts (local Mac paths)
```

> **Note:** Raw corpus data, Word2Vec models, and GTM topic outputs live on the WU cluster at `~/Final_Thesis_Clean/` and are not tracked in this repository. Only the final index CSVs (in `data/`) are committed.

---

## Pipeline

### Step 1 — Cleaning

**Script:** `code/cleaning/clean_world.py` → `code/cleaning/filter_region.py`
**SLURM:** `slurm/cleaning/`

The world Reuters corpus (1996–2025) is cleaned and tokenised:
- Lowercasing, punctuation removal, boilerplate/diary removal
- Bigram detection and concatenation (`trade_war`, `economic_sanctions`, …)
- Output: one `.txt.gz` per year, one article per line, plus aligned metadata `.jsonl.gz`

The world corpus is then filtered to US-centric articles via `filter_region.py`.

---

### Step 2 — Word2Vec Training

**Script:** `code/training/word2vec/train_w2v.py`
**SLURM:** `slurm/training/train_word2vec.slurm`

Word2Vec CBOW trained on the US cleaned corpus:

| Hyperparameter | Value |
|---|---|
| Architecture | CBOW (`sg=0`) |
| Embedding dimension | 64 |
| Context window | 18 |
| Min word count | 20 |
| Negative samples | 10 |
| Training epochs | 100 |

---

### Step 3 — Guided Topic Modeling (GTM-6)

**Script:** `code/training/gtm/gtm.py`
**SLURM:** `slurm/training/run_gtm.slurm` (baseline) · `run_gtm_v2.slurm` (robustness)

The GTM algorithm expands six geoeconomic sub-topics from seed words in the 64-dimensional embedding space (cluster size 100, gravity 1.5).

#### The 6 Topics

| # | Topic | Positive seeds | Negative seeds |
|---|---|---|---|
| 1 | **Sanctions** | `economic_sanctions`, `targeted_sanctions` | `sanctions_relief`, `sanctions_waiver` |
| 2 | **Trade Coercion** | `trade_war`, `retaliatory_tariffs` | `trade_deal`, `trade_pact` |
| 3 | **Export Controls** | `export_ban`, `entity_list` | `export_license`, `export_licenses` |
| 4 | **Financial Coercion** | `asset_freeze`, `secondary_sanctions` | `debt_relief` |
| 5 | **Embargo** | `trade_embargo`, `oil_embargo` | `lift_embargo`, `lifting_sanctions` |
| 6 | **Retaliation** | `retaliation`, `countermeasures` | `concessions`, `goodwill_gesture` |

**GTM v2 (robustness):** same 6 topics, alternative seed words emphasising ratcheting sanctions, anti-dumping, semiconductor denial, extraterritorial sanctions, energy weaponisation, and diplomatic retaliation.

---

### Step 4 — Index Construction

**Script:** `code/index/build_index.py`
**SLURM:** `slurm/index/build_index_us.slurm`

The GEP index measures the **daily share of articles classified as geoeconomic**:

$$\text{GEP}_d = \frac{n_{\text{GEP articles},\, d}}{n_{\text{articles},\, d}}$$

An article is a GEP article if it contains keywords from at least $K$ distinct topic categories:

$$\mathbf{1}[\text{GEP}_i] = \mathbf{1}\!\left[\left|\{q : \exists\, w \in \text{topic}_q \cap \text{article}_i\}\right| \geq K\right]$$

#### Why threshold-based

| | Score-based (old) | MIN-K threshold (new) |
|---|---|---|
| Scale | Raw weighted freq (~10⁻⁴) | Proportion ∈ [0,1], interpretable |
| Stationarity | Requires differencing | Stationary in levels (ADF confirmed) |
| Regression | Must use ΔGEP | Can use GEP levels directly |

#### Robustness variants

| File | K | Interpretation |
|---|---|---|
| `GEP_*_min1.csv` | ≥ 1 topic | Broadest — any GEP mention |
| **`GEP_*_Robust_min2.csv`** | **≥ 2 topics** | **Baseline (official index)** |
| `GEP_*_Robust_min3.csv` | ≥ 3 topics | Stricter |
| `GEP_*_Robust_min4.csv` | ≥ 4 topics | Most conservative |
| `GEP_*_gtm_v2.csv` | ≥ 2 topics | Alternative GTM seed words |

Country-level indices (Japan, UK, Germany, Russia, Iran, China) are built with `build_index_country.py`, adding a country-name filter on top of the topic threshold.

---

## Analysis Scripts

All scripts in `analysis/` use paths relative to the repo root (`../data/`) — no hardcoded local paths. They run locally (Mac) using the CSVs in `data/`.

| Script | What it produces |
|---|---|
| `plot_index.py` | Normalised monthly + daily index plots, 2025 zoom, rolling vol, distribution, ACF/PACF, annual stats, heatmap |
| `plot_comparisons.py` | GEP vs GPR, EPU, VIX, S&P 500 — z-score overlays, rolling correlations, cross-correlations, HAC regressions |
| `plot_industries.py` | FF49 industry exposures (levels + first-diff, with/without GPR); JKP factor exposures |
| `plot_robustness.py` | Individual variant plots + overlay comparison + GTM v2 |
| `plot_countries.py` | 3×2 country panel + individual country plots |

External data needed in `data/external/`: GPR xls (Caldara & Iacoviello), EPU csv/xlsx (Baker-Bloom-Davis), JKP factor csvs. FF3/FF49 and VIX/S&P 500 download automatically.

---

## Key Results

**1. GEP is a contemporaneous indicator, not a predictor.** Predictive regressions at h=1 are uniformly insignificant. GEP reflects market reactions to geoeconomic events.

**2. The sign of the GEP effect flips across regimes:**

| Period | Effect on returns | Interpretation |
|---|---|---|
| Pre-GFC (1996–2007) | Negative* | Geoeconomic news hurts markets |
| Post-GFC (2012–2021) | Positive** (with FF3) | Geopolitical risk premium |
| Russia–Ukraine (2022–2023) | Negative* | Return to fear-driven reaction |

**3. GEP and GPR are complementary.** Low but positive correlation — GEP captures economic coercion, GPR captures military/geopolitical threat.

**4. Industry exposure is heterogeneous.** Defense, Oil, and Gold load positively on GEP; Consumer Goods, Retail, and Financials load negatively.

---

## Cluster Workflow (WU HPC)

```bash
# Push → cluster
git add . && git commit -m "..." && git push
ssh wucluster "git -C ~/Master_Thesis pull"

# Run pipeline (submit SLURM jobs in order)
sbatch slurm/cleaning/clean_world.slurm
sbatch --dependency=afterok:<job_id> slurm/cleaning/filter_us.slurm
sbatch --dependency=afterok:<job_id> slurm/training/train_word2vec.slurm
sbatch --dependency=afterok:<job_id> slurm/training/run_gtm.slurm
sbatch --dependency=afterok:<job_id> slurm/index/build_index_us.slurm

# Pull results → local
scp wucluster:~/Final_Thesis_Clean/output/GEP_Index_US/GEP_*_Robust_min2.csv data/gep_us/
git add data/ && git commit -m "update index" && git push
```

---

## References

- Dangl, T., Halling, M. & Salbrechter, S. (2025). *The Price of Physical Climate Risk Estimated from Public News via Guided Topic Modeling*.
- Dangl, T. & Salbrechter, S. (2023). *Guided Topic Modeling with Word2Vec: A Technical Note*.
- Caldara, D. & Iacoviello, M. (2022). *Measuring Geopolitical Risk*. American Economic Review, 112(4), 1194–1225.
- Baker, S., Bloom, N. & Davis, S. (2016). *Measuring Economic Policy Uncertainty*. Quarterly Journal of Economics, 131(4), 1593–1636.
