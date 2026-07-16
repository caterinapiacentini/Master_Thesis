# Master Thesis — Caterina Piacentini

## The Geoeconomic Impact on Stock Market Indices
WU Vienna University of Economics and Business

This thesis builds a **Geoeconomic Pressure (GEP) Index** from Reuters newswire
text (1996–2025) and studies how it relates to financial markets. The index
tracks the monthly/daily share of news coverage about geoeconomic stress —
trade coercion, sanctions, export controls, financial coercion, embargoes,
retaliation.

The method extends Dangl & Salbrechter (2023): Word2Vec embeddings + Guided
Topic Modeling (GTM) + a threshold-based article classification.

## Repository layout

```
code/       cluster pipeline: cleaning -> Word2Vec -> GTM -> index construction
slurm/      SLURM job scripts that run code/ on the WU cluster
data/       final index CSVs (raw corpus and trained models live on the
            cluster, not in this repo — see "Data" below)
analysis/   local scripts (Mac) that turn the CSVs into plots and regressions
Archive/    superseded scripts, kept for reference
```

## Pipeline

1. **Cleaning** — `code/cleaning/clean_world.py`, `filter_region.py`
   Reuters corpus cleaned and tokenised, then filtered to a region (US, Europe, ...).

2. **Word2Vec** — `code/training/word2vec/train_w2v.py`
   CBOW, 64-dim, window 18, 100 epochs, trained on the cleaned US corpus.

3. **Guided Topic Modeling** — `code/training/gtm/gtm.py`
   Expands 6 geoeconomic sub-topics (sanctions, trade coercion, export
   controls, financial coercion, embargo, retaliation) from seed words in
   the embedding space.

4. **Index construction** — `code/index/build_index.py`, `build_index_country.py`
   `GEP_d` = share of articles that hit words from ≥K distinct topics on day d.
   Baseline uses K=2; K=1,3,4 and an alternative seed set (GTM v2) are
   robustness checks. Country versions add a name filter (Japan, UK,
   Germany, Russia, Iran, China).

Each step runs on the cluster via the matching script in `slurm/`.

## Data

Raw corpus, Word2Vec models, and GTM topic outputs live on the WU cluster
at `~/Final_Thesis_Clean/` and aren't tracked here — only the final index
CSVs in `data/` are committed. The raw input is an external RTRS (Reuters)
newswire archive; each cluster script documents in its own docstring what
it reads and writes.

## Analysis

Everything in `analysis/` runs locally against the CSVs in `data/`.

| Script | What it produces |
|---|---|
| `plot_index.py` | Main index plots (monthly, daily, 2025 zoom) + descriptive stats |
| `plot_countries.py` | Country-level index plots |
| `plot_comparisons.py` | GEP vs GPR / EPU / VIX / S&P 500 |
| `plot_industries.py` | FF49 industry and JKP factor regressions |
| `plot_robustness.py` | min-1/3/4 and GTM v2 robustness checks |
| `fetch_data.py` | Downloads/caches VIX, S&P 500, Fama-French, JKP data |

`data/external/` needs the GPR (Caldara & Iacoviello) and EPU
(Baker-Bloom-Davis) files added manually; everything else downloads
automatically via `fetch_data.py`.

## Findings

GEP looks like a contemporaneous indicator rather than a predictor —
predictive regressions at h=1 are consistently insignificant. Its
relationship with returns flips sign across regimes (negative pre-GFC,
positive post-GFC, negative again during Russia–Ukraine), and it's only
weakly correlated with GPR, since GEP captures economic coercion while GPR
captures military/geopolitical threat. Industry exposure is uneven: Defense,
Oil and Gold load positively on GEP; Consumer Goods, Retail and Financials
load negatively.

## Cluster workflow

```bash
# push -> cluster
git add . && git commit -m "..." && git push
ssh wucluster "git -C ~/Master_Thesis pull"

# run pipeline (submit in order)
sbatch slurm/cleaning/clean_world.slurm
sbatch --dependency=afterok:<job_id> slurm/cleaning/filter_us.slurm
sbatch --dependency=afterok:<job_id> slurm/training/train_word2vec.slurm
sbatch --dependency=afterok:<job_id> slurm/training/run_gtm.slurm
sbatch --dependency=afterok:<job_id> slurm/index/build_index_us.slurm

# pull results -> local
scp wucluster:~/Final_Thesis_Clean/output/GEP_Index_US/GEP_*_Robust_min2.csv data/gep_us/
git add data/ && git commit -m "update index" && git push
```

## References

- Dangl, T., Halling, M. & Salbrechter, S. (2025). *The Price of Physical Climate Risk Estimated from Public News via Guided Topic Modeling*.
- Dangl, T. & Salbrechter, S. (2023). *Guided Topic Modeling with Word2Vec: A Technical Note*.
- Caldara, D. & Iacoviello, M. (2022). *Measuring Geopolitical Risk*. American Economic Review, 112(4), 1194–1225.
- Baker, S., Bloom, N. & Davis, S. (2016). *Measuring Economic Policy Uncertainty*. Quarterly Journal of Economics, 131(4), 1593–1636.
