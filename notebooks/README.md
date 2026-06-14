# Notebooks

Analysis and figure generation for the genome-minimizer-2 paper (Cell Systems
revision).

The interactive notebooks are written with **[marimo](https://docs.marimo.io)** —
reactive Python notebooks stored as plain `.py` files (so they diff and review
like code). See the [marimo docs](https://docs.marimo.io) for how to install,
edit, and run them.

## What's where

| Path | What it is |
|------|------------|
| `*.py` with `@app.cell` | marimo notebook **source** (`systems_analysis.py`, `statistical_analysis.py`) — the canonical, runnable form. |
| `figures.py` | a headless script that writes the per-variant training-loss figures. |
| `figures/` | scratch figure/table outputs (gitignored, regenerable). |
| `training_summary/` | curated paper figures + `run_stats.csv` (tracked). |

## marimo notebooks

| Source | What it does |
|--------|--------------|
| `systems_analysis.py` | Two-tier evaluation of designed genomes plus the integrated size–viability frontier. Compares `real`, `random`, `v3`. |
| `statistical_analysis.py` | Hypergeometric enrichment of core-genome and essential genes across every cohort. |

Run or edit interactively:

```bash
uv run marimo edit notebooks/systems_analysis.py
```

## The two-tier evaluation

`systems_analysis.py` is a **single two-tier evaluation**, a deterministic
minutes-not-hours stand-in for the whole-cell-model viability eval:

- **Tier 1 — KEGG module coverage.** Map each genome's genes to KEGG b-numbers
  and score completeness over the ~110 *E. coli* functional modules.
- **Tier 2 — iML1515 FBA.** Knock out the metabolic genes a genome lacks and
  solve for growth on glucose minimal medium.

The tiers are **bundled in one notebook** because the integrated analysis at the
end (the size–viability frontier and chosen operating point) needs both at once.

## Figure scripts

`figures.py` renders the per-variant training-loss figures (train + validation
total loss, Cell-Press styling) from the local W&B export into
`training_summary/<variant>_total_loss.png`:

```bash
uv run python notebooks/figures.py
```

## Data dependencies

These read local data that is **not** shipped in the repo (gitignored):

- `data/F4_complete_presence_absence.csv` — pangenome presence/absence
- `data/essential_genes.csv` — literature essential genes
- `evaluation/data/<variant>/<variant>_gene_lists_with_essentials.npy` — per-cohort gene lists
- `data/kegg/` — KEGG module cache; fetched automatically from the KEGG REST API on first run
- `data/fba/iML1515.xml` — the iML1515 metabolic model; download once from [BiGG](http://bigg.ucsd.edu/models/iML1515)
- `paper_figures/data/` — local W&B export (`paper_figures/extract_wandb.py`), for `figures.py`

(`data/gene_prevalence.npy` is written by `systems_analysis.py` on first run as a cache — not a prerequisite.)

The training data and the per-cohort gene lists are downloaded by the top-level
`setup-data` mode (see the [main README](../README.md#get-the-data)):

```bash
uv run python main.py --mode setup-data
```

Or regenerate the samples from the trained checkpoints on HuggingFace:

```bash
uv run python -m genome_minimizer_2.sampling vae    --variant v3 --num-samples 100 --output evaluation/data/v3
uv run python -m genome_minimizer_2.sampling random --num-samples 100 --output evaluation/data/random
```
