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
| `__marimo__/*.ipynb` | executed **Jupyter clones** of those notebooks, tracked so they render on GitHub without running anything. Regenerate after editing (see below). |
| `figures.py` | a headless script that writes the per-variant training-loss figures. |
| `figures/` | scratch figure/table outputs (gitignored, regenerable). |
| `training_summary/` | curated paper figures + `run_stats.csv` (tracked). |

`__marimo__/` also holds marimo's `*.html` render and `session/` cache, which are
**not** tracked.

## marimo notebooks

| Source | Clone | What it does |
|--------|-------|--------------|
| `systems_analysis.py` | [`__marimo__/systems_analysis.ipynb`](__marimo__/systems_analysis.ipynb) | Two-tier evaluation of designed genomes plus the integrated size–viability frontier. Compares `real`, `random`, `v3`, `v4_opt`. |
| `statistical_analysis.py` | [`__marimo__/statistical_analysis.ipynb`](__marimo__/statistical_analysis.ipynb) | Hypergeometric enrichment of core-genome and essential genes across every cohort. |

Edit interactively, then **recompile the clone** after changes:

```bash
uv run marimo edit notebooks/systems_analysis.py
uv run marimo export ipynb notebooks/systems_analysis.py -o notebooks/__marimo__/systems_analysis.ipynb
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
- `data/kegg/`, `data/fba/iML1515.xml`, `data/gene_prevalence.npy` — KEGG + FBA caches
- `paper_figures/data/` — local W&B export (`paper_figures/extract_wandb.py`), for `figures.py`
- `evaluation/data/<variant>/<variant>_gene_lists_with_essentials.npy` — per-cohort gene lists

The cached generations (per-cohort gene lists and VAE samples) are published on
the HF bucket **[McClain/minimal_genomes](https://huggingface.co/buckets/McClain/minimal_genomes)**
(requires `huggingface_hub>=1.8.0`):

```bash
hf buckets sync hf://buckets/McClain/minimal_genomes/ ./evaluation/data
```

Or regenerate them from the trained checkpoints:

```bash
uv run python -m genome_minimizer_2.sampling vae    --variant v4_opt --num-samples 100 --output evaluation/data/v4_opt
uv run python -m genome_minimizer_2.sampling random --num-samples 100 --output evaluation/data/random
```
