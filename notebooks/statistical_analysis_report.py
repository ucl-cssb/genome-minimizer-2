#!/usr/bin/env python3
"""
Gene-enrichment report for VAE-designed genomes (script form of statistical_analysis.py).

Hypergeometric enrichment of core-genome and essential genes in each cohort's
gene lists, against the full pangenome background. "Enriched" means a cohort
carries more of the target genes than expected if its genes were drawn at random
from the pangenome. Prints a per-variant summary and writes the full per-sample
results + the summary to CSV.

OUTPUT
    notebooks/figures/enrichment_per_sample.csv
    notebooks/figures/enrichment_summary.csv

DATA SOURCE
    data/F4_complete_presence_absence.csv   pangenome presence/absence matrix
    data/essential_genes.csv                literature essential genes
    evaluation/data/<variant>/<variant>_gene_lists_with_essentials.npy
        per-cohort gene lists (produced by genome_minimizer_2.sampling)

Run:  uv run python notebooks/statistical_analysis_report.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import hypergeom

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EVAL_DATA = REPO_ROOT / "evaluation" / "data"
OUT_DIR = REPO_ROOT / "notebooks" / "figures"

VARIANTS = ["real", "v0", "v1", "v2", "v3", "v4", "v4_opt", "random"]
CORE_THRESHOLD = 0.95
N_REAL_STRAINS = 200   # real cohort = this many random pangenome strains
REAL_SEED = 42         # matches the notebook / systems_analysis.py real-strain draw


def load_pangenome():
    path = DATA_DIR / "F4_complete_presence_absence.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path, index_col=0).drop(index="Lineage", errors="ignore")


def load_gene_lists(pangenome_df):
    """Return {variant: list[list[gene]]} for every cohort in VARIANTS.

    The 'real' cohort is N_REAL_STRAINS random pangenome strains (seed REAL_SEED);
    the rest are the sampled gene lists on disk.
    """
    rng = np.random.default_rng(REAL_SEED)
    strain_cols = pangenome_df.columns.tolist()
    chosen = rng.choice(len(strain_cols), size=N_REAL_STRAINS, replace=False)
    chosen_names = [strain_cols[i] for i in sorted(chosen)]
    real_df = pangenome_df[chosen_names]
    real_gene_lists = [
        real_df.index[real_df[col].astype(int) == 1].tolist()
        for col in real_df.columns
    ]

    def _load(variant):
        if variant == "random":
            p = EVAL_DATA / "random" / "random_gene_lists_with_essentials.npy"
        else:
            p = EVAL_DATA / variant / f"{variant}_gene_lists_with_essentials.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — generate it with genome_minimizer_2.sampling"
            )
        return [list(x) for x in np.load(p, allow_pickle=True)]

    gene_lists = {v: _load(v) for v in VARIANTS if v != "real"}
    gene_lists["real"] = real_gene_lists
    return gene_lists


def hypergeometric_enrichment(sample_genes, target_genes, population_genes):
    """One-sided hypergeometric test: is `target` over-represented in `sample`?"""
    M = len(population_genes)
    n = len(target_genes & population_genes)
    N = len(sample_genes & population_genes)
    k = len(sample_genes & target_genes)
    p = hypergeom.sf(k - 1, M, n, N)
    exp = N * n / M
    return {
        "observed": k,
        "expected": round(exp, 1),
        "enrichment": round(k / exp, 3) if exp > 0 else 0,
        "p_value": p,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pangenome_df = load_pangenome()
    n_strains = pangenome_df.shape[1]
    pangenome_genes = pangenome_df.index.tolist()
    prevalence = pangenome_df.sum(axis=1) / n_strains
    core_genes = set(prevalence[prevalence >= CORE_THRESHOLD].index.tolist())
    print(f"Pangenome: {n_strains:,} strains, {len(pangenome_genes):,} genes")
    print(f"Core genome (>={CORE_THRESHOLD:.0%} of strains): {len(core_genes):,} genes")

    essential_df = pl.read_csv(DATA_DIR / "essential_genes.csv")
    essential_genes = set(essential_df["# gene"].to_list())
    print(f"Essential genes: {len(essential_genes):,}")

    gene_lists = load_gene_lists(pangenome_df)
    print("\nGene lists loaded:")
    for v in VARIANTS:
        sizes = [len(g) for g in gene_lists[v]]
        print(f"  {v:>7}: n={len(gene_lists[v])}, "
              f"mean={np.mean(sizes):.0f} genes [{min(sizes)}-{max(sizes)}]")

    all_genes = set(pangenome_genes)
    rows = []
    for variant in VARIANTS:
        for i, genes in enumerate(gene_lists[variant]):
            gene_set = set(genes)
            core_res = hypergeometric_enrichment(gene_set, core_genes, all_genes)
            rows.append({"variant": variant, "sample_id": i, "test": "core_genome", **core_res})
            ess_res = hypergeometric_enrichment(gene_set, essential_genes, all_genes)
            rows.append({"variant": variant, "sample_id": i, "test": "essential_genes", **ess_res})

    enrichment_df = pl.DataFrame(rows)
    summary = (
        enrichment_df.group_by(["variant", "test"])
        .agg([
            pl.col("enrichment").mean().round(3).alias("mean_enrichment"),
            pl.col("observed").mean().round(0).alias("mean_observed"),
            (pl.col("p_value") < 0.005).mean().mul(100).round(1).alias("pct_significant"),
        ])
        .sort(["test", "variant"])
    )

    print("\nEnrichment summary (mean per variant x test):")
    with pl.Config(tbl_rows=-1):
        print(summary)

    per_sample_path = OUT_DIR / "enrichment_per_sample.csv"
    summary_path = OUT_DIR / "enrichment_summary.csv"
    enrichment_df.write_csv(per_sample_path)
    summary.write_csv(summary_path)
    print(f"\nwrote {per_sample_path.relative_to(REPO_ROOT)}")
    print(f"wrote {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
