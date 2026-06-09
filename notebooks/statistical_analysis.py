import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import numpy as np
    import re
    from scipy.stats import hypergeom
    from pathlib import Path
    import altair as alt

    return Path, hypergeom, mo, np, pd, pl, re


@app.cell
def _(mo):
    mo.md("""
    # Genome Minimizer Analysis — Cell Systems Revision

    ## Table of Contents
    1. [Gene Enrichment](#enrichment) — hypergeometric tests for core/essential genes
    """)
    return


@app.cell
def _(Path):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    EVAL_DATA = PROJECT_ROOT / "evaluation" / "data"
    return DATA_DIR, EVAL_DATA


@app.cell
def _(DATA_DIR, mo, pd, re):
    # Pangenome matrix: genes (rows) x strains (columns)
    pangenome_df = pd.read_csv(DATA_DIR / "F4_complete_presence_absence.csv", index_col=0)
    pangenome_df = pangenome_df.drop(index="Lineage", errors="ignore")
    pangenome_genes = pangenome_df.index.tolist()
    n_strains = pangenome_df.shape[1]

    # Core genome: genes present in ≥95% of strains
    prevalence = pangenome_df.sum(axis=1) / n_strains
    core_genes = set(prevalence[prevalence >= 0.95].index.tolist())

    def normalize_gene(_name: str):
        return re.sub(r"_\d+$", "", _name).lower()

    mo.md(f"""
    **Pangenome:** {n_strains:,} strains, {len(pangenome_genes):,} genes  
    **Core genome (≥95%):** {len(core_genes):,} genes
    """)
    return core_genes, pangenome_df, pangenome_genes


@app.cell
def _(DATA_DIR, mo, pl):
    essential_df = pl.read_csv(DATA_DIR / "essential_genes.csv")
    essential_genes = set(essential_df["# gene"].to_list())
    mo.md(f"**Essential genes:** {len(essential_genes):,}")
    return (essential_genes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Random Genomes** In order to test if the VAE was actually learning anything we compare to a random baseline. This is implemented in `genome_minimizer_2.sampling`. What is does it calculate the core genome (genes in >95% of the dataset) keeps all of those, then samples the remaining genes until we have as many genes as we need for comparision (generally ~3300 which is how many our best VAE produced).
    """)
    return


@app.cell
def _(EVAL_DATA, mo, np, pangenome_df):
    # Load real strains (200 random from pangenome, matching systems_analysis.py)
    _rng = np.random.default_rng(42)
    _strain_cols = pangenome_df.columns.tolist()
    _chosen = _rng.choice(len(_strain_cols), size=200, replace=False)
    _chosen_names = [_strain_cols[_i] for _i in sorted(_chosen)]
    _real_df = pangenome_df[_chosen_names]
    real_gene_lists = []
    for _col in _real_df.columns:
        _present = _real_df.index[_real_df[_col].astype(int) == 1].tolist()
        real_gene_lists.append(_present)

    # Load model gene lists
    def load_gene_lists(_variant: str):
        if _variant == "random":
            _path = EVAL_DATA / "random" / "random_gene_lists_with_essentials.npy"
        else:
            _path = EVAL_DATA / _variant / f"{_variant}_gene_lists_with_essentials.npy"
        return [list(x) for x in np.load(_path, allow_pickle=True)]

    variants = ["real", "v0", "v1", "v2", "v3", "v4", "v4_opt", "random"]
    gene_lists = {_v: load_gene_lists(_v) for _v in variants if _v != "real"}
    gene_lists["real"] = real_gene_lists

    _lines = []
    for _v in variants:
        _sizes = [len(_g) for _g in gene_lists[_v]]
        _lines.append(f"- **{_v}**: n={len(gene_lists[_v])}, mean={np.mean(_sizes):.0f} genes [{min(_sizes)}–{max(_sizes)}]")
    mo.md("**Gene lists loaded:**\n" + "\n".join(_lines))
    return gene_lists, variants


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <a id='enrichment'></a>
    ## 1. Gene Enrichment Analysis

    We want to see if our VAE is learning so we are running an enrichment analysis to test if there are greater than the expected number of genes in our sample relative to what we would expect from random. This code uses a hypergeometric test to ask whether each gene list contains more **core genes** or **essential genes** than expected by chance. “Enriched” means the sample has a higher fraction of those target genes than the full pangenome background. The p-value tells us how likely it would be to see that many or more target genes if the sample were randomly drawn from the pangenome; small p-values suggest significant enrichment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the test works

    For each genome we run a one-sided **hypergeometric test**: is the overlap
    with a target set (the core genome, or the essential genes) larger than
    expected if the genome's genes were a random draw — without replacement —
    from the pangenome?

    - **M** (`_M`) — pangenome size; the population we draw from.
    - **n** (`_n`) — target genes that exist in the pangenome (core, or essential).
    - **N** (`_N`) — this genome's size, restricted to genes present in the pangenome.
    - **k** (`_k`) — target genes actually in this genome (the observed overlap).

    `hypergeom.sf(k - 1, M, n, N)` is $P(X \ge k)$ — the chance of seeing **at
    least** `k` target genes by chance; small $p$ means significant enrichment.
    `expected` $= Nn/M$ is the overlap expected at random, and `enrichment`
    $= k / \text{expected}$ is the fold-enrichment over it. We compute this per
    genome for both targets, then aggregate per cohort to mean enrichment, mean
    observed overlap, and the share of genomes with $p < 0.005$.
    """)
    return


@app.cell
def _(
    core_genes,
    essential_genes,
    gene_lists,
    hypergeom,
    pangenome_genes,
    pl,
    variants,
):
    def hypergeometric_enrichment(_sample_genes: set, _target_genes: set, _population_genes: set):
        _M = len(_population_genes)
        _n = len(_target_genes & _population_genes)
        _N = len(_sample_genes & _population_genes)
        _k = len(_sample_genes & _target_genes)
        _p = hypergeom.sf(_k - 1, _M, _n, _N)
        _exp = _N * _n / _M
        return {"observed": _k, "expected": round(_exp, 1), "enrichment": round(_k / _exp, 3) if _exp > 0 else 0, "p_value": _p}

    _all_genes = set(pangenome_genes)
    _results = []
    for _variant in variants:
        for _i, _genes in enumerate(gene_lists[_variant]):
            _gene_set = set(_genes)
            _cr = hypergeometric_enrichment(_gene_set, core_genes, _all_genes)
            _results.append({"variant": _variant, "sample_id": _i, "test": "core_genome", **_cr})
            _er = hypergeometric_enrichment(_gene_set, essential_genes, _all_genes)
            _results.append({"variant": _variant, "sample_id": _i, "test": "essential_genes", **_er})

    enrichment_df = pl.DataFrame(_results)

    enrichment_df.group_by(["variant", "test"]).agg([
        pl.col("enrichment").mean().round(3).alias("mean_enrichment"),
        pl.col("observed").mean().round(0).alias("mean_observed"),
        (pl.col("p_value") < 0.005).mean().mul(100).round(1).alias("pct_significant"),
    ]).sort(["test", "variant"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TODO

    - Every cohort — including the random baseline — keeps the full >95% core
      genome plus repaired essential genes, so all cohorts read as strongly
      enriched and `pct_significant` saturates at 100%. On its own this does
      **not** separate the VAE from random.
    - Move to **pathway / GO-term enrichment** (functional categories) rather
      than raw core/essential membership, to test whether the VAE picks a
      coherent *functional* gene set beyond just keeping the core.
    - Compare against a **size-matched** random baseline at each genome size.
    """)
    return


if __name__ == "__main__":
    app.run()
