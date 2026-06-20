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


    return Path, hypergeom, mo, np, pd, pl, re


@app.cell
def _(mo):
    mo.md("""
    # Genome Minimizer Analysis — Cell Systems Revision

    ## Table of Contents
    1. [Core / Essential Gene Enrichment](#enrichment) — hypergeometric tests for core/essential genes
    2. [GO Enrichment](#go-analysis) — hypergeometric tests over functional categories, restricted to the annotatable-gene background
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
        # Strip Panaroo paralog suffixes (flmC_1, proP_4 -> flmc, prop) so named
        # genes match UniProt symbols. Leave unnamed clusters (group_NNNN) intact:
        # they carry no GO annotation, and stripping "_NNNN" would collapse all
        # ~45k of them into the single token "group".
        if _name.startswith("group_"):
            return _name.lower()
        return re.sub(r"_\d+$", "", _name).lower()

    mo.md(f"""
    **Pangenome:** {n_strains:,} strains, {len(pangenome_genes):,} genes  
    **Core genome (≥95%):** {len(core_genes):,} genes
    """)
    return core_genes, normalize_gene, pangenome_df, pangenome_genes


@app.cell
def _(DATA_DIR, mo, pl):
    essential_df = pl.read_csv(DATA_DIR / "essential_genes.csv")
    essential_genes = set(essential_df["# gene"].to_list())
    mo.md(f"**Essential genes:** {len(essential_genes):,}")
    return (essential_genes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Random genomes:** a structured baseline, not uniform noise. Each random genome is the _entire_ core genome plus accessory genes appended at random until it reaches a target size (~3300, roughly comparable to a VAE sample under a minimization term).
    """)
    return


@app.cell
def _(EVAL_DATA, mo, np, pangenome_df):
    # Load real strains (100 random from pangenome)
    _rng = np.random.default_rng(42)
    _strain_cols = pangenome_df.columns.tolist()
    _chosen = _rng.choice(len(_strain_cols), size=100, replace=False)
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


    variants = ["real", "v0", "v1", "v2", "v3", "random"]
    gene_lists = {_v: load_gene_lists(_v) for _v in variants if _v != "real"}
    gene_lists["real"] = real_gene_lists

    _lines = []
    for _v in variants:
        _sizes = [len(_g) for _g in gene_lists[_v]]
        _lines.append(f"- **{_v}**: n={len(gene_lists[_v])}, mean={np.mean(_sizes):.0f} genes ")
    mo.md("**Gene lists loaded:**\n" + "\n".join(_lines))
    return gene_lists, variants


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data sources & gene-name crosswalk

    Every count below comes from three inputs that use **different gene-naming
    conventions**, so a crosswalk is needed before any set intersection is
    meaningful. Getting this wrong fails silently (empty intersections, or counts
    that exceed their population), so it is spelled out here.

    **1. Pangenome — `data/F4_complete_presence_absence.csv`.** A binary
    presence/absence matrix (verified 0/1), 55,039 gene clusters × strains, from
    Panaroo. ~45.5k clusters are unnamed (`group_NNNN`); ~9.5k carry gene symbols.
    This is the hypergeometric **population** $N$. The **core genome** = clusters
    present in ≥95% of strains (3,091).

    **2. Essential genes — `data/essential_genes.csv`.** 358 *E. coli* essential
    gene symbols (`mreC`, `pyrH`, …). Only **316 map into this pangenome**; the
    other 42 have no cluster here. Since the population is the pangenome, only
    those 316 are testable, so $k$ is counted *within the population* — otherwise
    the force-injected essentials give $k=358 > K=316$ and the test breaks.

    **3. GO annotations — `data/kegg/uniprot_eco_go.tsv`.** A UniProt *E. coli*
    export: per protein, space-separated gene-name synonyms → GO IDs
    (biological-process / molecular-function / cellular-component) plus
    descriptions. Inverted to a GO-term → gene-set map: 3,946 terms over 15,983
    gene names.

    ### The crosswalk: `normalize_gene`

    Pangenome symbols carry Panaroo paralog suffixes (`flmC_1`, `proP_4`); UniProt
    uses the bare symbol (`flmC`). `normalize_gene` lowercases and strips a trailing
    `_<digits>` so paralogs collapse to their base symbol and match GO. Unnamed
    `group_*` clusters are left **intact** — they carry no GO annotation, and
    stripping their numeric suffix would collapse all ~45k of them into the single
    token `"group"` (a silent bug that deflates $N$ and, if `"group"` ever matched a
    GO key, would absorb every unnamed gene). After normalization there are 50,767
    distinct genes, of which **3,125 carry ≥1 GO term**.

    ### Numbers at a glance

    | quantity | value |
    |---|---|
    | pangenome clusters (population $N$) | 55,039 |
    | core genes (≥95% prevalence) | 3,091 |
    | essential symbols / in pangenome | 358 / 316 |
    | GO terms | 3,946 |
    | annotatable genes (≥1 GO term) | 3,125 |
    | unannotated genes removed from GO background | 47,642 |

    ### Caveats (read before interpreting any bar)

    - **The shared core genome dominates every test, including GO.** The sampler
      keeps the *entire* core genome and force-injects essentials into every
      generated genome *and* into the random baseline, so all variants saturate on
      core/essential enrichment (k ≈ K; even the un-injected `real` strains
      saturate naturally — core 3069/3091, essential 310/316). Because core genes
      are heavily annotated (cytosol, ribosome, translation), this carries into the
      GO analysis: the GO profiles of `v3` and `random` are nearly identical (same
      top terms, Spearman ≈ 0.92 on per-term mean -log10(p)). As constructed, none
      of these tests isolate a VAE-specific signal — they confirm the pipeline
      preserves known biology. A VAE-vs-baseline contrast would need the *accessory*
      genome (core/essential removed from both sample and background) or a direct
      v3-vs-random differential test.
    - **-log10(p) scales with genome size**, so it is not comparable across variants
      of different size (mean annotatable draws: real ≈ 2815, v3 ≈ 2431, random ≈
      2531). For cross-variant comparison use the size-robust `enrichment_ratio`
      (fold over expected), not -log10(p).
    - **No multiple-testing correction**, and GO terms are hierarchically nested
      (cytosol ⊂ cytoplasm), so the top-20 list is a descriptive ranking, not 20
      independent significant findings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Gene Enrichment Analysis

    To test whether the VAE is learning real biology, we ask whether each generated gene list contains more **core genes** or **essential genes** than expected by chance. A hypergeometric test models drawing the genome's genes (without replacement) from the full pangenome; "enriched" means the sample holds a higher fraction of target genes than that background. Small p-values mean the observed overlap would be unlikely under random draws.

    The test is per sample. For one genome, the p-value is the upper-tail (survival) probability of seeing at least $k$ target genes — the hypergeometric survival function:

    $$P(X \geq k) = 1 - HyperGeoCDF(k-1, N, K, n)$$

    where

    $N$ — population size (number of pangenome genes)

    $K$ — number of target genes present in the population

    $n$ — number of draws (genome length)

    $k$ — number of target genes observed in this genome

    **Numerical note.** These p-values are astronomically small (e.g. p ~ 1e-4000 for a genome carrying essentially the whole core genome), so they underflow float64. We compute `hypergeom.logsf`, which returns the *natural* log of the survival function, then convert to -log10(p) via `-ln(p) * log10(e)`. The plotted values are -log10(p) in the hundreds-to-thousands — large but correct; bigger means more significant. Combining per-sample p-values into a single principled statistic (Fisher / Stouffer) is the next step; for now we report the mean of -log10(p).
    """)
    return


@app.cell
def _(hypergeom, np):
    # Shared upper-tail hypergeometric significance, returned as -log10(p).
    # logsf gives the *natural* log of the survival function P(X >= k), so the
    # tiny p-values here never underflow float64; ln(p) * log10(e) = log10(p).
    _LOG10_E = np.log10(np.e)

    def neg_log10_hypergeom_sf(_k, _N, _K, _n):
        # Fail loudly on inconsistent parameters. The classic silent bug is k > K
        # (counting target hits that aren't in the population) — scipy then returns
        # logsf = -inf, which quietly poisons the mean instead of erroring.
        if not (0 <= _k <= _K and _k <= _n <= _N):
            raise ValueError(f"inconsistent hypergeometric params: k={_k}, K={_K}, n={_n}, N={_N}")
        return -hypergeom.logsf(_k - 1, _N, _K, _n) * _LOG10_E

    return (neg_log10_hypergeom_sf,)


@app.cell
def _(
    core_genes,
    essential_genes,
    gene_lists,
    neg_log10_hypergeom_sf,
    pangenome_genes,
    pl,
    variants,
):
    def hypergeometric_enrichment(_sample_genes: set, _target_genes: set, _population_genes: set):
        _N = len(_population_genes)
        _K = len(_target_genes & _population_genes)
        _n = len(_sample_genes & _population_genes)
        # k must be counted *within the population*. Essentials are force-injected
        # into every genome, but 42/358 have no pangenome cluster; counting them
        # would make k > K and break the test (-inf). Intersect with the population.
        _k = len(_sample_genes & _target_genes & _population_genes)
        return {"observed": _k, "neg_log10_p": neg_log10_hypergeom_sf(_k, _N, _K, _n)}

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
    enrichment_df
    return (enrichment_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Combining P-Values

    Mean of per-sample -log₁₀(p) values, computed from `logsf` to avoid float64 underflow.
    """)
    return


@app.cell
def _(enrichment_df, pl):
    # Group by variant and test — use neg_log10_p computed from logsf (no underflow)
    combined_pvals = (
        enrichment_df
        .group_by(["variant", "test"])
        .agg(pl.col("neg_log10_p").mean().alias("mean_neg_log10_p"))
    )
    combined_pvals
    return (combined_pvals,)


@app.cell
def _(combined_pvals):
    import altair as alt

    # Core (~4000) and essential (~350) live on very different scales, so give
    # each test its own y-axis (independent scale per facet) instead of sharing one.
    _sort = ["real", "v0", "v1", "v2", "v3", "random"]
    _chart = (
        alt.Chart(combined_pvals)
        .mark_bar()
        .encode(
            x=alt.X("variant:N", title="Variant", sort=_sort),
            y=alt.Y("mean_neg_log10_p:Q", title="Mean -log₁₀(p-value)"),
            color=alt.Color("test:N", title="Test", legend=None),
        )
        .properties(width=250, height=300)
        .facet(column=alt.Column("test:N", title=""))
        .resolve_scale(y="independent")
        .properties(title="Core / Essential Gene Enrichment")
    )

    _chart
    return (alt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. GO Term Enrichment Analysis

    Test whether gene lists are enriched for specific functional categories (GO
    terms) rather than just core/essential membership. The population is restricted
    to the **annotatable background** (genes with ≥1 GO term) — see *Data sources &
    gene-name crosswalk* above for why, and read the caveats there before comparing
    variants: the GO signal is core-dominated and `v3` ≈ `random`.
    """)
    return


@app.cell
def _(DATA_DIR, mo, normalize_gene, pd):
    # Load GO annotations
    _go_df = pd.read_csv(DATA_DIR / "kegg" / "uniprot_eco_go.tsv", sep="\t")

    # Build gene -> GO term mapping
    _gene_to_go = {}
    for _, _row in _go_df.iterrows():
        _gene_names = str(_row["Gene Names"]).split()
        _go_ids = str(_row["Gene Ontology IDs"]).split("; ") if pd.notna(_row["Gene Ontology IDs"]) else []

        for _gene in _gene_names:
            _norm_gene = normalize_gene(_gene)
            if _norm_gene not in _gene_to_go:
                _gene_to_go[_norm_gene] = set()
            _gene_to_go[_norm_gene].update(_go_ids)

    # Build GO term -> genes mapping and GO ID -> description mapping
    _go_to_genes = {}
    _go_descriptions = {}

    for _, _row in _go_df.iterrows():
        # Parse all GO columns to extract descriptions
        for _col in ["Gene Ontology (biological process)", "Gene Ontology (molecular function)", "Gene Ontology (cellular component)"]:
            if pd.notna(_row[_col]):
                _entries = str(_row[_col]).split("; ")
                for _entry in _entries:
                    if "[GO:" in _entry:
                        _desc = _entry.split(" [GO:")[0].strip()
                        _go_id = "GO:" + _entry.split("[GO:")[1].rstrip("]")
                        _go_descriptions[_go_id] = _desc

    for _gene, _go_set in _gene_to_go.items():
        for _go_id in _go_set:
            if _go_id not in _go_to_genes:
                _go_to_genes[_go_id] = set()
            _go_to_genes[_go_id].add(_gene)

    # Use ALL GO terms (no filtering)
    gene_to_go = _gene_to_go
    go_to_genes = _go_to_genes
    go_descriptions = _go_descriptions

    mo.md(f"""
    **GO annotations loaded:**
    - {len(_gene_to_go):,} genes with GO annotations
    - {len(_go_to_genes):,} total GO terms
    """)
    return gene_to_go, go_descriptions, go_to_genes


@app.cell
def _():
    import seaborn as sns
    import matplotlib.pyplot as plt

    return plt, sns


@app.cell
def _(
    gene_lists,
    gene_to_go,
    go_to_genes,
    neg_log10_hypergeom_sf,
    normalize_gene,
    pangenome_genes,
    pl,
    variants,
):
    # Population = pangenome genes that have at least one GO annotation
    _all_norm = set(normalize_gene(_g) for _g in pangenome_genes)
    annotatable_genes = _all_norm & set(gene_to_go.keys())

    _go_results_annot = []
    _N = len(annotatable_genes)
    for _variant in variants:
        for _sample_i, _genes in enumerate(gene_lists[_variant]):
            _sample_norm = set(normalize_gene(_g) for _g in _genes) & annotatable_genes

            for _go_id, _go_gene_set in go_to_genes.items():
                _K = len(_go_gene_set & annotatable_genes)
                _n = len(_sample_norm)
                _k = len(_sample_norm & _go_gene_set)

                if _k > 0:
                    _go_results_annot.append({
                        "variant": _variant,
                        "sample_id": _sample_i,
                        "go_term": _go_id,
                        "observed": _k,
                        "expected": (_n * _K) / _N,
                        "neg_log10_p": neg_log10_hypergeom_sf(_k, _N, _K, _n),
                    })

    go_enrichment_annot_df = pl.DataFrame(_go_results_annot)
    go_enrichment_annot_df
    return annotatable_genes, go_enrichment_annot_df


@app.cell
def _(annotatable_genes, mo, normalize_gene, pangenome_genes):
    _all_norm = set(normalize_gene(_g) for _g in pangenome_genes)
    mo.md(f"""
    **Background restriction:**
    - Full pangenome (normalized): {len(_all_norm):,} genes
    - Annotatable (have GO annotation): {len(annotatable_genes):,} genes
    - Removed from background: {len(_all_norm) - len(annotatable_genes):,} unannotated genes
    """)
    return


@app.cell
def _(
    annotatable_genes,
    gene_lists,
    go_to_genes,
    mo,
    normalize_gene,
    np,
    pl,
    variants,
):
    # Diagnostic: N, K, n, k summary per variant (annotatable background)
    _N = len(annotatable_genes)
    _rows = []
    for _variant in variants:
        _n_vals = []
        for _genes in gene_lists[_variant]:
            _sample_norm = set(normalize_gene(_g) for _g in _genes) & annotatable_genes
            _n_vals.append(len(_sample_norm))
        _rows.append({
            "variant": _variant,
            "N (population)": _N,
            "n (mean sample draws)": np.mean(_n_vals),
            "n (min)": np.min(_n_vals),
            "n (max)": np.max(_n_vals),
        })

    # K values: number of genes per GO term in the annotatable background
    _K_vals = [len(_gs & annotatable_genes) for _gs in go_to_genes.values()]
    _k_summary = f"K across {len(go_to_genes)} GO terms: min={np.min(_K_vals)}, median={np.median(_K_vals):.0f}, mean={np.mean(_K_vals):.1f}, max={np.max(_K_vals)}"

    _df = pl.DataFrame(_rows)
    mo.vstack([
        mo.md(f"""
    **Hypergeometric parameters (annotatable background):**

    - **N** (population size) = {_N:,} annotatable genes
    - **K** (GO term gene counts): {_k_summary}
    """),
        _df,
    ])
    return


@app.cell
def _(go_descriptions, go_enrichment_annot_df, pl):
    go_combined_annot = (
        go_enrichment_annot_df
        .group_by(["variant", "go_term"])
        .agg(
            pl.col("neg_log10_p").mean().alias("mean_neg_log10_p"),
            pl.col("observed").mean().alias("avg_observed"),
            pl.col("expected").mean().alias("avg_expected"),
        )
        .with_columns(
            pl.col("go_term").replace(go_descriptions).alias("go_description"),
            (pl.col("avg_observed") / pl.col("avg_expected")).alias("enrichment_ratio"),
        )
    )
    go_combined_annot
    return (go_combined_annot,)


@app.cell
def _(go_combined_annot, mo, pl):
    _top_n = 20
    _top_annot = []
    for _variant in go_combined_annot["variant"].unique():
        _var = go_combined_annot.filter(pl.col("variant") == _variant)
        _top_annot.append(_var.sort("mean_neg_log10_p", descending=True).head(_top_n))

    top_go_annot = pl.concat(_top_annot)
    mo.md(f"### Top {_top_n} enriched GO terms per variant (annotatable background)")
    return (top_go_annot,)


@app.cell
def _(alt, top_go_annot):
    _chart = alt.Chart(top_go_annot).mark_bar(color="steelblue").encode(
        x=alt.X("mean_neg_log10_p:Q", title="Mean -log₁₀(p-value)"),
        y=alt.Y("go_description:N", title="GO Term", sort="-x"),
        tooltip=["go_description", "go_term", "mean_neg_log10_p", "enrichment_ratio", "avg_observed", "avg_expected"],
        facet=alt.Facet("variant:N", columns=2),
    ).properties(width=350, height=350, title="GO Enrichment (annotatable background)")

    _chart
    return


@app.cell
def _(go_combined_annot, pl, plt, sns):
    _v3_annot = go_combined_annot.filter(pl.col("variant") == "v3")
    _sorted = _v3_annot.sort("mean_neg_log10_p", descending=True).head(20)
    _plot_df = _sorted.to_pandas()

    _fig, _ax = plt.subplots(figsize=(10, max(6, len(_plot_df) * 0.3)))
    sns.barplot(data=_plot_df, y="go_description", x="mean_neg_log10_p", color="steelblue", ax=_ax)
    _ax.set_xlabel("Mean -log₁₀(p-value)", fontsize=12)
    _ax.set_ylabel("GO Term", fontsize=12)
    _ax.set_title("GO Term Enrichment — v3", fontsize=14, fontweight="bold")
    _ax.grid(axis="x", alpha=0.3, linestyle="--")
    _ax.set_axisbelow(True)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Per-Sample P-Value Distributions (annotatable background)

    Select the top 10 GO terms by mean -log₁₀(p) for v3, then show boxplots of
    the individual per-sample -log₁₀(p) values across all 100 genomes.
    """)
    return


@app.cell
def _(go_combined_annot, go_descriptions, go_enrichment_annot_df, pl):
    # Top 10 GO terms for v3 by mean -log10(p)
    _v3_combined = go_combined_annot.filter(pl.col("variant") == "v3")
    _top10_go_ids = (
        _v3_combined
        .sort("mean_neg_log10_p", descending=True)
        .head(10)["go_term"]
        .to_list()
    )

    # Pull the raw per-sample values for these terms (v3 only)
    _per_sample = (
        go_enrichment_annot_df
        .filter(
            (pl.col("variant") == "v3") &
            (pl.col("go_term").is_in(_top10_go_ids))
        )
        .with_columns(
            pl.col("go_term").replace(go_descriptions).alias("go_description"),
        )
    )

    # Order descriptions by mean rank
    _desc_order = (
        _per_sample.group_by("go_description")
        .agg(pl.col("neg_log10_p").mean().alias("mean_val"))
        .sort("mean_val", descending=True)["go_description"]
        .to_list()
    )

    boxplot_data = _per_sample
    boxplot_desc_order = _desc_order
    return boxplot_data, boxplot_desc_order


@app.cell
def _(boxplot_data, boxplot_desc_order, plt, sns):

    _plot_df = boxplot_data.to_pandas()

    _fig, _ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=_plot_df,
        y="go_description",
        x="neg_log10_p",
        order=boxplot_desc_order,
        color="steelblue",
        fliersize=2,
        ax=_ax,
    )
    _ax.set_xlabel("-log₁₀(p-value)", fontsize=12)
    _ax.set_ylabel("GO Term", fontsize=12)
    _ax.set_title("Per-Sample GO Enrichment — v3",
                   fontsize=13, fontweight="bold")
    _ax.grid(axis="x", alpha=0.3, linestyle="--")
    _ax.set_axisbelow(True)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Same distributions, BH-FDR corrected

    The boxplot above plots raw -log₁₀(p). Below, the same top GO terms for v3 are
    re-plotted as Benjamini–Hochberg q-values: within each genome the p-values are
    corrected across all GO terms that genome hits (the test family), and we show
    -log₁₀(q). The dashed line marks the 5% FDR cutoff — boxes to its left are not
    significant after correction.
    """)
    return


@app.cell
def _(boxplot_data, go_enrichment_annot_df, np, pl, plt, sns):
    # Benjamini-Hochberg q-values, computed per genome across every GO term that
    # genome hits (the test family), for v3 — then restricted to the same top
    # terms shown in the p-value boxplot above.
    def _bh_q(_p):
        _p = np.asarray(_p, dtype=float)
        _m = _p.size
        _o = np.argsort(_p)
        _ranked = _p[_o] * _m / np.arange(1, _m + 1)
        _q_sorted = np.minimum.accumulate(_ranked[::-1])[::-1]
        _q = np.empty(_m)
        _q[_o] = np.clip(_q_sorted, 0.0, 1.0)
        return _q

    _parts = []
    for _grp in go_enrichment_annot_df.filter(pl.col("variant") == "v3").partition_by("sample_id"):
        # reconstruct p from -log10(p); underflow -> 0 is fine (most significant)
        _p = np.power(10.0, -_grp["neg_log10_p"].to_numpy())
        _parts.append(_grp.with_columns(
            pl.Series("neg_log10_q", -np.log10(np.clip(_bh_q(_p), 1e-323, 1.0)))
        ))
    _v3_q = pl.concat(_parts)

    _terms = boxplot_data["go_term"].unique().to_list()
    _desc = boxplot_data.select(["go_term", "go_description"]).unique()
    _q_df = (
        _v3_q.filter(pl.col("go_term").is_in(_terms))
        .join(_desc, on="go_term", how="left")
        .to_pandas()
    )
    _order = (
        _q_df.groupby("go_description")["neg_log10_q"].mean()
        .sort_values(ascending=False).index.tolist()
    )

    _fig, _ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=_q_df,
        y="go_description",
        x="neg_log10_q",
        order=_order,
        color="steelblue",
        fliersize=2,
        ax=_ax,
    )
    _ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="5% FDR")
    _ax.legend(loc="lower right")
    _ax.set_xlabel("-log₁₀(q)  (BH-FDR)", fontsize=12)
    _ax.set_ylabel("GO Term", fontsize=12)
    _ax.set_title("Per-Sample GO Enrichment — v3 (BH-FDR q-values)",
                   fontsize=13, fontweight="bold")
    _ax.grid(axis="x", alpha=0.3, linestyle="--")
    _ax.set_axisbelow(True)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
