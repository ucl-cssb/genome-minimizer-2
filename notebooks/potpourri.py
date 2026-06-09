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

    return Path, alt, hypergeom, mo, np, pd, pl, re


@app.cell
def _(mo):
    mo.md("""
    # Genome Minimizer Analysis — Cell Systems Revision

    ## Table of Contents
    1. [Gene Enrichment](#enrichment) — hypergeometric tests for core/essential genes
    2. [KEGG Module Coverage](#kegg) — metabolic pathway completeness
    3. [ Viability](#) — iML1515 growth rate predictions
    4. [Reviewer Responses](#reviewer) — precomputed metrics for revision
    """)
    return


@app.cell
def _(Path):
    PROJECT_ROOT = Path("/Users/mcclainthiel/Projects/PhD/genome-minimizer-2")
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
    **Random Genomes** In order to test if the VAE was actually learning anything we compare to a random baseline. This is implemented in `genome_minimizer_2.sampling`. What is does it calculate the core genome (genes in >95% of the dataset) keeps all of those, then samples the remaining genes until we have as many genes as we need for comparision (generally ~3300 which is how many our best VAE produced).
    """)
    return


@app.cell
def _(EVAL_DATA, mo, np, pangenome_df):
    # Load real strains (200 random from pangenome, matching tier1_kegg.py)
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


@app.cell
def _(hypergeom):
    hypergeom.sf

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TODO
    Calculate this relative to GO (gene ontology) terms to get pathway enrichment as opposed to core genome

    mg1655 genes (refference) - deleted from this is the question
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
    $$N = \sim550000$$ pangenome
    Need to match these to GO terms

    For Given GO term -> K Genes

    Set of N genes deletes (from mg1655) / kept (from pangeome)

    We need to do this per go term but with a subset of GO terms.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    <a id='kegg'></a>
    ## 2. KEGG Module Coverage
    """)
    return


@app.cell
def _(DATA_DIR):
    import requests
    KEGG_CACHE = DATA_DIR / "kegg"
    KEGG_CACHE.mkdir(exist_ok=True)

    def fetch_kegg(_endpoint: str, _cache_file: str):
        _cache_path = KEGG_CACHE / _cache_file
        if _cache_path.exists():
            return _cache_path.read_text()
        _url = f"https://rest.kegg.jp/{_endpoint}"
        _resp = requests.get(_url)
        _resp.raise_for_status()
        _cache_path.write_text(_resp.text)
        return _resp.text

    eco_genes_text = fetch_kegg("list/eco", "eco_genes.tsv")
    eco_module_links = fetch_kegg("link/module/eco", "eco_module_links.tsv")
    return eco_genes_text, eco_module_links, requests


@app.cell
def _(eco_genes_text, eco_module_links, mo, normalize_gene, pangenome_genes):
    # KEGG symbol → b-number (matches tier1_kegg.py: split on "," for aliases)
    symbol_to_bnum = {}
    for _line in eco_genes_text.strip().splitlines():
        _parts = _line.split("\t")
        if len(_parts) < 4:
            continue
        _bnum = _parts[0].removeprefix("eco:")
        _info = _parts[3]
        _symbol_part = _info.split(";")[0].strip()
        for _sym in _symbol_part.split(","):
            _key = _sym.strip().lower()
            if _key:
                symbol_to_bnum[_key] = _bnum

    # Module definitions
    module_genes = {}
    for _line in eco_module_links.strip().splitlines():
        _parts = _line.split("\t")
        if len(_parts) >= 2:
            _bnum = _parts[0].removeprefix("eco:")
            _module_id = _parts[1].removeprefix("md:")
            module_genes.setdefault(_module_id, set()).add(_bnum)

    # Pangenome → KEGG crosswalk (matches tier1_kegg.py)
    pangenome_to_bnum = {}
    _unmatched = []
    for _gene in pangenome_genes:
        if _gene.startswith("group_"):
            continue
        _key = _gene.lower()
        if _key in symbol_to_bnum:
            pangenome_to_bnum[_gene] = symbol_to_bnum[_key]
            continue
        _norm = normalize_gene(_gene)
        if _norm in symbol_to_bnum:
            pangenome_to_bnum[_gene] = symbol_to_bnum[_norm]
            continue
        _unmatched.append(_gene)

    _n_named = sum(1 for _g in pangenome_genes if not _g.startswith("group_"))
    mo.md(f"""
    **KEGG crosswalk:** {len(pangenome_to_bnum)} / {_n_named} named genes mapped ({len(pangenome_to_bnum)/_n_named*100:.1f}%)  
    **Unique b-numbers:** {len(set(pangenome_to_bnum.values()))}  
    **KEGG modules:** {len(module_genes)}
    """)
    return module_genes, pangenome_to_bnum


@app.cell
def _(gene_lists, module_genes, pangenome_to_bnum, pl, variants):
    # Score every genome against every module (matches tier1_kegg.py score_genome)
    def score_genome(_gene_names: list):
        _bnums = set()
        for _g in _gene_names:
            _b = pangenome_to_bnum.get(_g)
            if _b is not None:
                _bnums.add(_b)
        return {_mid: len(_bnums & _mgenes) / len(_mgenes) if _mgenes else 0.0
                for _mid, _mgenes in module_genes.items()}

    _module_rows = []
    for _variant in variants:
        for _i, _genes in enumerate(gene_lists[_variant]):
            _scores = score_genome(_genes)
            for _mid, _comp in _scores.items():
                _module_rows.append({"variant": _variant, "sample_id": _i, "module_id": _mid, "completeness": _comp})

    module_long_df = pl.DataFrame(_module_rows)

    # Summary table
    module_summary = (
        module_long_df.group_by(["variant", "sample_id"])
        .agg([
            (pl.col("completeness") >= 0.80).sum().alias("n_modules_80pct"),
            (pl.col("completeness") >= 0.99).sum().alias("n_modules_100pct"),
        ])
    )
    module_summary.group_by("variant").agg([
        pl.col("n_modules_80pct").mean().round(1).alias("mean_80pct"),
        pl.col("n_modules_80pct").min().alias("min_80pct"),
        pl.col("n_modules_80pct").max().alias("max_80pct"),
        pl.col("n_modules_100pct").mean().round(1).alias("mean_100pct"),
        pl.col("n_modules_100pct").min().alias("min_100pct"),
        pl.col("n_modules_100pct").max().alias("max_100pct"),
    ]).sort("variant")
    return module_long_df, module_summary


@app.cell
def _(alt, module_long_df, pl, variants):
    _per_genome = module_long_df.group_by(["variant", "sample_id"]).agg(
        (pl.col("completeness") >= 0.99).sum().alias("n_modules")
    )

    alt.Chart(_per_genome.to_pandas()).transform_density(
        "n_modules", as_=["n_modules", "density"], groupby=["variant"],
        extent=[0, 112], bandwidth=2.0
    ).mark_area(orient="horizontal", opacity=0.7).encode(
        y=alt.Y("n_modules:Q", title="# modules ≥99% complete"),
        x=alt.X("density:Q", stack="center", title=None, axis=None),
        color="variant:N",
        column=alt.Column("variant:N", sort=variants)
    ).properties(width=80, height=250)
    return


@app.cell
def _(mo):
    mo.md("""
    <a id=''></a>
    ## 3.  Viability (iML1515)
    """)
    return


@app.cell
def _(DATA_DIR, mo, normalize_gene, pangenome_genes):
    import cobra
    import warnings
    warnings.filterwarnings("ignore", message="Solver status")

    _DIR = DATA_DIR / ""
    iml = cobra.io.read_sbml_model(str(_DIR / "fba/iML1515.xml"))
    wt_growth = iml.optimize().objective_value

    # Build iML1515 ↔ pangenome crosswalk using synonyms from SBML annotations
    # (matches tier1_kegg.py — primary name + refseq_synonym, not just KEGG symbols)
    iml_bnum_to_names = {}
    for _g in iml.genes:
        if _g.id == "s0001":
            continue
        _names = set()
        if _g.name:
            _names.add(_g.name.lower())
        for _syn in _g.annotation.get("refseq_synonym", []):
            if isinstance(_syn, str):
                _names.add(_syn.lower())
        if _names:
            iml_bnum_to_names[_g.id] = _names

    _pangenome_norm_set = {normalize_gene(_g) for _g in pangenome_genes}
    iml_pangenome_known = {_bnum for _bnum, _names in iml_bnum_to_names.items()
                           if _names & _pangenome_norm_set}

    mo.md(f"""
    **iML1515:** {len(iml.genes)} genes, {len(iml.reactions)} reactions, WT growth {wt_growth:.4f} h⁻¹  
    **Crosswalk:** {len(iml_pangenome_known)} / {len(iml_bnum_to_names)} iML1515 genes matched to pangenome ({len(iml_pangenome_known)/len(iml_bnum_to_names)*100:.1f}%)
    """)
    return iml, iml_bnum_to_names, iml_pangenome_known


@app.cell
def _(
    gene_lists,
    iml,
    iml_bnum_to_names,
    iml_pangenome_known,
    normalize_gene,
    pl,
    variants,
):
    #  for every genome (matches tier1_kegg.py logic exactly)
    __rows = []
    for _variant in variants:
        for _i, _genes in enumerate(gene_lists[_variant]):
            _genome_norm = {normalize_gene(_g) for _g in _genes}
            _present_b = {_bnum for _bnum in iml_pangenome_known
                          if iml_bnum_to_names[_bnum] & _genome_norm}
            _absent = iml_pangenome_known - _present_b
            with iml as _m:
                for _gid in _absent:
                    _m.genes.get_by_id(_gid).knock_out()
                _sol = _m.optimize()
                _growth = (
                    _sol.objective_value
                    if _sol.status == "optimal" and _sol.objective_value is not None
                    else 0.0
                )
            _growth = float(_growth)
            if abs(_growth) < 1e-6:
                _growth = 0.0
            __rows.append({
                "variant": _variant, "sample_id": _i,
                "n_iml_present": len(_present_b), "n_iml_absent": len(_absent),
                "growth_rate": _growth,
            })

    fba_df = pl.DataFrame(__rows)

    fba_df.group_by("variant").agg([
        pl.col("growth_rate").mean().round(3).alias("mean_growth"),
        pl.col("growth_rate").median().round(3).alias("median_growth"),
        (pl.col("growth_rate") > 0.01).sum().alias("n_viable"),
        pl.len().alias("n_total"),
        pl.col("n_iml_present").mean().round(0).alias("mean_iml_genes"),
    ]).with_columns(
        (pl.col("n_viable") / pl.col("n_total") * 100).round(1).alias("pct_viable")
    ).sort("variant")
    return (fba_df,)


@app.cell
def _(alt, fba_df, variants):
    alt.Chart(fba_df.to_pandas()).transform_density(
        "growth_rate", as_=["growth_rate", "density"], groupby=["variant"],
        extent=[-0.05, 0.95], bandwidth=0.025
    ).mark_area(orient="horizontal", opacity=0.7).encode(
        y=alt.Y("growth_rate:Q", title=" growth rate (h⁻¹)"),
        x=alt.X("density:Q", stack="center", title=None, axis=None),
        color="variant:N",
        column=alt.Column("variant:N", sort=variants)
    ).properties(width=80, height=280)
    return


@app.cell
def _(mo):
    mo.md("""
    <a id='reviewer'></a>
    ## 4. Reviewer Response Analyses
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Section 2.1: Missing metadata

    - **Raw samples:** 7,512
    - **Retained:** 5,953
    - **Removed:** 1,559 (20.75%)
    - **Missing phylogroup among retained:** 0
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Section 2.1: Essential gene quantification

    - **Literature essentials:** 358
    - **Represented (after matching):** 327
    - **Not represented:** 31
    - **Counts per genome:** 284–325, median 323
    - **Genomes with ≥95% essentials:** 5,937 / 5,953 = 99.73%
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Section 2.2: Phylogroup clustering (v0 vs v1)

    **Adjusted Rand Index** (k-means clusters vs phylogroup):
    - v0: 0.3349
    - v1: 0.4599
    - delta: +0.1250

    Phylogroup labels are more recoverable from v1 latent space.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Q3: Raw matrix baseline

    - Included raw presence-absence baseline to separate latent-space structure from sparse binary matrix structure
    - **Silhouette:** nan (not informative)
    - **Between/within centroid ratio:** v0 = 1.2924, v1 = 1.2189
    - Do not claim better centroid separation from this metric
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Loss definitions

    - **Reconstruction loss:** BCE between decoder probabilities $\hat{x}$ and binary presence vector $x$, summed over genes/genomes
    - **KL divergence:** Closed-form KL between encoder posterior $q_\phi(z|x)$ and standard normal prior $\mathcal{N}(0, I)$, summed over latent dimensions/genomes
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Train/val/test split

    **Split:** 4,167 train / 1,190 validation / 596 test
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### W&B metrics (v0–v3)
    """)
    return


@app.cell
def _(pl):
    pl.DataFrame({
        "variant": ["v0", "v1", "v2", "v3"],
        "test_f1": [0.9843, 0.9853, 0.9849, 0.9121],
        "test_accuracy": [0.9973, 0.9975, 0.9974, 0.9861],
    })
    return


@app.cell
def _(mo):
    mo.md("""
    ### v3 gamma schedule

    **Code:** `gamma_start=2.0`, `gamma_end=0.1`, `weight=1.0`
    Manuscript text saying "2 -> 1" is incorrect.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Latent space usage (active units)

    **Held-out active units** (both Var(mu) > 1e-2 and mean KL > 1e-2):
    """)
    return


@app.cell
def _(pl):
    pl.DataFrame({
        "variant": ["v0", "v1", "v2", "v3"],
        "latent_dim": [64, 32, 32, 32],
        "active_units_var": [64, 32, 29, 32],
        "active_units_kl": [64, 32, 29, 32],
    })
    return


@app.cell
def _(mo):
    mo.md("""
    No evidence of posterior collapse. v3 uses all 32 latent dimensions.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Tier 1 vs Tier 2: KEGG module coverage vs FBA viability
    """)
    return


@app.cell
def _(fba_df, module_summary, pl):
    # Join KEGG and FBA results for 2D comparison
    tier_comparison = (
        module_summary
        .join(fba_df, on=["variant", "sample_id"], how="inner")
        .filter(pl.col("variant").is_in(["real", "random", "v4"]))
    )

    tier_comparison.group_by("variant").agg([
        pl.col("n_modules_100pct").mean().round(1).alias("mean_modules_100pct"),
        pl.col("growth_rate").mean().round(3).alias("mean_growth"),
        (pl.col("growth_rate") > 0.01).mean().mul(100).round(1).alias("pct_viable"),
    ]).sort("variant")
    return (tier_comparison,)


@app.cell
def _(alt, tier_comparison):
    # Scatter plot with jitter for binary FBA
    _scatter = (
        alt.Chart(tier_comparison.to_pandas())
        .mark_circle(opacity=0.6, size=60)
        .encode(
            x=alt.X("n_modules_100pct:Q", title="# KEGG modules 100% complete (Tier 1)", scale=alt.Scale(domain=[50, 80])),
            y=alt.Y("growth_rate:Q", title="FBA growth rate h⁻¹ (Tier 2)"),
            color=alt.Color("variant:N", title="Source", scale=alt.Scale(
                domain=["real", "random", "v4"],
                range=["#2ca02c", "#d62728", "#9467bd"]
            )),
            yOffset="jitter:Q"
        )
        .transform_calculate(jitter="(random() - 0.5) * 0.03")
        .properties(width=500, height=350, title="Tier 1 (KEGG) vs Tier 2 (FBA)")
    )
    _scatter
    return


@app.cell
def _(alt, pl, tier_comparison):
    # Ridge plot by viability
    _tier_comp_binned = tier_comparison.with_columns(
        pl.when(pl.col("growth_rate") > 0.01).then(pl.lit("viable")).otherwise(pl.lit("inviable")).alias("viability_bin")
    )

    _ridge = (
        alt.Chart(_tier_comp_binned.to_pandas())
        .transform_density("n_modules_100pct", as_=["n_modules_100pct", "density"], groupby=["variant", "viability_bin"], bandwidth=2.0)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("n_modules_100pct:Q", title="# KEGG modules 100% complete"),
            y=alt.Y("density:Q", title=None, axis=None),
            color=alt.Color("variant:N", scale=alt.Scale(domain=["real", "random", "v4"], range=["#2ca02c", "#d62728", "#9467bd"])),
            row=alt.Row("viability_bin:N", title="FBA outcome", sort=["viable", "inviable"])
        )
        .properties(width=500, height=120, title="KEGG by FBA outcome")
    )
    _ridge
    return


@app.cell
def _(mo):
    mo.md("""
    <a id='go'></a>
    ## 5. Gene Ontology Enrichment
    """)
    return


@app.cell
def _(DATA_DIR, requests):
    # Download UniProt GO annotations
    GO_CACHE = DATA_DIR / "kegg" / "uniprot_eco_go.tsv"

    if not GO_CACHE.exists():
        _url = "https://rest.uniprot.org/uniprotkb/stream"
        _params = {"format": "tsv", "fields": "gene_names,gene_primary,go_id,go_p,go_f,go_c", "query": "(organism_id:83333) AND (reviewed:true)"}
        _resp = requests.get(_url, params=_params)
        _resp.raise_for_status()
        GO_CACHE.write_text(_resp.text)

    go_annotations_text = GO_CACHE.read_text()
    return (go_annotations_text,)


@app.cell
def _(go_annotations_text, mo, pd):
    from io import StringIO
    go_raw = pd.read_csv(StringIO(go_annotations_text), sep="\t")
    go_raw["b_number"] = go_raw["Gene Names"].str.extract(r"(b\d{4})")
    go_expanded = go_raw.copy()
    go_expanded["go_ids"] = go_expanded["Gene Ontology IDs"].str.split("; ")
    go_map = go_expanded.explode("go_ids")[["b_number", "Gene Names (primary)", "go_ids"]].dropna()
    go_map.columns = ["b_number", "gene_symbol", "go_id"]

    mo.md(f"""**UniProt GO annotations:** {go_map['b_number'].nunique()} genes, {len(go_map):,} annotations, {go_map['go_id'].nunique()} unique GO terms""")
    return (go_map,)


@app.cell
def _(gene_lists, go_map, hypergeom, pangenome_to_bnum, pd, pl):
    from statsmodels.stats.multitest import multipletests

    # 10 biologically relevant GO terms for minimal genome design
    _selected_go_terms = [
        "GO:0005524",  # ATP binding (446 genes)
        "GO:0002181",  # cytoplasmic translation (55 genes)
        "GO:0006355",  # regulation of transcription (242 genes)
        "GO:0006260",  # DNA replication (37 genes)
        "GO:0051301",  # cell division (62 genes)
        "GO:0006979",  # response to oxidative stress (52 genes)
        "GO:0003735",  # structural constituent of ribosome (57 genes)
        "GO:0006281",  # DNA repair (48 genes)
        "GO:0071555",  # cell wall organization (77 genes)
        "GO:0009252",  # peptidoglycan biosynthesis (48 genes)
    ]

    # Compute baseline from real strains (not full pangenome)
    _real_bnums_all = set()
    for _genes in gene_lists["real"]:
        _real_bnums_all.update(pangenome_to_bnum.get(_g) for _g in _genes if _g in pangenome_to_bnum)
    _real_bnums_all = _real_bnums_all & set(go_map["b_number"].dropna())
    _N_baseline = len(_real_bnums_all)

    # Pre-compute genes per GO term in the real strain baseline
    _go_to_genes_baseline = {}
    for _term in _selected_go_terms:
        _go_genes = set(go_map[go_map["go_id"] == _term]["b_number"])
        _go_to_genes_baseline[_term] = _go_genes & _real_bnums_all

    # Test v3 and random vs. real baseline
    _test_variants = ["v3", "random"]
    _go_results = []

    for _variant in _test_variants:
        for _i, _genes in enumerate(gene_lists[_variant]):
            # Map study genes to b-numbers
            _study_bnums = {pangenome_to_bnum.get(_g) for _g in _genes if _g in pangenome_to_bnum}
            _study_bnums = _study_bnums & set(go_map["b_number"].dropna())
            if len(_study_bnums) == 0:
                continue
            _n_study = len(_study_bnums)

            # Test each selected GO term
            for _go_term in _selected_go_terms:
                _baseline_genes = _go_to_genes_baseline[_go_term]
                _K_baseline = len(_baseline_genes)
                _k_study = len(_study_bnums & _baseline_genes)

                # Hypergeometric test: P(X >= k | N, K, n)
                _pval = hypergeom.sf(_k_study - 1, _N_baseline, _K_baseline, _n_study)
                _enrichment = (_k_study / _n_study) / (_K_baseline / _N_baseline) if _K_baseline > 0 and _n_study > 0 else 0

                _go_results.append({
                    "variant": _variant,
                    "sample_id": _i,
                    "go_id": _go_term,
                    "k_study": _k_study,
                    "n_study": _n_study,
                    "K_population": _K_baseline,
                    "enrichment": _enrichment,
                    "p_value": _pval
                })

    go_enrichment_df = pl.DataFrame(_go_results)

    # FDR correction per variant
    _fdr_corrected = []
    for _variant in _test_variants:
        _subset = go_enrichment_df.filter(pl.col("variant") == _variant)
        if len(_subset) == 0:
            continue
        _pvals = _subset["p_value"].to_numpy()
        _, _fdr, _, _ = multipletests(_pvals, method="fdr_bh")
        _subset_df = _subset.to_pandas()
        _subset_df["fdr"] = _fdr
        _fdr_corrected.append(_subset_df)

    go_enrichment_fdr = pl.from_pandas(pd.concat(_fdr_corrected, ignore_index=True)) if _fdr_corrected else pl.DataFrame()
    return (go_enrichment_fdr,)


@app.cell
def _(go_enrichment_fdr, mo):
    mo.md(f"""
    **GO enrichment results:** {len(go_enrichment_fdr):,} tests across 2 variants (v3, random) vs. real strains baseline

    Testing 10 core cellular functions: ATP binding, translation, transcription regulation, DNA replication, 
    cell division, oxidative stress response, ribosome structure, DNA repair, cell wall organization, 
    and peptidoglycan biosynthesis.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Pathway Enrichment Visualization

    Comparing GO term enrichment for v3 and random variants vs. real baseline.
    """)
    return


@app.cell
def _(go_enrichment_fdr, pl):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # GO enrichment visualization - compare v3 and random vs real
    _go_names_map = {
        "GO:0005524": "ATP binding",
        "GO:0002181": "cytoplasmic translation",
        "GO:0006355": "transcription regulation",
        "GO:0006260": "DNA replication",
        "GO:0051301": "cell division",
        "GO:0006979": "oxidative stress response",
        "GO:0003735": "ribosome structure",
        "GO:0006281": "DNA repair",
        "GO:0071555": "cell wall organization",
        "GO:0009252": "peptidoglycan biosynthesis",
    }

    # Prepare plot data - only v3 and random (real is baseline)
    go_plot_df = (
        go_enrichment_fdr
        .filter(pl.col("variant").is_in(["v3", "random"]))
        .with_columns([
            pl.col("go_id").replace(_go_names_map).alias("go_name"),
        ])
        .group_by(["variant", "go_name"])
        .agg([
            pl.col("enrichment").mean().alias("mean_enrichment"),
            (pl.col("fdr") < 0.05).sum().alias("n_significant"),
            pl.len().alias("n_total")
        ])
        .with_columns(
            (pl.col("n_significant") / pl.col("n_total") * 100).alias("pct_significant")
        )
        .to_pandas()
    )
    return go_plot_df, plt


@app.cell
def _(go_plot_df, plt):
    # Plot 1: Mean enrichment for v3
    _fig1, _ax1 = plt.subplots(figsize=(8, 6))
    _v3_data = go_plot_df[go_plot_df["variant"] == "v3"].sort_values("mean_enrichment")

    _bars = _ax1.barh(_v3_data["go_name"], _v3_data["mean_enrichment"], color="#d62728", alpha=0.8)
    _ax1.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Baseline (1.0)")
    _ax1.set_xlabel("Mean enrichment vs. E. coli pangenome", fontsize=12)
    _ax1.set_ylabel("")
    _ax1.set_title("GO Term Enrichment: v3 (minimal genome)", fontsize=14, fontweight="bold")
    _ax1.set_xlim(0.7, 1.1)
    _ax1.legend()
    _ax1.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    _fig1.gca()
    return


@app.cell
def _(go_plot_df, plt):
    # Plot 2: Mean enrichment for random
    _fig2, _ax2 = plt.subplots(figsize=(8, 6))
    _random_data = go_plot_df[go_plot_df["variant"] == "random"].sort_values("mean_enrichment")

    _bars = _ax2.barh(_random_data["go_name"], _random_data["mean_enrichment"], color="#7f7f7f", alpha=0.8)
    _ax2.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Baseline (1.0)")
    _ax2.set_xlabel("Mean enrichment vs. E. coli pangenome", fontsize=12)
    _ax2.set_ylabel("")
    _ax2.set_title("GO Term Enrichment: random (random gene selection)", fontsize=14, fontweight="bold")
    _ax2.set_xlim(0.7, 1.1)
    _ax2.legend()
    _ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    _fig2.gca()
    return


@app.cell
def _(go_plot_df, plt):
    # Plot 3: Significance for v3
    _fig3, _ax3 = plt.subplots(figsize=(8, 6))
    _v3_sig = go_plot_df[go_plot_df["variant"] == "v3"].sort_values("pct_significant")

    _bars = _ax3.barh(_v3_sig["go_name"], _v3_sig["pct_significant"], color="#d62728", alpha=0.8)
    _ax3.set_xlabel("% samples with FDR < 0.05", fontsize=12)
    _ax3.set_ylabel("")
    _ax3.set_title("Statistical Significance: v3", fontsize=14, fontweight="bold")
    _ax3.set_xlim(0, 100)
    _ax3.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    _fig3.gca()
    return


@app.cell
def _(go_plot_df, plt):
    # Plot 4: Significance for random
    _fig4, _ax4 = plt.subplots(figsize=(8, 6))
    _random_sig = go_plot_df[go_plot_df["variant"] == "random"].sort_values("pct_significant")

    _bars = _ax4.barh(_random_sig["go_name"], _random_sig["pct_significant"], color="#7f7f7f", alpha=0.8)
    _ax4.set_xlabel("% samples with FDR < 0.05", fontsize=12)
    _ax4.set_ylabel("")
    _ax4.set_title("Statistical Significance: random", fontsize=14, fontweight="bold")
    _ax4.set_xlim(0, 100)
    _ax4.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    _fig4.gca()
    return


@app.cell
def _():
    print("Test cell added to bottom of notebook")
    return


@app.cell
def _():
    print("Test cell added to bottom of notebook - nb test successful!")
    return


if __name__ == "__main__":
    app.run()
