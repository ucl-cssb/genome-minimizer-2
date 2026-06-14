import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _():
    import json
    import re
    import time
    from pathlib import Path

    import altair as alt
    import cobra
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import requests

    return Path, alt, cobra, mo, np, pd, pl, re, requests, time


@app.cell
def _(mo):
    mo.md(r"""
    # Two-tier eval for VAE-designed genomes

    Lightweight replacement for the vEcoli whole-cell-model eval: KEGG
    functional coverage plus iML1515 FBA. Both are deterministic, cached, and
    runnable in minutes.

    - **Tier 1 — KEGG module coverage.** ≈110 small functional modules.
    - **Tier 2 — iML1515 FBA.** Knock out absent metabolic genes and solve
      for growth on glucose minimal medium.

    Sources compared throughout:
    - **real** — 100 strains sampled from the pangenome matrix
    - **random** — frequency-weighted core+accessory baseline
    - **[v3](https://huggingface.co/UCL-CSSB/genome-minimizer-2/tree/v3)** — HF UCL-CSSB `v3/final.pt` (epoch 2363, no essential-gene loss; latent_dim=32, hidden_dim=512), 100 samples, seed 42

    v3 is sampled with essential gene repair, meaning the lit essential genes are added back in at inference time as a post-processing step. "Random" means we took the core genome then appended accessory genes to it at random until we hit the expected number of total genes.

    The `real` cohort is a reference distribution sampled from the same
    pangenome used for training, not an independent holdout. That is fine for
    asking whether generated genomes look biologically plausible relative to
    known *E. coli* gene-content variation; it should not be presented as a
    generalization test.
    """)
    return


@app.cell
def _(Path):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    KEGG_CACHE = DATA_DIR / "kegg"
    FBA_DIR = DATA_DIR / "fba"
    EVAL_DATA = PROJECT_ROOT / "evaluation" / "data"

    KEGG_CACHE.mkdir(parents=True, exist_ok=True)
    return DATA_DIR, EVAL_DATA, FBA_DIR, KEGG_CACHE

@app.cell
def _(mo, DATA_DIR, gene_order, np, pl):
    _prev_path = DATA_DIR / "gene_prevalence.npy"
    _gene_freq = np.load(_prev_path)

    sl_core_threshold = 0.95
    sl_core_mask = _gene_freq > sl_core_threshold

    core_genome_summary_df = pl.DataFrame(
        {
            "definition": ["> 95% prevalence"],
            "n_core_genes": [int(sl_core_mask.sum())],
            "n_total_genes": [int(len(gene_order))],
            "pct_pangenome": [round(float(100 * sl_core_mask.mean()), 2)],
        }
    )

    print(
        f"Core genome size using >95% prevalence: "
        f"{int(sl_core_mask.sum())} genes"
    )

    mo.md(
        f"""
        ### Core genome size

        Using the notebook's random-baseline definition of the core genome:

        - **Core genes >95% prevalence:** `{core_genome_summary_df['n_core_genes'][0]}`
        - **Total pangenome genes:** `{core_genome_summary_df['n_total_genes'][0]}`
        """
    )

    core_genome_summary_df
    return core_genome_summary_df, sl_core_mask

@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Pull KEGG module definitions

    KEGG **modules** (M-numbers) are small functional units — e.g. M00001 =
    "Glycolysis (Embden-Meyerhof pathway), glucose => pyruvate". They cover
    525 *E. coli* genes across 112 modules with median size 5 genes.

    Three KEGG REST hits:

    1. `list/module` → all 570 KEGG modules with names
    2. `list/eco` → every *E. coli* gene KEGG knows about, with its primary
       gene symbol (used for the symbol → b-number crosswalk in §2)
    3. `link/module/eco` → one bulk file mapping every *E. coli* gene to
       every module it appears in

    All responses are cached as flat TSVs under `data/kegg/`. Re-running the
    notebook hits zero network. Cache is committed so the analysis is
    reproducible without internet access.
    """)
    return


@app.cell
def _(KEGG_CACHE, requests, time):
    KEGG_BASE = "https://rest.kegg.jp"

    def fetch_kegg(endpoint: str, cache_path):
        if cache_path.exists():
            return cache_path.read_text()
        url = f"{KEGG_BASE}/{endpoint}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        text = r.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
        time.sleep(0.34)
        return text

    modules_text = fetch_kegg("list/module", KEGG_CACHE / "modules.tsv")
    eco_genes_text = fetch_kegg("list/eco", KEGG_CACHE / "eco_genes.tsv")
    eco_module_links_text = fetch_kegg(
        "link/module/eco", KEGG_CACHE / "eco_module_links.tsv"
    )
    return eco_genes_text, eco_module_links_text, modules_text


@app.cell
def _(eco_module_links_text):
    module_genes = {}
    for _line in eco_module_links_text.strip().splitlines():
        _parts = _line.split("\t")
        if len(_parts) < 2:
            continue
        _gene = _parts[0].removeprefix("eco:")
        _mid = _parts[1].removeprefix("md:eco_").removeprefix("md:")
        module_genes.setdefault(_mid, set()).add(_gene)

    _sizes = sorted(len(v) for v in module_genes.values())
    print(
        f"E. coli modules: {len(module_genes)} | "
        f"unique genes across modules: {len({g for s in module_genes.values() for g in s})} | "
        f"median module size: {_sizes[len(_sizes) // 2]} genes | "
        f"min={_sizes[0]} max={_sizes[-1]}"
    )
    return (module_genes,)


@app.cell
def _(module_genes, modules_text, pl):
    _eco_module_ids = set(module_genes.keys())
    _rows = []
    for _line in modules_text.strip().splitlines():
        _parts = _line.split("\t")
        if len(_parts) < 2:
            continue
        _mid = _parts[0].strip()
        if _mid in _eco_module_ids:
            _rows.append({"module_id": _mid, "name": _parts[1]})

    modules_df = pl.DataFrame(_rows)
    print(f"E. coli modules with names resolved: {modules_df.height}")
    modules_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Crosswalk pangenome gene names → KEGG b-numbers

    The pangenome (`F4_complete_presence_absence.csv`) labels genes with
    Panaroo-style symbols (e.g. `accA`, `aaeA_1`, `aaeA_2` for paralogs) plus
    ~45k `group_XXXX` clusters for accessory genes that didn't get a name.
    KEGG indexes everything by *b-number* (e.g. `b0185` = `accA`).

    For each named pangenome gene we (1) try an exact symbol match against
    KEGG, (2) try after stripping the Panaroo paralog suffix `_N`. The
    `group_XXXX` clusters are skipped — they're accessory and almost never
    in KEGG metabolic modules anyway.

    The unmatched named genes are dominated by conjugation / plasmid genes
    (`virB*`, `traD*`, `mbeA`, …) which is expected: KEGG metabolic modules
    don't include horizontal-transfer machinery.
    """)
    return


@app.cell
def _(eco_genes_text):
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

    print(f"KEGG symbol→b-number entries: {len(symbol_to_bnum)}")
    return (symbol_to_bnum,)


@app.cell
def _(DATA_DIR, pd, re):
    def normalize_gene(name: str) -> str:
        return re.sub(r"_\d+$", "", name).lower()

    pangenome_index = pd.read_csv(
        DATA_DIR / "F4_complete_presence_absence.csv",
        index_col=0,
        usecols=[0],
    ).index
    pangenome_genes = [g for g in pangenome_index if g != "Lineage"]

    n_named = sum(1 for g in pangenome_genes if not g.startswith("group_"))
    n_groups = sum(1 for g in pangenome_genes if g.startswith("group_"))
    print(
        f"Pangenome: {len(pangenome_genes)} entries "
        f"({n_named} named, {n_groups} group_XXXX clusters)"
    )
    return normalize_gene, pangenome_genes


@app.cell
def _(normalize_gene, pangenome_genes, symbol_to_bnum):
    pangenome_to_bnum = {}
    unmatched_named = []
    for _g in pangenome_genes:
        if _g.startswith("group_"):
            continue
        _key = _g.lower()
        if _key in symbol_to_bnum:
            pangenome_to_bnum[_g] = symbol_to_bnum[_key]
            continue
        _norm = normalize_gene(_g)
        if _norm in symbol_to_bnum:
            pangenome_to_bnum[_g] = symbol_to_bnum[_norm]
            continue
        unmatched_named.append(_g)

    n_named_total = sum(1 for g in pangenome_genes if not g.startswith("group_"))
    print(
        f"Crosswalk: {len(pangenome_to_bnum)} / {n_named_total} named pangenome genes "
        f"matched to KEGG ({len(pangenome_to_bnum) / n_named_total * 100:.1f}%)"
    )
    print(f"Unique b-numbers covered: {len(set(pangenome_to_bnum.values()))}")
    print(f"\nFirst 20 unmatched named genes: {unmatched_named[:20]}")
    assert 2000 <= len(pangenome_to_bnum) <= 6000, (
        f"Match count {len(pangenome_to_bnum)} outside sanity range [2000, 6000] — "
        "inspect crosswalk before proceeding."
    )
    return (pangenome_to_bnum,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Load all genome samples

    Provenance check:
    - `v3` reproduces UCL-CSSB `genome-minimizer-2` branch `v3`, `final.pt` (epoch 2363), sampled with seed 42.
    """)
    return


@app.cell
def _(DATA_DIR, EVAL_DATA, np, pd):
    def _load_real_strains(n_strains: int = 100, seed: int = 42):
        path = DATA_DIR / "F4_complete_presence_absence.csv"
        header = pd.read_csv(path, nrows=0).columns.tolist()
        strain_cols = [c for c in header[1:]]
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(strain_cols), size=n_strains, replace=False)
        chosen_names = [strain_cols[i] for i in sorted(chosen)]
        df = pd.read_csv(path, index_col=0, usecols=[header[0], *chosen_names])
        df = df.drop(index="Lineage", errors="ignore")
        out = []
        for col in df.columns:
            present = df.index[df[col].astype(int).values == 1].tolist()
            out.append((col, present))
        return out

    real_strains = _load_real_strains(n_strains=100, seed=42)

    def _load(p):
        return list(np.load(p, allow_pickle=True))

    sample_sources = {
        "real": [(name, genes) for name, genes in real_strains],
        "random": [
            (f"random_{i:03d}", list(g))
            for i, g in enumerate(
                _load(EVAL_DATA / "random" / "random_gene_lists_with_essentials.npy")
            )
        ],
        "v3": [
            (f"v3_{i:03d}", list(g))
            for i, g in enumerate(
                _load(EVAL_DATA / "v3" / "v3_gene_lists_with_essentials.npy")
            )
        ],
    }
    for _src, _items in sample_sources.items():
        _sizes = [len(genes) for _, genes in _items]
        print(
            f"{_src:>8s}: n={len(_items):3d}  genes/genome mean={int(np.mean(_sizes))} "
            f"min={min(_sizes)} max={max(_sizes)}"
        )
    return (sample_sources,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Genome-size distribution

    First sanity check: generated genomes should land in a plausible gene-count
    range before we interpret functional coverage. The random baseline is flat
    because it was generated with a fixed `--target-gene-count`.
    """)
    return


@app.cell
def _(alt, np, pl, sample_sources):
    _size_rows = [
        {"source": _src, "genome_id": _gid, "n_genes": len(_genes)}
        for _src, _items in sample_sources.items()
        for _gid, _genes in _items
    ]
    sizes_df = pl.DataFrame(_size_rows)

    _SRC_COLORS = {
        "real": "#2ca02c",
        "random": "#d62728",
        "v3": "#1f77b4",
    }
    _SRC_ORDER = ["real", "random", "v3"]

    _all_n = sizes_df["n_genes"].to_numpy()
    _bin_edges = np.linspace(_all_n.min() - 1, _all_n.max() + 1, 41)

    _hist = (
        alt.Chart(sizes_df.to_pandas())
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "n_genes:Q",
                bin=alt.Bin(extent=[float(_bin_edges[0]), float(_bin_edges[-1])], step=float(_bin_edges[1] - _bin_edges[0])),
                title="genes per genome",
            ),
            y=alt.Y("count():Q", title="# genomes"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=_SRC_ORDER, range=[_SRC_COLORS[s] for s in _SRC_ORDER]),
            ),
            row=alt.Row("source:N", sort=_SRC_ORDER),
        )
        .properties(width=600, height=80)
        .resolve_scale(y="independent")
    )
    _hist
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Score every genome against every module

    For each genome we map its gene names to KEGG b-numbers, then for each
    of the ~110 *E. coli* modules compute

    $$
    \text{completeness}_{g, m} = \frac{|G_g \cap M_m|}{|M_m|}
    $$

    where $G_g$ is the set of b-numbers in genome $g$ and $M_m$ is the set
    of b-numbers KEGG assigns to module $m$. Result lives in `long_df` —
    one row per (genome, module).

    `summary_df` rolls up to one row per genome with two thresholds:
    **# modules ≥ 80% complete** (lenient — allows one missing gene in
    a 5-gene module) and **# modules = 100% complete (≥ 99% threshold)**
    (strict — every gene present). Modules are small enough that 99% is
    effectively the "fully reconstructed" criterion.
    """)
    return


@app.cell
def _(module_genes, pangenome_to_bnum, pl, sample_sources):
    def score_genome(gene_names):
        bnums = set()
        for g in gene_names:
            b = pangenome_to_bnum.get(g)
            if b is not None:
                bnums.add(b)
        return {
            pid: len(bnums & pgenes) / len(pgenes) if pgenes else 0.0
            for pid, pgenes in module_genes.items()
        }

    rows = []
    for _src, _items in sample_sources.items():
        for _gid, _genes in _items:
            _scores = score_genome(_genes)
            for _pid, _frac in _scores.items():
                rows.append(
                    {
                        "source": _src,
                        "genome_id": _gid,
                        "module_id": _pid,
                        "completeness": _frac,
                    }
                )

    long_df = pl.DataFrame(rows)
    print(f"long_df: {long_df.shape}")
    long_df.head()
    return (long_df,)


@app.cell
def _(long_df, pl):
    summary_df = (
        long_df.group_by(["source", "genome_id"])
        .agg(
            [
                pl.col("completeness").mean().alias("mean_completeness"),
                (pl.col("completeness") >= 0.8).sum().alias("n_modules_80pct"),
                (pl.col("completeness") >= 0.99).sum().alias("n_modules_100pct"),
            ]
        )
        .sort(["source", "genome_id"])
    )
    summary_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Distribution: modules above the threshold

    The default 0.99 threshold is strict: for median 5-gene KEGG modules it is
    effectively "all genes present." The random baseline is strong because it
    keeps the full >95%-prevalence core genome and frequency-weights accessory
    genes; it is not a naive uniform-random baseline.
    """)
    return


@app.cell
def _(mo):
    threshold = mo.ui.slider(
        0.5, 1.0, value=0.99, step=0.01, label="module completeness threshold"
    )
    threshold
    return (threshold,)


@app.cell
def _(alt, long_df, pl, threshold):
    _t = threshold.value
    per_genome = (
        long_df.group_by(["source", "genome_id"])
        .agg((pl.col("completeness") >= _t).sum().alias("n_modules"))
        .sort(["source", "n_modules"])
    )

    _real_min = per_genome.filter(pl.col("source") == "real")["n_modules"].min()

    _SOURCE_ORDER = ["real", "random", "v3"]
    _SOURCE_COLORS = {
        "real": "#2ca02c",
        "random": "#d62728",
        "v3": "#1f77b4",
    }

    _violin = (
        alt.Chart(per_genome.to_pandas())
        .transform_density(
            "n_modules",
            as_=["n_modules", "density"],
            groupby=["source"],
            extent=[0, 112],
            steps=200,
            bandwidth=2.0,
            counts=True,
        )
        .mark_area(orient="horizontal")
        .encode(
            y=alt.Y("n_modules:Q", title=f"# modules ≥ {_t:.2f} complete"),
            x=alt.X(
                "density:Q",
                stack="center",
                impute=None,
                title=None,
                axis=alt.Axis(labels=False, ticks=False, grid=False),
            ),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=list(_SOURCE_COLORS.keys()),
                    range=list(_SOURCE_COLORS.values()),
                ),
                legend=None,
            ),
            column=alt.Column(
                "source:N", sort=_SOURCE_ORDER, header=alt.Header(titleOrient="bottom")
            ),
        )
        .properties(
            width=120,
            height=300,
            title=f"Module coverage distribution (threshold = {_t:.2f})",
        )
    )

    print(f"min(real n_modules) at threshold={_t:.2f}: {_real_min}")
    _violin
    return (per_genome,)


@app.cell
def _(per_genome, pl):
    descriptive = (
        per_genome.group_by("source")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("n_modules").mean().round(1).alias("mean"),
                pl.col("n_modules").quantile(0.25).alias("q25"),
                pl.col("n_modules").median().alias("median"),
                pl.col("n_modules").quantile(0.75).alias("q75"),
                pl.col("n_modules").min().alias("min"),
                pl.col("n_modules").max().alias("max"),
            ]
        )
        .sort("source")
    )
    descriptive
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Tier 2 — FBA on iML1515

    Mechanistic check via flux balance analysis. iML1515 is the standard
    genome-scale metabolic model of *E. coli* K-12 MG1655 (1,516 genes,
    2,712 reactions, 1,877 metabolites; Monk et al. 2017, *Nat Biotechnol*).
    Wild-type growth on glucose minimal medium is ≈ 0.877 h⁻¹.

    For each genome we knock out every iML1515 gene that is *absent* from
    the genome's gene list, then solve the LP. The growth rate that comes
    out is a mechanistic upper bound on viability — non-zero means the
    genome has the metabolic capacity to grow on the default medium.

    Caveats:
    - iML1515 covers ~1,500 of the ~4,500 *E. coli* genes — only metabolic
      enzymes / transporters. Regulatory, structural, and information-
      processing genes aren't in scope. So FBA can say "metabolism is
      intact" but not "the cell is viable."
    - Default medium is glucose minimal aerobic. Different conditions ⇒
      different essentiality patterns.
    """)
    return


@app.cell
def _(FBA_DIR, cobra):
    iml = cobra.io.read_sbml_model(str(FBA_DIR / "iML1515.xml"))
    iml_gene_ids = {g.id for g in iml.genes}
    wt_growth = iml.optimize().objective_value
    print(
        f"iML1515: {len(iml.genes)} genes, {len(iml.reactions)} reactions | "
        f"WT growth = {wt_growth:.4f} h⁻¹"
    )
    return iml, wt_growth


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Build a richer iML1515 ↔ pangenome crosswalk

    The KEGG `list/eco` endpoint only gives the *primary* gene symbol per
    b-number. The pangenome often uses a synonym instead — e.g. iML1515
    says `b3639 = dfp`, but the pangenome calls it `coaBC`; iML1515 says
    `b0174 = ispU`, pangenome calls it `uppS`. With KEGG primary symbols
    only, every real strain looks like it's missing ~10 essentials and
    FBA flatlines to zero.

    Fix: pull every (`g.name`, `g.annotation['refseq_synonym']`) tuple
    directly from the iML1515 SBML and match against the pangenome's
    normalized names (lowercase + strip Panaroo `_N` suffix). One
    special case: `s0001` is iML1515's pseudo-gene for spontaneous
    reactions — never knock it out.
    """)
    return


@app.cell
def _(iml, normalize_gene, pangenome_genes):
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

    pangenome_norm_set = {normalize_gene(g) for g in pangenome_genes}
    iml_pangenome_known = {
        bnum
        for bnum, names in iml_bnum_to_names.items()
        if names & pangenome_norm_set
    }
    iml_unmatched = set(iml_bnum_to_names) - iml_pangenome_known
    print(
        f"iML1515: {len(iml_bnum_to_names)} gene b-numbers (excl. s0001). "
        f"{len(iml_pangenome_known)} ({len(iml_pangenome_known) / len(iml_bnum_to_names) * 100:.1f}%) "
        f"have at least one synonym in the pangenome universe."
    )
    print(
        f"{len(iml_unmatched)} iML1515 genes have no pangenome name match at all "
        "(naming gap) — treated as always present."
    )
    return iml_bnum_to_names, iml_pangenome_known


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Run FBA for every genome

    Only the **iML1515 genes whose name appears anywhere in the pangenome**
    are eligible for knock-out. iML1515 genes with no pangenome name match
    at all are treated as always present — we can't tell apart a real
    absence from a naming gap, so we give the benefit of the doubt.

    For each genome:

    1. Normalize all of its gene names (lowercase + strip `_N`).
    2. For each *eligible* iML1515 b-number, mark it *present* if any of
       its synonyms appears in that set.
    3. Knock out the eligible-and-absent b-numbers inside a `with model:`
       block and solve the LP.

    Each FBA solve is ~10 ms; total runtime for ~400 genomes is under a
    minute.
    """)
    return


@app.cell
def _(
    iml,
    iml_bnum_to_names,
    iml_pangenome_known,
    normalize_gene,
    pl,
    sample_sources,
):
    fba_rows = []
    for _src, _items in sample_sources.items():
        for _gid, _genes in _items:
            _genome_norm = {normalize_gene(g) for g in _genes}
            present_b = {
                bnum
                for bnum in iml_pangenome_known
                if iml_bnum_to_names[bnum] & _genome_norm
            }
            absent_in_iml = iml_pangenome_known - present_b
            with iml as m:
                for _gene_id in absent_in_iml:
                    m.genes.get_by_id(_gene_id).knock_out()
                _sol = m.optimize()
                _growth = (
                    _sol.objective_value
                    if _sol.status == "optimal" and _sol.objective_value is not None
                    else 0.0
                )
            _growth = float(_growth)
            if abs(_growth) < 1e-6:
                _growth = 0.0
            fba_rows.append(
                {
                    "source": _src,
                    "genome_id": _gid,
                    "n_iml_present": len(present_b),
                    "n_iml_absent": len(absent_in_iml),
                    "growth_rate": _growth,
                }
            )

    fba_df = pl.DataFrame(fba_rows)
    print(f"fba_df: {fba_df.shape}")
    fba_df.head()
    return (fba_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Growth-rate distribution

    Violin plot of predicted growth rate per genome — denser regions are
    wider. Dashed line: WT iML1515 growth (≈ 0.877). Anything > 0 is
    mechanistically viable on glucose minimal; anything close to WT is
    metabolically intact. Use this view rather than a strip plot — the
    growth distribution is multi-modal (zero, mid-cost, near-WT) and dot
    overlap obscures the structure.
    """)
    return


@app.cell
def _(alt, fba_df, wt_growth):
    _SRC_ORDER = ["real", "random", "v3"]
    _SRC_COLORS = {
        "real": "#2ca02c",
        "random": "#d62728",
        "v3": "#1f77b4",
    }

    violin_fba = (
        alt.Chart(fba_df.to_pandas())
        .transform_density(
            "growth_rate",
            as_=["growth_rate", "density"],
            groupby=["source"],
            extent=[-0.05, 0.95],
            steps=300,
            bandwidth=0.025,
            counts=True,
        )
        .mark_area(orient="horizontal")
        .encode(
            y=alt.Y("growth_rate:Q", title="FBA predicted growth (h⁻¹)"),
            x=alt.X(
                "density:Q",
                stack="center",
                impute=None,
                title=None,
                axis=alt.Axis(labels=False, ticks=False, grid=False),
            ),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=list(_SRC_COLORS.keys()),
                    range=list(_SRC_COLORS.values()),
                ),
                legend=None,
            ),
            column=alt.Column(
                "source:N", sort=_SRC_ORDER, header=alt.Header(titleOrient="bottom")
            ),
        )
        .properties(width=120, height=320, title="iML1515 FBA growth distribution")
    )
    print(f"WT iML1515 growth: {wt_growth:.4f} h⁻¹")
    violin_fba
    return


@app.cell
def _(fba_df, pl):
    fba_summary = (
        fba_df.group_by("source")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("growth_rate").mean().round(3).alias("mean_growth"),
                pl.col("growth_rate").median().round(3).alias("median_growth"),
                pl.col("growth_rate").std().round(3).alias("std_growth"),
                (pl.col("growth_rate") > 0.01).sum().alias("n_viable"),
                (pl.col("growth_rate") > 0.1).sum().alias("n_growing"),
            ]
        )
        .with_columns(
            (pl.col("n_viable") / pl.col("n") * 100).round(1).alias("pct_viable"),
            (pl.col("n_growing") / pl.col("n") * 100).round(1).alias("pct_growing"),
        )
        .sort("source")
    )
    fba_summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Tier-1 vs Tier-2 agreement

    Compact joint view: module coverage at ≥80% versus FBA growth. Density in
    the upper-right means the two tiers agree.
    """)
    return


@app.cell
def _(fba_df, long_df, pl):
    _t80 = (
        long_df.group_by(["source", "genome_id"])
        .agg((pl.col("completeness") >= 0.8).sum().alias("n_modules_80pct"))
    )
    joined = fba_df.join(_t80, on=["source", "genome_id"])
    return (joined,)


@app.cell
def _(alt, joined):
    _SRC_ORDER = ["real", "random", "v3"]
    (
        alt.Chart(joined.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X(
                "n_modules_80pct:Q",
                bin=alt.Bin(maxbins=20, extent=[60, 95]),
                title="Tier 1: # modules ≥ 80% complete",
            ),
            y=alt.Y(
                "growth_rate:Q",
                bin=alt.Bin(maxbins=15, extent=[-0.05, 0.95]),
                title="Tier 2: FBA growth (h⁻¹)",
            ),
            color=alt.Color(
                "count():Q",
                scale=alt.Scale(scheme="viridis", type="symlog"),
                title="# genomes",
            ),
            tooltip=[
                alt.Tooltip("source:N"),
                alt.Tooltip("count():Q", title="n genomes"),
            ],
        )
        .properties(width=240, height=240)
        .facet(column=alt.Column("source:N", sort=_SRC_ORDER))
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 10. Operating point: the 0.5 decode threshold is the problem

    Everything above thresholds the VAE decoder at 0.5 — an arbitrary cut that
    discards the model's *continuous* gene ranking. It slices through the band of
    genes v3 scores 0.4–0.5, which turn out to be essential metabolic genes, so
    the 0.5 genome looks no better than random. Sweeping the threshold traces the
    genome-size ↔ viability frontier; comparing against a **size-matched** random
    baseline at each point is the fair test of whether the VAE's gene *selection*
    beats a frequency-weighted baseline of the same gene *count*.
    """)
    return


@app.cell
def _(DATA_DIR, EVAL_DATA, np, pangenome_genes):
    import sys as _sys

    _root = DATA_DIR.parent
    for _p in (str(_root / "src"), str(_root / "evaluation")):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    # Reuse the canonical sample/repair code (same source of truth as the data
    # files generated by genome_minimizer_2.sampling / the binary_converter
    # pipeline) rather than reimplementing it here.
    from genome_minimizer_2.explore_data.binary_converter import (
        add_essential_genes,
        load_essential_set,
    )
    from genome_minimizer_2.sampling import compute_gene_frequencies, sample_random_genomes

    v3_continuous = np.load(EVAL_DATA / "v3" / "v3_samples.npy")
    gene_order = np.asarray(pangenome_genes)
    assert v3_continuous.shape[1] == gene_order.size, (
        "v3 sample columns must align with the pangenome gene order"
    )

    _essential_set = load_essential_set(str(DATA_DIR / "essential_genes.csv"))

    # Per-gene frequencies (prevalence), cached to avoid re-reading the full
    # matrix. The cache stores exactly what compute_gene_frequencies returns, so
    # the cached and canonical paths sample identically.
    _prev_path = DATA_DIR / "gene_prevalence.npy"
    if _prev_path.exists():
        _gene_names, _gene_freq = gene_order, np.load(_prev_path)
    else:
        _gene_names, _gene_freq = compute_gene_frequencies(
            DATA_DIR / "F4_complete_presence_absence.csv"
        )
        np.save(_prev_path, _gene_freq)

    def make_random(target_size, n=50, seed=42):
        return sample_random_genomes(
            _gene_names, _gene_freq, int(target_size), n, np.random.default_rng(seed)
        )

    def repair_essentials(gene_lists):
        return [add_essential_genes(g, _essential_set) for g in gene_lists]

    return gene_order, make_random, repair_essentials, v3_continuous

@app.cell
def _(
    iml,
    iml_bnum_to_names,
    iml_pangenome_known,
    module_genes,
    normalize_gene,
    pangenome_to_bnum,
):
    def fba_growth(gene_names):
        _norm = {normalize_gene(g) for g in gene_names}
        _present = {b for b in iml_pangenome_known if iml_bnum_to_names[b] & _norm}
        with iml as _m:
            for _absent in iml_pangenome_known - _present:
                _m.genes.get_by_id(_absent).knock_out()
            _sol = _m.optimize()
            _val = _sol.objective_value if _sol.status == "optimal" and _sol.objective_value else 0.0
        return 0.0 if abs(_val) < 1e-6 else float(_val)

    def n_modules_complete(gene_names, frac):
        _bn = {pangenome_to_bnum[g] for g in gene_names if g in pangenome_to_bnum}
        return sum(len(_bn & mg) / len(mg) >= frac for mg in module_genes.values() if mg)

    return fba_growth, n_modules_complete


@app.cell
def _(
    fba_growth,
    gene_order,
    make_random,
    n_modules_complete,
    np,
    pl,
    repair_essentials,
    v3_continuous,
):
    _grid = [0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
    _rows = []
    for _t in _grid:
        _v3 = repair_essentials(
            [gene_order[v3_continuous[i] > _t].tolist() for i in range(v3_continuous.shape[0])]
        )
        _size = int(round(np.mean([len(x) for x in _v3])))
        _rnd = repair_essentials(make_random(_size, n=50, seed=42))
        for _src, _gl in (("v3", _v3), ("random", _rnd)):
            _growth = np.array([fba_growth(x) for x in _gl])
            _rows.append(
                {
                    "threshold": _t,
                    "source": _src,
                    "genome_size": int(round(np.mean([len(x) for x in _gl]))),
                    "n_modules_80": round(float(np.mean([n_modules_complete(x, 0.8) for x in _gl])), 1),
                    "fba_viable_pct": round(float(100 * np.mean(_growth > 0.01)), 0),
                }
            )
    sweep_df = pl.DataFrame(_rows)
    sweep_df
    return (sweep_df,)


@app.cell
def _(alt, sweep_df):
    alt.Chart(sweep_df.to_pandas()).mark_line(point=True).encode(
        x=alt.X("genome_size:Q", title="genes per genome", scale=alt.Scale(zero=False)),
        y=alt.Y("fba_viable_pct:Q", title="FBA viable (%)"),
        color=alt.Color(
            "source:N",
            scale=alt.Scale(domain=["v3", "random"], range=["#1f77b4", "#d62728"]),
        ),
        tooltip=["threshold", "source", "genome_size", "fba_viable_pct", "n_modules_80"],
    ).properties(
        width=480, height=300, title="Size–viability frontier: v3 vs size-matched random"
    )
    return


@app.cell
def _(mo, pl, sweep_df):
    _v3 = sweep_df.filter(pl.col("source") == "v3")
    _viable = _v3.filter(pl.col("fba_viable_pct") >= 60).sort("genome_size")
    best_threshold = (
        float(_viable["threshold"][0])
        if _viable.height
        else float(_v3.sort("fba_viable_pct", descending=True)["threshold"][0])
    )
    best_size = int(_v3.filter(pl.col("threshold") == best_threshold)["genome_size"][0])
    mo.md(
        f"""
        ### Chosen operating point: threshold **{best_threshold:.2f}** → ~{best_size} genes

        Rule: the most aggressive cut (smallest genome) that still keeps a clear
        majority (≥60%) of samples FBA-viable — ~{100 * (1 - best_size / 4739):.0f}% smaller
        than a real strain, viable, and ahead of size-matched random on both tiers.
        """
    )
    return best_size, best_threshold


@app.cell
def _(mo):
    mo.md(r"""
    # 11. Two-tier eval at the chosen operating point

    Re-run both tiers with v3 decoded at the chosen threshold and the random
    baseline regenerated to the **same** genome size, so the comparison isolates
    gene *selection* rather than gene *count*. `real` is the wild-type reference.
    """)
    return


@app.cell
def _(
    best_size,
    best_threshold,
    fba_growth,
    gene_order,
    make_random,
    n_modules_complete,
    np,
    pl,
    repair_essentials,
    sample_sources,
    v3_continuous,
):
    _real = [genes for _, genes in sample_sources["real"]]
    _v3 = repair_essentials(
        [gene_order[v3_continuous[i] > best_threshold].tolist() for i in range(v3_continuous.shape[0])]
    )
    _rnd = repair_essentials(make_random(best_size, n=100, seed=42))
    _cohorts = [("real", _real), (f"random@{best_size}", _rnd), (f"v3@{best_threshold:.2f}", _v3)]

    _rows = []
    for _name, _gl in _cohorts:
        _growth = np.array([fba_growth(x) for x in _gl])
        _rows.append(
            {
                "cohort": _name,
                "n": len(_gl),
                "genome_size": int(round(np.mean([len(x) for x in _gl]))),
                "mean_modules_80": round(float(np.mean([n_modules_complete(x, 0.8) for x in _gl])), 1),
                "mean_modules_99": round(float(np.mean([n_modules_complete(x, 0.99) for x in _gl])), 1),
                "fba_viable_pct": round(float(100 * np.mean(_growth > 0.01)), 1),
                "median_growth": round(float(np.median(_growth)), 3),
            }
        )
    reanalysis_df = pl.DataFrame(_rows)
    reanalysis_df
    return (reanalysis_df,)


@app.cell
def _(alt, reanalysis_df):
    _pdf = reanalysis_df.to_pandas().melt(
        id_vars="cohort",
        value_vars=["fba_viable_pct", "mean_modules_80"],
        var_name="metric",
        value_name="value",
    )
    alt.Chart(_pdf).mark_bar().encode(
        x=alt.X("cohort:N", title=None, sort=None),
        y=alt.Y("value:Q", title=None),
        color=alt.Color("cohort:N", legend=None),
        column=alt.Column("metric:N", title=None),
        tooltip=["cohort", "metric", "value"],
    ).resolve_scale(y="independent").properties(width=170, height=240)
    return

# ---------------------------------------------------------------------
# 12. Synthetic lethality screen
# ---------------------------------------------------------------------

@app.cell
def _(mo):
    mo.md(r"""
    # 12. BioGRID negative-genetic-interaction screen

    For each sampled genome, count known
    pairwise incompatibility interactions for which both genes are absent.

    BioGRID did not contain rows explicitly labelled `Synthetic Lethality` in
    the *E. coli* file used here. We therefore use BioGRID `Negative Genetic`
    interactions as a pragmatic proxy for pairwise deletion risks.

    Expected input:

    `data/synthetic_lethality/ecoli_synthetic_lethal_pairs.tsv`

    Required columns:

    - `gene_a`
    - `gene_b`

    Optional columns such as `interaction_type`, `score`, `publication`, and
    `source` are preserved only in the source file; the screen itself needs only
    the gene-pair columns.
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    sl_dir = DATA_DIR / "synthetic_lethality"
    sl_dir.mkdir(parents=True, exist_ok=True)

    sl_pairs_path = sl_dir / "ecoli_synthetic_lethal_pairs.tsv"

    if not sl_pairs_path.exists():
        _template = pd.DataFrame(
            columns=["gene_a", "gene_b", "interaction_type", "score", "source"]
        )
        _template.to_csv(sl_pairs_path, sep="\t", index=False)
        raise FileNotFoundError(
            f"Created a template file at: {sl_pairs_path}\n"
            "Fill it with BioGRID E. coli Negative Genetic interaction pairs, "
            "then re-run this cell.\n"
            "Required columns: gene_a, gene_b"
        )

    sl_raw_pairs_df = pd.read_csv(sl_pairs_path, sep=None, engine="python")
    sl_raw_pairs_df.columns = [
        str(_c).strip().lower() for _c in sl_raw_pairs_df.columns
    ]

    _rename = {}
    for _c in sl_raw_pairs_df.columns:
        _c_clean = _c.replace("-", "_").replace(" ", "_")
        if _c_clean in {
            "gene1",
            "gene_1",
            "gene_a",
            "interactor_a",
            "interactora",
            "query",
            "query_gene",
        }:
            _rename[_c] = "gene_a"
        elif _c_clean in {
            "gene2",
            "gene_2",
            "gene_b",
            "interactor_b",
            "interactorb",
            "array",
            "array_gene",
            "target",
            "target_gene",
        }:
            _rename[_c] = "gene_b"

    sl_raw_pairs_df = sl_raw_pairs_df.rename(columns=_rename)

    if not {"gene_a", "gene_b"}.issubset(sl_raw_pairs_df.columns):
        raise ValueError(
            "Interaction file must contain columns called gene_a and gene_b, "
            "or recognisable alternatives such as gene1/gene2."
        )

    sl_raw_pairs_df = (
        sl_raw_pairs_df[["gene_a", "gene_b"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"Raw BioGRID interaction pairs loaded: {len(sl_raw_pairs_df)}")
    sl_raw_pairs_df.head()
    return sl_dir, sl_pairs_path, sl_raw_pairs_df


@app.cell
def _(normalize_gene, pangenome_genes, pangenome_to_bnum, sl_raw_pairs_df, pd, re):
    """
    Map BioGRID interaction-pair identifiers onto the pangenome naming system.

    Downstream, the screen uses normalized pangenome names:
    - lowercase
    - Panaroo suffixes such as _1, _2 stripped

    b-numbers are resolved through the existing pangenome_to_bnum crosswalk.
    """
    sl_pangenome_norm_set = {normalize_gene(_g) for _g in pangenome_genes}

    _bnum_to_pangenome_norms = {}
    for _gene, _bnum in pangenome_to_bnum.items():
        _bnum_to_pangenome_norms.setdefault(str(_bnum).lower(), set()).add(
            normalize_gene(_gene)
        )

    def _sl_clean_identifier(x):
        _x = str(x).strip()
        _x = _x.replace("eco:", "")
        _x = _x.replace("bnumber:", "")
        _x = _x.split(";")[0].split(",")[0].strip()
        return _x

    def _sl_resolve_identifier(x):
        """
        Return a set of normalized pangenome identifiers corresponding to x.

        Supports:
        - gene symbols, e.g. accA
        - Panaroo-like names, e.g. accA_1
        - b-numbers, e.g. b0185
        """
        _x = _sl_clean_identifier(x)
        _xl = _x.lower()

        if re.fullmatch(r"b\d{4}", _xl):
            return _bnum_to_pangenome_norms.get(_xl, {_xl})

        return {normalize_gene(_x)}

    _sl_mapped_pairs = []
    _sl_unmapped_pairs = []

    for _, _row in sl_raw_pairs_df.iterrows():
        _a_set = _sl_resolve_identifier(_row["gene_a"])
        _b_set = _sl_resolve_identifier(_row["gene_b"])

        _found = False
        for _a in _a_set:
            for _b in _b_set:
                if _a == _b:
                    continue
                if _a in sl_pangenome_norm_set and _b in sl_pangenome_norm_set:
                    _sl_mapped_pairs.append(tuple(sorted((_a, _b))))
                    _found = True

        if not _found:
            _sl_unmapped_pairs.append((_row["gene_a"], _row["gene_b"]))

    sl_pairs_df = (
        pd.DataFrame(_sl_mapped_pairs, columns=["gene_a", "gene_b"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    sl_unmapped_pairs_df = pd.DataFrame(
        _sl_unmapped_pairs, columns=["gene_a", "gene_b"]
    )

    print(f"Mapped/testable BioGRID interaction pairs: {len(sl_pairs_df)}")
    print(f"Unmapped/excluded raw pairs: {len(sl_unmapped_pairs_df)}")

    if len(sl_pairs_df) == 0:
        raise ValueError(
            "No BioGRID interaction pairs mapped to the pangenome identifiers. "
            "Check whether the input file uses E. coli gene symbols or b-numbers."
        )

    sl_pairs_df.head()
    return sl_pairs_df, sl_pangenome_norm_set, sl_unmapped_pairs_df


# @app.cell
# def _(
#     best_size,
#     best_threshold,
#     gene_order,
#     make_random,
#     repair_essentials,
#     sample_sources,
#     v3_continuous,
#     np,
# ):
#     """
#     Define cohorts for the BioGRID interaction screen.

#     Main analysis:
#     - v3 genomes decoded at the selected operating point

#     Controls:
#     - real genomes
#     - size-matched random genomes
#     """
#     sl_v3_best_gene_lists = repair_essentials(
#         [
#             gene_order[v3_continuous[_i] > best_threshold].tolist()
#             for _i in range(v3_continuous.shape[0])
#         ]
#     )

#     sl_random_best_gene_lists = repair_essentials(
#         make_random(best_size, n=100, seed=42)
#     )

#     sl_cohort_gene_lists = {
#         "real": [
#             (_gid, _genes)
#             for _gid, _genes in sample_sources["real"]
#         ],
#         "random_size_matched": [
#             (f"random_{_i:03d}", _genes)
#             for _i, _genes in enumerate(sl_random_best_gene_lists)
#         ],
#         f"v3_threshold_{best_threshold:.2f}": [
#             (f"v3_{_i:03d}", _genes)
#             for _i, _genes in enumerate(sl_v3_best_gene_lists)
#         ],
#     }

#     for _sl_cohort_name, _sl_items in sl_cohort_gene_lists.items():
#         _sl_sizes = [len(_genes) for _, _genes in _sl_items]
#         print(
#             f"{_sl_cohort_name}: n={len(_sl_items)}, "
#             f"mean genes={np.mean(_sl_sizes):.1f}, "
#             f"min={min(_sl_sizes)}, max={max(_sl_sizes)}"
#         )

#     return sl_cohort_gene_lists, sl_random_best_gene_lists, sl_v3_best_gene_lists

@app.cell
def _(sample_sources, np):
    """
    Define cohorts for the BioGRID interaction screen.

    This version uses the ORIGINAL v3 sampled genomes loaded in section 3:
        sample_sources["v3"]

    It does NOT use:
        - v3_continuous
        - best_threshold
        - best_size
        - repair_essentials
        - make_random

    Therefore the screen is applied to the original v3 gene lists, not the
    later threshold-swept operating point.
    """

    sl_cohort_gene_lists = {
        "real": [
            (_gid, _genes)
            for _gid, _genes in sample_sources["real"]
        ],
        "random_original": [
            (_gid, _genes)
            for _gid, _genes in sample_sources["random"]
        ],
        "v3_original": [
            (_gid, _genes)
            for _gid, _genes in sample_sources["v3"]
        ],
    }

    for _sl_cohort_name, _sl_items in sl_cohort_gene_lists.items():
        _sl_sizes = [len(_genes) for _, _genes in _sl_items]
        print(
            f"{_sl_cohort_name}: n={len(_sl_items)}, "
            f"mean genes={np.mean(_sl_sizes):.1f}, "
            f"min={min(_sl_sizes)}, max={max(_sl_sizes)}"
        )

    return sl_cohort_gene_lists

@app.cell
def _(normalize_gene, pd, sl_cohort_gene_lists, sl_pairs_df, sl_pangenome_norm_set):
    """
    Count, for each genome, the number of BioGRID interaction pairs where
    both genes are absent.
    """

    def sl_codeleted_interaction_hits(gene_list, interaction_pairs_df):
        _sl_present = {normalize_gene(_g) for _g in gene_list}
        _sl_absent = sl_pangenome_norm_set - _sl_present

        _sl_hits = interaction_pairs_df[
            interaction_pairs_df["gene_a"].isin(_sl_absent)
            & interaction_pairs_df["gene_b"].isin(_sl_absent)
        ].copy()

        if len(_sl_hits):
            _sl_hits["pair"] = _sl_hits["gene_a"] + "-" + _sl_hits["gene_b"]

        return _sl_hits

    _sl_screen_rows = []
    _sl_hit_rows = []

    for _sl_cohort_name, _sl_genomes in sl_cohort_gene_lists.items():
        for _sl_genome_id, _sl_genes in _sl_genomes:
            _sl_hits = sl_codeleted_interaction_hits(_sl_genes, sl_pairs_df)

            _sl_screen_rows.append(
                {
                    "cohort": _sl_cohort_name,
                    "genome_id": _sl_genome_id,
                    "n_genes": len(_sl_genes),
                    "n_testable_interaction_pairs": len(sl_pairs_df),
                    "n_codeleted_interaction_pairs": len(_sl_hits),
                }
            )

            for _, _sl_hit in _sl_hits.iterrows():
                _sl_hit_rows.append(
                    {
                        "cohort": _sl_cohort_name,
                        "genome_id": _sl_genome_id,
                        "gene_a": _sl_hit["gene_a"],
                        "gene_b": _sl_hit["gene_b"],
                        "pair": _sl_hit["gene_a"] + "-" + _sl_hit["gene_b"],
                    }
                )

    sl_screen_df = pd.DataFrame(_sl_screen_rows)
    sl_hit_df = pd.DataFrame(_sl_hit_rows)

    print(f"Screened genomes: {len(sl_screen_df)}")
    print(f"Total co-deleted BioGRID interaction hits: {len(sl_hit_df)}")

    sl_screen_df.head()
    return sl_codeleted_interaction_hits, sl_hit_df, sl_screen_df


@app.cell
def _(pl, sl_screen_df):
    """
    Cohort-level summary table.
    """
    sl_summary_df = (
        pl.from_pandas(sl_screen_df)
        .group_by("cohort")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("n_genes").mean().round(1).alias("mean_genes"),
                pl.col("n_codeleted_interaction_pairs")
                .mean()
                .round(2)
                .alias("mean_codeleted_interactions"),
                pl.col("n_codeleted_interaction_pairs")
                .median()
                .alias("median_codeleted_interactions"),
                pl.col("n_codeleted_interaction_pairs")
                .max()
                .alias("max_codeleted_interactions"),
                (pl.col("n_codeleted_interaction_pairs") > 0)
                .sum()
                .alias("n_with_any_codeleted_interaction"),
            ]
        )
        .with_columns(
            (
                pl.col("n_with_any_codeleted_interaction")
                / pl.col("n")
                * 100
            )
            .round(1)
            .alias("pct_with_any_codeleted_interaction")
        )
        .sort("cohort")
    )

    sl_summary_df
    return sl_summary_df


@app.cell
def _(alt, sl_screen_df):
    """
    Distribution of co-deleted BioGRID interaction pairs per genome.
    """
    sl_hist_chart = (
        alt.Chart(sl_screen_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "n_codeleted_interaction_pairs:Q",
                bin=alt.Bin(maxbins=30),
                title="# co-deleted BioGRID negative genetic interaction pairs",
            ),
            y=alt.Y("count():Q", title="# genomes"),
            color=alt.Color("cohort:N"),
            row=alt.Row("cohort:N"),
            tooltip=["cohort", "count()"],
        )
        .properties(
            width=520,
            height=90,
            title="BioGRID negative genetic interaction deletions per genome",
        )
        .resolve_scale(y="independent")
    )

    sl_hist_chart
    return sl_hist_chart


@app.cell
def _(sl_dir, sl_hit_df, sl_screen_df, sl_summary_df):
    """
    Write analysis outputs.
    """
    sl_dir.mkdir(parents=True, exist_ok=True)

    sl_per_genome_path = sl_dir / "biogrid_negative_genetic_screen_per_genome.tsv"
    sl_hits_path = sl_dir / "biogrid_negative_genetic_codeleted_pairs.tsv"
    sl_summary_path = sl_dir / "biogrid_negative_genetic_screen_summary.tsv"

    sl_screen_df.to_csv(sl_per_genome_path, sep="\t", index=False)
    sl_hit_df.to_csv(sl_hits_path, sep="\t", index=False)
    sl_summary_df.write_csv(sl_summary_path, separator="\t")

    print("Wrote:")
    print(f"- {sl_per_genome_path}")
    print(f"- {sl_hits_path}")
    print(f"- {sl_summary_path}")

    return sl_hits_path, sl_per_genome_path, sl_summary_path


@app.cell
def _(mo, sl_summary_df):
    sl_manuscript_text = """
    As an additional post hoc screen, we compared the sampled genomes against
    curated BioGRID E. coli negative genetic interactions. BioGRID did not
    contain interactions explicitly annotated as synthetic lethality in the
    E. coli file analysed; therefore, negative genetic interactions were used
    as a proxy for known pairwise incompatibilities that may constrain genome
    reduction. For each genome, we counted interaction pairs for which both
    genes were absent. This analysis provides a check for known pairwise
    deletion risks, while recognising that available interaction data are
    incomplete, condition-dependent, and do not capture higher-order deletion
    effects.
    """

    mo.md(
        f"""
        ## Suggested manuscript sentence

        {sl_manuscript_text}

        Cohort-level summary:

        {sl_summary_df}
        """
    )
    return sl_manuscript_text

# @app.cell
# def _(
#     gene_order,
#     make_random,
#     normalize_gene,
#     np,
#     pl,
#     repair_essentials,
#     sl_pairs_df,
#     sl_pangenome_norm_set,
#     v3_continuous,
# ):
#     """
#     Sweep VAE decode threshold and count co-deleted BioGRID interaction pairs.

#     This mirrors the FBA size-viability frontier:
#     - For each v3 decode threshold, generate v3 genomes.
#     - Compute their mean genome size.
#     - Generate random genomes matched to that mean size.
#     - Count known BioGRID negative-genetic-interaction pairs where both genes
#       are absent.
#     """

#     def _sl_count_codeleted_interactions(_gene_names):
#         _present = {normalize_gene(_g) for _g in _gene_names}
#         _absent = sl_pangenome_norm_set - _present

#         return int(
#             (
#                 sl_pairs_df["gene_a"].isin(_absent)
#                 & sl_pairs_df["gene_b"].isin(_absent)
#             ).sum()
#         )

#     _sl_grid = [0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
#     _sl_sweep_rows = []

#     for _sl_threshold in _sl_grid:
#         _sl_v3_gene_lists = repair_essentials(
#             [
#                 gene_order[v3_continuous[_i] > _sl_threshold].tolist()
#                 for _i in range(v3_continuous.shape[0])
#             ]
#         )

#         _sl_target_size = int(
#             round(np.mean([len(_genes) for _genes in _sl_v3_gene_lists]))
#         )

#         _sl_random_gene_lists = repair_essentials(
#             make_random(_sl_target_size, n=100, seed=42)
#         )

#         for _sl_source, _sl_gene_lists in (
#             ("v3", _sl_v3_gene_lists),
#             ("random", _sl_random_gene_lists),
#         ):
#             _sl_counts = np.array(
#                 [
#                     _sl_count_codeleted_interactions(_genes)
#                     for _genes in _sl_gene_lists
#                 ]
#             )

#             _sl_sweep_rows.append(
#                 {
#                     "threshold": _sl_threshold,
#                     "source": _sl_source,
#                     "genome_size": int(
#                         round(np.mean([len(_genes) for _genes in _sl_gene_lists]))
#                     ),
#                     "mean_codeleted_interactions": round(float(np.mean(_sl_counts)), 2),
#                     "median_codeleted_interactions": round(float(np.median(_sl_counts)), 2),
#                     "max_codeleted_interactions": int(np.max(_sl_counts)),
#                     "pct_with_any_codeleted_interaction": round(
#                         float(100 * np.mean(_sl_counts > 0)), 1
#                     ),
#                 }
#             )

#     sl_sweep_df = pl.DataFrame(_sl_sweep_rows)
#     sl_sweep_df
#     return sl_sweep_df

# @app.cell
# def _(alt, sl_sweep_df):
#     sl_sweep_chart = (
#         alt.Chart(sl_sweep_df.to_pandas())
#         .mark_line(point=True)
#         .encode(
#             x=alt.X(
#                 "genome_size:Q",
#                 title="genes per genome",
#                 scale=alt.Scale(zero=False),
#             ),
#             y=alt.Y(
#                 "mean_codeleted_interactions:Q",
#                 title="mean co-deleted BioGRID interaction pairs",
#             ),
#             color=alt.Color(
#                 "source:N",
#                 scale=alt.Scale(
#                     domain=["v3", "random"],
#                     range=["#1f77b4", "#d62728"],
#                 ),
#             ),
#             tooltip=[
#                 "threshold",
#                 "source",
#                 "genome_size",
#                 "mean_codeleted_interactions",
#                 "median_codeleted_interactions",
#                 "max_codeleted_interactions",
#                 "pct_with_any_codeleted_interaction",
#             ],
#         )
#         .properties(
#             width=480,
#             height=300,
#             title="Size–interaction burden frontier: v3 vs size-matched random",
#         )
#     )

#     sl_sweep_chart
#     return sl_sweep_chart

@app.cell
def _(
    gene_order,
    make_random,
    normalize_gene,
    np,
    pl,
    repair_essentials,
    sl_pairs_df,
    sl_pangenome_norm_set,
    v3_continuous,
):
    """
    Sweep VAE decode threshold and count co-deleted BioGRID interaction pairs.

    For each threshold:
    - generate v3 genomes
    - compute their mean genome size
    - generate size-matched random genomes
    - count co-deleted interaction pairs per genome
    - summarise with median and IQR
    """

    def _sl_count_codeleted_interactions(_gene_names):
        _present = {normalize_gene(_g) for _g in _gene_names}
        _absent = sl_pangenome_norm_set - _present

        return int(
            (
                sl_pairs_df["gene_a"].isin(_absent)
                & sl_pairs_df["gene_b"].isin(_absent)
            ).sum()
        )

    _sl_grid = [0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
    _sl_sweep_rows = []

    for _sl_threshold in _sl_grid:
        _sl_v3_gene_lists = repair_essentials(
            [
                gene_order[v3_continuous[_i] > _sl_threshold].tolist()
                for _i in range(v3_continuous.shape[0])
            ]
        )

        _sl_target_size = int(
            round(np.mean([len(_genes) for _genes in _sl_v3_gene_lists]))
        )

        _sl_random_gene_lists = repair_essentials(
            make_random(_sl_target_size, n=100, seed=42)
        )

        for _sl_source, _sl_gene_lists in (
            ("v3", _sl_v3_gene_lists),
            ("random", _sl_random_gene_lists),
        ):
            _sl_counts = np.array(
                [
                    _sl_count_codeleted_interactions(_genes)
                    for _genes in _sl_gene_lists
                ]
            )

            _sl_sweep_rows.append(
                {
                    "threshold": _sl_threshold,
                    "source": _sl_source,
                    "genome_size": int(
                        round(np.mean([len(_genes) for _genes in _sl_gene_lists]))
                    ),
                    "median_codeleted_interactions": round(
                        float(np.median(_sl_counts)), 2
                    ),
                    "q25_codeleted_interactions": round(
                        float(np.quantile(_sl_counts, 0.25)), 2
                    ),
                    "q75_codeleted_interactions": round(
                        float(np.quantile(_sl_counts, 0.75)), 2
                    ),
                    "max_codeleted_interactions": int(np.max(_sl_counts)),
                    "pct_with_any_codeleted_interaction": round(
                        float(100 * np.mean(_sl_counts > 0)), 1
                    ),
                }
            )

    sl_sweep_df = pl.DataFrame(_sl_sweep_rows)
    sl_sweep_df
    return sl_sweep_df

@app.cell
def _(alt, sl_sweep_df):
    _sl_pdf = sl_sweep_df.to_pandas()

    _sl_band = (
        alt.Chart(_sl_pdf)
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X(
                "genome_size:Q",
                title="genes per genome",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "q25_codeleted_interactions:Q",
                title="co-deleted BioGRID interaction pairs",
            ),
            y2="q75_codeleted_interactions:Q",
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=["v3", "random"],
                    range=["#1f77b4", "#d62728"],
                ),
            ),
            tooltip=[
                "threshold",
                "source",
                "genome_size",
                "q25_codeleted_interactions",
                "median_codeleted_interactions",
                "q75_codeleted_interactions",
                "max_codeleted_interactions",
                "pct_with_any_codeleted_interaction",
            ],
        )
    )

    _sl_line = (
        alt.Chart(_sl_pdf)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "genome_size:Q",
                title="genes per genome",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "median_codeleted_interactions:Q",
                title="co-deleted BioGRID interaction pairs",
            ),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=["v3", "random"],
                    range=["#1f77b4", "#d62728"],
                ),
            ),
            tooltip=[
                "threshold",
                "source",
                "genome_size",
                "q25_codeleted_interactions",
                "median_codeleted_interactions",
                "q75_codeleted_interactions",
                "max_codeleted_interactions",
                "pct_with_any_codeleted_interaction",
            ],
        )
    )

    sl_sweep_chart = (
        alt.layer(_sl_band, _sl_line)
        .properties(
            width=480,
            height=300,
            title="Size–interaction burden frontier: v3 vs size-matched random",
        )
    )

    sl_sweep_chart
    return sl_sweep_chart

@app.cell
def _(sl_dir, sl_sweep_df):
    sl_sweep_path = sl_dir / "biogrid_negative_genetic_size_sweep.tsv"
    sl_sweep_df.write_csv(sl_sweep_path, separator="\t")

    print(f"Wrote: {sl_sweep_path}")
    return sl_sweep_path

if __name__ == "__main__":
    app.run()
