#!/usr/bin/env python3
"""
Tier 1 of the two-tier evaluation — KEGG functional-module coverage.

For each genome we map its gene names to KEGG b-numbers and, for each of the
~110 E. coli KEGG modules, compute completeness = |genome ∩ module| / |module|.
Modules are small functional units (median ~5 genes), so a strict 0.99 threshold
is effectively "module fully reconstructed". We report how many modules each
cohort reconstructs vs the `real`, `random`, `v3`, and `v4_opt` baselines.

This is the standalone-script form of Tier 1 in `notebooks/systems_analysis.py`.
That notebook bundles Tier 1 + Tier 2 because the integrated size–viability
frontier analysis needs both tiers at once; run `fba_growth.py` for Tier 2.

OUTPUT (under notebooks/figures/, gitignored)
    kegg_genome_sizes.png     genome-size distribution per cohort
    kegg_module_coverage.png  module-coverage distribution (violin)
    kegg_summary.csv          per-cohort module-coverage summary

DATA SOURCE
    data/kegg/                            KEGG REST caches (auto-fetched if absent)
    data/F4_complete_presence_absence.csv pangenome presence/absence
    evaluation/data/<variant>/<variant>_gene_lists_with_essentials.npy

Run:  uv run python notebooks/kegg_coverage.py
"""
import re
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
KEGG_CACHE = DATA_DIR / "kegg"
EVAL_DATA = REPO_ROOT / "evaluation" / "data"
OUT_DIR = REPO_ROOT / "notebooks" / "figures"

KEGG_BASE = "https://rest.kegg.jp"
COMPLETENESS_THRESHOLD = 0.99
SRC_ORDER = ["real", "random", "v3", "v4_opt"]
SRC_COLORS = {"real": "#2ca02c", "random": "#d62728", "v3": "#1f77b4", "v4_opt": "#9467bd"}


def normalize_gene(name):
    return re.sub(r"_\d+$", "", name).lower()


def fetch_kegg(endpoint, cache_path):
    """Cached KEGG REST GET. Cache lives in data/kegg/ so reruns hit no network."""
    if cache_path.exists():
        return cache_path.read_text()
    r = requests.get(f"{KEGG_BASE}/{endpoint}", timeout=30)
    r.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(r.text)
    time.sleep(0.34)
    return r.text


def load_kegg_modules():
    """{module_id: set(b-numbers)} for every E. coli KEGG module."""
    KEGG_CACHE.mkdir(parents=True, exist_ok=True)
    links = fetch_kegg("link/module/eco", KEGG_CACHE / "eco_module_links.tsv")
    module_genes = {}
    for line in links.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        gene = parts[0].removeprefix("eco:")
        mid = parts[1].removeprefix("md:eco_").removeprefix("md:")
        module_genes.setdefault(mid, set()).add(gene)
    return module_genes


def build_crosswalk():
    """{pangenome gene name: KEGG b-number}, plus the pangenome gene list.

    Exact symbol match first, then after stripping the Panaroo paralog suffix
    `_N`. group_XXXX accessory clusters are skipped (rarely in KEGG modules).
    """
    eco_genes = fetch_kegg("list/eco", KEGG_CACHE / "eco_genes.tsv")
    symbol_to_bnum = {}
    for line in eco_genes.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        bnum = parts[0].removeprefix("eco:")
        for sym in parts[3].split(";")[0].strip().split(","):
            key = sym.strip().lower()
            if key:
                symbol_to_bnum[key] = bnum

    pangenome_index = pd.read_csv(
        DATA_DIR / "F4_complete_presence_absence.csv", index_col=0, usecols=[0]
    ).index
    pangenome_genes = [g for g in pangenome_index if g != "Lineage"]

    pangenome_to_bnum = {}
    for g in pangenome_genes:
        if g.startswith("group_"):
            continue
        if g.lower() in symbol_to_bnum:
            pangenome_to_bnum[g] = symbol_to_bnum[g.lower()]
        elif normalize_gene(g) in symbol_to_bnum:
            pangenome_to_bnum[g] = symbol_to_bnum[normalize_gene(g)]

    n_named = sum(1 for g in pangenome_genes if not g.startswith("group_"))
    print(f"Crosswalk: {len(pangenome_to_bnum)}/{n_named} named pangenome genes "
          f"matched to KEGG ({len(pangenome_to_bnum) / n_named * 100:.1f}%)")
    assert 2000 <= len(pangenome_to_bnum) <= 6000, (
        f"Match count {len(pangenome_to_bnum)} outside sanity range [2000, 6000] — "
        "inspect the crosswalk before proceeding."
    )
    return pangenome_to_bnum, pangenome_genes


def load_sample_sources():
    """{source: [(genome_id, [gene names])]} for real / random / v3 / v4_opt.

    `real` = 100 random pangenome strains (seed 42). Kept in sync with
    fba_growth.py and systems_analysis.py.
    """
    path = DATA_DIR / "F4_complete_presence_absence.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    strain_cols = header[1:]
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(strain_cols), size=100, replace=False)
    chosen_names = [strain_cols[i] for i in sorted(chosen)]
    df = (pd.read_csv(path, index_col=0, usecols=[header[0], *chosen_names])
          .drop(index="Lineage", errors="ignore"))
    sources = {"real": [(col, df.index[df[col].astype(int).values == 1].tolist())
                        for col in df.columns]}

    def _load(variant):
        p = EVAL_DATA / variant / f"{variant}_gene_lists_with_essentials.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — generate it with genome_minimizer_2.sampling")
        return [list(g) for g in np.load(p, allow_pickle=True)]

    for v in ("random", "v3", "v4_opt"):
        sources[v] = [(f"{v}_{i:03d}", g) for i, g in enumerate(_load(v))]

    for src, items in sources.items():
        sizes = [len(g) for _, g in items]
        print(f"{src:>8s}: n={len(items):3d}  mean={int(np.mean(sizes))} genes "
              f"[{min(sizes)}-{max(sizes)}]")
    return sources


def score_genome(gene_names, pangenome_to_bnum, module_genes):
    bnums = {pangenome_to_bnum[g] for g in gene_names if g in pangenome_to_bnum}
    return {mid: len(bnums & mg) / len(mg) if mg else 0.0
            for mid, mg in module_genes.items()}


def save_size_chart(sizes_df, out_path):
    n = sizes_df["n_genes"].to_numpy()
    edges = np.linspace(n.min() - 1, n.max() + 1, 41)
    chart = (
        alt.Chart(sizes_df.to_pandas()).mark_bar(opacity=0.7).encode(
            x=alt.X("n_genes:Q",
                    bin=alt.Bin(extent=[float(edges[0]), float(edges[-1])],
                                step=float(edges[1] - edges[0])),
                    title="genes per genome"),
            y=alt.Y("count():Q", title="# genomes"),
            color=alt.Color("source:N", scale=alt.Scale(
                domain=SRC_ORDER, range=[SRC_COLORS[s] for s in SRC_ORDER])),
            row=alt.Row("source:N", sort=SRC_ORDER),
        ).properties(width=600, height=80).resolve_scale(y="independent")
    )
    chart.save(str(out_path))


def save_coverage_violin(per_genome, threshold, out_path):
    chart = (
        alt.Chart(per_genome.to_pandas()).transform_density(
            "n_modules", as_=["n_modules", "density"], groupby=["source"],
            extent=[0, 112], steps=200, bandwidth=2.0, counts=True,
        ).mark_area(orient="horizontal").encode(
            y=alt.Y("n_modules:Q", title=f"# modules ≥ {threshold:.2f} complete"),
            x=alt.X("density:Q", stack="center", impute=None, title=None,
                    axis=alt.Axis(labels=False, ticks=False, grid=False)),
            color=alt.Color("source:N", scale=alt.Scale(
                domain=SRC_ORDER, range=[SRC_COLORS[s] for s in SRC_ORDER]), legend=None),
            column=alt.Column("source:N", sort=SRC_ORDER,
                              header=alt.Header(titleOrient="bottom")),
        ).properties(width=120, height=300,
                     title=f"Module coverage (threshold = {threshold:.2f})")
    )
    chart.save(str(out_path))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading KEGG modules + crosswalk...")
    module_genes = load_kegg_modules()
    n_genes = len({g for s in module_genes.values() for g in s})
    print(f"E. coli KEGG modules: {len(module_genes)} | unique genes: {n_genes}")
    pangenome_to_bnum, _ = build_crosswalk()

    print("\nLoading genome samples...")
    sources = load_sample_sources()

    size_rows = [{"source": s, "genome_id": gid, "n_genes": len(g)}
                 for s, items in sources.items() for gid, g in items]
    save_size_chart(pl.DataFrame(size_rows), OUT_DIR / "kegg_genome_sizes.png")
    print(f"wrote {(OUT_DIR / 'kegg_genome_sizes.png').relative_to(REPO_ROOT)}")

    print("\nScoring genomes against KEGG modules...")
    rows = []
    for src, items in sources.items():
        for gid, genes in items:
            for mid, frac in score_genome(genes, pangenome_to_bnum, module_genes).items():
                rows.append({"source": src, "genome_id": gid,
                             "module_id": mid, "completeness": frac})
    long_df = pl.DataFrame(rows)

    per_genome = (
        long_df.group_by(["source", "genome_id"])
        .agg((pl.col("completeness") >= COMPLETENESS_THRESHOLD).sum().alias("n_modules"))
        .sort(["source", "n_modules"])
    )
    save_coverage_violin(per_genome, COMPLETENESS_THRESHOLD,
                         OUT_DIR / "kegg_module_coverage.png")
    print(f"wrote {(OUT_DIR / 'kegg_module_coverage.png').relative_to(REPO_ROOT)}")

    summary = (
        per_genome.group_by("source").agg([
            pl.len().alias("n"),
            pl.col("n_modules").mean().round(1).alias("mean_modules"),
            pl.col("n_modules").median().alias("median_modules"),
            pl.col("n_modules").min().alias("min_modules"),
            pl.col("n_modules").max().alias("max_modules"),
        ]).sort("source")
    )
    print(f"\nModule coverage (# modules ≥ {COMPLETENESS_THRESHOLD:.2f} complete):")
    with pl.Config(tbl_rows=-1):
        print(summary)
    summary.write_csv(OUT_DIR / "kegg_summary.csv")
    print(f"wrote {(OUT_DIR / 'kegg_summary.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
