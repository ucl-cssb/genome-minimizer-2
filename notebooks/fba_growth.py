#!/usr/bin/env python3
"""
Tier 2 of the two-tier evaluation — iML1515 flux-balance-analysis growth.

iML1515 is the genome-scale metabolic model of E. coli K-12 MG1655 (Monk et al.
2017). For each genome we knock out every iML1515 gene that is *absent* from the
genome's gene list, then solve the LP for growth on glucose minimal medium. A
non-zero growth rate means the genome retains the metabolic capacity to grow —
a mechanistic upper bound on viability (FBA covers metabolism only, ~1,500 of
~4,500 genes, so it cannot speak to regulatory/structural essentiality).

This is the standalone-script form of Tier 2 in `notebooks/systems_analysis.py`;
run `kegg_coverage.py` for Tier 1.

OUTPUT (under notebooks/figures/, gitignored)
    fba_growth.png    growth-rate distribution per cohort (violin)
    fba_summary.csv   per-cohort viability summary

DATA SOURCE
    data/fba/iML1515.xml
    data/F4_complete_presence_absence.csv pangenome presence/absence
    evaluation/data/<variant>/<variant>_gene_lists_with_essentials.npy

Run:  uv run python notebooks/fba_growth.py
"""
import re
from pathlib import Path

import altair as alt
import cobra
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FBA_DIR = DATA_DIR / "fba"
EVAL_DATA = REPO_ROOT / "evaluation" / "data"
OUT_DIR = REPO_ROOT / "notebooks" / "figures"

SRC_ORDER = ["real", "random", "v3", "v4_opt"]
SRC_COLORS = {"real": "#2ca02c", "random": "#d62728", "v3": "#1f77b4", "v4_opt": "#9467bd"}
VIABLE_GROWTH = 0.01   # h⁻¹ — above this counts as "viable"


def normalize_gene(name):
    return re.sub(r"_\d+$", "", name).lower()


def load_pangenome_genes():
    idx = pd.read_csv(DATA_DIR / "F4_complete_presence_absence.csv",
                      index_col=0, usecols=[0]).index
    return [g for g in idx if g != "Lineage"]


def load_sample_sources():
    """{source: [(genome_id, [gene names])]} for real / random / v3 / v4_opt.

    `real` = 100 random pangenome strains (seed 42). Kept in sync with
    kegg_coverage.py and systems_analysis.py.
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


def build_iml_crosswalk(iml, pangenome_genes):
    """{b-number: set(synonyms)} for iML1515 genes, and the subset whose name
    appears anywhere in the pangenome (only those are eligible for knock-out;
    unmatched iML genes are treated as always present — a naming gap, not a real
    absence). s0001 (spontaneous-reaction pseudo-gene) is excluded.
    """
    bnum_to_names = {}
    for g in iml.genes:
        if g.id == "s0001":
            continue
        names = set()
        if g.name:
            names.add(g.name.lower())
        for syn in g.annotation.get("refseq_synonym", []):
            if isinstance(syn, str):
                names.add(syn.lower())
        if names:
            bnum_to_names[g.id] = names

    pangenome_norm = {normalize_gene(g) for g in pangenome_genes}
    known = {b for b, names in bnum_to_names.items() if names & pangenome_norm}
    print(f"iML1515: {len(bnum_to_names)} gene b-numbers (excl. s0001); "
          f"{len(known)} ({len(known) / len(bnum_to_names) * 100:.1f}%) have a "
          f"pangenome synonym and are knock-out eligible")
    return bnum_to_names, known


def fba_growth(iml, gene_names, bnum_to_names, known):
    """Knock out the eligible-and-absent iML1515 genes, solve, return growth."""
    norm = {normalize_gene(g) for g in gene_names}
    present = {b for b in known if bnum_to_names[b] & norm}
    with iml as m:
        for b in known - present:
            m.genes.get_by_id(b).knock_out()
        sol = m.optimize()
        val = (sol.objective_value
               if sol.status == "optimal" and sol.objective_value is not None else 0.0)
    return 0.0 if abs(val) < 1e-6 else float(val)


def save_growth_violin(fba_df, wt_growth, out_path):
    chart = (
        alt.Chart(fba_df.to_pandas()).transform_density(
            "growth_rate", as_=["growth_rate", "density"], groupby=["source"],
            extent=[-0.05, 0.95], steps=300, bandwidth=0.025, counts=True,
        ).mark_area(orient="horizontal").encode(
            y=alt.Y("growth_rate:Q", title="FBA predicted growth (h⁻¹)"),
            x=alt.X("density:Q", stack="center", impute=None, title=None,
                    axis=alt.Axis(labels=False, ticks=False, grid=False)),
            color=alt.Color("source:N", scale=alt.Scale(
                domain=SRC_ORDER, range=[SRC_COLORS[s] for s in SRC_ORDER]), legend=None),
            column=alt.Column("source:N", sort=SRC_ORDER,
                              header=alt.Header(titleOrient="bottom")),
        ).properties(width=120, height=320,
                     title=f"iML1515 FBA growth (WT ≈ {wt_growth:.3f} h⁻¹)")
    )
    chart.save(str(out_path))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pangenome_genes = load_pangenome_genes()

    print("Loading iML1515...")
    iml = cobra.io.read_sbml_model(str(FBA_DIR / "iML1515.xml"))
    wt_growth = iml.optimize().objective_value
    print(f"iML1515: {len(iml.genes)} genes, {len(iml.reactions)} reactions | "
          f"WT growth = {wt_growth:.4f} h⁻¹")
    bnum_to_names, known = build_iml_crosswalk(iml, pangenome_genes)

    print("\nLoading genome samples...")
    sources = load_sample_sources()

    print("\nRunning FBA for every genome...")
    rows = []
    for src, items in sources.items():
        for gid, genes in items:
            rows.append({"source": src, "genome_id": gid,
                         "growth_rate": fba_growth(iml, genes, bnum_to_names, known)})
    fba_df = pl.DataFrame(rows)

    save_growth_violin(fba_df, wt_growth, OUT_DIR / "fba_growth.png")
    print(f"wrote {(OUT_DIR / 'fba_growth.png').relative_to(REPO_ROOT)}")

    summary = (
        fba_df.group_by("source").agg([
            pl.len().alias("n"),
            pl.col("growth_rate").mean().round(3).alias("mean_growth"),
            pl.col("growth_rate").median().round(3).alias("median_growth"),
            pl.col("growth_rate").std().round(3).alias("std_growth"),
            (pl.col("growth_rate") > VIABLE_GROWTH).sum().alias("n_viable"),
        ]).with_columns(
            (pl.col("n_viable") / pl.col("n") * 100).round(1).alias("pct_viable")
        ).sort("source")
    )
    print(f"\nFBA viability (growth > {VIABLE_GROWTH} h⁻¹):")
    with pl.Config(tbl_rows=-1):
        print(summary)
    summary.write_csv(OUT_DIR / "fba_summary.csv")
    print(f"wrote {(OUT_DIR / 'fba_summary.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
