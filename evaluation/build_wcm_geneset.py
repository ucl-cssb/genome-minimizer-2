"""
Build the WCM gene set by intersecting pangenome columns with vEcoli gene symbols.

The paper says: "we defined the WCM gene set as the 1,872 genes explicitly
modelled in the WCM" and "we first restricted each binary presence-absence
genome vector to the WCM gene set."

This script finds which pangenome columns correspond to vEcoli-modelled genes,
giving us the equivalent restriction matrix.
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/vEcoli")


def normalize_gene(name):
    """Strip Panaroo paralog suffixes (_1, _2, etc) for matching."""
    return re.sub(r"_\d+$", "", name).lower()


def main():
    # Load pangenome gene names (they're row indices, not columns)
    # Matrix is genes x samples, with gene names as the row index
    pangenome = pd.read_csv("/data/F4_complete_presence_absence.csv", index_col=0, usecols=[0])
    pangenome_cols = np.array([g for g in pangenome.index if g != "Lineage"])
    print(f"Pangenome genes: {len(pangenome_cols)}")

    # Load vEcoli gene data from sim_data
    with open("/vEcoli/out/kb/simData.cPickle", "rb") as f:
        sim_data = pickle.load(f)

    gene_data = sim_data.process.replication.gene_data
    vecoli_symbols = set()
    for g in gene_data:
        vecoli_symbols.add(str(g["symbol"]).lower())
    print(f"vEcoli gene symbols: {len(vecoli_symbols)}")

    # Load synonyms from genes.tsv for broader matching
    genes_tsv = pd.read_csv(
        "/vEcoli/reconstruction/ecoli/flat/genes.tsv",
        sep="\t", comment="#", quoting=0,
    )
    genes_tsv.columns = [c.strip('"') for c in genes_tsv.columns]

    synonym_to_symbol = {}
    for _, row in genes_tsv.iterrows():
        symbol = str(row["symbol"]).strip('"').lower()
        synonym_to_symbol[symbol] = symbol
        synonyms_raw = row.get("synonyms", "")
        if isinstance(synonyms_raw, str) and synonyms_raw:
            synonyms_raw = synonyms_raw.strip('"').strip("'")
            try:
                synonyms = json.loads(synonyms_raw.replace("'", '"'))
                for syn in synonyms:
                    synonym_to_symbol[syn.strip('"').lower()] = symbol
            except (json.JSONDecodeError, ValueError):
                pass
    print(f"Total name entries (symbols + synonyms): {len(synonym_to_symbol)}")

    # Intersect: which pangenome genes match a vEcoli gene?
    # Try both exact match and normalized (strip _N suffix) match
    wcm_cols = []
    wcm_col_indices = []
    matched_symbols = set()

    for i, col in enumerate(pangenome_cols):
        col_lower = col.lower().strip()
        col_normalized = normalize_gene(col)

        # Try exact match first, then normalized
        matched = None
        if col_lower in synonym_to_symbol:
            matched = synonym_to_symbol[col_lower]
        elif col_normalized in synonym_to_symbol:
            matched = synonym_to_symbol[col_normalized]

        if matched and matched in vecoli_symbols:
            wcm_cols.append((i, col, matched))
            wcm_col_indices.append(i)
            matched_symbols.add(matched)

    print(f"\n=== Results ===")
    print(f"Pangenome columns matching vEcoli genes: {len(wcm_cols)}")
    print(f"Unique vEcoli genes matched: {len(matched_symbols)}")
    print(f"vEcoli genes NOT in pangenome: {len(vecoli_symbols - matched_symbols)}")
    unmatched = sorted(vecoli_symbols - matched_symbols)
    print(f"  First 20 unmatched: {unmatched[:20]}")

    # How does this compare to the paper's 1,872?
    print(f"\nPaper had 1,872 WCM genes. We found {len(matched_symbols)} unique matches.")
    if len(matched_symbols) > 1872:
        print(f"  vEcoli has {len(matched_symbols) - 1872} more genes than wcEcoli's WCM set")
    else:
        print(f"  {1872 - len(matched_symbols)} fewer than wcEcoli (expected — different model version)")

    # Check: for a typical v3 sample, how many genes survive the WCM restriction?
    v3_lists = np.load("/data/v3_gene_lists_with_essentials.npy", allow_pickle=True)
    wcm_col_set = {pangenome_cols[i].lower() for i in wcm_col_indices}

    restricted_sizes = []
    for genes in v3_lists:
        restricted = [g for g in genes if g.lower() in wcm_col_set]
        restricted_sizes.append(len(restricted))
    restricted_sizes = np.array(restricted_sizes)
    print(f"\nv3 samples restricted to WCM gene set:")
    print(f"  Mean: {restricted_sizes.mean():.0f} genes")
    print(f"  Min: {restricted_sizes.min()}, Max: {restricted_sizes.max()}")
    print(f"  As fraction of WCM set: {restricted_sizes.mean()/len(matched_symbols)*100:.1f}%")
    print(f"  Paper reported: 63-76% of 1,872 = {int(0.63*1872)}-{int(0.76*1872)} genes")

    # Save
    output = {
        "wcm_col_indices": wcm_col_indices,
        "wcm_col_names": [c[1] for c in wcm_cols],
        "wcm_vecoli_symbols": sorted(set(c[2] for c in wcm_cols)),
        "n_pangenome_cols": len(pangenome_cols),
        "n_vecoli_genes": len(vecoli_symbols),
        "n_wcm_cols": len(wcm_cols),
        "n_unique_matched": len(matched_symbols),
    }
    with open("/data/wcm_gene_set.json", "w") as f:
        json.dump(output, f, indent=2)
    np.save("/data/wcm_col_indices.npy", np.array(wcm_col_indices))
    print(f"\nSaved: wcm_gene_set.json, wcm_col_indices.npy")


if __name__ == "__main__":
    main()
