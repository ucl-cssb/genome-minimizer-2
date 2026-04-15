"""
Build the correct gene -> TU knockout mapping for vEcoli.

vEcoli perturbations work on transcription unit (TU) IDs, not gene/cistron IDs.
Chain: gene symbol -> cistron_id (EG*_RNA) -> TU IDs (via cistron_tu_mapping_matrix)

Also identifies which genes are actually modelled (have cistrons/TUs) vs just
annotated in the genome.

Saves mapping as JSON for use by eval scripts.
"""
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, "/vEcoli")


def main():
    with open("/vEcoli/out/kb/simData.cPickle", "rb") as f:
        sim_data = pickle.load(f)

    gene_data = sim_data.process.replication.gene_data
    tx = sim_data.process.transcription
    cistron_data = tx.cistron_data
    rna_data = tx.rna_data
    mapping_matrix = tx.cistron_tu_mapping_matrix  # (n_cistrons x n_TUs), sparse

    # Build cistron_id -> index
    cistron_ids = []
    for c in cistron_data:
        cistron_ids.append(str(c["id"]))
    cistron_to_idx = {cid: i for i, cid in enumerate(cistron_ids)}

    # Build TU index -> TU ID
    tu_ids = []
    for r in rna_data:
        tu_ids.append(str(r["id"]))

    print(f"Genes: {len(gene_data)}")
    print(f"Cistrons: {len(cistron_ids)}")
    print(f"TUs (RNAs): {len(tu_ids)}")
    print(f"Mapping matrix: {mapping_matrix.shape}")

    # For each gene, find its TU(s)
    gene_to_tus = {}
    modelled_genes = []
    unmodelled_genes = []

    for g in gene_data:
        symbol = str(g["symbol"])
        cistron_id = str(g["cistron_id"])

        if cistron_id not in cistron_to_idx:
            unmodelled_genes.append(symbol)
            continue

        cidx = cistron_to_idx[cistron_id]
        # Get TU indices for this cistron from the mapping matrix
        tu_indices = mapping_matrix[cidx].nonzero()[1]

        if len(tu_indices) == 0:
            unmodelled_genes.append(symbol)
            continue

        gene_tus = [tu_ids[ti] for ti in tu_indices]
        gene_to_tus[symbol.lower()] = gene_tus
        modelled_genes.append(symbol)

    print(f"\nModelled genes (have TUs): {len(modelled_genes)}")
    print(f"Unmodelled genes (no TUs): {len(unmodelled_genes)}")
    print(f"  First 10 unmodelled: {unmodelled_genes[:10]}")

    # Build synonym mapping from genes.tsv
    import pandas as pd
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

    # Verify: knock out "alr" gene
    test_gene = "alr"
    if test_gene in gene_to_tus:
        print(f"\nTest: '{test_gene}' -> TUs: {gene_to_tus[test_gene]}")
        for tu in gene_to_tus[test_gene]:
            print(f"  {tu} in rna_data: {tu in set(tu_ids)}")

    # All unique TU IDs that can be knocked out
    all_knockout_tus = set()
    for tus in gene_to_tus.values():
        all_knockout_tus.update(tus)
    print(f"\nTotal unique TUs that can be knocked out: {len(all_knockout_tus)}")

    # Save everything
    output = {
        "gene_to_tus": gene_to_tus,
        "synonym_to_symbol": synonym_to_symbol,
        "modelled_gene_symbols": [g.lower() for g in modelled_genes],
        "unmodelled_gene_symbols": [g.lower() for g in unmodelled_genes],
        "all_tu_ids": tu_ids,
        "n_modelled": len(modelled_genes),
        "n_unmodelled": len(unmodelled_genes),
        "n_tus": len(tu_ids),
    }

    with open("/data/vecoli_knockout_mapping.json", "w") as f:
        json.dump(output, f)
    print(f"\nSaved: vecoli_knockout_mapping.json")


if __name__ == "__main__":
    main()
