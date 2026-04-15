"""
wcEcoli evaluation v2 — uses BC4_func_genes_indices.csv and KO_index.

Matches the paper's methodology:
1. Restrict genome to the 1,870 WCM gene set (from BC4_func_genes_indices.csv)
2. For each absent gene, get its KO_index for adjust_final_expression
3. Skip WCM-essential genes (restore them even if absent from sample)
4. Apply multi-gene knockout via adjust_final_expression with KO_index
5. Run 20 generations

The KO_index maps directly to rna_data indices in wcEcoli, matching the
gene_knockout variant indexing scheme.

Usage (inside wcm-code container):
    python wcecoli_eval_v2.py \
        --gene-lists /data/v3_gene_lists_with_essentials.npy \
        --wcm-genes /data/BC4_func_genes_indices.csv \
        --essential-genes /data/essential_genes.csv \
        --sample-idx 0 \
        --sim-data /work/kb/simData.cPickle \
        --generations 20 \
        --output /results
"""

import argparse
import copy
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_gene(name):
    return re.sub(r"_\d+$", "", name).lower()


def load_wcm_gene_set(wcm_csv_path, essential_csv_path):
    """Load the WCM gene set and identify essential genes."""
    # WCM gene set with KO indices
    wcm_df = pd.read_csv(wcm_csv_path)
    wcm_df["gene_lower"] = wcm_df["gene_ko"].str.lower()

    # Literature essential genes (proxy for WCM-essentials)
    ess_df = pd.read_csv(essential_csv_path)
    col = "# gene" if "# gene" in ess_df.columns else "gene"
    essential_set = set(ess_df[col].str.strip().str.lower())

    # Mark essentials in WCM set
    wcm_df["is_essential"] = wcm_df["gene_lower"].isin(essential_set)

    n_essential = wcm_df["is_essential"].sum()
    print(f"WCM gene set: {len(wcm_df)} genes")
    print(f"WCM-essential (literature): {n_essential}")

    return wcm_df, essential_set


def get_knockout_indices_for_sample(gene_list, wcm_df, essential_set):
    """For a sample's gene list, determine which WCM genes to knock out.

    Logic (from notebook cell 7):
    - For each WCM gene absent from the sample (value == 0):
      - If WCM-essential: skip (restore it)
      - If KO_index is NaN: skip
      - Otherwise: add to knockout list

    Returns KO indices (1-based, matching rna_data indexing).
    """
    # Normalize sample gene names for matching
    present_lower = set()
    for g in gene_list:
        g_low = g.lower().strip()
        g_norm = normalize_gene(g)
        present_lower.add(g_low)
        present_lower.add(g_norm)

    ko_indices = []
    n_restored = 0
    n_absent_nonessential = 0

    for _, row in wcm_df.iterrows():
        gene = row["gene_lower"]
        ko_idx = row["KO_index"]
        is_essential = row["is_essential"]

        # Check if gene is present in the sample
        if gene in present_lower:
            continue  # Gene present, no knockout needed

        # Gene is absent from sample
        if is_essential:
            n_restored += 1
            continue  # Restore essential gene

        if pd.isna(ko_idx):
            continue  # No KO index available

        ko_indices.append(int(ko_idx))
        n_absent_nonessential += 1

    # KO_index from BC4_func_genes_indices.csv uses the gene_knockout variant
    # indexing: geneIndex = (index - 1) % (nGenes + 1)
    # where nGenes = len(rna_data). We need to apply this mapping.
    # The rna_data size isn't known here, so return raw KO indices
    # and let the caller apply the modulo.
    return ko_indices, n_restored, n_absent_nonessential


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-lists", type=Path, required=True)
    parser.add_argument("--wcm-genes", type=Path, required=True,
                        help="BC4_func_genes_indices.csv")
    parser.add_argument("--essential-genes", type=Path, required=True,
                        help="essential_genes.csv")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--sim-data", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    result_path = args.output / f"result_sample{args.sample_idx}_seed{args.seed}.json"
    if result_path.exists():
        print(f"Already done: {result_path}")
        return

    print(f"=== wcEcoli Evaluation v2 ===")
    print(f"Sample: {args.sample_idx} | Seed: {args.seed} | Gens: {args.generations}")

    # Load WCM gene set
    wcm_df, essential_set = load_wcm_gene_set(args.wcm_genes, args.essential_genes)

    # Load sample gene list
    gene_lists = np.load(str(args.gene_lists), allow_pickle=True)
    gene_list = list(gene_lists[args.sample_idx])
    print(f"Genes in sample: {len(gene_list)}")

    # Get knockout indices (raw KO_index values from CSV)
    raw_ko_indices, n_restored, n_ko = get_knockout_indices_for_sample(
        gene_list, wcm_df, essential_set,
    )
    n_wcm = len(wcm_df)
    print(f"Essential genes restored: {n_restored}")
    print(f"Genes knocked out: {n_ko}/{n_wcm} ({n_ko/n_wcm*100:.1f}%)")
    print(f"Genes retained: {n_wcm - n_ko - n_restored} present + {n_restored} restored")

    # Load and modify sim_data
    print("Loading sim_data...")
    with open(args.sim_data, "rb") as f:
        sim_data = pickle.load(f)

    # Convert KO_index to rna_data indices using gene_knockout variant formula:
    # geneIndex = (KO_index - 1) % (nGenes + 1)
    n_rnas = len(sim_data.process.transcription.rna_data)
    n_conditions = n_rnas + 1
    rna_indices = [(idx - 1) % n_conditions for idx in raw_ko_indices]
    # Deduplicate and validate
    rna_indices = sorted(set(i for i in rna_indices if i < n_rnas))
    print(f"rna_data indices after mapping: {len(rna_indices)} (rna_data size: {n_rnas})")

    print("Applying knockouts via adjust_final_expression...")
    ko_sim_data = copy.deepcopy(sim_data)
    # Use epsilon instead of 0 to avoid normalization issues
    ko_sim_data.adjust_final_expression(rna_indices, [1e-10] * len(rna_indices))

    # Save variant sim_data
    sample_dir = args.output / f"sample_{args.sample_idx}" / f"seed_{args.seed}"
    kb_dir = sample_dir / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)

    from wholecell.utils import constants
    variant_path = kb_dir / constants.SERIALIZED_SIM_DATA_FILENAME
    print(f"Saving variant sim_data...")
    with open(variant_path, "wb") as f:
        pickle.dump(ko_sim_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    del ko_sim_data

    # Run simulation
    print(f"\nRunning {args.generations}-gen simulation...")
    t0 = time.time()

    cmd = [
        sys.executable, "/wcEcoli/runscripts/manual/runSim.py",
        str(sample_dir),
        "-g", str(args.generations),
        "-s", str(args.seed),
        "-i", "1",
        "--length-sec", "10800",
        "--no-log-to-shell",
    ]
    print(f"CMD: {' '.join(cmd)}")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
    elapsed = time.time() - t0

    # Check viability — count completed generation directories
    gen_dirs = sorted(sample_dir.glob("wildtype_*/000000/generation_*"))
    gens_completed = len(gen_dirs)
    viable = (gens_completed >= args.generations) and (proc.returncode == 0)

    if proc.returncode != 0 and gens_completed < args.generations:
        stderr_tail = proc.stderr[-500:] if proc.stderr else "no stderr"
        print(f"Sim failed at gen {gens_completed + 1} (exit {proc.returncode})")
        print(f"stderr: {stderr_tail}")

    status = "VIABLE" if viable else "NON-VIABLE"
    print(f"\n=== Result: {status} ({gens_completed}/{args.generations} gens) ===")
    print(f"Wall time: {elapsed / 60:.1f} min")

    result = {
        "sample_idx": args.sample_idx,
        "seed": args.seed,
        "model": "wcEcoli",
        "model_version": "2909dfc001 (Feb 2023)",
        "target_generations": args.generations,
        "generations_completed": gens_completed,
        "viable": viable,
        "n_genes_in_list": len(gene_list),
        "n_wcm_genes": n_wcm,
        "n_knockouts": n_ko,
        "n_essentials_restored": n_restored,
        "total_wall_time_s": elapsed,
        "returncode": proc.returncode,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Result saved: {result_path}")

    # Cleanup bulky sim output
    for d in sample_dir.glob("wildtype_*"):
        shutil.rmtree(d, ignore_errors=True)
    variant_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
