"""
Generate random baseline genomes for comparison with VAE.

Reviewer request: "generate genomes of similar sizes by selecting the core
pangenome genes and randomly sampling from the accessory pool, then test
these in the WCM."

Strategy:
  1. Load the pangenome presence/absence matrix
  2. Identify core genes (present in >95% of strains) and accessory genes
  3. For each sample: keep all core genes + randomly sample accessory genes
     to match the mean gene count of the VAE variant being compared
  4. Add essential genes (same as VAE pipeline)

Usage:
    python evaluation/generate_random_baseline.py \
        --num-samples 100 \
        --target-gene-count 3300 \
        --output evaluation/data/random \
        --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genome_minimizer_2.explore_data.binary_converter import (
    load_files,
    check_essential_genes,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description="Generate random baseline genomes")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--target-gene-count", type=int, default=None,
                        help="Target gene count per sample. If not set, uses mean of pangenome.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/data/random"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--core-threshold", type=float, default=0.95,
                        help="Fraction of strains a gene must appear in to be 'core' (default: 0.95)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load pangenome
    print("Loading pangenome matrix...")
    pa_matrix = pd.read_csv(DATA_DIR / "F4_complete_presence_absence.csv", index_col=0)
    data = pa_matrix.drop(index=["Lineage"]).T  # strains x genes
    gene_names = np.array(data.columns)
    binary = data.values.astype(float)
    n_strains, n_genes = binary.shape
    print(f"Pangenome: {n_strains} strains x {n_genes} genes")

    # Identify core vs accessory
    gene_freq = binary.mean(axis=0)  # fraction of strains with each gene
    core_mask = gene_freq >= args.core_threshold
    accessory_mask = ~core_mask
    n_core = core_mask.sum()
    n_accessory = accessory_mask.sum()
    print(f"Core genes (>={args.core_threshold*100:.0f}% of strains): {n_core}")
    print(f"Accessory genes: {n_accessory}")

    core_genes = set(gene_names[core_mask])
    accessory_genes = gene_names[accessory_mask]

    # Determine target gene count
    if args.target_gene_count is not None:
        target = args.target_gene_count
    else:
        # Use mean gene count across real strains
        target = int(binary.sum(axis=1).mean())
    accessory_to_sample = max(0, target - n_core)
    print(f"Target gene count: {target} (core: {n_core} + accessory: {accessory_to_sample})")

    if accessory_to_sample > n_accessory:
        print(f"WARNING: target requires {accessory_to_sample} accessory genes "
              f"but only {n_accessory} available. Capping.")
        accessory_to_sample = n_accessory

    # Sample accessory genes weighted by their frequency in the pangenome
    # This is a fair baseline — genes that appear more often are more likely to be sampled
    accessory_freq = gene_freq[accessory_mask]
    accessory_prob = accessory_freq / accessory_freq.sum()

    print(f"\nGenerating {args.num_samples} random genomes...")
    gene_lists = []
    for i in range(args.num_samples):
        sampled_accessory = rng.choice(
            accessory_genes, size=accessory_to_sample, replace=False, p=accessory_prob
        )
        genes = sorted(set(list(core_genes) + list(sampled_accessory)))
        gene_lists.append(genes)
        if (i + 1) % 25 == 0:
            print(f"  generated {i + 1}/{args.num_samples}")

    sizes = [len(g) for g in gene_lists]
    print(f"Gene counts: mean={np.mean(sizes):.0f}, min={np.min(sizes)}, max={np.max(sizes)}")

    # Save raw gene lists
    raw_path = args.output / "random_gene_lists.npy"
    np.save(raw_path, np.array(gene_lists, dtype=object))
    print(f"Saved: {raw_path}")

    # Add essential genes
    print("\nAdding essential genes...")
    essential_set, id_lists = load_files(
        str(DATA_DIR / "essential_genes.csv"), str(raw_path)
    )
    filled_path = check_essential_genes(essential_set, id_lists, str(raw_path))

    # Summary
    final_lists = np.load(filled_path, allow_pickle=True)
    final_sizes = [len(g) for g in final_lists]
    print(f"\n--- Summary ---")
    print(f"Samples: {len(final_lists)}")
    print(f"Gene counts (with essentials): mean={np.mean(final_sizes):.0f}, "
          f"min={np.min(final_sizes)}, max={np.max(final_sizes)}")
    print(f"Total pangenome genes: {n_genes}")
    print(f"Mean reduction: {(1 - np.mean(final_sizes) / n_genes) * 100:.1f}%")
    print(f"\nReady for vEcoli eval: {filled_path}")


if __name__ == "__main__":
    main()
