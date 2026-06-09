"""
Sampling for genome-minimizer-2 — one module, two generators.

- VAE sampling: draw genomes from a trained VAE checkpoint (per variant, pulled
  from the HF Hub), threshold to presence/absence, convert to gene lists, and add
  back the literature essential genes.
- Random baseline: keep all core genes and frequency-weighted sample the accessory
  pool to a target gene count (the reviewer-requested baseline).

`compute_gene_frequencies` and `sample_random_genomes` are the single source of
truth for the random baseline and are imported directly by the analysis notebooks.
The heavy deps (torch, huggingface_hub, the VAE model, binary_converter) are
imported lazily inside the flows that need them, so importing those two helpers
stays numpy/pandas-only.

CLI (run with ``src`` on the path, e.g. ``cd src`` or ``PYTHONPATH=src``):
    python -m genome_minimizer_2.sampling vae    --variant v3  --num-samples 100 --output evaluation/data/v3
    python -m genome_minimizer_2.sampling random --num-samples 100 --output evaluation/data/random
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# src/genome_minimizer_2/sampling.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_DIM = 55039

# Architecture per variant.
VARIANT_DIMS = {
    "v0": {"hidden_dim": 1024, "latent_dim": 64},
    "v1": {"hidden_dim": 512, "latent_dim": 32},
    "v2": {"hidden_dim": 512, "latent_dim": 32},
    "v3": {"hidden_dim": 512, "latent_dim": 32},
    "v4": {"hidden_dim": 512, "latent_dim": 32},
    "v4_opt": {"hidden_dim": 512, "latent_dim": 32},
}

# Authoritative checkpoint location per variant: (repo_id, branch).
# v0-v4 live on the UCL-CSSB org repo; the hyperparameter-tuned v4_opt is on McClain.
VARIANT_REPO = {
    "v0": ("UCL-CSSB/genome-minimizer-2", "v0"),
    "v1": ("UCL-CSSB/genome-minimizer-2", "v1"),
    "v2": ("UCL-CSSB/genome-minimizer-2", "v2"),
    "v3": ("UCL-CSSB/genome-minimizer-2", "v3"),
    "v4": ("UCL-CSSB/genome-minimizer-2", "v4"),
    "v4_opt": ("McClain/genome-minimizer-2", "v4_opt"),
}


# --------------------------------------------------------------------------- #
# Random baseline (pure helpers; imported by the analysis notebooks)
# --------------------------------------------------------------------------- #
def compute_gene_frequencies(pa_csv_path):
    """Return (gene_names, gene_freq) from the pangenome presence/absence CSV.

    gene_freq[i] is the fraction of strains carrying gene i — the prevalence used
    for the core/accessory split and frequency-weighted accessory sampling.
    """
    pa_matrix = pd.read_csv(pa_csv_path, index_col=0)
    data = pa_matrix.drop(index=["Lineage"]).T  # strains x genes
    gene_names = np.array(data.columns)
    gene_freq = data.values.astype(float).mean(axis=0)
    return gene_names, gene_freq


def sample_random_genomes(gene_names, gene_freq, target, num_samples, rng, core_threshold=0.95):
    """All core genes (freq >= core_threshold) + a frequency-weighted accessory
    sample (without replacement) to reach `target` total genes, repeated
    `num_samples` times from a single `rng`. Single source of truth for the
    random baseline — used by main() and by analysis notebooks.
    """
    core_mask = gene_freq >= core_threshold
    core_genes = set(gene_names[core_mask])
    accessory_genes = gene_names[~core_mask]
    accessory_prob = gene_freq[~core_mask] / gene_freq[~core_mask].sum()
    accessory_to_sample = min(max(0, target - len(core_genes)), len(accessory_genes))
    gene_lists = []
    for _ in range(num_samples):
        sampled = rng.choice(
            accessory_genes, size=accessory_to_sample, replace=False, p=accessory_prob
        )
        gene_lists.append(sorted(set(list(core_genes) + list(sampled))))
    return gene_lists


# --------------------------------------------------------------------------- #
# Flows
# --------------------------------------------------------------------------- #
def sample_vae(variant, num_samples, output, seed=42):
    """Sample genomes from a trained VAE: download checkpoint -> sample prior ->
    decode -> threshold -> gene lists -> add essential genes.

    Writes `{variant}_samples.npy`, `{variant}_z.npy`, `{variant}_gene_lists.npy`
    and the `_with_essentials` variant under `output`. Returns the path to the
    essential-filled gene lists.
    """
    import torch
    from huggingface_hub import hf_hub_download

    from genome_minimizer_2.training.model import VAE
    from genome_minimizer_2.explore_data.binary_converter import (
        masks_to_gene_lists,
        load_files,
        check_essential_genes,
    )

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = VARIANT_DIMS[variant]
    print(f"Variant: {variant} | hidden={dims['hidden_dim']}, latent={dims['latent_dim']}")

    # Step 1: Download model
    repo_id, branch = VARIANT_REPO[variant]
    print(f"\n--- Step 1: Download {variant} checkpoint from {repo_id}@{branch} ---")
    ckpt_path = hf_hub_download(repo_id, "final.pt", revision=branch)
    print(f"Checkpoint: {ckpt_path}")

    # Step 2: Load model
    print("\n--- Step 2: Load model ---")
    model = VAE(INPUT_DIM, dims["hidden_dim"], dims["latent_dim"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded from epoch {ckpt['epoch']}")

    # Step 3: Sample from prior
    print(f"\n--- Step 3: Sample {num_samples} genomes ---")
    torch.manual_seed(seed)
    with torch.no_grad():
        z = torch.randn(num_samples, dims["latent_dim"])
        samples = model.decode(z).cpu().numpy()

    samples_path = out_dir / f"{variant}_samples.npy"
    np.save(samples_path, samples)
    print(f"Saved raw samples: {samples_path} (shape: {samples.shape})")

    # Save the latent draw so the exact sample set is recoverable without
    # relying on the RNG seed alone.
    z_path = out_dir / f"{variant}_z.npy"
    np.save(z_path, z.cpu().numpy())
    print(f"Saved latent z: {z_path} (shape: {tuple(z.shape)})")

    # Quick stats
    binary = (samples > 0.5).astype(int)
    gene_counts = binary.sum(axis=1)
    print(f"Gene counts: mean={gene_counts.mean():.0f}, min={gene_counts.min()}, max={gene_counts.max()}")

    # Step 4: Convert to gene lists
    print("\n--- Step 4: Convert to gene lists ---")
    pa_matrix = pd.read_csv(DATA_DIR / "F4_complete_presence_absence.csv", index_col=0)
    cols = pa_matrix.drop(index=["Lineage"]).T.columns

    gene_lists_path = out_dir / f"{variant}_gene_lists.npy"
    masks_to_gene_lists(str(samples_path), cols, str(gene_lists_path))

    # Step 5: Add essential genes
    print("\n--- Step 5: Add essential genes ---")
    essential_set, id_lists = load_files(
        str(DATA_DIR / "essential_genes.csv"), str(gene_lists_path)
    )
    filled_path = check_essential_genes(essential_set, id_lists, str(gene_lists_path))
    print(f"Final gene lists: {filled_path}")

    # Summary
    final_lists = np.load(filled_path, allow_pickle=True)
    sizes = [len(g) for g in final_lists]
    print("\n--- Summary ---")
    print(f"Variant: {variant}")
    print(f"Samples: {len(final_lists)}")
    print(f"Gene counts (with essentials): mean={np.mean(sizes):.0f}, min={np.min(sizes)}, max={np.max(sizes)}")
    print(f"Total E. coli genes: {INPUT_DIM}")
    print(f"Mean reduction: {(1 - np.mean(sizes) / INPUT_DIM) * 100:.1f}%")
    return filled_path


def generate_random_baseline(num_samples, output, target_gene_count=None, seed=42,
                             core_threshold=0.95):
    """Generate the random core+accessory baseline and add essential genes.

    Writes `random_gene_lists.npy` and its `_with_essentials` variant under
    `output`. Returns the path to the essential-filled gene lists.
    """
    from genome_minimizer_2.explore_data.binary_converter import (
        load_files,
        check_essential_genes,
    )

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("Loading pangenome matrix...")
    gene_names, gene_freq = compute_gene_frequencies(DATA_DIR / "F4_complete_presence_absence.csv")
    n_genes = len(gene_names)
    n_core = int((gene_freq >= core_threshold).sum())
    print(f"Pangenome: {n_genes} genes")
    print(f"Core genes (>={core_threshold * 100:.0f}% of strains): {n_core}")
    print(f"Accessory genes: {n_genes - n_core}")

    # Default target: mean genes per strain (== sum of per-gene frequencies)
    target = target_gene_count if target_gene_count is not None else int(gene_freq.sum())
    print(f"Target gene count: {target} (core: {n_core} + accessory: {max(0, target - n_core)})")

    print(f"\nGenerating {num_samples} random genomes...")
    gene_lists = sample_random_genomes(
        gene_names, gene_freq, target, num_samples, rng, core_threshold
    )
    sizes = [len(g) for g in gene_lists]
    print(f"Gene counts: mean={np.mean(sizes):.0f}, min={np.min(sizes)}, max={np.max(sizes)}")

    raw_path = out_dir / "random_gene_lists.npy"
    np.save(raw_path, np.array(gene_lists, dtype=object))
    print(f"Saved: {raw_path}")

    print("\nAdding essential genes...")
    essential_set, id_lists = load_files(
        str(DATA_DIR / "essential_genes.csv"), str(raw_path)
    )
    filled_path = check_essential_genes(essential_set, id_lists, str(raw_path))

    final_lists = np.load(filled_path, allow_pickle=True)
    final_sizes = [len(g) for g in final_lists]
    print("\n--- Summary ---")
    print(f"Samples: {len(final_lists)}")
    print(f"Gene counts (with essentials): mean={np.mean(final_sizes):.0f}, "
          f"min={np.min(final_sizes)}, max={np.max(final_sizes)}")
    print(f"Total pangenome genes: {n_genes}")
    print(f"Mean reduction: {(1 - np.mean(final_sizes) / n_genes) * 100:.1f}%")
    return filled_path


def main():
    parser = argparse.ArgumentParser(description="Sample genomes (VAE or random baseline)")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_vae = sub.add_parser("vae", help="Sample genomes from a trained VAE checkpoint")
    p_vae.add_argument("--variant", type=str, default="v3", choices=VARIANT_DIMS.keys())
    p_vae.add_argument("--num-samples", type=int, default=10)
    p_vae.add_argument("--output", type=Path, default=Path("evaluation/data"))
    p_vae.add_argument("--seed", type=int, default=42)

    p_rand = sub.add_parser("random", help="Generate the random core+accessory baseline")
    p_rand.add_argument("--num-samples", type=int, default=100)
    p_rand.add_argument("--target-gene-count", type=int, default=None,
                        help="Target gene count per sample. If not set, uses mean of pangenome.")
    p_rand.add_argument("--output", type=Path, default=Path("evaluation/data/random"))
    p_rand.add_argument("--seed", type=int, default=42)
    p_rand.add_argument("--core-threshold", type=float, default=0.95,
                        help="Fraction of strains a gene must appear in to be 'core' (default: 0.95)")

    args = parser.parse_args()
    if args.mode == "vae":
        sample_vae(args.variant, args.num_samples, args.output, args.seed)
    elif args.mode == "random":
        generate_random_baseline(args.num_samples, args.output, args.target_gene_count,
                                 args.seed, args.core_threshold)


if __name__ == "__main__":
    main()
