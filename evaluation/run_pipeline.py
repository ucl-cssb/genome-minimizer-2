"""
Full pipeline: sample genomes from trained VAE -> convert to gene lists -> save for vEcoli eval.

Usage:
    cd genome-minimizer-2
    python evaluation/run_pipeline.py --variant v3 --num-samples 10 --output evaluation/data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genome_minimizer_2.training.model import VAE
from genome_minimizer_2.explore_data.binary_converter import (
    masks_to_gene_lists,
    load_files,
    check_essential_genes,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Model dimensions per variant
VARIANT_DIMS = {
    "v0": {"hidden_dim": 1024, "latent_dim": 64},
    "v1": {"hidden_dim": 512, "latent_dim": 32},
    "v2": {"hidden_dim": 512, "latent_dim": 32},
    "v3": {"hidden_dim": 512, "latent_dim": 32},
    "v4": {"hidden_dim": 512, "latent_dim": 32},
}
INPUT_DIM = 55039


def main():
    parser = argparse.ArgumentParser(description="Sample VAE genomes and convert to gene lists")
    parser.add_argument("--variant", type=str, default="v3", choices=VARIANT_DIMS.keys())
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("evaluation/data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = VARIANT_DIMS[args.variant]
    print(f"Variant: {args.variant} | hidden={dims['hidden_dim']}, latent={dims['latent_dim']}")

    # Step 1: Download model
    print(f"\n--- Step 1: Download {args.variant} checkpoint from HF ---")
    ckpt_path = hf_hub_download(
        "McClain/genome-minimizer-2", "final.pt", revision=args.variant
    )
    print(f"Checkpoint: {ckpt_path}")

    # Step 2: Load model
    print("\n--- Step 2: Load model ---")
    model = VAE(INPUT_DIM, dims["hidden_dim"], dims["latent_dim"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded from epoch {ckpt['epoch']}")

    # Step 3: Sample from prior
    print(f"\n--- Step 3: Sample {args.num_samples} genomes ---")
    torch.manual_seed(args.seed)
    with torch.no_grad():
        z = torch.randn(args.num_samples, dims["latent_dim"])
        samples = model.decode(z).cpu().numpy()

    samples_path = out_dir / f"{args.variant}_samples.npy"
    np.save(samples_path, samples)
    print(f"Saved raw samples: {samples_path} (shape: {samples.shape})")

    # Quick stats
    binary = (samples > 0.5).astype(int)
    gene_counts = binary.sum(axis=1)
    print(f"Gene counts: mean={gene_counts.mean():.0f}, min={gene_counts.min()}, max={gene_counts.max()}")

    # Step 4: Convert to gene lists
    print("\n--- Step 4: Convert to gene lists ---")
    pa_matrix = pd.read_csv(DATA_DIR / "F4_complete_presence_absence.csv", index_col=0)
    cols = pa_matrix.drop(index=["Lineage"]).T.columns

    gene_lists_path = out_dir / f"{args.variant}_gene_lists.npy"
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
    print(f"\n--- Summary ---")
    print(f"Variant: {args.variant}")
    print(f"Samples: {len(final_lists)}")
    print(f"Gene counts (with essentials): mean={np.mean(sizes):.0f}, min={np.min(sizes)}, max={np.max(sizes)}")
    print(f"Total E. coli genes: {INPUT_DIM}")
    print(f"Mean reduction: {(1 - np.mean(sizes) / INPUT_DIM) * 100:.1f}%")
    print(f"\nReady for vEcoli eval: {filled_path}")


if __name__ == "__main__":
    main()
