#!/usr/bin/env python3
"""
Download the data needed to run genome-minimizer-2 from the public
UCL-CSSB/genome-minimizer-2 HuggingFace bucket. No login required.

Training data lands in data/; pre-computed samples land in evaluation/data/<variant>/.
"""

import os
import sys
from pathlib import Path

BUCKET = "hf://buckets/UCL-CSSB/genome-minimizer-2"

# (bucket_path, local_path_relative_to_repo_root)
TRAINING_FILES = [
    (f"{BUCKET}/data/F4_complete_presence_absence.csv", "data/F4_complete_presence_absence.csv"),
    (f"{BUCKET}/data/accessionID_phylogroup_BD.csv",    "data/accessionID_phylogroup_BD.csv"),
    (f"{BUCKET}/data/essential_genes.csv",              "data/essential_genes.csv"),
    (f"{BUCKET}/data/BC4_func_genes_indices.csv",       "data/BC4_func_genes_indices.csv"),
    (f"{BUCKET}/data/GCF_000005845.2.gbff",             "data/GCF_000005845.2.gbff"),
    # GO annotations, needed by the statistical_analysis notebook's GO enrichment.
    (f"{BUCKET}/data/kegg/uniprot_eco_go.tsv",          "data/kegg/uniprot_eco_go.tsv"),
]

# Pre-computed VAE samples (v0–v3) and the random baseline, as read by the notebooks.
SAMPLE_FILES = []
for _v in ("v0", "v1", "v2", "v3"):
    for _f in ("samples", "gene_lists", "gene_lists_with_essentials"):
        SAMPLE_FILES.append(
            (f"{BUCKET}/samples/{_v}/{_v}_{_f}.npy", f"evaluation/data/{_v}/{_v}_{_f}.npy")
        )
for _f in ("gene_lists", "gene_lists_with_essentials"):
    SAMPLE_FILES.append(
        (f"{BUCKET}/samples/random/random_{_f}.npy", f"evaluation/data/random/random_{_f}.npy")
    )


def _download(fs, files, root, force):
    ok = True
    for bucket_path, rel_local in files:
        local_path = os.path.join(root, rel_local)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path) and not force:
            print(f"  ✓ {rel_local} ({os.path.getsize(local_path):,} bytes)")
            continue

        print(f"  downloading {rel_local} ...", end=" ", flush=True)
        try:
            fs.get_file(bucket_path, local_path)
            print(f"done ({os.path.getsize(local_path):,} bytes)")
        except Exception as e:
            print(f"FAILED\n    {e}")
            ok = False
    return ok


def setup_data(project_root=None, force=False, training_only=False):
    """
    Download data from the public UCL-CSSB/genome-minimizer-2 HF bucket.

    Args:
        project_root: repo root; defaults to three levels above this file.
        force: re-download files that already exist.
        training_only: download training data only, skip pre-computed samples.

    Returns:
        True if every requested file is present afterwards.
    """
    from huggingface_hub import HfFileSystem

    if project_root is None:
        project_root = str(Path(__file__).parent.parent.parent)

    fs = HfFileSystem()

    print("Training data:")
    ok = _download(fs, TRAINING_FILES, project_root, force)

    if not training_only:
        print("\nPre-computed samples (v0–v3, random):")
        ok = _download(fs, SAMPLE_FILES, project_root, force) and ok

    print("\n✓ All files ready." if ok else "\n✗ Some downloads failed — check your connection and retry.")
    return ok


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download genome-minimizer-2 data from HuggingFace")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--training-data-only", action="store_true",
                        help="Download training data only, skip pre-computed samples")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Repo root to write into (default: repo root)")
    args = parser.parse_args()

    ok = setup_data(project_root=args.data_dir, force=args.force, training_only=args.training_data_only)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
