"""
vEcoli-based viability evaluation for genome-minimizer-2 pipeline.

Takes gene lists (output of binary converter) and runs whole-cell E. coli
simulations via vEcoli to determine if minimized genomes produce viable cells.

Key metric: does the simulated cell divide?

Usage:
    cd evaluation
    uv run evaluate.py \
        --gene-lists ../data/seq_out_with_essentials.npy \
        --replicates 3 \
        --output results.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray


# ---------------------------------------------------------------------------
# Gene mapping: pipeline names <-> vEcoli gene IDs
# ---------------------------------------------------------------------------

def find_vecoli_root() -> Path:
    """Locate the vEcoli installation directory."""
    import ecoli

    ecoli_path = Path(ecoli.__file__).resolve().parent
    # vEcoli root is one level above the ecoli package
    vecoli_root = ecoli_path.parent
    if (vecoli_root / "reconstruction").is_dir():
        return vecoli_root
    raise FileNotFoundError(
        f"Could not find vEcoli root (reconstruction/ dir) from {ecoli_path}"
    )


def load_vecoli_genes(vecoli_root: Path) -> pd.DataFrame:
    """Load vEcoli gene reference from reconstruction/ecoli/flat/genes.tsv."""
    genes_path = vecoli_root / "reconstruction" / "ecoli" / "flat" / "genes.tsv"
    if not genes_path.exists():
        raise FileNotFoundError(f"genes.tsv not found at {genes_path}")
    return pd.read_csv(genes_path, sep="\t")


def build_gene_mapping(
    genes_df: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """Build mapping from gene symbol -> {id, rna_id} using vEcoli genes.tsv.

    The genes.tsv has columns: id, symbol, synonyms, left_end_pos,
    right_end_pos, direction, rna_ids.

    Returns dict mapping lowercase gene symbol to:
        {"gene_id": "EG10001", "rna_id": "EG10001_RNA", "symbol": "alr"}
    Also indexes by synonyms (b-numbers, etc).
    """
    mapping: dict[str, dict[str, str]] = {}

    for _, row in genes_df.iterrows():
        gene_id = row["id"].strip('"')
        symbol = row["symbol"].strip('"')
        rna_ids_raw = row["rna_ids"]

        # Parse rna_ids - stored as string repr of list like '["EG10001_RNA"]'
        if isinstance(rna_ids_raw, str):
            rna_ids_raw = rna_ids_raw.strip('"').strip("'")
            # Handle JSON-like list
            try:
                rna_ids = json.loads(rna_ids_raw.replace("'", '"'))
            except json.JSONDecodeError:
                rna_ids = [f"{gene_id}_RNA"]
        else:
            rna_ids = [f"{gene_id}_RNA"]

        rna_id = rna_ids[0].strip('"') if rna_ids else f"{gene_id}_RNA"

        entry = {"gene_id": gene_id, "rna_id": rna_id, "symbol": symbol}

        # Index by symbol (lowercase for case-insensitive matching)
        mapping[symbol.lower()] = entry

        # Also index by synonyms
        synonyms_raw = row.get("synonyms", "")
        if isinstance(synonyms_raw, str) and synonyms_raw:
            synonyms_raw = synonyms_raw.strip('"').strip("'")
            try:
                synonyms = json.loads(synonyms_raw.replace("'", '"'))
            except json.JSONDecodeError:
                synonyms = []
            for syn in synonyms:
                syn_clean = syn.strip('"').lower()
                if syn_clean not in mapping:
                    mapping[syn_clean] = entry

    return mapping


def get_knockout_rna_ids(
    present_genes: list[str],
    gene_mapping: dict[str, dict[str, str]],
) -> list[str]:
    """Given genes PRESENT in a minimal genome, return RNA IDs to knock out.

    The knockout set = all vEcoli genes NOT in the provided gene list.
    """
    # Genes present in this minimal genome (lowercase for matching)
    present_lower = {g.lower() for g in present_genes}

    # All vEcoli genes that are NOT in the minimal genome
    all_entries = {v["rna_id"]: v for v in gene_mapping.values()}
    knockout_rna_ids = []
    seen = set()
    for entry in all_entries.values():
        rna_id = entry["rna_id"]
        symbol = entry["symbol"].lower()
        if symbol not in present_lower and rna_id not in seen:
            knockout_rna_ids.append(rna_id)
            seen.add(rna_id)

    return knockout_rna_ids


# ---------------------------------------------------------------------------
# vEcoli simulation execution
# ---------------------------------------------------------------------------

def run_parca(vecoli_root: Path, out_dir: Path, cpus: int = 1) -> Path:
    """Run ParCa (Parameter Calculator) to generate sim_data.

    Returns path to the sim_data pickle file.
    """
    sim_data_path = out_dir / "kb" / "simData.cPickle"
    if sim_data_path.exists():
        print(f"  ParCa output already exists: {sim_data_path}")
        return sim_data_path

    print("  Running ParCa (this may take a while)...")
    parca_script = vecoli_root / "runscripts" / "parca.py"

    config = {
        "out_dir": str(out_dir),
        "parca_options": {"cpus": cpus},
    }
    config_path = out_dir / "parca_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))

    result = subprocess.run(
        [sys.executable, str(parca_script), "--config", str(config_path)],
        cwd=str(vecoli_root),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ParCa stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"ParCa failed with return code {result.returncode}")

    if not sim_data_path.exists():
        # Try alternate location
        alt_paths = list(out_dir.rglob("simData.cPickle"))
        if alt_paths:
            sim_data_path = alt_paths[0]
        else:
            raise FileNotFoundError(
                f"ParCa completed but simData.cPickle not found in {out_dir}"
            )

    print(f"  ParCa complete: {sim_data_path}")
    return sim_data_path


def apply_knockout_variant(sim_data: Any, knockout_rna_ids: list[str]) -> Any:
    """Apply gene knockouts by setting genetic_perturbations on sim_data.

    vEcoli implements knockouts by setting synthesis probabilities to 0
    for specified RNA IDs via the genetic_perturbations attribute.
    """
    sim_data_copy = copy.deepcopy(sim_data)
    # Map RNA IDs to zero synthesis probability
    sim_data_copy.genetic_perturbations = {
        rna_id: 0.0 for rna_id in knockout_rna_ids
    }
    return sim_data_copy


def run_single_simulation(
    sim_data: Any,
    vecoli_root: Path,
    sim_out_dir: Path,
    seed: int = 0,
    max_time: float = 10_000.0,
    generations: int = 1,
) -> dict[str, Any]:
    """Run a single vEcoli simulation and return viability results.

    Args:
        sim_data: Fitted simulation data (with knockouts applied).
        vecoli_root: Path to vEcoli installation.
        sim_out_dir: Output directory for this simulation.
        seed: Random seed.
        max_time: Max simulation time in seconds (sim time, not wall time).
        generations: Number of generations to simulate.

    Returns:
        Dict with keys: divided, division_time, error, sim_out_dir
    """
    sim_out_dir.mkdir(parents=True, exist_ok=True)

    # Save variant sim_data as pickle for the simulation to load
    variant_path = sim_out_dir / "variant_sim_data.cPickle"
    with open(variant_path, "wb") as f:
        pickle.dump(sim_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Build simulation config
    config = {
        "sim_data_path": str(variant_path),
        "seed": seed,
        "generations": generations,
        "emitter": "parquet",
        "emitter_arg": {"out_dir": str(sim_out_dir / "output")},
        "experiment_id": f"eval_seed{seed}",
        "single_daughters": True,
    }
    config_path = sim_out_dir / "sim_config.json"
    config_path.write_text(json.dumps(config))

    sim_script = vecoli_root / "ecoli" / "experiments" / "ecoli_master_sim.py"
    result = subprocess.run(
        [sys.executable, str(sim_script), "--config", str(config_path)],
        cwd=str(vecoli_root),
        capture_output=True,
        text=True,
        timeout=int(max_time * 60),  # wall-clock timeout
    )

    # Check for division
    divided = False
    division_time = None
    error = None

    # Method 1: Check for division_time.sh (written on DivisionDetected)
    div_time_files = list(sim_out_dir.rglob("division_time.sh"))
    if div_time_files:
        divided = True
        try:
            content = div_time_files[0].read_text()
            # Format: export DIVISION_TIME=<seconds>
            for line in content.splitlines():
                if "DIVISION_TIME" in line:
                    division_time = float(line.split("=")[1].strip())
        except (ValueError, IndexError):
            pass

    # Method 2: Check for daughter state files
    if not divided:
        daughter_files = list(sim_out_dir.rglob("daughter_state_*.json"))
        if daughter_files:
            divided = True

    # Check for errors
    if result.returncode != 0 and not divided:
        error = result.stderr[-500:] if result.stderr else "Unknown error"

    return {
        "divided": divided,
        "division_time": division_time,
        "error": error,
        "sim_out_dir": str(sim_out_dir),
    }


# ---------------------------------------------------------------------------
# Ray-parallelized evaluation
# ---------------------------------------------------------------------------

@ray.remote
def ray_run_simulation(
    sim_data_bytes: bytes,
    vecoli_root_str: str,
    sim_out_dir_str: str,
    seed: int,
    max_time: float,
) -> dict[str, Any]:
    """Ray remote task: run one vEcoli simulation replicate."""
    sim_data = pickle.loads(sim_data_bytes)
    return run_single_simulation(
        sim_data=sim_data,
        vecoli_root=Path(vecoli_root_str),
        sim_out_dir=Path(sim_out_dir_str),
        seed=seed,
        max_time=max_time,
    )


def aggregate_results(
    sample_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-sample viability statistics from replicate results."""
    n_total = len(sample_results)
    n_divided = sum(1 for r in sample_results if r["divided"])
    division_times = [
        r["division_time"]
        for r in sample_results
        if r["divided"] and r["division_time"] is not None
    ]
    errors = [r["error"] for r in sample_results if r["error"] is not None]

    return {
        "n_replicates": n_total,
        "n_divided": n_divided,
        "viability_rate": n_divided / n_total if n_total > 0 else 0.0,
        "mean_division_time": np.mean(division_times) if division_times else None,
        "median_division_time": np.median(division_times) if division_times else None,
        "n_errors": len(errors),
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate minimized E. coli genomes using vEcoli whole-cell simulation."
    )
    parser.add_argument(
        "--gene-lists",
        type=Path,
        required=True,
        help="Path to .npy file with array of gene name lists (pipeline output).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Number of simulation replicates per genome sample (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.csv"),
        help="Output CSV path for viability results (default: results.csv).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work"),
        help="Working directory for simulation outputs (default: work/).",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (default: start local cluster).",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=720.0,
        help="Max wall-clock minutes per simulation (default: 720 = 12 hours).",
    )
    parser.add_argument(
        "--parca-cpus",
        type=int,
        default=1,
        help="CPUs for ParCa parameter calculation (default: 1).",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Comma-separated sample indices to evaluate (default: all).",
    )
    args = parser.parse_args()

    # --- Setup ---
    print("=== vEcoli Genome Viability Evaluation ===\n")

    vecoli_root = find_vecoli_root()
    print(f"vEcoli root: {vecoli_root}")

    args.work_dir.mkdir(parents=True, exist_ok=True)

    # --- Load gene lists ---
    print(f"Loading gene lists from {args.gene_lists}")
    gene_lists = np.load(args.gene_lists, allow_pickle=True)
    n_samples = len(gene_lists)
    print(f"  Found {n_samples} genome samples")

    # Select samples
    if args.samples:
        sample_indices = [int(i) for i in args.samples.split(",")]
    else:
        sample_indices = list(range(n_samples))
    print(f"  Evaluating {len(sample_indices)} samples with {args.replicates} replicates each")

    # --- Build gene mapping ---
    print("\nBuilding gene name mapping...")
    genes_df = load_vecoli_genes(vecoli_root)
    gene_mapping = build_gene_mapping(genes_df)
    n_vecoli_genes = len({v["gene_id"] for v in gene_mapping.values()})
    print(f"  Mapped {n_vecoli_genes} vEcoli genes ({len(gene_mapping)} total name entries)")

    # --- Run ParCa ---
    print("\nPreparing ParCa (parameter calculation)...")
    parca_out = args.work_dir / "parca"
    sim_data_path = run_parca(vecoli_root, parca_out, cpus=args.parca_cpus)

    print("  Loading baseline sim_data...")
    with open(sim_data_path, "rb") as f:
        baseline_sim_data = pickle.load(f)

    # --- Initialize Ray ---
    print(f"\nInitializing Ray (address={args.ray_address or 'local'})...")
    ray.init(address=args.ray_address, ignore_reinit_error=True)
    print(f"  Ray cluster: {ray.cluster_resources()}")

    # --- Submit simulation tasks ---
    print("\nSubmitting simulation tasks...")
    futures_by_sample: dict[int, list] = {}

    for sample_idx in sample_indices:
        gene_list = list(gene_lists[sample_idx])
        knockout_rna_ids = get_knockout_rna_ids(gene_list, gene_mapping)
        n_present = len(gene_list)
        n_knockout = len(knockout_rna_ids)
        print(f"  Sample {sample_idx}: {n_present} genes present, {n_knockout} knocked out")

        # Apply knockouts to sim_data
        variant_sim_data = apply_knockout_variant(baseline_sim_data, knockout_rna_ids)
        sim_data_bytes = pickle.dumps(variant_sim_data)

        # Submit replicates
        futures = []
        for rep in range(args.replicates):
            sim_out = args.work_dir / f"sample_{sample_idx}" / f"rep_{rep}"
            future = ray_run_simulation.remote(
                sim_data_bytes=sim_data_bytes,
                vecoli_root_str=str(vecoli_root),
                sim_out_dir_str=str(sim_out),
                seed=sample_idx * 1000 + rep,
                max_time=args.max_time,
            )
            futures.append(future)
        futures_by_sample[sample_idx] = futures

    # --- Collect results ---
    total_tasks = sum(len(f) for f in futures_by_sample.values())
    print(f"\nWaiting for {total_tasks} simulations to complete...")

    all_rows = []
    for sample_idx in sample_indices:
        futures = futures_by_sample[sample_idx]
        replicate_results = ray.get(futures)
        agg = aggregate_results(replicate_results)

        row = {
            "sample_idx": sample_idx,
            "n_genes": len(gene_lists[sample_idx]),
            **agg,
        }
        all_rows.append(row)

        status = "VIABLE" if agg["viability_rate"] > 0 else "NON-VIABLE"
        print(
            f"  Sample {sample_idx}: {status} "
            f"({agg['n_divided']}/{agg['n_replicates']} divided, "
            f"viability={agg['viability_rate']:.1%})"
        )

    # --- Save results ---
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # --- Summary ---
    print("\n=== Summary ===")
    n_viable = sum(1 for r in all_rows if r["viability_rate"] > 0)
    print(f"Viable genomes: {n_viable}/{len(all_rows)}")
    print(f"Mean viability rate: {results_df['viability_rate'].mean():.1%}")
    if results_df["mean_division_time"].notna().any():
        mean_div = results_df["mean_division_time"].dropna().mean()
        print(f"Mean division time (viable): {mean_div:.1f}s")

    ray.shutdown()


if __name__ == "__main__":
    main()
