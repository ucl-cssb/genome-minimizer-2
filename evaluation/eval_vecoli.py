"""
Single-sample vEcoli evaluation with CORRECT TU-based knockouts.

Previous version used cistron IDs (EG*_RNA) which don't match vEcoli's
perturbation system. This version maps gene symbols -> TU IDs via the
knockout mapping built by build_knockout_mapping.py.

Usage (inside container):
    python eval_single_v2.py \
        --gene-lists /data/v3_gene_lists_with_essentials.npy \
        --ko-mapping /data/vecoli_knockout_mapping.json \
        --sample-idx 0 \
        --sim-data /vEcoli/out/kb/simData.cPickle \
        --output /results
"""

import argparse
import copy
import json
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def normalize_gene(name):
    """Strip Panaroo paralog suffixes (_1, _2, etc)."""
    return re.sub(r"_\d+$", "", name).lower()


def get_knockout_tus(present_genes, ko_mapping):
    """Given genes PRESENT in a genome, return TU IDs to knock out.

    A TU is knocked out if ALL genes it contains are absent.
    If a TU contains any present gene, it stays active (operon logic).
    """
    gene_to_tus = ko_mapping["gene_to_tus"]
    synonym_to_symbol = ko_mapping["synonym_to_symbol"]
    all_tu_ids = set(ko_mapping["all_tu_ids"])
    modelled_symbols = set(ko_mapping["modelled_gene_symbols"])

    # Map present gene names to vEcoli symbols
    present_symbols = set()
    for g in present_genes:
        g_lower = g.lower().strip()
        g_norm = normalize_gene(g)
        # Try exact, then normalized, then synonym lookup
        if g_lower in modelled_symbols:
            present_symbols.add(g_lower)
        elif g_norm in modelled_symbols:
            present_symbols.add(g_norm)
        elif g_lower in synonym_to_symbol:
            sym = synonym_to_symbol[g_lower]
            if sym in modelled_symbols:
                present_symbols.add(sym)
        elif g_norm in synonym_to_symbol:
            sym = synonym_to_symbol[g_norm]
            if sym in modelled_symbols:
                present_symbols.add(sym)

    # Find TUs where ALL constituent genes are absent
    # First, find all TUs that have at least one PRESENT gene
    active_tus = set()
    for sym in present_symbols:
        if sym in gene_to_tus:
            for tu in gene_to_tus[sym]:
                active_tus.add(tu)

    # Knockout = all TUs NOT active
    knockout_tus = all_tu_ids - active_tus

    return knockout_tus, present_symbols


def run_sim(sim_data_path, knockout_tus, output_dir, seed=0):
    """Run a single vEcoli simulation with TU-based knockouts."""
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and apply knockouts to sim_data
    with open(sim_data_path, "rb") as f:
        sim_data = pickle.load(f)

    # genetic_perturbations maps TU ID -> synthesis probability
    # Use tiny epsilon instead of 0.0 to avoid breaking vEcoli's
    # transcription probability normalization (assertion in initial_conditions.py)
    sim_data.genetic_perturbations = {tu: 1e-10 for tu in knockout_tus}

    # Save variant sim_data
    variant_path = output_dir / "variant_simData.cPickle"
    with open(variant_path, "wb") as f:
        pickle.dump(sim_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    del sim_data  # free memory

    # Write config
    config = {
        "sim_data_path": str(variant_path),
        "seed": seed,
        "generations": 1,
        "single_daughters": True,
        "emitter": "parquet",
        "emitter_arg": {"out_dir": str(output_dir / "output")},
        "experiment_id": f"eval_s{seed}",
        "fail_at_max_duration": True,
        "n_init_sims": 1,
    }
    config_path = output_dir / "sim_config.json"
    config_path.write_text(json.dumps(config))

    sim = EcoliSim.from_file(str(config_path))
    sim.build_ecoli()

    divided = False
    error = None
    try:
        sim.run()
        error = "max_duration_reached"
    except SystemExit:
        divided = True
    except Exception as e:
        error = str(e)[:500]

    # Clean up variant pickle (64MB each)
    variant_path.unlink(missing_ok=True)

    return divided, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-lists", type=Path, required=True)
    parser.add_argument("--ko-mapping", type=Path, required=True)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--sim-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result_path = args.output / f"result_sample{args.sample_idx}_seed{args.seed}.json"
    if result_path.exists():
        print(f"Already done: {result_path}")
        return

    print(f"=== vEcoli Evaluation (v2 — TU knockouts) ===")
    print(f"Sample: {args.sample_idx} | Seed: {args.seed}")

    # Load knockout mapping
    with open(args.ko_mapping) as f:
        ko_mapping = json.load(f)
    n_modelled = ko_mapping["n_modelled"]
    n_tus = ko_mapping["n_tus"]

    # Load gene lists
    gene_lists = np.load(args.gene_lists, allow_pickle=True)
    gene_list = list(gene_lists[args.sample_idx])
    print(f"Genes in sample: {len(gene_list)}")

    # Compute knockouts
    knockout_tus, present_symbols = get_knockout_tus(gene_list, ko_mapping)
    n_present = len(present_symbols)
    n_ko_tus = len(knockout_tus)
    n_active_tus = n_tus - n_ko_tus
    print(f"Mapped to vEcoli: {n_present}/{n_modelled} modelled genes present")
    print(f"TUs knocked out: {n_ko_tus}/{n_tus} ({n_ko_tus/n_tus*100:.1f}%)")
    print(f"TUs active: {n_active_tus}")

    # Run simulation
    sim_dir = args.output / f"sample_{args.sample_idx}" / f"seed_{args.seed}"
    print(f"\nRunning simulation -> {sim_dir}")
    t0 = time.time()
    divided, error = run_sim(args.sim_data, knockout_tus, sim_dir, seed=args.seed)
    elapsed = time.time() - t0

    status = "VIABLE" if divided else f"NON-VIABLE ({error})"
    print(f"\n=== Result: {status} ===")
    print(f"Wall time: {elapsed / 60:.1f} min")

    # Save result
    result = {
        "sample_idx": args.sample_idx,
        "seed": args.seed,
        "n_genes_in_list": len(gene_list),
        "n_genes_mapped": n_present,
        "n_modelled_genes": n_modelled,
        "n_tus_total": n_tus,
        "n_tus_knocked_out": n_ko_tus,
        "n_tus_active": n_active_tus,
        "divided": divided,
        "error": error,
        "wall_time_s": elapsed,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Result saved: {result_path}")

    # Clean up bulky sim output
    for subdir in ["history", "configuration", "output"]:
        for d in sim_dir.rglob(subdir):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
