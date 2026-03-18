# vEcoli Viability Evaluation

Evaluates minimized E. coli genomes from the genome-minimizer-2 pipeline using
[vEcoli](https://github.com/CovertLab/vEcoli) whole-cell simulation. For each
candidate minimal genome, the script runs replicated simulations and reports
whether the cell divides (= viable).

## Setup

This directory has its own `pyproject.toml` because vEcoli's dependencies
(specific numpy/scipy versions) would conflict with the main project.

```bash
cd evaluation
uv sync
```

Verify the install:

```bash
uv run python -c "import ecoli; print('vEcoli OK')"
uv run python -c "import ray; ray.init(); print('Ray OK'); ray.shutdown()"
```

## Usage

```bash
# Evaluate all samples with 3 replicates each
uv run evaluate.py \
    --gene-lists ../data/seq_out_with_essentials.npy \
    --replicates 3 \
    --output results.csv

# Evaluate specific samples
uv run evaluate.py \
    --gene-lists ../data/seq_out_with_essentials.npy \
    --samples 0,1,5 \
    --replicates 3

# Connect to an existing Ray cluster
uv run evaluate.py \
    --gene-lists ../data/seq_out_with_essentials.npy \
    --ray-address auto

# Adjust wall-clock timeout per simulation (default: 720 min = 12 hours)
uv run evaluate.py \
    --gene-lists ../data/seq_out_with_essentials.npy \
    --max-time 360
```

## How It Works

1. **Gene mapping**: Maps pipeline gene names to vEcoli gene/RNA IDs using
   `reconstruction/ecoli/flat/genes.tsv`.
2. **ParCa**: Runs the Parameter Calculator once to generate baseline simulation
   data (`sim_data`). Cached in `work/parca/` for reuse.
3. **Knockout variant**: For each minimal genome, genes *not* in the gene list
   are knocked out by setting their transcription synthesis probability to zero
   via `sim_data.genetic_perturbations`.
4. **Simulation**: Each replicate runs as a Ray task, executing the full
   vEcoli simulation with the knockout variant applied.
5. **Viability check**: A simulation is "viable" if the cell divides
   (detected via `division_time.sh` or daughter state files in output).
6. **Aggregation**: Per-sample viability rate, mean/median division time.

## Output

`results.csv` columns:

| Column | Description |
|---|---|
| `sample_idx` | Index into the input gene list array |
| `n_genes` | Number of genes in the minimal genome |
| `n_replicates` | Number of simulation replicates |
| `n_divided` | Replicates where cell divided |
| `viability_rate` | Fraction of replicates that divided |
| `mean_division_time` | Mean time to division (seconds, viable only) |
| `median_division_time` | Median time to division (seconds, viable only) |
| `n_errors` | Replicates that errored out |

## Compute Requirements

Each vEcoli simulation takes ~5-12 hours on a single CPU core. Plan accordingly:

- **Local testing**: Use `--samples 0 --replicates 1` for a smoke test.
- **Cluster**: Point `--ray-address` at a Ray cluster on g6-big or similar.
- **Full evaluation**: N samples × R replicates × ~8 hours average per sim.
