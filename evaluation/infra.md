# Whole-Cell Model Evaluation Infrastructure — Myriad (UCL HPC)

## Overview

Evaluation of minimal genomes uses whole-cell E. coli simulations on UCL's
Myriad cluster via Apptainer containers. Two models are supported:

- **vEcoli** (1.1.0): Port of wcEcoli to Vivarium. 4,538 modelled genes.
  Too permissive for genome minimization (100% viability at tested knockout depths).
- **wcEcoli** (Feb 2023): Original whole-cell model. ~1,870 WCM gene set.
  Paper-comparable evaluation. 20 generations, gene knockouts via `adjust_final_expression`.

## Connection

```bash
ssh myriad   # config in ~/.ssh/config, user ucbt042, ProxyJump via ucl-gateway
```

## Directory layout on Myriad

```
~/Scratch/genome-minimizer/
├── python_3.12.12-bookworm.sif   # vEcoli container
├── wcm-code.sif                  # wcEcoli container (Feb 2023, commit 2909dfc001)
├── vecoli_venv/                   # vEcoli Python venv
├── vEcoli/                        # vEcoli source (C extensions compiled)
│   └── out/kb/simData.cPickle     # vEcoli ParCa output
├── wcEcoli/                       # wcEcoli source (Feb 2023)
├── wcecoli_out/                   # wcEcoli ParCa output
│   ├── kb/simData.cPickle
│   └── cache/
├── evaluation/                    # Eval scripts (synced from repo)
├── data/                          # Gene lists, WCM gene set, etc.
│   ├── BC4_func_genes_indices.csv # WCM gene set (1,870 genes with KO_index)
│   ├── essential_genes.csv        # Literature essential genes (358)
│   ├── F4_complete_presence_absence.csv
│   └── {variant}_gene_lists_with_essentials.npy
└── results_*/                     # Evaluation output
```

## Evaluation scripts

| Script | Description |
|--------|-------------|
| `run_pipeline.py` | Sample VAE genomes, convert to gene lists, add essentials |
| `generate_random_baseline.py` | Generate frequency-weighted random baseline |
| `eval_wcecoli.py` | wcEcoli evaluation (paper-comparable, 20 gens) |
| `eval_vecoli.py` | vEcoli evaluation (TU-based knockouts) |
| `build_knockout_mapping.py` | Build vEcoli gene->TU mapping |
| `build_wcm_geneset.py` | Intersect pangenome with vEcoli genes |
| `eval_*_array.sh` | SGE array job wrappers |

## wcEcoli evaluation (paper-comparable)

Uses `BC4_func_genes_indices.csv` for the WCM gene set (1,870 genes) and
`KO_index` values for `adjust_final_expression`. Essential genes (literature
set) are restored before evaluation.

```bash
# Submit all 100 samples for a variant
sed 's/VARIANT/v3/g' evaluation/eval_wcecoli_array.sh | qsub
```

Key parameters matching the paper:
- 20 generations, seed=0
- Knockout via `adjust_final_expression` with factor `1e-10`
- `KO_index` mapped to rna_data indices: `(KO_index - 1) % (n_rnas + 1)`
- Essential genes restored (not knocked out even if absent)

## SGE job submission

```bash
#$ -l h_rt=48:0:0    # walltime (20 gens takes ~4-8h)
#$ -l mem=8G          # per-core memory
#$ -t 1-100           # array job
#$ -N job_name
#$ -wd /home/ucbt042/Scratch/genome-minimizer
```

Submit: `qsub script.sh` | Check: `qstat` | Delete: `qdel <job-id>`

## Data storage

- **HF Bucket**: `hf://buckets/McClain/minimal_genomes/` (samples, gene lists, results)
- **Sync up**: `hf buckets sync ./local hf://buckets/McClain/minimal_genomes/path/`
- **Sync down**: `hf buckets sync hf://buckets/McClain/minimal_genomes/path/ ./local`

## Gotchas

- **Compute nodes can't write to $HOME** — always use `~/Scratch/`
- **vEcoli knockouts use TU IDs** (`TU0-*[c]`), not cistron IDs (`EG*_RNA`)
- **wcEcoli `KO_index`** needs modulo: `(KO_index - 1) % (n_rnas + 1)`
- **Use `1e-10` not `0.0`** for knockout factor to avoid assertion errors
- **wcEcoli needs writable `/wcEcoli/cache/`** — bind mount it
- **`--cleanenv`** required to prevent Intel `icc` leaking from host
- **Disk fills fast** — wcEcoli sim output is huge, clean up after collecting results
