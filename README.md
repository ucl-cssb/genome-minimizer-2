# 🧬 GENOME MINIMIZER 2

VAE-powered minimal genome generation pipeline for E. coli.

## Pipeline Overview

```
Data Files → [Preprocess] → [Explore] → [Training] → [Sample] → [Minimize]
```

1. **Preprocess**: Extract essential gene positions from literature
2. **Explore**: Analyze dataset distributions and generate visualizations  
3. **Training**: Train VAE models with different configurations
4. **Sample**: Generate synthetic genomes from trained models
5. **Binary converter**: Converts binary synthetic genomes into lists with gene names
6. **Minimize**: Create actual minimized genome sequences

## Setup

This project is run with [uv](https://docs.astral.sh/uv/). Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and sync — `uv sync` creates the virtualenv, installs every dependency
from `pyproject.toml`, and installs the package itself (uv manages the Python
version too, so no manual `python`/`pip` steps):

```bash
git clone https://github.com/ucl-cssb/genome-minimizer-2
cd genome-minimizer-2
uv sync
```

Run any command with `uv run` (e.g. `uv run python main.py ...`), or activate the
env once with `source .venv/bin/activate` and drop the prefix. Next, fetch the
inputs — see [Data Setup](#data-setup).

## Quick Start

After [Data Setup](#data-setup), the full pipeline runs end to end:

```bash
# 1. Essential-gene positions from the reference genome
uv run python main.py --mode preprocess

# 2. Train a VAE (presets v0–v4; --no-hf-upload skips the HF checkpoint upload)
uv run python main.py --mode training --preset v0 --epochs 1 --no-hf-upload

# 3. Sample genomes from the trained model
uv run python main.py --mode sample \
    --model-path models/trained_models/v0_model/saved_VAE_v0.pt \
    --genes-path src/genome_minimizer_2/data/essential_genes/essential_gene_positions.pkl \
    --num-samples 100

# 4. Convert the binary samples to gene-name lists
uv run python main.py --mode convert-samples \
    --genes-path models/v0_model/sampling_results/v0_binary_samples_default.npy

# 5. Build minimized FASTA sequences (one record per genome)
uv run python main.py --mode minimizer \
    --genes-path seq_out_with_essentials.npy \
    --genome-path data/GCF_000005845.2.gbff \
    --single-file --output-file results.fasta
```

Also available: `explore` (dataset figures) and `experiment` (custom-config
training). The analysis notebooks and the standalone sampling module are under
[Analysis & sampling](#analysis--sampling).

## Data Setup

`essential_genes.csv` and `BC4_func_genes_indices.csv` ship in the repo. The
larger inputs — pangenome matrix, phylogroups, the *E. coli* MG1655 reference
genome, and the KEGG/FBA caches — are on the HF bucket
[McClain/minimal_genomes](https://huggingface.co/buckets/McClain/minimal_genomes)
(needs `huggingface_hub>=1.8.0`):

```bash
hf buckets sync hf://buckets/McClain/minimal_genomes/data ./data
```

The per-cohort gene lists the notebooks read (`evaluation/data/<variant>/`) are
regenerated from the trained checkpoints with `genome_minimizer_2.sampling`
(see [`notebooks/`](notebooks/README.md)).

After syncing, `data/` holds:
```
data/
├── F4_complete_presence_absence.csv    # pangenome presence/absence (bucket)
├── accessionID_phylogroup_BD.csv       # phylogroup classifications (bucket)
├── GCF_000005845.2.gbff                # E. coli MG1655 reference genome (bucket)
├── essential_genes.csv                 # literature essential genes (in repo)
└── kegg/   fba/iML1515.xml             # KEGG modules + FBA model (bucket)
```

## Commands

| Mode | Purpose | Example |
|------|---------|---------|
| `preprocess` | Extract essential gene positions | `python main.py --mode preprocess` |
| `explore` | Generate data analysis plots | `python main.py --mode explore` |
| `training` | Train VAE models | `python main.py --mode training --preset v0` |
| `sample` | Generate synthetic genomes | `python main.py --mode sample --model-path MODEL.pt --genes-path GENES.pkl` |
| `convert-samples` | Converts binary synthetic genomes to a list of gene names | `python main.py --mode convert-samples --genes-path BINARY_SAMPLES.npy` |
| `minimizer` | Create FASTA sequences | `python main.py --mode minimizer --genes-path SAMPLES.npy --single-file` |

## Parameters by Mode

### Preprocess
```bash
python main.py --mode preprocess [--force-reprocess]
```
- `--force-reprocess`: Regenerate even if files exist

### Training
```bash
python main.py --mode training --preset PRESET [--epochs N]
```
- `--preset v0/v1/v2/v3/v4`: Model architecture (required)
- `--epochs N`: Training epochs (default: 10000)

### Experiment
```bash
python main.py --mode experiment [--interactive]
```
- `--interactive`: Prompt for custom parameters

### Sample
```bash
python main.py --mode sample --model-path PATH --genes-path PATH [OPTIONS]
```
**Required:**
- `--model-path`: Trained model (.pt file)
- `--genes-path`: Essential gene positions (.pkl file)

**Optional:**
- `--num-samples N`: Number of genomes (default: 1)
- `--sampling-mode default/focused`: Strategy (default: default)
- `--noise-level N`: Noise for focused sampling (default: 0.1)
- `--genome-path`: Reference genome (.gb file)

### Binary converter
```bash
python main.py --mode convert-samples --genes-path PATH    
```

**Required:**
- `--genes-path`: Binary samples (.npy file)

**Optional:**
- `--output_file`: Output file name (default: seq_out.npy and seq_out_with_essentials.npy)

### Minimizer
```bash
python main.py --mode minimizer --genes-path PATH [OPTIONS]
```
**Required:**
- `--genes-path`: Path to a .npy file containing lists of gene names (one list per sample); Note: this is not a binary mask! pass actual gene IDs/names like the ones in the original presence absence matrix referred to in the paper.
- `--genome-path`: Reference genome (.gb, .gbff, .genbank files allowed)

**Optional:**
- `--single-file`: Output single FASTA file with all sequences (if not specified one FASTA file per sequence is generated)
- `--output-file`: Specific output filename (default for one file minimized_genomes_default.fasta, for multiple minimized_default_XXXX.fasta)
- `--output-dir`: Directory for outputs (default: ./minimized_genomes)
- `--model-name`: Label for file naming (default: "default")

**Examples:**

Single combined FASTA (generated output file - minimized_default.fasta):
```python 
python main.py --mode minimizer \
  --genes-path data/data_full_validated_IDS.npy \
  --genome-path data/GCF_000005845.2.gbff \
  --single-file \
  --output-file minimized_genomes/minimized_default.fasta \
```

Single combined FASTA (generated output file - minimized_genomes_67.fasta):
```python 
python main.py --mode minimizer \
  --genes-path data/data_full_validated_IDS.npy \
  --genome-path data/GCF_000005845.2.gbff \
  --single-file \
  --model-name 67
```

Multiple FASTAs (one per sample) into a directory (generated output files - minimized_default_XXXX.fasta):
```python 
python main.py --mode minimizer \
  --genes-path data/data_full_validated_IDS.npy \
  --genome-path data/GCF_000005845.2.gbff \
  --output-dir minimized_genomes/ \
  --model-name default
```

## Model Architectures

| Preset | Architecture | Features |
|--------|--------------|----------|
| v0 | 1024→64 | Linear KL annealing |
| v1 | 512→32 | + Gene abundance + L1 regularization |
| v2 | 512→32 | + Cosine annealing |
| v3 | 512→32 | + Weighted abundance |
| v4 | 512→32 | + Essential-gene preservation loss |

The hyperparameter-tuned `v4_opt` variant (lr 7.5e-4, essential-gene weight 0.5)
shares the v4 architecture; see `sweeps/` for the W&B sweep configs.

## Checkpoints

- **HF Hub**: [https://huggingface.co/UCL-CSSB/genome-minimizer-2](https://huggingface.co/UCL-CSSB/genome-minimizer-2) — model checkpoints, one branch per preset (`v0`–`v4`). The tuned `v4_opt` variant lives on [McClain/genome-minimizer-2](https://huggingface.co/McClain/genome-minimizer-2) (branch `v4_opt`).

Checkpoints are saved every 500 epochs (configurable via `checkpoint_every` in `ExperimentConfig`) and include full training state (model, optimizer, scheduler) for resumable training.

To download a checkpoint:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("UCL-CSSB/genome-minimizer-2", "final.pt", revision="v3")
```

## Analysis & sampling

- **Reproducible sampling** — `genome_minimizer_2.sampling` is a single module for
  both VAE sampling and the random baseline used in the paper:
  ```bash
  uv run python -m genome_minimizer_2.sampling vae    --variant v4_opt --num-samples 100
  uv run python -m genome_minimizer_2.sampling random --num-samples 100
  ```
- **Cached generations** — the per-cohort gene lists / VAE samples used by the
  analyses are on the HF bucket [McClain/minimal_genomes](https://huggingface.co/buckets/McClain/minimal_genomes)
  (`hf buckets sync hf://buckets/McClain/minimal_genomes/ ./evaluation/data`).
- **Notebooks & figures** — see [`notebooks/`](notebooks/README.md). The
  `systems_analysis` notebook is the two-tier evaluation (KEGG module coverage +
  iML1515 FBA growth); `statistical_analysis` covers gene-enrichment.
- **Experiment tracking** — training logs to W&B (`mcclain/genome-minimizer-2`); the
  hyperparameter sweep configs are in `sweeps/`.

## Output Structure

```
├── data/essential_genes/           # Preprocessing results
├── models/
│   ├── trained_models/v0_model/    # Saved model weights
│   └── v0_model/
│       ├── figures/                # Training plots
│       └── sampling_results/       # Generated samples
└── [output-dir]/                   # Final FASTA files
```

## Troubleshooting

- **Missing files**: Pipeline automatically checks and shows missing files
- **Import errors**: Ensure virtual environment is activated
- **GPU issues**: Auto-detects GPU/CPU availability
- **Path errors**: Use absolute paths for model/data files

**Pipeline flow**: preprocess → training → sample → minimizer