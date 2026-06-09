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

Clone the repository:
```bash
git clone https://github.com/ucl-cssb/genome-minimizer-2
cd genome-minimizer-2
```

Install with uv (recommended) or a plain virtualenv — both are supported.

### Option A — uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtualenv and dependencies from
`pyproject.toml`:
```bash
uv sync                                # create .venv, install deps + the package
uv run python main.py --mode preprocess
```
Prefix commands with `uv run`, or activate the env: `source .venv/bin/activate`.

### Option B — plain virtualenv + pip

For environments without uv, use the pinned `requirements.txt`:
```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # pinned dependencies
pip install -e . --no-deps             # install the genome_minimizer_2 package
```

Either path lets you run the commands below as `python main.py ...` (with the env
activated) or `uv run python main.py ...`.

## Quick Start

```bash
python main.py --mode preprocess
python main.py --mode training --preset v0 --epochs 1
python main.py --mode sample --model-path models/trained_models/v0_model/saved_VAE_v0.pt --genes-path data/essential_genes/essential_gene_positions.pkl --num-samples 100
python main.py --mode convert-samples --genes-path data/binary_samples_default.npy
python main.py --mode minimizer --genes-path data/seq_out.npy --single-file --output-file results.fasta
```

All above commands can also be run with `uv run` prefix, for example:
```bash
uv run python main.py --mode preprocess
```

## Data Setup

Place these files in `data/`:
```
data/
├── F4_complete_presence_absence.csv    # Gene presence/absence matrix
├── accessionID_phylogroup_BD.csv       # Phylogroup classifications
├── essential_genes.csv                 # Essential genes from literature
└── wild_type_sequence.gb               # E. coli reference genome
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