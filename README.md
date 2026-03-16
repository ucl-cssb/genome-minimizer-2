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

### Prerequisites
- Python >=3.9,<3.11
- pip (usually comes with Python)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd genome-minimizer-2
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install project**:
   ```bash
   pip install -e .
   ```

## Quick Start

```bash
python main.py --mode preprocess
python main.py --mode training --preset v0 --epochs 1
python main.py --mode sample --model-path models/trained_models/v0_model/saved_VAE_v0.pt --genes-path data/essential_genes/essential_gene_positions.pkl --num-samples 100
python main.py --mode convert-samples --genes-path data/binary_samples_default.npy
python main.py --mode minimizer --genes-path data/seq_out.npy --single-file --output-file results.fasta
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
- `--preset v0/v1/v2/v3`: Model architecture (required)
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

## Experiment Tracking & Checkpoints

- **W&B**: [wandb.ai/mcclain/genome-minimizer-2](https://wandb.ai/mcclain/genome-minimizer-2) — training loss curves, per-component losses, learning rate, and test metrics (F1, accuracy)
- **HF Hub**: [huggingface.co/McClain/genome-minimizer-2](https://huggingface.co/McClain/genome-minimizer-2) — model checkpoints, one branch per preset (`v0`, `v1`, `v2`, `v3`)

Checkpoints are saved every 500 epochs (configurable via `checkpoint_every` in `ExperimentConfig`) and include full training state (model, optimizer, scheduler) for resumable training.

To download a checkpoint:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("McClain/genome-minimizer-2", "final.pt", revision="v3")
```

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