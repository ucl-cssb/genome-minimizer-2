# 🧬 Genome Minimizer 2

VAE-powered minimal genome generation pipeline for *E. coli*.

## Install

This project uses [uv](https://docs.astral.sh/uv/). Install it once, then sync:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/ucl-cssb/genome-minimizer-2
cd genome-minimizer-2
uv sync
```

`uv sync` creates the virtualenv, installs every dependency, and installs the
package — uv manages the Python version too. On a CUDA machine it pulls the GPU
build of PyTorch automatically; on macOS it falls back to the CPU build.

Prefix commands with `uv run` (e.g. `uv run python main.py ...`), or activate the
env once with `source .venv/bin/activate` and drop the prefix.

## Get the data

All inputs live on the public HuggingFace bucket
[UCL-CSSB/genome-minimizer-2](https://huggingface.co/buckets/UCL-CSSB/genome-minimizer-2).
One command downloads everything — no login, no manual sync:

```bash
uv run python main.py --mode setup-data
```

This fetches the training data into `data/` and pre-computed samples
(v0–v3 + random baseline) into `evaluation/data/`. Add `--training-data-only`
to skip the samples.

## Two ways to use it

### A. Build genomes from existing samples (no training)

`setup-data` already gave you gene lists for each preset. Go straight to genome construction:

```bash
uv run python main.py --mode minimizer \
    --genes-path evaluation/data/v3/v3_gene_lists_with_essentials.npy \
    --genome-path data/GCF_000005845.2.gbff \
    --single-file --output-file results.fasta
```

To regenerate samples for any preset from the trained checkpoints on HuggingFace
(this is the reproducible path used in the paper):

```bash
uv run python -m genome_minimizer_2.sampling vae    --variant v3 --num-samples 100
uv run python -m genome_minimizer_2.sampling random --num-samples 100
```

### B. Train your own VAE

```bash
# 1. Essential-gene positions from the reference genome (run once)
uv run python main.py --mode preprocess

# 2. Train a preset (W&B and HF upload are off by default)
uv run python main.py --mode training --preset v3 --epochs 10000

# 3. Sample from your trained checkpoint
uv run python main.py --mode sample \
    --model-path models/trained_models/v3_model/saved_VAE_v3.pt \
    --num-samples 100

# 4. Convert binary samples to gene-name lists
uv run python main.py --mode convert-samples \
    --genes-path models/v3_model/sampling_results/v3_binary_samples_default.npy

# 5. Build minimized FASTA sequences
uv run python main.py --mode minimizer \
    --genes-path seq_out_with_essentials.npy \
    --genome-path data/GCF_000005845.2.gbff \
    --single-file --output-file results.fasta
```

## Commands

| Mode | Purpose |
|------|---------|
| `setup-data` | Download training data + samples from HuggingFace (`--training-data-only` to skip samples) |
| `preprocess` | Extract essential gene positions from the reference genome (run once) |
| `explore` | Dataset analysis plots |
| `training` | Train a VAE preset |
| `experiment` | Train with a fully custom config (`--interactive` to be prompted) |
| `sample` | Sample from a locally trained checkpoint |
| `convert-samples` | Convert binary genome masks to gene-name lists |
| `minimizer` | Build minimized FASTA sequences from gene lists |

### Training

```bash
uv run python main.py --mode training --preset PRESET [--epochs N] [--wandb] [--hf-upload]
```

Each preset is a hardcoded configuration. `--epochs` overrides the epoch count;
everything else (architecture, loss schedule) is fixed per preset — use
`--mode experiment` for full control.

| Preset | Architecture | Loss |
|--------|--------------|------|
| `v0` | 1024 → 64 | Reconstruction + KL (linear annealing) |
| `v1` | 512 → 32 | + gene abundance + L1 |
| `v2` | 512 → 32 | + cosine KL annealing |
| `v3` | 512 → 32 | + weighted abundance |

W&B hyperparameter sweep configs are in `sweeps/`.

W&B logging and HF Hub checkpoint upload are **off by default**. Enable with
`--wandb` (needs a W&B login) and `--hf-upload` (needs write access to the repo).

### Minimizer

```bash
uv run python main.py --mode minimizer --genes-path PATH --genome-path PATH [OPTIONS]
```

- `--genes-path`: `.npy` of gene-name lists (one list per genome) — actual gene
  IDs as in the presence/absence matrix, **not** a binary mask.
- `--genome-path`: reference genome (`.gb` / `.gbff` / `.genbank`).
- `--single-file`: one combined FASTA (default: one file per genome).
- `--output-file` / `--output-dir` / `--model-name`: output naming.

## Checkpoints

Trained checkpoints are on HuggingFace:
[UCL-CSSB/genome-minimizer-2](https://huggingface.co/UCL-CSSB/genome-minimizer-2),
one branch per preset (`v0`–`v3`).
Each `final.pt` is a full training-state checkpoint (model + optimizer + scheduler).
The sampling module loads these directly — you don't download them by hand.

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("UCL-CSSB/genome-minimizer-2", "final.pt", revision="v3")
```

## Analysis & notebooks

- **Sampling module** — `genome_minimizer_2.sampling` reproduces the VAE samples
  and random baseline used in the paper (pulls checkpoints from HuggingFace).
- **Notebooks** — see [`notebooks/`](notebooks/README.md): `systems_analysis` is
  the two-tier evaluation (KEGG module coverage + iML1515 FBA growth);
  `statistical_analysis` covers gene-enrichment.

## Output layout

```
data/                                # training data (from setup-data)
evaluation/data/{variant}/           # pre-computed samples (from setup-data)
data/essential_genes/                # preprocess output (.pkl)
models/trained_models/{preset}_model/   # saved .pt checkpoint
models/{preset}_model/figures/       # training plots
models/{preset}_model/sampling_results/ # raw binary samples
```
