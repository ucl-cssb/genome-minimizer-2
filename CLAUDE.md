# genome-minimizer-2

VAE-based minimal E. coli genome design. Cell Systems submission (CELL-SYSTEMS-D-25-00724).

## Storage

- **Models**: `McClain/genome-minimizer-2` on HF Hub (branches v0-v4)
- **Data bucket**: `McClain/minimal_genomes` on HF Buckets (private)
  - Sync up: `hf buckets sync ./local_dir hf://buckets/McClain/minimal_genomes/path/`
  - Sync down: `hf buckets sync hf://buckets/McClain/minimal_genomes/path/ ./local_dir`
  - List: `hf buckets ls hf://buckets/McClain/minimal_genomes/`
  - Requires `huggingface_hub>=1.8.0` for bucket support
- **W&B**: `mcclain/genome-minimizer-2`

## Bucket layout

```
hf://buckets/McClain/minimal_genomes/
├── v0/samples/          # VAE samples + gene lists per variant
├── v1/samples/
├── v2/samples/
├── v3/samples/
├── random/samples/      # Random baseline gene lists
├── results/             # vEcoli viability results (JSON per sample)
│   ├── v0/
│   ├── v1/
│   ├── v2/
│   ├── v3/
│   └── random/
└── minimized_genomes/   # DNA sequences for viable genomes (post-minimizer)
```

## vEcoli evaluation (Myriad HPC)

See `evaluation/infra.md` for full details. Key commands:

```bash
ssh myriad
qsub ~/Scratch/genome-minimizer/evaluation/eval_job.sh   # single sample
# For array: add #$ -t 1-100 to eval_job.sh
```

## Model dimensions

| Variant | hidden_dim | latent_dim | input_dim |
|---------|-----------|-----------|-----------|
| v0      | 1024      | 64        | 55039     |
| v1-v4   | 512       | 32        | 55039     |
