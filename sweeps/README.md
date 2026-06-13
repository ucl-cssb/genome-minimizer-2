# sweeps/

W&B sweep configurations for the genome-minimizer-2 VAE. Each `*.yaml` is a
self-contained [W&B sweep](https://docs.wandb.ai/guides/sweeps) that trains the
model and optimises `test/f1_overall` (gene-level reconstruction F1 on the
596-genome test split).

## How the harness works

Every config drives the same entry point:

```
main.py --mode=experiment --trainer-version=vN <fixed flags> ${args}
```

- `--mode=experiment` → `run_custom_experiment` → `setup_experiment_config`
  (`src/genome_minimizer_2/utils/custom_config.py`) builds the config, trains,
  evaluates on the test set, and logs `test/f1_overall` + `test/accuracy_overall`.
- **v0–v3 differ by *loss function*, not hyperparameters** (see
  `training/training/trainer.py:create_vN_trainer`). `--trainer-version` picks the
  loss; the sweep varies hyperparameters on top of it.
- Parameter keys are **kebab-case** (`learning-rate`, not `learning_rate`) so the
  `${args}` macro emits flags argparse accepts (`--learning-rate=…`). W&B does
  *not* translate underscores — using underscore keys silently produces flags the
  CLI rejects.

## Running one

```bash
wandb sweep sweeps/<config>.yaml                       # prints a sweep id
wandb agent mcclain/genome-minimizer-2/<sweep_id>      # runs the agent (needs a GPU)
```

Run the agent on a GPU box (g6-big / AWS DLAMI). Launch multiple agents to
parallelise. A 1-epoch smoke test (`--n-epochs=1`) before a full sweep is cheap
insurance that the chosen `--trainer-version` path is wired correctly.

## Configs

| file | base loss | method | sweeps over | status |
|------|-----------|--------|-------------|--------|
| `v0_grid.yaml` | v0 (recon + linear KL) | grid (18) | hidden-dim, latent-dim, learning-rate | **not yet run** |
| `baseline_v1_seeds.yaml` | v1 | grid (3) | random-state | run → `m6ktu3xf` |

### `v0_grid.yaml` — basic HPO on v0 (paper)
3 × 3 × 2 = 18-run grid over **hidden-dim** {256, 512, 1024}, **latent-dim**
{32, 64, 128}, **learning-rate** {0.01, 0.001}; single seed (12345). The v0 loss
(reconstruction + linear-annealed KL) is held fixed, so this isolates capacity /
LR. Use the W&B sweep **Parameter Importance** panel for the feature-importance
report. **Not yet run.**

### `baseline_v1_seeds.yaml` — Phase B noise floor → `m6ktu3xf`
v1 trained with 3 seeds, everything else fixed. The spread of `test/f1_overall`
across seeds is the **noise floor** for interpreting hyperparameter sweeps.
Result: 2/3 finished, F1 = **0.9862 / 0.9872** → seed noise ≈ **0.001**.

## Note on orphan W&B sweeps

Four other sweep ids exist in the W&B project with **no config here** —
`xcat8dyn`, `y6i2wdrp`, `88x3mgbe`, `u2et2gmh`. All are `random_state`-only grids
that **crashed on launch (0 logged steps)**; they are false starts and carry no
usable data.
