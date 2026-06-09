#!/usr/bin/env python3
"""
Generate the paper training-loss figures, one PNG per model variant.

Each figure is a single self-contained panel (no subplots / compound figures):
the train and validation total-loss curves for one VAE variant (v0-v3), to be
assembled into multi-panel layouts separately. Styling follows Cell Press
figure guidelines (Arial, single-column width, 300+ dpi, minimal chrome).

OUTPUT
    notebooks/training_summary/<variant>_total_loss.png   e.g. v0_total_loss.png

DATA SOURCE
    Reads the local W&B export under paper_figures/data/ (index.json + the
    per-run history CSVs) produced by paper_figures/extract_wandb.py. The
    authoritative runs are the UCL-CSSB canonical v0-v3 (the paper checkpoints).

NOTES ON THE DATA
    - `total` is the full training objective; its composition differs by variant
      (v0: recon+KL; v1-v3: +gene_abundance+L1), so absolute loss scales are not
      comparable across variants. Each panel stands alone.
    - The held-out curve is the VALIDATION split (logged every epoch). The test
      split was scored once at the final epoch (F1/accuracy only); there is no
      per-epoch test loss, so no "test loss" curve can be drawn from this data.

Run:  uv run python notebooks/figures.py
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# --- paths ---
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "paper_figures" / "data"
OUT_DIR = Path(__file__).resolve().parent / "training_summary"

# authoritative runs = the UCL-CSSB canonical checkpoints
AUTH_PROJECT = "ucl-cssb/Genome-Minimizer"
VARIANTS = ["v0", "v1", "v2", "v3"]

# colourblind-safe (Okabe-Ito); distinguished by colour AND linestyle for B/W
TRAIN_COLOR = "#0072B2"   # blue
VAL_COLOR = "#D55E00"     # vermillion

# Cell Press house style: Arial, single-column figure, light spines, 600 dpi
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

FIGSIZE = (3.3, 2.5)   # inches; ~single-column width for Cell Press


def load_history():
    """Return {variant: DataFrame} for the canonical UCL-CSSB runs."""
    index_path = DATA_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found. Run paper_figures/extract_wandb.py first "
            "to pull the W&B tables."
        )
    index = json.loads(index_path.read_text())
    chosen = {
        r["variant"]: r for r in index
        if r["project"] == AUTH_PROJECT and r["category"] == "canonical"
        and r["variant"] in VARIANTS and r["history_path"]
    }
    missing = [v for v in VARIANTS if v not in chosen]
    if missing:
        raise KeyError(f"no canonical history for {missing} in {AUTH_PROJECT}")
    return {v: pd.read_csv(DATA_DIR / chosen[v]["history_path"]) for v in VARIANTS}


def make_total_loss_figure(variant, df, out_dir):
    """One PNG: train + validation total loss vs epoch for a single variant."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["epoch"], df["train/total"], color=TRAIN_COLOR, lw=1.5,
            label="train")
    ax.plot(df["epoch"], df["val/total"], color=VAL_COLOR, lw=1.5, ls="--",
            label="validation")
    ax.set_title(variant)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / f"{variant}_total_loss.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    for variant in VARIANTS:
        out_path = make_total_loss_figure(variant, history[variant], OUT_DIR)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
