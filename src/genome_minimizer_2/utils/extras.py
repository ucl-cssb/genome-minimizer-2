#!/usr/bin/env python3
"""
Extra utility functions for VAE training and evaluation.
"""

# Import mlibraries
import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Import modules
from src.genome_minimizer_2.training.model import VAE

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_essential_vs_total(essential_counts, total_counts, output_path):
    """Plot relationship between essential and total gene counts."""
    plt.figure(figsize=(4,4))
    plt.scatter(total_counts, essential_counts, color='violet')
    sns.regplot(x=total_counts, y=essential_counts, scatter=False, color='black')
    plt.xlabel("Genome size")
    plt.ylabel("Essential genes")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def write_samples_to_dataframe(binary_generated_samples, all_genes, output_file):
        """Convert binary generated samples to a DataFrame with genes as rows and samples as columns."""
        df = pd.DataFrame(binary_generated_samples, columns=all_genes)
        df.index = [f"Sample_{i+1}" for i in range(df.shape[0])]
        df = df.transpose()  # Transpose to get genes x samples
        df.columns = [f"Sample_{i+1}" for i in range(df.shape[1])]  # Rename columns
        df = df.reset_index()  # Move gene names from index to a column
        df = df.rename(columns={'index': 'Gene'})  # Rename that column to 'Gene'
        df.to_csv(output_file, index=False)

def extract_prefix(gene):
    """Extract gene prefix from gene name."""
    match = re.match(r"([a-zA-Z0-9]+)", gene)
    if match:
        return match.group(1)
    return gene


def count_essential_genes(binary_generated_samples, essential_gene_positions):
    """
    Count essential genes in the generated samples

    Parameters:
    ----------
    binary_generated_samples - a 10000 x 55390 array with 10k samples and 55390 boolean values 
    corresponding to genes

    essential_gene_positions - a boolean, pre-calculated mask to spot essential genes

    Returns:
    -------
    essential_genes_count_per_sample - a 10000 element array where each element shows the total number 
    of essential genes in that sample
    """
    nsamples = binary_generated_samples.shape[0]
    binary_generated_samples = binary_generated_samples.astype(int)
    essential_genes_count_per_sample = np.zeros(nsamples, dtype=int)

    for sample_index in range(nsamples):
        present_essential_genes = 0
        
        for _, positions in essential_gene_positions.items():
            if len(positions) == 1:
                pos = positions[0]
                if pos < binary_generated_samples.shape[1]:
                    if binary_generated_samples[sample_index, pos] != 0:
                        present_essential_genes += 1
            else:
                for pos in positions:
                    if pos < binary_generated_samples.shape[1]:
                        if binary_generated_samples[sample_index, pos] != 0:
                            present_essential_genes += 1
                            break

        essential_genes_count_per_sample[sample_index] = present_essential_genes

    return essential_genes_count_per_sample


def plot_essential_genes_distribution(essential_genes_count_per_sample, figure_name, plot_color, x_min=0, x_max=0):
    """
    Plot the frequency of essential genes of the samples

    Parameters:
    ----------
    essential_genes_count_per_sample - a 10000 element array where each element shows the total number 
    of essential genes in that sample (counted by count_essential_genes function)

    figure_name - name of the pdf figure

    plot_color - color of the plot

    Returns:
    -------
    None, saves a pdf image of the plot in the current working directory 
    """
    median = np.median(essential_genes_count_per_sample)
    min_value = np.min(essential_genes_count_per_sample)
    max_value = np.max(essential_genes_count_per_sample)

    plt.figure(figsize=(5,5))
    plt.hist(essential_genes_count_per_sample, color=plot_color, range=(x_min, x_max), bins=30)
    plt.xlim(x_min, x_max)
    plt.xlabel('Essential genes')
    plt.ylabel('Frequency')

    plt.axvline(median, color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    dummy_min = plt.Line2D([], [], color='black',  linewidth=2, label=f'Min: {min_value:.2f}')
    dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f'Max: {max_value:.2f}')

    handles = [plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}'), dummy_min, dummy_max]

    plt.legend(handles=handles, fontsize=6)
    plt.savefig(figure_name, format="pdf", bbox_inches="tight")


def plot_samples_distribution(binary_generated_samples, figure_name, plot_color, x_min=0, x_max=0):
    """
    Plot the frequency distribution of genome sizes

    Parameters:
    ----------
    binary_generated_samples - a 10000 x 55390 array with 10k samples and 55390 boolean values 
    corresponding to genes

    figure_name - name of the pdf figure

    plot_color - color of the plot

    Returns:
    -------
    None, saves a pdf image of the plot in the current working directory 
    """
    samples_size_sum = binary_generated_samples.sum(axis=1)

    median = np.median(samples_size_sum)
    min_value = np.min(samples_size_sum)
    max_value = np.max(samples_size_sum)

    plt.figure(figsize=(5,5))
    plt.hist(samples_size_sum, color=plot_color)
    plt.xlim(x_min, x_max)
    plt.xlabel('Genome size')
    plt.ylabel('Frequency')

    plt.axvline(median, color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    dummy_min = plt.Line2D([], [], color='black',  linewidth=2, label=f'Min: {min_value:.2f}')
    dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f'Max: {max_value:.2f}')

    handles = [plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}'), dummy_min, dummy_max]

    plt.legend(handles=handles, fontsize=6, loc='upper left')
    plt.savefig(figure_name, format="pdf", bbox_inches="tight")


def load_model(input_dim, hidden_dim, latent_dim, path_to_model):
    """
    Load a saved VAE model 

    Parameters:
    ----------
    input_dim - input dimension of the model 

    hidden_dim - hidden dimension of the model 

    latent_dim - latent dimension of the model 

    path_to_model - path to the model where its stored

    Returns:
    -------
    model - return the loaded model so it can then be subsequently used for sampling
    """
    # Load trained model 
    model = VAE(input_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(path_to_model, weights_only=True))  
    model.eval()  

    return model


def sample_from_model(model, latent_dim, num_samples, device):
    """Sample new data from trained VAE model."""
    model.to(device)

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device) 
        generated_samples = model.decode(z).cpu().numpy() 

    threshold = 0.5
    binary_generated_samples = (generated_samples > threshold).astype(float)

    return binary_generated_samples, generated_samples, z

def get_latent_variables(model, data_loader, device):
    """
    Extract latent variables from a given model using a data loader.

    Parameters:
    ----------
    model - The neural network model to extract latent variables from.
    data_loader - The data loader providing the input data.
    device - The device (CPU or GPU) to perform computations on.

    Returns:
    -------
    An array of latent variables extracted from the model.
    """
    model.eval()
    latents = []
    with torch.no_grad():
        for data in data_loader:
            data = data[0].to(torch.float).to(device)
            mean, _ = model.encode(data)
            latents.append(mean.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    return latents


def plot_loss_vs_epochs_graph(epochs, train_loss_vals, val_loss_vals, fig_name):
    """
    Plot the training and validation loss versus epochs

    Parameters:
    ----------
    epochs - The list of epoch numbers
    train_loss_vals - The list of training loss values
    val_loss_vals - The list of validation loss values
    fig_name - The name of the file to save the plot

    Returns:
    -------
    None, saves a pdf image of the plot in the current working directory 
    """
    plt.figure(figsize=(4,4), dpi=300)
    plt.scatter(epochs, train_loss_vals, color='dodgerblue')
    plt.plot(epochs, train_loss_vals, label='Train Loss', color='dodgerblue')
    plt.scatter(epochs, val_loss_vals, color='darkorange')
    plt.plot(epochs, val_loss_vals, label='Validation Loss', color='darkorange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)
    plt.savefig(fig_name, format="pdf", bbox_inches="tight")
    plt.close()


# Publication styling shared by the component-breakdown figure.
_PUB_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "mathtext.default": "regular",
}

_TRAIN_COLOR = "#2c3e8c"   # deep blue
_VAL_COLOR = "#e08214"     # warm orange
_TEST_COLOR = "#3a8a5f"    # green (test split in distribution panels)

# Order components are laid out in; anything else is appended after these.
_COMPONENT_ORDER = [
    "total", "reconstruction", "kl_divergence",
    "gene_abundance", "l1_regularization", "l2_regularization", "essential_gene",
]

_PRETTY_NAMES = {
    "total": "Total loss",
    "reconstruction": "Reconstruction (BCE)",
    "kl_divergence": "KL divergence",
    "gene_abundance": "Gene abundance",
    "l1_regularization": "L1 regularisation",
    "l2_regularization": "L2 regularisation",
    "essential_gene": "Essential-gene",
}


def _pretty_component(name):
    return _PRETTY_NAMES.get(name, name.replace("_", " ").capitalize())


def plot_loss_components(train_losses, val_losses, fig_name, title=None):
    """
    Publication-quality breakdown of the VAE training loss into its components.

    Renders one small-multiple panel per active loss component (total,
    reconstruction, KL divergence, and any extra terms such as gene abundance
    or L1), each with its own y-scale so terms of very different magnitude stay
    readable. Train and validation curves share a single figure-level legend.

    Parameters
    ----------
    train_losses, val_losses : dict[str, list[float]]
        Per-epoch loss values keyed by component name, plus a 'total' key — i.e.
        the LossTracker.train_losses / val_losses dicts produced during training.
    fig_name : str
        Output path. The format is inferred from the extension (.pdf, .png, ...).
    title : str, optional
        Figure suptitle (e.g. the preset name).
    """
    # Keep only components that were actually active (skip all-zero terms like a
    # disabled L1), but always keep total/reconstruction/KL for context.
    always = {"total", "reconstruction", "kl_divergence"}
    ordered = [k for k in _COMPONENT_ORDER if k in train_losses]
    ordered += [k for k in train_losses if k not in _COMPONENT_ORDER]

    def _active(key):
        series = train_losses.get(key, [])
        return key in always or (len(series) > 0 and np.any(np.abs(series) > 1e-9))

    components = [k for k in ordered if _active(k)]
    if not components:
        return

    n = len(components)
    ncols = 2 if n == 4 else min(n, 3)  # 4 terms read better as a balanced 2x2
    nrows = int(np.ceil(n / ncols))

    with plt.rc_context(_PUB_RC):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(2.9 * ncols, 2.5 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        flat_axes = axes.flatten()
        train_handle = val_handle = None

        for ax, comp in zip(flat_axes, components):
            tr = np.asarray(train_losses.get(comp, []), dtype=float)
            va = np.asarray(val_losses.get(comp, []), dtype=float)
            ep = np.arange(1, len(tr) + 1)
            # For very short runs markers read better than thin lines.
            marker = "o" if len(ep) <= 25 else None

            (train_handle,) = ax.plot(
                ep, tr, color=_TRAIN_COLOR, lw=1.6, marker=marker, ms=3, label="Train",
            )
            if len(va) == len(tr) and len(va) > 0:
                (val_handle,) = ax.plot(
                    ep, va, color=_VAL_COLOR, lw=1.6, ls="--", marker=marker, ms=3,
                    label="Validation",
                )

            ax.set_title(_pretty_component(comp))
            ax.grid(True, axis="both", ls=":", lw=0.6, alpha=0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(x=0.02)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))

        # Shared axis labels along the figure edges.
        for r in range(nrows):
            axes[r][0].set_ylabel("Loss")
        for c in range(ncols):
            axes[nrows - 1][c].set_xlabel("Epoch")

        # Blank any unused panels in the grid.
        for ax in flat_axes[n:]:
            ax.set_visible(False)

        handles = [h for h in (train_handle, val_handle) if h is not None]
        if handles:
            # "outside lower center" reserves its own space under the panels, so
            # the legend never overlaps the bottom-row x-axis labels — which it
            # otherwise does in the short single-row (3-panel) layout.
            fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold")

        fig.savefig(fig_name, bbox_inches="tight")
        plt.close(fig)


# Export all functions
__all__ = [
    'create_dataloaders',
    'plot_essential_vs_total',
    'extract_prefix', 
    'count_essential_genes',
    'plot_essential_genes_distribution',
    'plot_samples_distribution',
    'load_model',
    'sample_from_model',
    'l1_regularization',
    'cosine_annealing_schedule',
    'get_latent_variables',
    'plot_loss_vs_epochs_graph',
    'plot_loss_components'
]