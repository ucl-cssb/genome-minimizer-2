#!/usr/bin/env python3
"""
Visualization functions for VAE
"""

# Import Libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List, Optional
import os
from src.genome_minimizer_2.utils.extras import (
    get_latent_variables, plot_loss_components,
    _PUB_RC, _TRAIN_COLOR, _VAL_COLOR, _TEST_COLOR,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_latent_space_pca(model, test_loader, config, test_phylogroups: np.ndarray, 
                         output_dir: str, n_components: int = 3,
                         show_plot: bool = True) -> pd.DataFrame:
    """
    Visualize latent space using PCA.
    
    Args:
        model: Trained VAE model
        test_loader: DataLoader for test data
        test_phylogroups: Array of phylogroup labels
        output_dir: Directory to save plots
        n_components: Number of PCA components
        show_plot: Whether to display the plot
        
    Returns:
        DataFrame with PCA coordinates and phylogroups
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get latent variables
    latents = get_latent_variables(model, test_loader, device)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(latents)
    
    # Create DataFrame
    column_names = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(data_pca, columns=column_names)
    df_pca['phylogroup'] = test_phylogroups
    
    if show_plot:
        #fig, axes = plt.subplots(1, 2, figsize=(8,4), dpi=300)
        #sns.scatterplot(x='PC1', y='PC2', hue=df_pca['phylogroup'], data=df_pca, ax=axes[0])
        #sns.scatterplot(x='PC2', y='PC3', hue=df_pca['phylogroup'], data=df_pca, ax=axes[1])
        # for ax in axes:
        #     handles, labels = ax.get_legend_handles_labels()
        #     ax.legend(handles, labels, fontsize=8)
        #     ax.set_aspect('equal', adjustable='box')
        
        fig, ax = plt.subplots(figsize=(5,5))
        sns.scatterplot(x='PC1', y='PC2', hue=df_pca['phylogroup'], data=df_pca, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=6)
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_aspect('equal', adjustable='box')
        
        #plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{config.trainer_version}_pca_latent_space_test_set.pdf"), 
                   format="pdf", bbox_inches="tight")
        plt.close()
        
        print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return df_pca


def plot_latent_dimensions_distribution(model, test_loader, output_dir: str) -> None:
    """
    Plot distribution of values in each latent dimension.
    
    Args:
        model: Trained VAE model
        test_loader: DataLoader for test data
        output_dir: Directory to save plots
        show_plot: Whether to display the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get latent variables
    latents = get_latent_variables(model, test_loader, device)
    
    # Plot distributions
    n_dims = latents.shape[1]
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows), dpi=300)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i in range(n_dims):
        axes[i].hist(latents[:, i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Latent Dim {i+1}', fontsize=10)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_dimensions_distribution.pdf"), 
               format="pdf", bbox_inches="tight")
    plt.close()


def plot_reconstruction_examples(model, test_loader, output_dir: str, 
                               n_examples: int = 5, show_plot: bool = True) -> None:
    """
    Plot examples of original vs reconstructed data.
    
    Args:
        model: Trained VAE model
        test_loader: DataLoader for test data
        output_dir: Directory to save plots
        n_examples: Number of examples to plot
        show_plot: Whether to display the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    examples_plotted = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if examples_plotted >= n_examples:
                break
                
            batch_data = batch[0].to(device)
            recon_x, mu, logvar = model(batch_data)
            
            # Take first few samples from batch
            batch_size = min(n_examples - examples_plotted, batch_data.shape[0])
            
            for i in range(batch_size):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Original data
                original = batch_data[i].cpu().numpy()
                ax1.plot(original, alpha=0.7)
                ax1.set_title(f'Original Sample {examples_plotted + 1}')
                ax1.set_xlabel('Gene Index')
                ax1.set_ylabel('Presence')
                
                # Reconstructed data
                reconstructed = recon_x[i].cpu().numpy()
                ax2.plot(reconstructed, alpha=0.7, color='orange')
                ax2.set_title(f'Reconstructed Sample {examples_plotted + 1}')
                ax2.set_xlabel('Gene Index')
                ax2.set_ylabel('Probability')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"reconstruction_example_{examples_plotted + 1}.pdf"), 
                           format="pdf", bbox_inches="tight")
                plt.close()
                
                examples_plotted += 1
                if examples_plotted >= n_examples:
                    break


def _distribution_panel(ax, train_vals, test_vals, xlabel, title):
    """Overlaid train-vs-test density histogram with mean markers."""
    train_vals = np.asarray(train_vals, dtype=float)
    test_vals = np.asarray(test_vals, dtype=float)
    lo = min(train_vals.min(), test_vals.min())
    hi = max(train_vals.max(), test_vals.max())
    bins = np.linspace(lo, hi, 28) if hi > lo else 28
    ax.hist(train_vals, bins=bins, color=_TRAIN_COLOR, alpha=0.55, density=True, label="Train")
    ax.hist(test_vals, bins=bins, color=_TEST_COLOR, alpha=0.55, density=True, label="Test")
    ax.axvline(train_vals.mean(), color=_TRAIN_COLOR, ls="--", lw=1.2)
    ax.axvline(test_vals.mean(), color=_TEST_COLOR, ls="--", lw=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.4)
    ax.legend()


def create_training_summary_plot(train_losses: List[float], val_losses: List[float],
                                f1_test: List[float], accuracy_test: List[float],
                                output_dir: str, model_name: str = "VAE",
                                f1_train: Optional[List[float]] = None,
                                accuracy_train: Optional[List[float]] = None) -> None:
    """
    Publication-quality training summary: total loss curve, train-vs-test
    distributions of per-sample reconstruction F1 and accuracy, and a compact
    stats table.

    Args:
        train_losses, val_losses: per-epoch total loss.
        f1_test, accuracy_test: per-sample reconstruction metrics on the test set.
        output_dir: directory to save into.
        model_name: title / filename stem.
        f1_train, accuracy_train: per-sample metrics on the train set. When given,
            the distribution panels overlay train vs test; otherwise test only.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Fall back to test-only if train metrics weren't supplied.
    f1_tr = np.asarray(f1_train if f1_train is not None else f1_test, dtype=float)
    acc_tr = np.asarray(accuracy_train if accuracy_train is not None else accuracy_test, dtype=float)
    f1_te = np.asarray(f1_test, dtype=float)
    acc_te = np.asarray(accuracy_test, dtype=float)

    with plt.rc_context(_PUB_RC):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), constrained_layout=True)

        # Total loss
        ep = np.arange(1, len(train_losses) + 1)
        axes[0, 0].plot(ep, train_losses, color=_TRAIN_COLOR, lw=1.6, label="Train")
        axes[0, 0].plot(ep, val_losses, color=_VAL_COLOR, lw=1.6, ls="--", label="Validation")
        axes[0, 0].set_title("Total loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].spines["top"].set_visible(False)
        axes[0, 0].spines["right"].set_visible(False)
        axes[0, 0].grid(True, ls=":", lw=0.6, alpha=0.4)
        axes[0, 0].legend()
        axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))

        # Train-vs-test metric distributions
        _distribution_panel(axes[0, 1], f1_tr, f1_te, "F1 score", "Reconstruction F1")
        _distribution_panel(axes[1, 0], acc_tr, acc_te, "Accuracy", "Reconstruction accuracy")

        # Compact stats table
        ax = axes[1, 1]
        ax.axis("off")
        rows = [
            ("", "Train", "Test"),
            ("F1 (mean)", f"{f1_tr.mean():.3f}", f"{f1_te.mean():.3f}"),
            ("Accuracy (mean)", f"{acc_tr.mean():.3f}", f"{acc_te.mean():.3f}"),
            ("Final loss", f"{train_losses[-1]:.2e}", f"{val_losses[-1]:.2e}"),
            ("Epochs", f"{len(train_losses)}", ""),
        ]
        tbl = ax.table(cellText=rows, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if r == 0:
                cell.set_text_props(fontweight="bold")
            if c == 0:
                cell.set_text_props(ha="left")

        fig.suptitle(f"{model_name} training summary", fontsize=12, fontweight="bold")
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(output_dir, f"{model_name}_training_summary.{ext}"),
                        bbox_inches="tight")
        plt.close(fig)


# Export all functions
__all__ = [
    'plot_latent_space_pca',
    'plot_latent_dimensions_distribution',
    'plot_reconstruction_examples',
    'create_training_summary_plot'
]