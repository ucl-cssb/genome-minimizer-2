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
from src.genome_minimizer_2.utils.extras import get_latent_variables

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


def create_training_summary_plot(train_losses: List[float], val_losses: List[float], 
                                f1_scores: List[float], accuracy_scores: List[float],
                                output_dir: str, model_name: str = "VAE") -> None:
    """
    Create a comprehensive summary plot of training results.
    
    Args:
        train_losses: Training loss values
        val_losses: Validation loss values
        f1_scores: Per-sample F1 scores
        accuracy_scores: Per-sample accuracy scores
        output_dir: Directory to save plots
        model_name: Name of the model for titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    
    # Training curves
    epochs = range(1, len(train_losses) + 1)
    axes[0, 0].plot(epochs, train_losses, label='Training Loss', color='blue', alpha=0.8)
    axes[0, 0].plot(epochs, val_losses, label='Validation Loss', color='red', alpha=0.8)
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'{model_name} Training Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 score distribution
    axes[0, 1].hist(f1_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.mean(f1_scores), color='darkgreen', linestyle='--', 
                       label=f'Mean: {np.mean(f1_scores):.3f}')
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('F1 Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy distribution
    axes[1, 0].hist(accuracy_scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(np.mean(accuracy_scores), color='darkviolet', linestyle='--', 
                       label=f'Mean: {np.mean(accuracy_scores):.3f}')
    axes[1, 0].set_xlabel('Accuracy Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Accuracy Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    {model_name} Training Summary

    Final Training Loss: {train_losses[-1]:.4f}
    Final Validation Loss: {val_losses[-1]:.4f}

    F1 Score Statistics:
    - Mean: {np.mean(f1_scores):.4f}
    - Std:  {np.std(f1_scores):.4f}
    - Min:  {np.min(f1_scores):.4f}
    - Max:  {np.max(f1_scores):.4f}

    Accuracy Statistics:
    - Mean: {np.mean(accuracy_scores):.4f}
    - Std:  {np.std(accuracy_scores):.4f}
    - Min:  {np.min(accuracy_scores):.4f}
    - Max:  {np.max(accuracy_scores):.4f}

    Total Epochs: {len(train_losses)}
    """
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                     fontsize=11, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_training_summary.pdf"), 
               format="pdf", bbox_inches="tight")
    plt.close()


# Export all functions
__all__ = [
    'plot_latent_space_pca',
    'plot_latent_dimensions_distribution',
    'plot_reconstruction_examples',
    'create_training_summary_plot'
]