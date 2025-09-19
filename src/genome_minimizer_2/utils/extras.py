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
    plt.hist(essential_genes_count_per_sample, color=plot_color)
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
    'plot_loss_vs_epochs_graph'
]