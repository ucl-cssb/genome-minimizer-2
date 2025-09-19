
#!/usr/bin/env python3
"""
Evaluation metrics for VAE
"""

# Import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from typing import Tuple, List
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_reconstruction_metrics(model, test_loader, threshold: float = 0.5) -> Tuple[float, float, List[float], List[float]]:
    """
    Calculate F1 scores and accuracy for VAE reconstructions.
    
    Args:
        model: Trained VAE model
        test_loader: DataLoader for test data
        threshold: Threshold for binarizing reconstructions
        
    Returns:
        Tuple of (overall_f1, overall_accuracy, per_sample_f1_scores, per_sample_accuracy_scores)
    """
    model.eval()
    all_recon_x = []
    all_test_data = []
    
    # Collect all reconstructions and original data
    with torch.no_grad():
        for batch in test_loader:
            batch_data = batch[0].to(device)
            recon_x, mu, logvar = model(batch_data)
            all_recon_x.append(recon_x.to(device))
            all_test_data.append(batch_data.to(device))
    
    all_recon_x = torch.cat(all_recon_x)
    all_test_data = torch.cat(all_test_data)
    
    # Binarize reconstructions
    recon_x_binarized = (all_recon_x > threshold).int()
    
    # Calculate overall metrics (flattened)
    all_test_data_np = all_test_data.cpu().numpy().flatten()
    recon_x_binarized_np = recon_x_binarized.cpu().numpy().flatten()
    
    overall_f1 = sklearn.metrics.f1_score(all_test_data_np, recon_x_binarized_np)
    overall_accuracy = sklearn.metrics.accuracy_score(all_test_data_np, recon_x_binarized_np)
    
    # Calculate per-sample metrics
    f1_scores = []
    accuracy_scores = []
    
    for genome_x, genome in zip(recon_x_binarized.cpu(), all_test_data.cpu().int()):
        f1_scores.append(sklearn.metrics.f1_score(genome_x.numpy(), genome.numpy()))
        accuracy_scores.append(sklearn.metrics.accuracy_score(genome_x.numpy(), genome.numpy()))
    
    return overall_f1, overall_accuracy, f1_scores, accuracy_scores


def generate_metric_histograms(f1_scores: List[float], accuracy_scores: List[float], config,
                               output_dir: str) -> None:
    """
    Generate and save histograms of F1 and accuracy scores.
    
    Args:
        f1_scores: List of per-sample F1 scores
        accuracy_scores: List of per-sample accuracy scores
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # F1 score histogram
    plt.figure(figsize=(4, 4), dpi=300)
    #plt.hist(f1_scores, color='dodgerblue', bins=30)
    plt.hist(f1_scores, color='dodgerblue')
    plt.xlabel("F1 score")
    plt.ylabel("Frequency")
    #plt.title("Distribution of F1 Scores")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.9, 1)
    # make square
    plt.tight_layout()

    # Add statistics text
    #mean_f1 = np.mean(f1_scores)
    mean_f1 = np.median(f1_scores)
    # std_f1 = np.std(f1_scores)
    plt.axvline(mean_f1, color='red', linestyle='--', alpha=0.8, label=f'Median: {mean_f1:.3f}')
    plt.legend()
    
    print("################### Here:", config.trainer_version)

    plt.savefig(os.path.join(output_dir, f"{config.trainer_version}_f1_score_frequency_test_set.pdf"), 
               format="pdf", bbox_inches="tight")
    plt.close()
    
    # Accuracy histogram
    plt.figure(figsize=(4, 4), dpi=300)
    plt.hist(accuracy_scores, color='dodgerblue')
    plt.xlabel("Accuracy Score")
    plt.ylabel("Frequency")
    #plt.title("Distribution of Accuracy Scores")
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_acc = np.mean(accuracy_scores)
    # std_acc = np.std(accuracy_scores)
    plt.axvline(mean_acc, color='darkred', linestyle='--', alpha=0.8, label=f'Mean: {mean_acc:.3f}')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f"{config.trainer_version}_accuracy_score_frequency_test_set.pdf"), 
               format="pdf", bbox_inches="tight")
    plt.close()


def print_metric_summary(config, overall_f1: float, overall_accuracy: float, 
                        f1_scores: List[float], accuracy_scores: List[float],
                        output_dir: str = None) -> None:
    """
    Print a comprehensive summary of metrics and optionally save to file.
    
    Args:
        overall_f1: Overall F1 score
        overall_accuracy: Overall accuracy
        f1_scores: Per-sample F1 scores
        accuracy_scores: Per-sample accuracy scores
        output_dir: Directory to save the summary file (optional)
    """
    from datetime import datetime
    from pathlib import Path
    
    # Create the summary report
    report = f"""
    ===============================================
    RECONSTRUCTION METRICS SUMMARY REPORT
    ===============================================
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Overall Metrics (flattened):
    - F1 Score: {overall_f1:.4f}
    - Accuracy: {overall_accuracy:.4f}

    Per-Sample Metrics:
    - F1 Score - Mean: {np.mean(f1_scores):.4f}, Std: {np.std(f1_scores):.4f}
    - F1 Score - Min: {np.min(f1_scores):.4f}, Max: {np.max(f1_scores):.4f}
    - Accuracy - Mean: {np.mean(accuracy_scores):.4f}, Std: {np.std(accuracy_scores):.4f}
    - Accuracy - Min: {np.min(accuracy_scores):.4f}, Max: {np.max(accuracy_scores):.4f}

    Sample Statistics:
    - Total samples: {len(f1_scores)}
    - Samples with F1 > 0.9: {sum(1 for f1 in f1_scores if f1 > 0.9)}
    - Samples with F1 < 0.5: {sum(1 for f1 in f1_scores if f1 < 0.5)}
    - Samples with Accuracy > 0.95: {sum(1 for acc in accuracy_scores if acc > 0.95)}
    - Samples with Accuracy < 0.8: {sum(1 for acc in accuracy_scores if acc < 0.8)}

    Detailed Statistics:
    F1 Score Percentiles:
    - 25th: {np.percentile(f1_scores, 25):.4f}
    - 50th (Median): {np.percentile(f1_scores, 50):.4f}
    - 75th: {np.percentile(f1_scores, 75):.4f}
    - 90th: {np.percentile(f1_scores, 90):.4f}
    - 95th: {np.percentile(f1_scores, 95):.4f}

    Accuracy Percentiles:
    - 25th: {np.percentile(accuracy_scores, 25):.4f}
    - 50th (Median): {np.percentile(accuracy_scores, 50):.4f}
    - 75th: {np.percentile(accuracy_scores, 75):.4f}
    - 90th: {np.percentile(accuracy_scores, 90):.4f}
    - 95th: {np.percentile(accuracy_scores, 95):.4f}
    ===============================================
    """
    
    # Print to console
    print(report)
    
    # Save to file if output directory is provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / f"{config.trainer_version}_metrics_summary.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✓ Metrics summary saved to: {report_file}")


def calculate_reconstruction_loss_breakdown(model, test_loader) -> dict:
    """
    Calculate detailed reconstruction loss breakdown.
    
    Args:
        model: Trained VAE model
        test_loader: DataLoader for test data
        
    Returns:
        Dictionary containing various loss metrics
    """
    model.eval()
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch_data = batch[0].to(device)
            recon_x, mu, logvar = model(batch_data)
            
            # Reconstruction loss
            recon_loss = torch.nn.functional.binary_cross_entropy(
                recon_x, batch_data, reduction='sum'
            )
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_samples += batch_data.shape[0]
    
    return {
        'avg_reconstruction_loss': total_recon_loss / total_samples,
        'avg_kl_divergence_loss': total_kl_loss / total_samples,
        'total_samples': total_samples
    }


# Export all functions
__all__ = [
    'calculate_reconstruction_metrics',
    'generate_metric_histograms', 
    'print_metric_summary',
    'calculate_reconstruction_loss_breakdown'
]