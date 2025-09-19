#!/usr/bin/env python3
"""
Config for experiments
"""

# Import libraries
import os
import torch
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Import modules
from src.genome_minimizer_2.utils.custom_config import ExperimentConfig
from src.genome_minimizer_2.training.training.trainer import v0, v1, v2, v3
from src.genome_minimizer_2.training.evaluation.metrics import (
    calculate_reconstruction_metrics, 
    generate_metric_histograms, 
    print_metric_summary
)
from src.genome_minimizer_2.training.model import VAE
from src.genome_minimizer_2.training.evaluation.visualise import (
    plot_latent_space_pca,
    create_training_summary_plot
)
from src.genome_minimizer_2.utils.extras import plot_loss_vs_epochs_graph
from src.genome_minimizer_2.utils.directories import (
    PROJECT_ROOT
)

# Import data loading function from data exploration
from src.genome_minimizer_2.explore_data.data_exploration import load_and_validate_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_v0_config() -> ExperimentConfig:
    """
    v0 model: 
    1024 hidden, 64 latent, linear KL annealing
    """
    return ExperimentConfig(
        hidden_dim=1024,
        latent_dim=64,
        n_epochs=10000,
        min_beta=0.1,
        max_beta=1.0,
        lambda_l1=0.0,  # No L1 for v0
        trainer_version="v0",
        experiment_name="v0_model"
    )


def get_v1_config() -> ExperimentConfig:
    """
    v1 model: 
    512 hidden, 32 latent, linear annealing + gene abundance + L1
    """
    return ExperimentConfig(
        hidden_dim=512,
        latent_dim=32,
        n_epochs=10000,
        min_beta=0.1,
        max_beta=1.0,
        gamma_start=1.0,
        gamma_end=0.1,
        lambda_l1=0.01,
        trainer_version="v1",
        experiment_name="v1_model"
    )


def get_v2_config() -> ExperimentConfig:
    """
    v2 model: 
    512 hidden, 32 latent, cosine annealing + gene abundance + L1
    """
    return ExperimentConfig(
        hidden_dim=512,
        latent_dim=32,
        n_epochs=10000,
        min_beta=0.0,
        max_beta=1.0,
        gamma_start=1.0,
        gamma_end=0.1,
        lambda_l1=0.01,
        trainer_version="v2",
        experiment_name="v2_model"
    )


def get_v3_config() -> ExperimentConfig:
    """
    v3 model: 
    512 hidden, 32 latent, cosine annealing + weighted gene abundance + L1
    """
    return ExperimentConfig(
        hidden_dim=512,
        latent_dim=32,
        n_epochs=10000,
        min_beta=0.1,
        max_beta=1.0,
        gamma_start=2.0,  # Increased gamma_start
        gamma_end=0.1,
        weight=1.0,
        lambda_l1=0.01,
        trainer_version="v3",
        experiment_name="v3_model"
    )


class IntegratedExperimentRunner:
    """Experiment runner"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Setup logging for this experiment
        self.logger = logging.getLogger(f"{__name__}.{config.experiment_name}")
        
        # Setup output directories
        self.figure_dir = os.path.join(PROJECT_ROOT, "models", config.experiment_name, "figures")
        self.model_dir = os.path.join(PROJECT_ROOT, "models", "trained_models", config.experiment_name)
        
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.logger.info(f"Created directories: {self.figure_dir}, {self.model_dir}")

        # Data storage
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.test_phylogroups = None
        self.model = None
        self.input_dim = None
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Using device: {self.device}")

    def display_config(self):
        """Display all configuration parameters in a formatted way and save to file"""
        from pathlib import Path
        from datetime import datetime
        
        # Create the configuration report
        config_lines = []
        config_lines.append("="*80)
        config_lines.append("EXPERIMENT CONFIGURATION")
        config_lines.append("="*80)
        config_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        config_lines.append("")
        
        # Group parameters by category
        categories = {
            "Model Parameters": ["hidden_dim", "latent_dim"],
            "Training Parameters": ["n_epochs", "batch_size", "learning_rate", "max_norm", "lambda_l1"],
            "Loss Scheduling": ["min_beta", "max_beta", "gamma_start", "gamma_end", "weight"],
            "Trainer": ["trainer_version"],
            "Scheduler": ["scheduler_step_size", "scheduler_gamma"],
            "Data Split": ["test_size", "val_ratio", "random_state"],
            "Output": ["experiment_name", "save_model", "generate_plots", "calculate_metrics", "explore_latent_space"]
        }
        
        for category, params in categories.items():
            config_lines.append(f"{category}:")
            config_lines.append("-" * len(category))
            for param in params:
                if hasattr(self.config, param):
                    value = getattr(self.config, param)
                    config_lines.append(f"  {param:<20}: {value}")
            config_lines.append("")
        
        config_lines.append("="*80)
        
        config_text = "\n".join(config_lines)
        
        # Print to console
        self.logger.info("Displaying experiment configuration:")
        print(config_text)
        
        # Save to file (always save since we have figure_dir)
        config_file = Path(self.figure_dir) / f"{self.config.experiment_name}_config.txt"
        with open(config_file, 'w') as f:
            f.write(config_text)
        
        self.logger.info(f"Configuration saved to: {config_file}")
    
    def prep_data(self):
        """Load and preprocess the dataset using the shared data loading function"""
        self.logger.info("Loading the dataset...")
        
        try:
            # Use the shared data loading function from data exploration
            _, merged_df, _ = load_and_validate_data()
            
            self.logger.info(f"Dataset loaded successfully: {merged_df.shape}")
            self.logger.info("Phylogroup distribution:")
            phylogroup_counts = merged_df['Phylogroup'].value_counts()
            for phylogroup, count in phylogroup_counts.items():
                self.logger.info(f"  {phylogroup}: {count}")
            
            # Extract data arrays
            data_array_t = merged_df.iloc[:, :-1].values  # All columns except phylogroup
            phylogroups_array = merged_df['Phylogroup'].values
            
            self.logger.info(f"Data array shape: {data_array_t.shape}")
            self.logger.info(f"Phylogroups array shape: {phylogroups_array.shape}")
            
            self.input_dim = data_array_t.shape[1]
            self.logger.info(f"Input dimension: {self.input_dim}")

            self.create_dataloaders(data_array_t, phylogroups_array, self.config.batch_size)
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def create_dataloaders(self, data_array, labels, batch_size):
        """Create DataLoaders for train, validation, and test splits."""
        self.logger.info("Creating data loaders...")
        
        data_tensor = torch.tensor(data_array, dtype=torch.float32)
        
        # Split data
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data_tensor, labels, test_size=0.3, random_state=12345
        )
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels, test_size=0.3333, random_state=12345
        )
        
        self.logger.info(f"Data splits - Train: {train_data.shape[0]}, Val: {val_data.shape[0]}, Test: {test_data.shape[0]}")
        
        # Create datasets
        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)
        test_dataset = TensorDataset(test_data)

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.test_phylogroups = test_labels
        
        self.logger.info(f"Created data loaders with batch size: {batch_size}")
    
    def setup_model_and_training(self):
        """Setup model, optimizer, and scheduler"""
        self.logger.info("Setting up model and training components...")
        self.logger.info(f"Model architecture: {self.input_dim} -> {self.config.hidden_dim} -> {self.config.latent_dim}")
        
        self.model = VAE(self.input_dim, self.config.hidden_dim, self.config.latent_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config.scheduler_step_size, 
            gamma=self.config.scheduler_gamma
        )
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    def train_model(self):
        """Train using original v0-v3 functions configs"""
        self.logger.info(f"Starting training with {self.config.trainer_version} configuration...")
        self.logger.info(f"Training for {self.config.n_epochs} epochs")
        
        folder = self.figure_dir + "/"  # Add trailing slash for compatibility
        
        try:
            if self.config.trainer_version == "v0":
                train_loss_vals, val_loss_vals, epochs = v0(
                    self.model, folder, self.optimizer, self.scheduler, 
                    self.config.n_epochs, self.train_loader, self.val_loader,
                    self.config.min_beta, self.config.max_beta, self.config.max_norm
                )
            elif self.config.trainer_version == "v1":
                train_loss_vals, val_loss_vals, epochs = v1(
                    self.model, folder, self.optimizer, self.scheduler,
                    self.config.n_epochs, self.train_loader, self.val_loader,
                    self.config.min_beta, self.config.max_beta,
                    self.config.gamma_start, self.config.gamma_end,
                    self.config.max_norm, self.config.lambda_l1
                )
            elif self.config.trainer_version == "v2":
                train_loss_vals, val_loss_vals, epochs = v2(
                    self.model, folder, self.optimizer, self.scheduler,
                    self.config.n_epochs, self.train_loader, self.val_loader,
                    self.config.min_beta, self.config.max_beta,
                    self.config.gamma_start, self.config.gamma_end,
                    self.config.max_norm, self.config.lambda_l1
                )
            elif self.config.trainer_version == "v3":
                train_loss_vals, val_loss_vals, epochs = v3(
                    self.model, folder, self.optimizer, self.scheduler,
                    self.config.n_epochs, self.train_loader, self.val_loader,
                    self.config.min_beta, self.config.max_beta,
                    self.config.gamma_start, self.config.gamma_end,
                    self.config.weight, self.config.max_norm, self.config.lambda_l1
                )
            else:
                raise ValueError(f"Unknown trainer version: {self.config.trainer_version}")
            
            self.results['train_loss_vals'] = train_loss_vals
            self.results['val_loss_vals'] = val_loss_vals
            self.results['epochs_trained'] = epochs
            
            self.logger.info(f"Training completed after {epochs} epochs")
            self.logger.info(f"Final train loss: {train_loss_vals[-1]:.4f}")
            self.logger.info(f"Final validation loss: {val_loss_vals[-1]:.4f}")
            
            # Save model
            if self.config.save_model:
                model_path = os.path.join(self.model_dir, f"saved_VAE_{self.config.trainer_version}.pt")
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model saved to {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def generate_comparison_plots(self):
        """Generate training vs validation loss plots"""
        if not self.config.generate_plots:
            self.logger.info("Skipping plot generation (disabled in config)")
            return
            
        self.logger.info("Generating comparison graphs...")
        try:
            epochs = np.linspace(1, self.results['epochs_trained'], num=self.results['epochs_trained'])
            name = os.path.join(self.figure_dir, f"{self.config.trainer_version}_train_val_loss.pdf")
            plot_loss_vs_epochs_graph(
                epochs=epochs, 
                train_loss_vals=self.results['train_loss_vals'], 
                val_loss_vals=self.results['val_loss_vals'], 
                fig_name=name
            )
            self.logger.info(f"Loss comparison plot saved to {name}")
        except Exception as e:
            self.logger.error(f"Error generating comparison plots: {e}")
    
    def calculate_metrics(self):
        """Calculate F1 scores and accuracy"""
        if not self.config.calculate_metrics:
            self.logger.info("Skipping metrics calculation (disabled in config)")
            return
            
        self.logger.info("Calculating F1 scores and accuracy...")
        
        try:
            overall_f1, overall_accuracy, f1_scores, accuracy_scores = calculate_reconstruction_metrics(
                self.model, self.test_loader
            )
            
            self.results['f1_overall'] = overall_f1
            self.results['accuracy_overall'] = overall_accuracy
            self.results['f1_scores_per_sample'] = f1_scores
            self.results['accuracy_scores_per_sample'] = accuracy_scores
            
            self.logger.info(f"Overall F1 Score: {overall_f1:.4f}")
            self.logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
            self.logger.info(f"F1 Score range: {np.min(f1_scores):.4f} - {np.max(f1_scores):.4f}")
            self.logger.info(f"Accuracy range: {np.min(accuracy_scores):.4f} - {np.max(accuracy_scores):.4f}")
            
            print_metric_summary(self.config, overall_f1, overall_accuracy, f1_scores, accuracy_scores, self.figure_dir)
            
            if self.config.generate_plots:
                generate_metric_histograms(f1_scores, accuracy_scores, self.config, self.figure_dir)
                self.logger.info("Metric histograms generated")
                
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
    
    def explore_latent_space(self):
        """Explore latent space with PCA"""
        if not self.config.explore_latent_space:
            self.logger.info("Skipping latent space exploration (disabled in config)")
            return
            
        self.logger.info("Exploring latent space...")
        
        try:
            # PCA visualization
            df_pca = plot_latent_space_pca(
                self.model, self.test_loader, self.config, self.test_phylogroups, 
                self.figure_dir, show_plot=self.config.generate_plots
            )
            
            self.results['pca_data'] = df_pca
            self.logger.info("Latent space PCA analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error exploring latent space: {e}")
    
    def generate_summary_plot(self):
        """Generate comprehensive summary plot"""
        if not self.config.generate_plots or 'f1_scores_per_sample' not in self.results:
            self.logger.info("Skipping summary plot generation")
            return
            
        self.logger.info("Generating summary plot...")
        try:
            create_training_summary_plot(
                self.results['train_loss_vals'],
                self.results['val_loss_vals'],
                self.results['f1_scores_per_sample'],
                self.results['accuracy_scores_per_sample'],
                self.figure_dir,
                self.config.experiment_name
            )
            self.logger.info("Summary plot generated")
        except Exception as e:
            self.logger.error(f"Error generating summary plot: {e}")
    
    def run_complete_experiment(self):
        """Run the complete experiment pipeline"""
        self.logger.info(f"** START OF EXPERIMENT: {self.config.experiment_name} **")
        
        try:
            self.prep_data()
            self.setup_model_and_training()
            self.display_config()
            self.train_model()
            self.generate_comparison_plots()
            self.calculate_metrics()
            self.explore_latent_space()
            self.generate_summary_plot()
            
            self.logger.info(f"** EXPERIMENT {self.config.experiment_name} COMPLETED SUCCESSFULLY **")
            
        except Exception as e:
            self.logger.error(f"** EXPERIMENT {self.config.experiment_name} FAILED: {e} **")
            raise
            
        return self.results
    
    def _get_predictions_from_output(self, reconstruction):
        """Extract predictions from model output"""
        if reconstruction.shape[-1] > 1:  # Multi-dimensional output
            predictions = (reconstruction > 0.5).float()
            return predictions.argmax(dim=-1) if predictions.dim() > 1 else predictions
        else:
            return (reconstruction > 0.5).float().squeeze()

    def _calculate_reconstruction_loss(self, reconstruction, target):
        """Calculate reconstruction loss"""
        if hasattr(self, 'criterion'):
            return self.criterion(reconstruction, target)
        else:
            if target.dtype == torch.float32:
                return torch.nn.functional.mse_loss(reconstruction, target, reduction='mean')
            else:
                return torch.nn.functional.binary_cross_entropy_with_logits(reconstruction, target.float(), reduction='mean')

    def _calculate_kl_loss(self, mu, logvar):
        """Calculate KL divergence loss for VAE"""
        if mu is None or logvar is None:
            return 0.0
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss / mu.size(0)  # Average over batch

    def create_model(self):
        """Create the VAE model architecture"""
        try:
            from src.genome_minimizer_2.training.model import VAE
            
            if not hasattr(self, 'input_dim'):
                self.prep_data()
            
            self.model = VAE(
                input_dim=self.input_dim,
                hidden_dim=self.config.hidden_dim,
                latent_dim=self.config.latent_dim
            ).to(self.device)
            
            self.logger.info(f"Created VAE model: {self.input_dim} -> {self.config.hidden_dim} -> {self.config.latent_dim}")
            
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            raise