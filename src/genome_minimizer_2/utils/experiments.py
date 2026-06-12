#!/usr/bin/env python3
"""
Config for experiments
"""

# Import libraries
import os
import tempfile
import torch
import numpy as np
import logging
import wandb
from dataclasses import fields
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import HfApi

# Import modules
from src.genome_minimizer_2.utils.custom_config import ExperimentConfig
from src.genome_minimizer_2.training.training.trainer import (
    v0, v1, v2, v3, v4,
    create_v0_trainer, create_v1_trainer, create_v2_trainer, create_v3_trainer,
    create_v4_trainer,
)
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
from src.genome_minimizer_2.utils.extras import plot_loss_vs_epochs_graph, plot_loss_components
from src.genome_minimizer_2.utils.directories import (
    PROJECT_ROOT,
    ESSENTIAL_GENES_POSITIONS,
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


def get_v4_config() -> ExperimentConfig:
    """
    v4 model:
    Same as v3 (512 hidden, 32 latent, cosine annealing + weighted gene abundance + L1)
    plus an essential gene preservation loss that pushes all known essential
    gene outputs toward 1.  LR schedule relaxed so learning doesn't stall
    early (step_size=2000, gamma=0.5 → 5 halvings over 10k epochs).
    """
    return ExperimentConfig(
        hidden_dim=512,
        latent_dim=32,
        n_epochs=10000,
        min_beta=0.1,
        max_beta=1.0,
        gamma_start=2.0,
        gamma_end=0.1,
        weight=1.0,
        lambda_l1=0.01,
        trainer_version="v4",
        experiment_name="v4_model",
        scheduler_step_size=2000,
        scheduler_gamma=0.5,
    )


class IntegratedExperimentRunner:
    """Experiment runner"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config

        # Setup logging for this experiment
        self.logger = logging.getLogger(f"{__name__}.{config.experiment_name}")

        # Seed all RNGs from config so --random-state actually does something.
        torch.manual_seed(config.random_state)
        np.random.seed(config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_state)

        # Setup output directories
        self.figure_dir = os.path.join(PROJECT_ROOT, "models", config.experiment_name, "figures")
        self.model_dir = os.path.join(PROJECT_ROOT, "models", "trained_models", config.experiment_name)

        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger.info(f"Created directories: {self.figure_dir}, {self.model_dir}")

        # HF Hub — one branch per preset, or override via config.hf_branch (e.g. v4_opt for tuned variants).
        self.hf_branch = config.hf_branch or config.trainer_version
        if config.hf_upload:
            self.hf_api = HfApi()
            self.hf_api.create_repo(config.hf_repo_id, exist_ok=True)
            self.hf_api.create_branch(config.hf_repo_id, branch=self.hf_branch, exist_ok=True)
            self.logger.info(f"HF Hub uploads enabled → {config.hf_repo_id}@{self.hf_branch}")
        else:
            self.hf_api = None
            self.logger.info("HF Hub uploads disabled (config.hf_upload=False)")

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
            
            # Extract data arrays. .to_numpy() (not .values) so pyarrow-backed
            # columns convert cleanly — .values returns an arrow array sklearn can't index.
            data_array_t = merged_df.iloc[:, :-1].to_numpy()  # All columns except phylogroup
            phylogroups_array = merged_df['Phylogroup'].to_numpy()
            
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
        
        # Split data — uses config.random_state and config.test_size/val_ratio so seed sweeps work.
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data_tensor, labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels,
            test_size=self.config.val_ratio,
            random_state=self.config.random_state,
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
    
    def _save_checkpoint(self, model, optimizer, scheduler, epoch, path_in_repo):
        """Save a checkpoint locally and (if enabled) upload to HF Hub on the preset branch."""
        if not self.config.hf_upload:
            return
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            tmp_path = f.name
        self.hf_api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=path_in_repo,
            repo_id=self.config.hf_repo_id,
            revision=self.hf_branch,
        )
        os.unlink(tmp_path)
        self.logger.info(f"Checkpoint uploaded to {self.config.hf_repo_id}@{self.hf_branch}/{path_in_repo}")

    def _make_checkpoint_fn(self):
        """Create a checkpoint callback for the trainer."""
        if not self.config.hf_upload:
            return None
        checkpoint_every = self.config.checkpoint_every
        if checkpoint_every <= 0:
            return None
        version = self.config.trainer_version

        def checkpoint_fn(model, optimizer, scheduler, epoch):
            if epoch % checkpoint_every == 0:
                path_in_repo = f"checkpoint-epoch-{epoch}.pt"
                self._save_checkpoint(model, optimizer, scheduler, epoch, path_in_repo)

        return checkpoint_fn

    def _load_essential_gene_indices(self) -> list[int]:
        """Load flattened essential gene column indices from the preprocessed pickle."""
        import pickle
        # Check both possible paths (directories.py constant and actual preprocess output)
        candidates = [
            ESSENTIAL_GENES_POSITIONS,
            os.path.join(PROJECT_ROOT, "data", "essential_genes", "essential_gene_positions.pkl"),
        ]
        pkl_path = None
        for p in candidates:
            if os.path.exists(p):
                pkl_path = p
                break
        if pkl_path is None:
            raise FileNotFoundError(
                f"Essential gene positions not found. Checked: {candidates}. "
                "Run --mode preprocess first."
            )
        with open(pkl_path, "rb") as f:
            positions_dict = pickle.load(f)
        # Flatten {gene_name: [idx, ...]} → sorted unique list of ints
        indices = sorted({idx for idxs in positions_dict.values() for idx in idxs})
        self.logger.info(f"Loaded {len(indices)} essential gene indices from {pkl_path}")
        return indices

    def train_model(self):
        """Train using v0-v4 trainer configs with HF Hub checkpointing."""
        self.logger.info(f"Starting training with {self.config.trainer_version} configuration...")
        self.logger.info(f"Training for {self.config.n_epochs} epochs")

        cfg = self.config
        folder = self.figure_dir + "/"

        try:
            # Create trainer
            if cfg.trainer_version == "v0":
                trainer = create_v0_trainer(
                    self.model, self.optimizer, self.scheduler,
                    cfg.n_epochs, cfg.max_norm, cfg.min_beta, cfg.max_beta,
                )
            elif cfg.trainer_version == "v1":
                trainer = create_v1_trainer(
                    self.model, self.optimizer, self.scheduler,
                    cfg.n_epochs, cfg.max_norm, cfg.lambda_l1,
                    cfg.min_beta, cfg.max_beta, cfg.gamma_start, cfg.gamma_end,
                )
            elif cfg.trainer_version == "v2":
                trainer = create_v2_trainer(
                    self.model, self.optimizer, self.scheduler,
                    cfg.n_epochs, cfg.max_norm, cfg.lambda_l1,
                    cfg.min_beta, cfg.max_beta, cfg.gamma_start, cfg.gamma_end,
                )
            elif cfg.trainer_version == "v3":
                trainer = create_v3_trainer(
                    self.model, self.optimizer, self.scheduler,
                    cfg.n_epochs, cfg.max_norm, cfg.lambda_l1,
                    cfg.min_beta, cfg.max_beta, cfg.gamma_start, cfg.gamma_end, cfg.weight,
                )
            elif cfg.trainer_version == "v4":
                essential_indices = self._load_essential_gene_indices()
                trainer = create_v4_trainer(
                    self.model, self.optimizer, self.scheduler,
                    cfg.n_epochs, cfg.max_norm, cfg.lambda_l1,
                    essential_indices,
                    cfg.min_beta, cfg.max_beta, cfg.gamma_start, cfg.gamma_end, cfg.weight,
                    cfg.essential_weight,
                )
            else:
                raise ValueError(f"Unknown trainer version: {cfg.trainer_version}")

            # Set checkpoint callback
            trainer.checkpoint_fn = self._make_checkpoint_fn()

            # Train
            train_loss_vals, val_loss_vals, epochs = trainer.train(
                self.train_loader, self.val_loader, folder
            )

            self.results['train_loss_vals'] = train_loss_vals
            self.results['val_loss_vals'] = val_loss_vals
            self.results['epochs_trained'] = epochs
            # Per-component loss histories (reconstruction, KL, etc.) for the
            # component-breakdown plot. The tracker keeps them; train() only
            # returns the totals.
            self.results['train_loss_components'] = dict(trainer.loss_tracker.train_losses)
            self.results['val_loss_components'] = dict(trainer.loss_tracker.val_losses)

            self.logger.info(f"Training completed after {epochs} epochs")
            self.logger.info(f"Final train loss: {train_loss_vals[-1]:.4f}")
            self.logger.info(f"Final validation loss: {val_loss_vals[-1]:.4f}")

            # Save final model locally and to HF Hub
            if cfg.save_model:
                model_path = os.path.join(self.model_dir, f"saved_VAE_{cfg.trainer_version}.pt")
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model saved to {model_path}")

                path_in_repo = "final.pt"
                self._save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epochs, path_in_repo
                )

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

            # Publication-quality breakdown into loss components (total,
            # reconstruction, KL, ...). Both PDF (for the paper) and PNG.
            if 'train_loss_components' in self.results:
                base = os.path.join(self.figure_dir, f"{self.config.trainer_version}_loss_components")
                for ext in ("pdf", "png"):
                    plot_loss_components(
                        self.results['train_loss_components'],
                        self.results['val_loss_components'],
                        fig_name=f"{base}.{ext}",
                        title=f"VAE training — preset {self.config.trainer_version}",
                    )
                self.logger.info(f"Loss-component plot saved to {base}.pdf / .png")
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

            # Same per-sample metrics on the train set, for the train-vs-test
            # distribution panels in the summary figure.
            _, _, f1_train, accuracy_train = calculate_reconstruction_metrics(
                self.model, self.train_loader
            )
            self.results['f1_scores_per_sample_train'] = f1_train
            self.results['accuracy_scores_per_sample_train'] = accuracy_train
            
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
                self.config.experiment_name,
                f1_train=self.results.get('f1_scores_per_sample_train'),
                accuracy_train=self.results.get('accuracy_scores_per_sample_train'),
            )
            self.logger.info("Summary plot generated")
        except Exception as e:
            self.logger.error(f"Error generating summary plot: {e}")

    def save_numeric_data(self):
        """Persist the numbers behind the figures as CSVs — source data for the
        loss-component and summary plots, so they can be re-plotted or analysed
        without retraining."""
        import csv
        ver = self.config.trainer_version

        # Per-epoch loss components (train + val): one row per epoch.
        tr = self.results.get('train_loss_components')
        va = self.results.get('val_loss_components') or {}
        if tr:
            comps = ['total'] + [c for c in tr if c != 'total']
            n_epochs = len(tr.get('total', []))
            path = os.path.join(self.figure_dir, f"{ver}_loss_history.csv")
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['epoch'] + [f'train_{c}' for c in comps] + [f'val_{c}' for c in comps])
                for i in range(n_epochs):
                    row = [i + 1]
                    row += [tr[c][i] for c in comps]
                    row += [va[c][i] if c in va and i < len(va[c]) else '' for c in comps]
                    w.writerow(row)
            self.logger.info(f"Loss history saved to {path}")

        # Per-sample reconstruction metrics (train + test): one row per sample.
        def _rows(split, f1_key, acc_key):
            f1 = self.results.get(f1_key)
            acc = self.results.get(acc_key)
            if f1 is None or acc is None:
                return []
            return [(split, float(a), float(b)) for a, b in zip(f1, acc)]

        rows = (_rows('train', 'f1_scores_per_sample_train', 'accuracy_scores_per_sample_train')
                + _rows('test', 'f1_scores_per_sample', 'accuracy_scores_per_sample'))
        if rows:
            path = os.path.join(self.figure_dir, f"{ver}_reconstruction_metrics.csv")
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['split', 'f1', 'accuracy'])
                w.writerows(rows)
            self.logger.info(f"Per-sample metrics saved to {path}")

    def _upload_model_card(self):
        """Generate and upload a model card (README.md) to this run's HF branch.

        v0–v4 are distinguished by *loss function*. Branches whose name does NOT match
        the trainer_version (e.g. ``v4_opt``) are hyperparameter-tuned variants of the
        loss-equivalent baseline branch. The card spells out which deltas matter.
        """
        if not self.config.hf_upload:
            self.logger.info("Skipping model card upload (hf_upload=False)")
            return

        cfg = self.config
        v = cfg.trainer_version
        branch = self.hf_branch
        is_tuned_variant = branch != v

        loss_descriptions = {
            "v0": "Reconstruction + KL divergence (linear annealing)",
            "v1": "Reconstruction + KL divergence (linear) + Gene abundance + L1 regularization",
            "v2": "Reconstruction + KL divergence (cosine) + Gene abundance + L1 regularization",
            "v3": "Reconstruction + KL divergence (cosine) + Weighted gene abundance + L1 regularization",
            "v4": "Reconstruction + KL divergence (cosine) + Weighted gene abundance + Essential gene preservation + L1 regularization",
        }

        # Default hyperparameters per loss-equivalent baseline (used to surface deltas).
        baseline_hparams = {
            "v0": {"hidden_dim": 1024, "latent_dim": 64,  "learning_rate": 1e-3, "batch_size": 32, "lambda_l1": 0.0,  "gamma_start": 1.0, "essential_weight": 0.0, "scheduler_step_size": 20,   "scheduler_gamma": 0.5, "random_state": 12345},
            "v1": {"hidden_dim": 512,  "latent_dim": 32,  "learning_rate": 1e-3, "batch_size": 32, "lambda_l1": 0.01, "gamma_start": 1.0, "essential_weight": 0.0, "scheduler_step_size": 20,   "scheduler_gamma": 0.5, "random_state": 12345},
            "v2": {"hidden_dim": 512,  "latent_dim": 32,  "learning_rate": 1e-3, "batch_size": 32, "lambda_l1": 0.01, "gamma_start": 1.0, "essential_weight": 0.0, "scheduler_step_size": 20,   "scheduler_gamma": 0.5, "random_state": 12345},
            "v3": {"hidden_dim": 512,  "latent_dim": 32,  "learning_rate": 1e-3, "batch_size": 32, "lambda_l1": 0.01, "gamma_start": 2.0, "essential_weight": 0.0, "scheduler_step_size": 20,   "scheduler_gamma": 0.5, "random_state": 12345},
            "v4": {"hidden_dim": 512,  "latent_dim": 32,  "learning_rate": 1e-3, "batch_size": 32, "lambda_l1": 0.01, "gamma_start": 2.0, "essential_weight": 1.0, "scheduler_step_size": 2000, "scheduler_gamma": 0.5, "random_state": 12345},
        }

        arch = "1024 → 64" if cfg.hidden_dim == 1024 else f"{cfg.hidden_dim} → {cfg.latent_dim}"
        params = sum(p.numel() for p in self.model.parameters())

        f1 = self.results.get("f1_overall", "N/A")
        acc = self.results.get("accuracy_overall", "N/A")
        epochs_trained = self.results.get("epochs_trained", "N/A")
        if isinstance(f1, float):
            f1 = f"{f1:.4f}"
            acc = f"{acc:.4f}"

        wandb_url = ""
        try:
            if wandb.run is not None:
                wandb_url = wandb.run.get_url() or ""
        except Exception:
            wandb_url = ""

        # Build the "differences from baseline" block for tuned variants.
        deltas_block = ""
        if is_tuned_variant:
            base = baseline_hparams.get(v, {})
            current = {k: getattr(cfg, k) for k in base.keys()}
            diffs = [(k, base[k], current[k]) for k in base if base[k] != current[k]]
            if diffs:
                rows = "\n".join(f"| `{k}` | {b} | {c} |" for k, b, c in diffs)
            else:
                rows = "| _no deltas detected_ | — | — |"
            deltas_block = f"""
## Differences from baseline (`{v}`)

This branch (**`{branch}`**) is a **hyperparameter-tuned variant** of `{v}`. The loss
function is identical to `{v}`. Only the hyperparameters below differ:

| Hyperparameter | `{v}` baseline | `{branch}` |
|---|---|---|
{rows}

For loss-function deltas between numbered versions (v0 → v1 → … → v4), see the
respective baseline branches.
"""

        header_title = f"Genome Minimizer 2 — {branch}"
        intro = (
            f"Hyperparameter-tuned variant of the **{v}** loss configuration. "
            f"Same loss as `{v}`; differs only in hyperparameters (see below)."
            if is_tuned_variant else
            f"VAE model for generating minimal *E. coli* genomes, trained with the **{v}** loss configuration."
        )

        card = f"""---
library_name: pytorch
tags:
  - vae
  - genomics
  - genome-minimization
  - e-coli
---

# {header_title}

{intro}

## Model Details

| | |
|---|---|
| **Branch** | `{branch}` |
| **Loss configuration** | `{v}` ({loss_descriptions[v]}) |
| **Architecture** | VAE: 55,039 → {arch} |
| **Parameters** | {params:,} |
| **Epochs trained** | {epochs_trained} |
| **Test F1 (overall)** | {f1} |
| **Test Accuracy (overall)** | {acc} |
{deltas_block}
## Training Configuration

| Parameter | Value |
|---|---|
| Hidden dim | {cfg.hidden_dim} |
| Latent dim | {cfg.latent_dim} |
| Learning rate | {cfg.learning_rate} |
| Batch size | {cfg.batch_size} |
| Beta range | {cfg.min_beta} → {cfg.max_beta} |
| Gamma range | {cfg.gamma_start} → {cfg.gamma_end} |
| L1 lambda | {cfg.lambda_l1} |
| Gene-abundance weight (v3+) | {cfg.weight} |
| Essential-gene weight (v4+) | {cfg.essential_weight} |
| Scheduler step / gamma | {cfg.scheduler_step_size} / {cfg.scheduler_gamma} |
| Random state | {cfg.random_state} |
| Checkpoint every | {cfg.checkpoint_every} epochs |

## Files

- `checkpoint-epoch-N.pt` — periodic checkpoints (model, optimizer, scheduler state)
- `final.pt` — final checkpoint after training

## Usage

```python
from huggingface_hub import hf_hub_download
import torch
from src.genome_minimizer_2.training.model import VAE

path = hf_hub_download("{cfg.hf_repo_id}", "final.pt", revision="{branch}")
checkpoint = torch.load(path, map_location="cpu")

model = VAE(input_dim=55039, hidden_dim={cfg.hidden_dim}, latent_dim={cfg.latent_dim})
model.load_state_dict(checkpoint["model_state_dict"])
```

## Links

- [W&B project](https://wandb.ai/mcclain/genome-minimizer-2){f" — [this run]({wandb_url})" if wandb_url else ""}
- [GitHub repository](https://github.com/ucl-cssb/genome-minimizer-2)
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(card)
            tmp_path = f.name
        self.hf_api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=cfg.hf_repo_id,
            revision=self.hf_branch,
        )
        os.unlink(tmp_path)
        self.logger.info(f"Model card uploaded to {cfg.hf_repo_id}@{self.hf_branch}/README.md")

    def run_complete_experiment(self):
        """Run the complete experiment pipeline"""
        self.logger.info(f"** START OF EXPERIMENT: {self.config.experiment_name} **")

        # Init wandb — only when explicitly enabled, so a fresh checkout trains
        # without a W&B login. Everything below guards on wandb.run, so a skipped
        # init just means no logging.
        if self.config.wandb_log:
            config_dict = {f.name: getattr(self.config, f.name) for f in fields(self.config)}
            wandb.init(
                project="genome-minimizer-2",
                name=self.config.experiment_name,
                config=config_dict,
            )
        # Make sweep runs distinguishable in the W&B UI: <experiment_name>-<run_id[:6]>
        if wandb.run is not None and getattr(wandb.run, "id", None):
            wandb.run.name = f"{self.config.experiment_name}-{wandb.run.id[:6]}"
            existing_tags = list(wandb.run.tags or ())
            for tag in (self.config.trainer_version, self.hf_branch):
                if tag and tag not in existing_tags:
                    existing_tags.append(tag)
            wandb.run.tags = tuple(existing_tags)

        try:
            self.prep_data()
            self.setup_model_and_training()
            self.display_config()
            self.train_model()
            self.generate_comparison_plots()
            self.calculate_metrics()
            self.explore_latent_space()
            self.generate_summary_plot()
            self.save_numeric_data()

            # Log final metrics to wandb
            if wandb.run is not None and 'f1_overall' in self.results:
                wandb.log({
                    "test/f1_overall": self.results['f1_overall'],
                    "test/accuracy_overall": self.results['accuracy_overall'],
                })

            # Upload model card to HF branch
            self._upload_model_card()

            self.logger.info(f"** EXPERIMENT {self.config.experiment_name} COMPLETED SUCCESSFULLY **")

        except Exception as e:
            self.logger.error(f"** EXPERIMENT {self.config.experiment_name} FAILED: {e} **")
            raise
        finally:
            if wandb.run is not None:
                wandb.finish()

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