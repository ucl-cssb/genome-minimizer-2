#!/usr/bin/env python3

"""
Training components
"""

# Import libraries
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Import the loss components 
from .loss_components import (
    LossComponent, ReconstructionLoss, KLDivergenceLoss, 
    GeneAbundanceLoss, L1RegularizationLoss
)

# Set deveice
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    n_epochs: int
    max_norm: float
    lambda_l1: float = 0.0
    patience: int = 10
    min_delta: float = 1e-4
    print_every: int = 100


class LossTracker:
    """Tracks and manages loss components during training"""
    
    def __init__(self, loss_components: List[LossComponent]):
        self.loss_components = loss_components
        self.train_losses = {comp.get_name(): [] for comp in loss_components}
        self.val_losses = {comp.get_name(): [] for comp in loss_components}
        self.train_losses['total'] = []
        self.val_losses['total'] = []
    
    def compute_total_loss(self, recon_x, data, mu, logvar, model, epoch, batch_idx, 
                          is_training=True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss and individual components"""
        individual_losses = {}
        total_loss = torch.tensor(0.0, device=device)
        
        for component in self.loss_components:
            loss = component.compute_loss(recon_x, data, mu, logvar, model, epoch, batch_idx)
            individual_losses[component.get_name()] = loss.item()
            total_loss += loss
        
        individual_losses['total'] = total_loss.item()
        return total_loss, individual_losses
    
    def update_epoch_losses(self, epoch_losses: Dict[str, float], is_training=True):
        """Update loss tracking for the epoch"""
        losses_dict = self.train_losses if is_training else self.val_losses
        for name, loss_val in epoch_losses.items():
            losses_dict[name].append(loss_val)


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
    
    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            return self.epochs_no_improve >= self.patience


class VAETrainer:
    """Main training class that uses loss components"""
    
    def __init__(self, model: nn.Module, optimizer, scheduler, config: TrainingConfig):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.loss_tracker = None
        self.early_stopping = EarlyStopping(config.patience, config.min_delta)
    
    def setup_loss_components(self, loss_components: List[LossComponent]):
        """Setup loss components for training"""
        self.loss_tracker = LossTracker(loss_components)
        
        # Update n_epochs in components that need it
        for component in loss_components:
            if hasattr(component, 'n_epochs'):
                component.n_epochs = self.config.n_epochs
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {name: 0.0 for name in self.loss_tracker.train_losses.keys()}
        
        for batch_idx, batch in enumerate(train_loader):
            data = batch[0].to(torch.float).to(device)
            self.optimizer.zero_grad()
            
            recon_x, mu, logvar = self.model(data)
            total_loss, individual_losses = self.loss_tracker.compute_total_loss(
                recon_x, data, mu, logvar, self.model, epoch, batch_idx, is_training=True
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm)
            self.optimizer.step()
            
            # Accumulate losses
            for name, loss_val in individual_losses.items():
                epoch_losses[name] += loss_val
        
        # Average losses
        dataset_size = len(train_loader.dataset)
        for name in epoch_losses:
            epoch_losses[name] /= dataset_size
            
        return epoch_losses
    
    def validate_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {name: 0.0 for name in self.loss_tracker.val_losses.keys()}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = self.model(data)
                
                total_loss, individual_losses = self.loss_tracker.compute_total_loss(
                    recon_x, data, mu, logvar, self.model, epoch, batch_idx, is_training=False
                )
                
                # Accumulate losses
                for name, loss_val in individual_losses.items():
                    epoch_losses[name] += loss_val
        
        # Average losses
        dataset_size = len(val_loader.dataset)
        for name in epoch_losses:
            epoch_losses[name] /= dataset_size
            
        return epoch_losses
    
    def train(self, train_loader, val_loader, folder: str = "./") -> Tuple[List[float], List[float], int]:
        """Main training loop"""
        if self.loss_tracker is None:
            raise ValueError("Loss components not set up. Call setup_loss_components first.")
        
        for epoch in range(self.config.n_epochs):
            # Training
            train_losses = self.train_epoch(train_loader, epoch)
            self.loss_tracker.update_epoch_losses(train_losses, is_training=True)
            
            # Validation
            val_losses = self.validate_epoch(val_loader, epoch)
            self.loss_tracker.update_epoch_losses(val_losses, is_training=False)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print progress
            if (epoch + 1) % self.config.print_every == 0:
                print(f"Epoch {epoch + 1}:")
                print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]}")
                print(f"  Train Loss: {train_losses['total']}")
                print(f"  Validation Loss: {val_losses['total']}")
            
            # Early stopping
            if self.early_stopping.should_stop(val_losses['total']):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        return (self.loss_tracker.train_losses['total'], 
                self.loss_tracker.val_losses['total'], 
                epoch + 1)


# FACTORY FUNCTIONS THAT USE CONFIG PRE-DEFINED PARAMS AND LOSS COMPONENTS
def create_v0_trainer(model, optimizer, scheduler, n_epochs, max_norm, 
                     beta_start, beta_end):
    """Create trainer equivalent to v0 function using loss components"""
    config = TrainingConfig(n_epochs=n_epochs, max_norm=max_norm, lambda_l1=0.0)
    trainer = VAETrainer(model, optimizer, scheduler, config)
    
    loss_components = [
        ReconstructionLoss(),
        KLDivergenceLoss(scheduler_type="linear", min_beta=beta_start, max_beta=beta_end)
    ]
    
    trainer.setup_loss_components(loss_components)
    return trainer


def create_v1_trainer(model, optimizer, scheduler, n_epochs, max_norm, lambda_l1,
                     beta_start=0.1, beta_end=1.0, gamma_start=1.0, gamma_end=0.1):
    """Create trainer equivalent to v1 function using loss components"""
    config = TrainingConfig(n_epochs=n_epochs, max_norm=max_norm, lambda_l1=lambda_l1)
    trainer = VAETrainer(model, optimizer, scheduler, config)
    
    loss_components = [
        ReconstructionLoss(),
        KLDivergenceLoss(scheduler_type="linear", min_beta=beta_start, max_beta=beta_end),
        GeneAbundanceLoss(gamma_start=gamma_start, gamma_end=gamma_end),
        L1RegularizationLoss(lambda_l1=lambda_l1)
    ]
    
    trainer.setup_loss_components(loss_components)
    return trainer


def create_v2_trainer(model, optimizer, scheduler, n_epochs, max_norm, lambda_l1,
                     min_beta=0.0, max_beta=1.0, gamma_start=1.0, gamma_end=0.1):
    """Create trainer equivalent to v2 function using loss components"""
    config = TrainingConfig(n_epochs=n_epochs, max_norm=max_norm, lambda_l1=lambda_l1, patience=10)
    trainer = VAETrainer(model, optimizer, scheduler, config)
    
    loss_components = [
        ReconstructionLoss(),
        KLDivergenceLoss(scheduler_type="cosine", min_beta=min_beta, max_beta=max_beta, T=10),
        GeneAbundanceLoss(gamma_start=gamma_start, gamma_end=gamma_end),
        L1RegularizationLoss(lambda_l1=lambda_l1)
    ]
    
    trainer.setup_loss_components(loss_components)
    return trainer


def create_v3_trainer(model, optimizer, scheduler, n_epochs, max_norm, lambda_l1,
                     min_beta=0.1, max_beta=1.0, gamma_start=2.0, gamma_end=0.1, weight=1.0):
    """Create trainer equivalent to v3 function using loss components"""
    config = TrainingConfig(n_epochs=n_epochs, max_norm=max_norm, lambda_l1=lambda_l1, 
                           patience=20, print_every=100)
    trainer = VAETrainer(model, optimizer, scheduler, config)
    
    loss_components = [
        ReconstructionLoss(),
        KLDivergenceLoss(scheduler_type="cosine", min_beta=min_beta, max_beta=max_beta, T=50),
        GeneAbundanceLoss(gamma_start=gamma_start, gamma_end=gamma_end, weight=weight),
        L1RegularizationLoss(lambda_l1=lambda_l1)
    ]
    
    trainer.setup_loss_components(loss_components)
    return trainer


# ORIGINAL V0-V3 FUNCTIONS THAT USE LOSS COMPONENTS
def v0(model, folder, optimizer, scheduler, n_epochs, train_loader, val_loader, 
       beta_start, beta_end, max_norm):
    """Original v0 function using modular loss components"""
    trainer = create_v0_trainer(model, optimizer, scheduler, n_epochs, max_norm, 
                               beta_start, beta_end)
    return trainer.train(train_loader, val_loader, folder)


def v1(model, folder, optimizer, scheduler, n_epochs, train_loader, val_loader, 
       beta_start, beta_end, gamma_start, gamma_end, max_norm, lambda_l1):
    """Original v1 function using modular loss components"""
    trainer = create_v1_trainer(model, optimizer, scheduler, n_epochs, max_norm, lambda_l1,
                               beta_start, beta_end, gamma_start, gamma_end)
    return trainer.train(train_loader, val_loader, folder)


def v2(model, folder, optimizer, scheduler, n_epochs, train_loader, val_loader, 
       min_beta, max_beta, gamma_start, gamma_end, max_norm, lambda_l1):
    """Original v2 function using modular loss components"""
    trainer = create_v2_trainer(model, optimizer, scheduler, n_epochs, max_norm, lambda_l1,
                               min_beta, max_beta, gamma_start, gamma_end)
    return trainer.train(train_loader, val_loader, folder)


def v3(model, folder, optimizer, scheduler, n_epochs, train_loader, val_loader, 
       min_beta, max_beta, gamma_start, gamma_end, weight, max_norm, lambda_l1):
    """Original v3 function using modular loss components"""
    trainer = create_v3_trainer(model, optimizer, scheduler, n_epochs, max_norm, lambda_l1,
                               min_beta, max_beta, gamma_start, gamma_end, weight)
    return trainer.train(train_loader, val_loader, folder)


# BUILDER PATTERN 
class VAETrainerBuilder:
    """Builder class that uses loss_components.py"""
    
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_components = []
        self.config_params = {}
    
    def epochs(self, n_epochs: int):
        self.config_params['n_epochs'] = n_epochs
        return self
    
    def gradient_clipping(self, max_norm: float):
        self.config_params['max_norm'] = max_norm
        return self
    
    def early_stopping(self, patience: int = 10, min_delta: float = 1e-4):
        self.config_params['patience'] = patience
        self.config_params['min_delta'] = min_delta
        return self
    
    def print_every(self, epochs: int):
        self.config_params['print_every'] = epochs
        return self
    
    def with_reconstruction_loss(self):
        """Add reconstruction loss from loss_components.py"""
        self.loss_components.append(ReconstructionLoss())
        return self
    
    def with_kl_loss(self, scheduler_type: str = "linear", min_beta: float = 0.0, 
                     max_beta: float = 1.0, T: int = 10):
        """Add KL divergence loss from loss_components.py"""
        self.loss_components.append(
            KLDivergenceLoss(scheduler_type=scheduler_type, min_beta=min_beta, 
                           max_beta=max_beta, T=T)
        )
        return self
    
    def with_gene_abundance_loss(self, gamma_start: float = 0.0, gamma_end: float = 1.0, 
                                weight: float = 1.0):
        """Add gene abundance loss from loss_components.py"""
        self.loss_components.append(
            GeneAbundanceLoss(gamma_start=gamma_start, gamma_end=gamma_end, weight=weight)
        )
        return self
    
    def with_l1_regularization(self, lambda_l1: float):
        """Add L1 regularization from loss_components.py"""
        self.config_params['lambda_l1'] = lambda_l1
        self.loss_components.append(L1RegularizationLoss(lambda_l1=lambda_l1))
        return self
    
    def with_custom_loss(self, loss_component: LossComponent):
        """Add any custom loss component"""
        self.loss_components.append(loss_component)
        return self
    
    def build(self) -> VAETrainer:
        """Build the trainer"""
        config_defaults = {
            'n_epochs': 1,
            'max_norm': 1.0,
            'lambda_l1': 0.0,
            'patience': 10,
            'min_delta': 1e-4,
            'print_every': 10
        }
        
        for key, default_val in config_defaults.items():
            if key not in self.config_params:
                self.config_params[key] = default_val
        
        config = TrainingConfig(**self.config_params)
        trainer = VAETrainer(self.model, self.optimizer, self.scheduler, config)
        trainer.setup_loss_components(self.loss_components)
        return trainer


# Export everything
__all__ = [
    'VAETrainer', 'VAETrainerBuilder', 'TrainingConfig',
    'create_v0_trainer', 'create_v1_trainer', 'create_v2_trainer', 'create_v3_trainer',
    'v0', 'v1', 'v2', 'v3'
]