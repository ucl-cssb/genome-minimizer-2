#!/usr/bin/env python3
"""
Loss components for VAE training.
"""

# Import libraries
import torch
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LossComponent(ABC):
    """Abstract base class for loss components"""
    
    @abstractmethod
    def compute_loss(self, recon_x: torch.Tensor, data: torch.Tensor, 
                    mu: torch.Tensor, logvar: torch.Tensor, 
                    model: nn.Module, epoch: int, batch_idx: int) -> torch.Tensor:
        """
        Compute the loss for this component.
        
        Args:
            recon_x: Reconstructed data
            data: Original data
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            model: The VAE model
            epoch: Current epoch number
            batch_idx: Current batch index
            
        Returns:
            Loss tensor
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this loss component"""
        pass


class ReconstructionLoss(LossComponent):
    """Binary cross-entropy reconstruction loss"""
    
    def compute_loss(self, recon_x, data, mu, logvar, model, epoch, batch_idx):
        return nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
    
    def get_name(self):
        return "reconstruction"


class KLDivergenceLoss(LossComponent):
    """KL divergence loss with various scheduling options"""
    
    def __init__(self, scheduler_type="linear", min_beta=0.0, max_beta=1.0, T=10):
        """
        Initialize KL divergence loss.
        
        Args:
            scheduler_type: Type of beta scheduling ("linear", "cosine", "constant")
            min_beta: Minimum beta value
            max_beta: Maximum beta value
            T: Period for cosine annealing
        """
        self.scheduler_type = scheduler_type
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.T = T
        self.counter = 0
        self.n_epochs = 1000  # Will be updated when trainer is created
    
    def compute_loss(self, recon_x, data, mu, logvar, model, epoch, batch_idx):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.scheduler_type == "linear":
            beta = self.min_beta + (self.max_beta - self.min_beta) * epoch / self.n_epochs
        elif self.scheduler_type == "cosine":
            t = epoch * 32 + self.counter
            beta = cosine_annealing_schedule(t, self.T, self.min_beta, self.max_beta)
            self.counter += 1
        else:
            beta = self.max_beta
            
        return beta * kl_loss
    
    def get_name(self):
        return "kl_divergence"


class GeneAbundanceLoss(LossComponent):
    """Gene abundance loss with linear annealing"""
    
    def __init__(self, gamma_start=0.0, gamma_end=1.0, weight=1.0):
        """
        Initialize gene abundance loss.
        
        Args:
            gamma_start: Initial gamma value
            gamma_end: Final gamma value
            weight: Weight multiplier for the loss
        """
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.weight = weight
        self.n_epochs = 1000  # Will be updated when trainer is created
    
    def compute_loss(self, recon_x, data, mu, logvar, model, epoch, batch_idx):
        gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * epoch / self.n_epochs
        total_gene_number = recon_x.sum(axis=0)
        total_gene_number_loss = torch.sum(torch.abs(total_gene_number))
        return self.weight * gamma * total_gene_number_loss
    
    def get_name(self):
        return "gene_abundance"


class L1RegularizationLoss(LossComponent):
    """L1 regularization loss"""
    
    def __init__(self, lambda_l1=0.0):
        """
        Initialize L1 regularization loss.
        
        Args:
            lambda_l1: L1 regularization strength
        """
        self.lambda_l1 = lambda_l1
    
    def compute_loss(self, recon_x, data, mu, logvar, model, epoch, batch_idx):
        if self.lambda_l1 == 0.0:
            return torch.tensor(0.0, device=device)
        return l1_regularization(model, self.lambda_l1)
    
    def get_name(self):
        return "l1_regularization"


class L2RegularizationLoss(LossComponent):
    """L2 (weight decay) regularization loss"""
    
    def __init__(self, lambda_l2: float = 0.01):
        """
        Initialize L2 regularization loss.
        
        Args:
            lambda_l2: L2 regularization strength
        """
        self.lambda_l2 = lambda_l2
    
    def compute_loss(self, recon_x, data, mu, logvar, model, epoch, batch_idx):
        if self.lambda_l2 == 0.0:
            return torch.tensor(0.0, device=device)
        
        l2_penalty = 0
        for param in model.parameters():
            l2_penalty += torch.sum(param ** 2)
        return self.lambda_l2 * l2_penalty
    
    def get_name(self):
        return "l2_regularization"


def l1_regularization(model, lambda_l1):
    """
    Compute the L1 regularization term for a given model

    Parameters:
    ----------
    model - The neural network model whose parameters are to be regularized

    lambda_l1 - The regularization strength parameter for L1 regularization

    Returns:
    -------
    The computed L1 regularization term.
    """
    l1_penalty = 0.0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
    return lambda_l1 * l1_penalty


def cosine_annealing_schedule(t, T, min_beta, max_beta):
    """
    Compute the value of beta using a cosine annealing schedule

    Parameters:
    ----------
    t - The current time step or epoch
    T - The total number of time steps or epochs
    min_beta - The minimum value of beta
    max_beta - The maximum value of beta

    Returns:
    -------
    The computed beta value for the given time step.
    """
    return min_beta + (max_beta - min_beta) / 2 * (1 + np.cos(np.pi * (t % T) / T))


# Export all loss components
__all__ = [
    'LossComponent',
    'ReconstructionLoss', 
    'KLDivergenceLoss', 
    'GeneAbundanceLoss',
    'L1RegularizationLoss',
    'L2RegularizationLoss', 
]