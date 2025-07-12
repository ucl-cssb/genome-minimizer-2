#!/usr/bin/env python3
"""
MAIN MODEL
"""

# Import libraries
import torch 
import torch.nn as nn

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE):
    This class implements a Variational Autoencoder (VAE), a generative model 
    that learns a latent representation of input data. The VAE consists of an encoder 
    that maps the input data to a latent space, a reparameterization step that introduces 
    stochasticity, and a decoder that reconstructs the data from the latent space.

    Attributes:
    -----------
    encoder - A sequence of fully connected layers with Batch Normalization and ReLU activations,
    used to map the input data to a hidden representation.
    
    mean_layer - A fully connected layer that maps the hidden representation 
    to the mean of the latent distribution.
    
    logvar_layer - A fully connected layer that maps the hidden representation to the logarithm 
    of the variance of the latent distribution.

    decoder - A sequence of fully connected layers with Batch Normalization, ReLU activations, 
    and a final Sigmoid activation, used to reconstruct the input data from the latent space.
    
    Methods:
    --------
    encode(x) - Encodes the input `x` into its latent representation by passing it through the encoder 
    and outputting the mean and log-variance of the latent distribution.
    
    reparameterization(mean, logvar) - Performs the reparameterization trick, where random noise is sampled 
    and combined with the mean and log-variance to generate a latent vector `z`.

    decode(z) - Decodes the latent vector `z` back into the reconstructed data `x_hat`.

    forward(x) - The forward pass of the VAE. Encodes the input `x` into a latent vector, 
    applies the reparameterization trick, and decodes the latent vector back into 
    a reconstruction of the input. Returns the reconstructed data, mean, and log-variance.

    _initialize_weights() - Initializes the weights of the linear layers using Xavier uniform initialization, 
    and sets the biases to zero where applicable.

    Parameters:
    -----------
    input_dim - Dimensionality of the input data (number of features).
    
    hidden_dim - Dimensionality of the hidden layers used in the encoder and decoder.
    
    latent_dim - Dimensionality of the latent space (size of the encoded representation).
    
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self._initialize_weights()
     
    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)