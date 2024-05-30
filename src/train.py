import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from src.beta_vae import BetaVAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class BetaVAETrainer:
    def __init__(self, in_channels, latent_dim, hidden_dims, device, log_dir):
        self.model = BetaVAE(in_channels, latent_dim, hidden_dims, device)
        self.device = device
        self.log_dir = log_dir

    def train(self, dataset, num_epochs, batch_size, beta, learning_rate):
        self.load_data(dataset, batch_size)
        self.model.initilize_weights()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.beta = beta
        self.reconstruction_loss = nn.BCELoss(reduction="sum")
        for epoch in range(num_epochs):
            for batch_idx, (x, _) in enumerate(self.loaded_data):
                x = x.to(self.device)
                x_reconstructed, mu, log_var = self.model.forward(x)

                loss = self.calculate_loss(x_reconstructed, x, mu, log_var)

                self.backpropagation(loss)

    def calculate_loss(self, x_reconstructed, x, mu, log_var):
        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim = 1)
        return reconstruction_loss + self.beta * kl_divergence

    def backpropagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



