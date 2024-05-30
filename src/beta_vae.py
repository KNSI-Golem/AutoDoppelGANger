import torch.nn as nn
import torch

class BetaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims, beta, device):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.encoder = nn.Sequential(
            *self.build_encoder(in_channels)
        )
        self.mu, self.sigma = self.build_distr_params(self)
        self.decoder_input = self.build_decoder_input(self)
        self.decoder = nn.Sequential(
            *self.build_decoder(),
        )
        if device.type == "cuda":
            self.cuda()

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparametrize(mu, sigma)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, sigma

    def encode(self, x):
        output = self.encoder(x)
        output = torch.flatten(output, start_dim=1)
        mu = self.mu(output)
        sigma = self.sigma(output)
        return mu, sigma

    def reparametrize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):
        output = self.decoder_input(z)
        output = output.view(-1, 512, 2, 2)
        output = self.decoder(output)
        return output

    def build_encoder(self, in_channels):
        blocks = []
        for h_dim in self.hidden_dims:
            blocks.append(self.encoder_block(in_channels, h_dim))
            in_channels = h_dim
        return blocks

    def build_distr_params(self):
        return (nn.Linear(self.hidden_dims[-1]*4, self.latent_dim),
               nn.Linear(self.hidden_dims[-1]*4, self.latent_dim))

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def build_decoder_input(self):
        return nn.Linear(self.latent_dim, self.hidden_dims[-1] * 4)

    def build_decoder(self):
        blocks = []
        self.hidden_dims.reverse()
        in_channels = self.hidden_dims[0]
        self.hidden_dims.pop(0)
        for h_dim in self.hidden_dims:
            blocks.append(self.decoder_block(h_dim, in_channels))
            in_channels = h_dim
        blocks.append(self.decoder_final(self))
        return blocks

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def decoder_final(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1],
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels= 3,
                        kernel_size= 3, padding= 1),
            nn.Tanh()
        )

