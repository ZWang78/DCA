import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """A basic convolutional block: Conv -> GroupNorm -> GELU."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.relu = nn.GELU()
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class ResBlock(nn.Module):
    """A simple residual block for the autoencoder."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    """Encodes an image into a latent representation (mean and log variance)."""
    def __init__(self, in_channels, ae_channels, num_res_blocks, latent_dim):
        super().__init__()
        self.initial_conv = ConvBlock(in_channels, ae_channels[0])
        self.down_blocks = nn.ModuleList()
        channels = ae_channels[0]
        for i, out_ch in enumerate(ae_channels):
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(channels))
            if i != len(ae_channels) - 1:
                self.down_blocks.append(ConvBlock(channels, channels, stride=2))
        self.final_conv = nn.Conv2d(ae_channels[-1], 2 * latent_dim, kernel_size=1)

    def forward(self, x):
        h = self.initial_conv(x)
        for block in self.down_blocks:
            h = block(h)
        return self.final_conv(h)

class Decoder(nn.Module):
    """Decodes a latent representation back into an image."""
    def __init__(self, latent_dim, ae_channels, num_res_blocks, out_channels_img):
        super().__init__()
        initial_stage_channels = ae_channels[-1]
        self.initial_conv = nn.Conv2d(latent_dim, initial_stage_channels, kernel_size=1)
        self.up_blocks = nn.ModuleList()
        current_channels = initial_stage_channels
        num_resolutions = len(ae_channels)

        for i in range(num_resolutions):
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResBlock(current_channels))
            if i != num_resolutions - 1:
                self.up_blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    ConvBlock(current_channels, current_channels)
                ))

        self.final_conv = nn.Conv2d(current_channels, out_channels_img, kernel_size=3, padding=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, z):
        h = self.initial_conv(z)
        for block in self.up_blocks:
            h = block(h)
        recon = self.final_conv(h)
        return self.final_activation(recon)

class Autoencoder(nn.Module):
    """Variational Autoencoder (VAE) architecture."""
    def __init__(self, in_channels, ae_channels, num_res_blocks, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, ae_channels, num_res_blocks, latent_dim)
        self.decoder = Decoder(latent_dim, ae_channels, num_res_blocks, in_channels)
        self.latent_dim = latent_dim

    def forward(self, x):
        encoded = self.encoder(x)
        mean, logvar = torch.chunk(encoded, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        recon = self.decoder(z)
        return recon, mean, logvar, z
