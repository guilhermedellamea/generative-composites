import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator Network for WGAN-GP.

    This generator accepts a latent vector and a condition vector,
    processes them through a fully connected layer followed by a
    sequence of transposed convolutions to generate an image.

    Args:
        n_channels (int): Number of output image channels.
        resolution (int): Target spatial resolution (must be divisible by 2**nup).
        initial_channels (int): Number of base channels before upsampling.
        latent_dim (int): Size of the latent noise vector.
        condition_size (int): Size of the conditioning vector.
        nup (int): Number of upsampling layers (each doubling spatial size).
    """

    def __init__(
        self,
        n_channels: int,
        resolution: int,
        initial_channels: int,
        latent_dim: int = 256,
        condition_size: int = 1,
        nup: int = 6,
    ):
        super().__init__()

        self.spatial_dim = resolution // 2**nup
        self.channel_dim = initial_channels * 2**nup
        self.output_channels = n_channels

        mlp_output_dim = self.spatial_dim * self.spatial_dim * self.channel_dim

        self.fc = nn.Linear(latent_dim + condition_size, mlp_output_dim)
        self.norm1 = nn.BatchNorm1d(mlp_output_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        layers = []
        current_channels = self.channel_dim
        for _ in range(nup):
            next_channels = current_channels // 2
            layers.append(
                nn.ConvTranspose2d(
                    current_channels,
                    next_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(next_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_channels = next_channels

        self.upsample_blocks = nn.Sequential(*layers)

        self.final_conv = nn.Conv2d(
            current_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.output_activation = nn.Tanh()

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, condition], dim=1)
        x = self.act(self.norm1(self.fc(x)))
        x = x.view(-1, self.channel_dim, self.spatial_dim, self.spatial_dim)
        x = self.upsample_blocks(x)
        x = self.final_conv(x)
        return self.output_activation(x)


class Critic(nn.Module):
    """
    Critic for WGAN-GP with optional normalization.

    Args:
        n_channels (int): Number of input image channels.
        resolution (int): Input spatial resolution.
        initial_channels (int): Number of base channels.
        kernel_size (int): Size of convolutional kernels.
        ndown (int): Number of downsampling layers.
        norm (str): Type of normalization ("BN", "IN", or None).
    """

    def __init__(
        self,
        n_channels: int,
        resolution: int,
        initial_channels: int,
        kernel_size: int,
        ndown: int = 6,
        norm: str = None,
    ):
        super().__init__()

        layers = []
        s = resolution
        c = initial_channels
        next_c = initial_channels * 2

        # Initial convolution
        layers.append(
            nn.Conv2d(n_channels, c, kernel_size, stride=2, padding=1, bias=False)
        )
        if norm == "BN":
            layers.append(nn.BatchNorm2d(c, track_running_stats=False))
        elif norm == "IN":
            layers.append(nn.InstanceNorm2d(c, affine=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        s //= 2

        # Downsampling blocks
        for _ in range(ndown):
            layers.append(
                nn.Conv2d(c, next_c, kernel_size, stride=2, padding=1, bias=False)
            )
            if norm == "BN":
                layers.append(nn.BatchNorm2d(next_c, track_running_stats=False))
            elif norm == "IN":
                layers.append(nn.InstanceNorm2d(next_c, affine=True))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            c = next_c
            next_c *= 2
            s //= 2

        self.down_blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(s * s * c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).view(-1)

    def use_checkpointing(self):
        self.down_blocks = torch.utils.checkpoint(self.down_blocks)
        self.fc = torch.utils.checkpoint(self.fc)


if __name__ == "__main__":
    # Minimal check
    latent_dim = 512
    batch_size = 2
    condition_size = 1
    resolution = 512

    generator = ConditionalGenerator(
        n_channels=1,
        resolution=resolution,
        initial_channels=64,
        latent_dim=latent_dim,
        condition_size=condition_size,
        nup=6,
    )

    z = torch.randn(batch_size, latent_dim)
    condition = torch.randn(batch_size, condition_size)
    fake_img = generator(z, condition)

    print(f"Generated image shape: {fake_img.shape}")

    critic = Critic(
        n_channels=1,
        resolution=resolution,
        initial_channels=64,
        kernel_size=4,
        ndown=6,
        norm="BN",
    )
    real_img = torch.randn(batch_size, 1, resolution, resolution)
    critic_output = critic(real_img)
    print(f"Critic output shape: {critic_output.shape}")
