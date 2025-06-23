import torch
import torch.nn as nn

# =========================
# Utility Modules
# =========================


def get_norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    """Returns normalization layer based on the type."""
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm_type is None or norm_type.lower() == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")


class MLPEmbedding(nn.Module):
    """1D MLP used for embedding scalar inputs (e.g. timestep, condition)."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.embed(x.view(-1, self.embed[0].in_features))


# =========================
# U-Net Components
# =========================


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and optional skip connection."""

    def __init__(
        self, in_channels, out_channels, use_residual=False, norm_type="batch"
    ):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            get_norm_layer(norm_type, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            get_norm_layer(norm_type, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.block(x) if self.use_residual else self.block(x)


class DownSampleBlock(nn.Module):
    """Downsampling block using ResidualBlock + MaxPool."""

    def __init__(self, in_channels, out_channels, norm_type):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(in_channels, out_channels, norm_type=norm_type),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.model(x)


class UpSampleBlock(nn.Module):
    """Upsampling block using ConvTranspose2D and residual layers."""

    def __init__(self, in_channels, out_channels, norm_type):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ResidualBlock(out_channels, out_channels, norm_type=norm_type),
            ResidualBlock(out_channels, out_channels, norm_type=norm_type),
        )

    def forward(self, x, skip_connection):
        x = torch.cat((x, skip_connection), dim=1)
        return self.model(x)


# =========================
# Generative U-Nets
# =========================


class ContextualUNet(nn.Module):
    """
    U-Net architecture with conditioning on both timestep and a scalar structural property.

    Args:
        base_channels (int): Number of input/output channels.
        n_feat (int): Number of base feature channels. Default is 128.
        n_down (int): Number of downsampling steps. Default is 2.
        norm_type (str): Normalization type ('batch', 'instance', etc.). Default is 'instance'.
    """

    def __init__(self, base_channels, n_feat=128, n_down=2, norm_type="instance"):
        super().__init__()
        self.in_channels = base_channels
        self.n_feat = n_feat
        initial_resolution = 512

        # Initial convolution
        self.init_conv = ResidualBlock(
            base_channels, n_feat, use_residual=True, norm_type=norm_type
        )

        # Downsampling path
        self.downs = [DownSampleBlock(n_feat, n_feat, norm_type=norm_type)]
        n_feat_i = n_feat
        resolutions = [initial_resolution // 2]

        for _ in range(n_down - 1):
            self.downs.append(
                DownSampleBlock(n_feat_i, 2 * n_feat_i, norm_type=norm_type)
            )
            n_feat_i *= 2
            resolutions.append(resolutions[-1] // 2)

        self.downs = nn.ModuleList(self.downs)
        avg_size = resolutions[-1] // 16

        # Bottleneck and decoder initialization
        self.to_vec = nn.Sequential(nn.AvgPool2d(avg_size), nn.GELU())
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(n_feat_i, n_feat_i, avg_size, avg_size),
            nn.GroupNorm(avg_size, n_feat_i),
            nn.ReLU(),
        )

        # Embeddings for timestep and conditioning variable
        self.time_embeds = nn.ModuleList(
            [MLPEmbedding(1, 2**i * n_feat) for i in range(n_down)]
        )
        self.context_embeds = nn.ModuleList(
            [MLPEmbedding(1, 2**i * n_feat) for i in range(n_down)]
        )

        # Upsampling path
        self.ups = nn.ModuleList()
        for i in range(n_down):
            in_ch = (2 ** (i + 1)) * n_feat
            out_ch = (2 ** max(i - 1, 0)) * n_feat
            self.ups.append(UpSampleBlock(in_ch, out_ch, norm_type=norm_type))

        self.ups = nn.ModuleList(list(reversed(self.ups)))

        # Final convolution to project back to base_channels
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, base_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, c, t):
        """
        Forward pass of the contextual U-Net.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).
            c (Tensor): Conditioning scalar of shape (B, 1).
            t (Tensor): Normalized timestep of shape (B, 1).

        Returns:
            Tensor: Denoised output image.
        """
        x_in = self.init_conv(x)

        # Downsampling with skip connections
        downs = [x_in]
        for down in self.downs:
            downs.append(down(downs[-1]))

        # Bottleneck vector and upsampling initialization
        hiddenvec = self.to_vec(downs[-1])
        x = self.up0(hiddenvec)

        # Prepare embeddings
        c = c.view(-1, 1)
        t = t.view(-1, 1)
        cembeds = [emb(c).view(c.size(0), -1, 1, 1) for emb in self.context_embeds]
        tembeds = [emb(t).view(t.size(0), -1, 1, 1) for emb in self.time_embeds]

        cembeds.reverse()
        tembeds.reverse()
        downs.reverse()

        # Upsampling with skip connections and embeddings
        for i, up in enumerate(self.ups):
            x = up(cembeds[i] * x + tembeds[i], downs[i])

        return self.out(torch.cat((x, x_in), dim=1))


class DamageConditionedUNet(nn.Module):
    """Diffusion U-Net with spatial conditioning from a damage mask."""

    def __init__(self, in_channels, base_channels=256, norm_type="instance"):
        super().__init__()

        self.init_conv = ResidualBlock(
            in_channels, base_channels, use_residual=True, norm_type=norm_type
        )
        self.down1 = DownSampleBlock(base_channels, base_channels, norm_type)
        self.down2 = DownSampleBlock(base_channels, base_channels * 2, norm_type)
        self.bottleneck = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        self.temb1 = MLPEmbedding(1, 2 * base_channels + 1)
        self.temb2 = MLPEmbedding(1, base_channels + 1)

        self.pool1 = nn.MaxPool2d(4)
        self.pool2 = nn.MaxPool2d(2)

        self.decoder_input = nn.Sequential(
            nn.ConvTranspose2d(2 * base_channels, 2 * base_channels, 8, 8),
            nn.GroupNorm(8, 2 * base_channels),
            nn.ReLU(),
        )

        self.up1 = UpSampleBlock(4 * base_channels + 1, base_channels, norm_type)
        self.up2 = UpSampleBlock(2 * base_channels + 1, base_channels, norm_type)

        self.final_conv = nn.Sequential(
            nn.Conv2d(2 * base_channels + 1, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, t, damage_mask):
        x0 = self.init_conv(x)
        d1 = self.down1(x0)
        d2 = self.down2(d1)
        z = self.bottleneck(d2)

        temb1 = self.temb1(t).view(-1, 2 * x0.shape[1] + 1, 1, 1)
        temb2 = self.temb2(t).view(-1, x0.shape[1] + 1, 1, 1)

        x = self.decoder_input(z)
        x = self.up1(torch.cat((x, self.pool1(damage_mask)), 1) + temb1, d2)
        x = self.up2(torch.cat((x, self.pool2(damage_mask)), 1) + temb2, d1)
        return self.final_conv(torch.cat((x, damage_mask, x0), 1))


# =========================
# Evaluators / Predictors
# =========================


class DoubleConv(nn.Module):
    """Double convolutional block used in ResNet evaluators."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        leakyrelu=False,
        dropout_rate=None,
        norm_type="batch",
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        Norm2d = {"batch": nn.BatchNorm2d, "instance": nn.InstanceNorm2d}.get(
            norm_type, None
        )

        def block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)]
            if Norm2d:
                layers.append(Norm2d(out_c))
            layers.append(
                nn.LeakyReLU(0.01, inplace=True) if leakyrelu else nn.ReLU(inplace=True)
            )
            return layers

        self.double_conv = nn.Sequential(
            *block(in_channels, mid_channels), *block(mid_channels, out_channels)
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()

    def forward(self, x):
        return self.dropout(self.double_conv(x))


class DownSkip(nn.Module):
    """Downsampling block with skip connection."""

    def __init__(
        self,
        in_channels,
        out_channels,
        leakyrelu=False,
        dropout_rate=None,
        norm_type="batch",
    ):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            leakyrelu=leakyrelu,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
        )

    def forward(self, x):
        x_down = self.maxpool(x)
        x_conv = self.conv(x_down)
        return torch.cat([x_down, x_conv], dim=1)


class MLP(nn.Module):
    """Generic MLP block used for regression head."""

    def __init__(
        self, in_features, hidden_size, out_features, leakyrelu=False, dropout_rate=None
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(0.01, inplace=True) if leakyrelu else nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate else nn.Identity(),
            nn.Linear(hidden_size, out_features),
        )

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    """ResNet-style evaluator used for predicting scalar properties."""

    def __init__(
        self,
        n_channels,
        n_labels,
        resolution,
        initial_channels,
        ndown,
        mlp_hidden_layers=32,
        leakyrelu=False,
        dropout_rate=None,
        norm_type="batch",
    ):

        super().__init__()
        self.inc = DoubleConv(
            n_channels,
            initial_channels,
            leakyrelu=leakyrelu,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
        )

        s = resolution
        c = initial_channels
        layers = []

        for _ in range(ndown):
            layers.append(
                DownSkip(
                    c,
                    2 * c,
                    leakyrelu=leakyrelu,
                    dropout_rate=dropout_rate,
                    norm_type=norm_type,
                )
            )
            s //= 2
            c *= 2

        self.downs = nn.Sequential(*layers)
        self.mlp = MLP(
            s**2 * c,
            mlp_hidden_layers,
            n_labels,
            leakyrelu=leakyrelu,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        x = self.inc(x)
        x = self.downs(x)
        return self.mlp(x.view(x.size(0), -1))
