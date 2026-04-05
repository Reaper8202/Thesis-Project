"""
U-Net architecture for density map regression.
Sized for GTX 1660 Super (6GB VRAM) with 256x256 inputs.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two convolution layers with BatchNorm and LeakyReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net for density map regression.

    Input:  (B, 1, 256, 256)  — grayscale image
    Output: (B, 1, 256, 256)  — predicted density map (non-negative)

    Design notes:
    - LeakyReLU(0.1) instead of ReLU avoids dead neurons on a small dataset.
    - AvgPool2d instead of MaxPool2d preserves spatial density in downsampled
      feature maps; MaxPool discards 75% of activations per 2x2 region, which
      is unsuitable for density-preserving regression.
    - Softplus(beta=10) output instead of ReLU gives smooth, non-negative
      density values with non-zero gradients everywhere (ReLU zero-gradients
      prevent correction of over-suppressed regions).
    - Dropout2d on encoder + bottleneck for small-dataset regularisation.
    """

    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()

        f = base_filters  # 32

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)        # 32
        self.enc2 = ConvBlock(f, f * 2)               # 64
        self.enc3 = ConvBlock(f * 2, f * 4)           # 128
        self.enc4 = ConvBlock(f * 4, f * 8)           # 256

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)    # 512

        # Dropout for regularization (small dataset)
        self.drop_enc1 = nn.Dropout2d(p=0.1)
        self.drop_enc2 = nn.Dropout2d(p=0.1)
        self.drop_enc3 = nn.Dropout2d(p=0.1)
        self.drop_enc4 = nn.Dropout2d(p=0.1)
        self.drop_bottleneck = nn.Dropout2d(p=0.2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)          # 256 (concat with enc4)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)           # 128

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)           # 64

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)               # 32

        # Output: 1x1 conv to single channel density map
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        # Softplus output: smooth non-negative activation.
        # beta=10 closely approximates ReLU while keeping gradients non-zero
        # everywhere, allowing the model to recover from over-suppression.
        self.out_act = nn.Softplus(beta=10)

        # AvgPool: preserves total density energy across the pooled region.
        # MaxPool would keep only the largest activation and discard 75% of
        # spatial density information, hurting count accuracy.
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.drop_enc1(self.enc1(x))                    # (B, 32, 256, 256)
        e2 = self.drop_enc2(self.enc2(self.pool(e1)))        # (B, 64, 128, 128)
        e3 = self.drop_enc3(self.enc3(self.pool(e2)))        # (B, 128, 64, 64)
        e4 = self.drop_enc4(self.enc4(self.pool(e3)))        # (B, 256, 32, 32)

        # Bottleneck
        b = self.drop_bottleneck(self.bottleneck(self.pool(e4)))   # (B, 512, 16, 16)

        # Decoder with skip connections
        d4 = self.up4(b)                     # (B, 256, 32, 32)
        d4 = torch.cat([d4, e4], dim=1)      # (B, 512, 32, 32)
        d4 = self.dec4(d4)                   # (B, 256, 32, 32)

        d3 = self.up3(d4)                    # (B, 128, 64, 64)
        d3 = torch.cat([d3, e3], dim=1)      # (B, 256, 64, 64)
        d3 = self.dec3(d3)                   # (B, 128, 64, 64)

        d2 = self.up2(d3)                    # (B, 64, 128, 128)
        d2 = torch.cat([d2, e2], dim=1)      # (B, 128, 128, 128)
        d2 = self.dec2(d2)                   # (B, 64, 128, 128)

        d1 = self.up1(d2)                    # (B, 32, 256, 256)
        d1 = torch.cat([d1, e1], dim=1)      # (B, 64, 256, 256)
        d1 = self.dec1(d1)                   # (B, 32, 256, 256)

        # Output
        out = self.out_conv(d1)              # (B, 1, 256, 256)
        out = self.out_act(out)              # Non-negative, smooth density

        return out


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    model = UNet()
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output min: {y.min().item():.4f}, max: {y.max().item():.4f}")
