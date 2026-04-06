"""
U-Net architecture for density map regression + count regression.
Sized for GTX 1660 Super (6GB VRAM) with 256x256 inputs.

Dual-head design:
  - Decoder head: predicts spatial density map (used for visualization)
  - Count head:   branches from bottleneck, directly regresses dot count
                  (used as the primary count output at inference)
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
    U-Net for density map regression + direct count regression.

    Input:  (B, 1, 256, 256)  — grayscale image
    Output: tuple of
        density_map: (B, 1, 256, 256)  — predicted density map (non-negative)
        count:       (B,)               — direct count prediction from bottleneck

    Design notes:
    - Dual-head: the count head branches from the bottleneck (pre-dropout) and
      applies GlobalAvgPool → FC layers → Softplus. This gives the count a direct
      gradient path that isn't diluted by the spatial density map loss. The old
      approach of summing the density map collapsed to predicting the dataset mean
      because MSE on a mostly-zero map has near-zero gradient almost everywhere.
    - LeakyReLU(0.1) instead of ReLU avoids dead neurons on a small dataset.
    - AvgPool2d instead of MaxPool2d preserves spatial density in downsampled
      feature maps.
    - Softplus(beta=10) output for density map: smooth non-negative activation.
    - Dropout2d on encoder + bottleneck for regularisation.
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

        # Dropout for regularization
        self.drop_enc1 = nn.Dropout2d(p=0.1)
        self.drop_enc2 = nn.Dropout2d(p=0.1)
        self.drop_enc3 = nn.Dropout2d(p=0.1)
        self.drop_enc4 = nn.Dropout2d(p=0.1)
        self.drop_bottleneck = nn.Dropout2d(p=0.2)

        # Count regression head — branches from pre-dropout bottleneck features.
        # GlobalAvgPool collapses (B, 512, 16, 16) → (B, 512), then two FC layers
        # regress to a single count value. Using pre-dropout features gives the
        # count head a richer gradient signal than the post-dropout features used
        # by the decoder.
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (B, f*16, 1, 1)
            nn.Flatten(),               # (B, f*16 = 512)
            nn.Linear(f * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Softplus(beta=1),        # count ≥ 0; beta=1 gives smoother gradient near 0
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        # Output: 1x1 conv → density map
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)
        self.out_act = nn.Softplus(beta=10)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.drop_enc1(self.enc1(x))                    # (B, 32,  256, 256)
        e2 = self.drop_enc2(self.enc2(self.pool(e1)))        # (B, 64,  128, 128)
        e3 = self.drop_enc3(self.enc3(self.pool(e2)))        # (B, 128,  64,  64)
        e4 = self.drop_enc4(self.enc4(self.pool(e3)))        # (B, 256,  32,  32)

        # Bottleneck — keep pre-dropout features for count head
        b_feat = self.bottleneck(self.pool(e4))              # (B, 512,  16,  16)
        b = self.drop_bottleneck(b_feat)

        # Count head branches from pre-dropout bottleneck
        count = self.count_head(b_feat).squeeze(1)           # (B,)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))  # (B, 256, 32, 32)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 128, 64, 64)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 64, 128, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 32, 256, 256)

        density_map = self.out_act(self.out_conv(d1))         # (B, 1, 256, 256)

        return density_map, count


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = UNet()
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 1, 256, 256)
    density_map, count = model(x)
    print(f"Input shape:       {x.shape}")
    print(f"Density map shape: {density_map.shape}")
    print(f"Count shape:       {count.shape}")
    print(f"Count values:      {count.tolist()}")
