"""
U-Net for binary dot segmentation.

Input:  (B, 1, 256, 256)  — grayscale fluorescence image
Output: (B, 2, 256, 256)  — logits for [background, dot] classes

Loss:   CrossEntropyLoss(weight=[1, pos_weight])
Count:  sum(softmax[:, 1]) / (pi * mask_radius^2)

This mirrors the DeepTrack2 tutorial approach: train a binary segmentation
model with class-weighted cross-entropy (background:dot = 1:10) so the
network has a strong, balanced gradient signal even though dot pixels are
a tiny fraction of the image. Density map MSE was collapsing to predicting
the dataset mean because background dominates the loss.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm and LeakyReLU."""

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
    U-Net for binary dot segmentation.

    Output is raw logits (no softmax) — CrossEntropyLoss handles normalisation.
    At inference, apply softmax and threshold channel 1 (dot class) at 0.001
    (following the DeepTrack2 tutorial default).

    Design notes:
    - LeakyReLU(0.1) prevents dead neurons.
    - AvgPool2d instead of MaxPool preserves spatial density in feature maps.
    - Dropout2d on encoder + bottleneck for regularisation.
    - No output activation: logits fed directly to CrossEntropyLoss.
    """

    def __init__(self, in_channels=1, out_channels=2, base_filters=32):
        super().__init__()

        f = base_filters  # 32

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)        # 32
        self.enc2 = ConvBlock(f, f * 2)               # 64
        self.enc3 = ConvBlock(f * 2, f * 4)           # 128
        self.enc4 = ConvBlock(f * 4, f * 8)           # 256

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)    # 512

        # Dropout
        self.drop_enc1      = nn.Dropout2d(p=0.1)
        self.drop_enc2      = nn.Dropout2d(p=0.1)
        self.drop_enc3      = nn.Dropout2d(p=0.1)
        self.drop_enc4      = nn.Dropout2d(p=0.1)
        self.drop_bottleneck = nn.Dropout2d(p=0.2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        # Output: raw logits for [background, dot]
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.drop_enc1(self.enc1(x))
        e2 = self.drop_enc2(self.enc2(self.pool(e1)))
        e3 = self.drop_enc3(self.enc3(self.pool(e2)))
        e4 = self.drop_enc4(self.enc4(self.pool(e3)))

        # Bottleneck
        b = self.drop_bottleneck(self.bottleneck(self.pool(e4)))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)   # (B, 2, H, W) — raw logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = UNet()
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}  (logits, no activation)")
