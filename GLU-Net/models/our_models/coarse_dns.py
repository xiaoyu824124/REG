"""Shared 2D DNS on GLU-Net's 16x16 deep features.

Adapted from Mok et al., CVPR 2024: centre-free direct and dilated neighbour
self-similarity followed by learned feature squeezing. The paper uses 3D
six-neighbourhoods and anatomy-aware contrastive training; this module uses
2D four-neighbourhoods and a zero-gated residual for pretrained GLU-Net.
It is not a reproduction of the complete DSIR method.
"""

import torch
from torch import nn
from torch.nn import functional as F


class CoarseDNS(nn.Module):
    """Compute [B,C,8,H,W] DNS, squeeze C, then project back to C channels."""

    def __init__(self, channels=512, hidden=128, radii=(1, 2), eps=1e-6):
        super().__init__()
        self.channels = channels
        self.radii = tuple(radii)
        self.eps = eps
        if not self.radii or any(radius < 1 for radius in self.radii):
            raise ValueError("DNS radii must be positive")
        self.channel_squeeze = nn.Linear(channels, 1)
        self.projection = nn.Sequential(
            nn.Conv2d(4 * len(self.radii), hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1))
        self.gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def four_neighbours(features, radius):
        _, _, height, width = features.shape
        padded = F.pad(features, (radius, radius, radius, radius),
                       mode="replicate")
        north = padded[:, :, :height, radius:radius + width]
        east = padded[:, :, radius:radius + height, 2 * radius:2 * radius + width]
        south = padded[:, :, 2 * radius:2 * radius + height, radius:radius + width]
        west = padded[:, :, radius:radius + height, :width]
        return north, east, south, west

    def descriptor(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(f"Expected [B,{self.channels},H,W] features")
        _, _, height, width = features.shape
        descriptors = []
        for radius in self.radii:
            if radius >= min(height, width):
                raise ValueError("DNS radius must be smaller than the feature map")
            north, east, south, west = self.four_neighbours(features, radius)
            # Each pair has Euclidean offset sqrt(2)*radius; the centre is
            # excluded. Keep all deep feature channels until squeezing.
            distances = torch.stack(((north - east).square(),
                                     (east - south).square(),
                                     (south - west).square(),
                                     (west - north).square()), dim=2)
            noise = distances.mean(dim=2, keepdim=True).clamp_min(self.eps)
            descriptors.append(torch.exp(-distances / noise))
        return torch.cat(descriptors, dim=2)

    def forward_one(self, features):
        descriptor = self.descriptor(features)  # [B,C,P,H,W]
        squeezed = self.channel_squeeze(
            descriptor.permute(0, 2, 3, 4, 1)).squeeze(-1)  # [B,P,H,W]
        structure = self.projection(squeezed)  # [B,C,H,W]
        return features + self.gate * structure

    def forward(self, target, source):
        if target.shape != source.shape:
            raise ValueError("DNS target and source features must have equal shapes")
        return self.forward_one(target), self.forward_one(source)
