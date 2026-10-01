"""2D Deep Neighbourhood Self-similarity for 16x16 GLU-Net features.

This adapts the two-ring, centre-free DNS idea of Mok et al. (CVPR 2024)
from 3D medical images to 2D visible/infrared features. It is not their 3D
MASR-Net or a reproduction of its anatomy-aware training procedure.
"""

import torch
from torch import nn
from torch.nn import functional as F


class CoarseDNS(nn.Module):
    """Shared 2D neighbour self-similarity with a zero-initialized residual gate."""

    def __init__(self, channels=512, hidden=64, radii=(1, 2)):
        super().__init__()
        self.radii = tuple(radii)
        if not self.radii or any(radius < 1 for radius in self.radii):
            raise ValueError("DNS radii must be positive")
        descriptor_channels = 4 * len(self.radii)
        self.squeeze = nn.Sequential(
            nn.Conv2d(descriptor_channels, hidden, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def descriptor(self, features):
        if features.ndim != 4:
            raise ValueError("Expected [batch, channels, height, width] features")
        _, _, height, width = features.shape
        descriptors = []
        for radius in self.radii:
            if radius >= min(height, width):
                raise ValueError("DNS radius must be smaller than the feature map")
            padded = F.pad(features, (radius, radius, radius, radius),
                           mode="replicate")
            north = padded[:, :, :height, radius:radius + width]
            east = padded[:, :, radius:radius + height, 2 * radius:2 * radius + width]
            south = padded[:, :, 2 * radius:2 * radius + height, radius:radius + width]
            west = padded[:, :, radius:radius + height, :width]
            # Pair neighbours around the centre; the centre feature is excluded.
            neighbours = (north, east, south, west)
            squared_distances = [
                (neighbours[index] - neighbours[(index + 1) % 4]).square().mean(dim=1)
                for index in range(4)
            ]
            distances = torch.stack(squared_distances, dim=1)
            noise_scale = distances.mean(dim=1, keepdim=True).clamp_min(1e-6)
            descriptors.append(torch.exp(-distances / noise_scale))
        return torch.cat(descriptors, dim=1)

    def forward_one(self, features):
        structure = self.squeeze(self.descriptor(features))
        return features + self.gate * structure

    def forward(self, target, source):
        if target.shape != source.shape:
            raise ValueError("DNS target and source features must have equal shapes")
        return self.forward_one(target), self.forward_one(source)
