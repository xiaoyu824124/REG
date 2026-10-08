"""One pass of 128/256 local refinement; every flow is in 512px units.

VI is the target and IR is the sampled source. There is no deformable
convolution, learned sampling offset, or recurrent update in this module.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .coarse_dns import CoarseDNS
from .local_recurrent_128_256 import LocalScale


class LocalNoDCNMultiScale(nn.Module):
    def __init__(self, width=16, hidden=32):
        super().__init__()
        self.project128 = nn.Conv2d(128, width, 1)
        self.project256 = nn.Conv2d(64, width, 1)
        self.fuse128 = nn.Conv2d(width * 2, width, 3, padding=1)
        self.fuse256 = nn.Conv2d(width * 2, width, 3, padding=1)
        self.dns128 = CoarseDNS(width, hidden=width)
        self.dns256 = CoarseDNS(width, hidden=width)
        self.scale128 = LocalScale(width, radius=2)
        self.scale256 = LocalScale(width, radius=1)
        self.body128 = nn.Sequential(nn.Conv2d(width + 2, hidden, 3, padding=1),
                                     nn.LeakyReLU(.1), nn.Conv2d(hidden, hidden, 3, padding=1),
                                     nn.LeakyReLU(.1))
        self.body256 = nn.Sequential(nn.Conv2d(width + 2, hidden, 3, padding=1),
                                     nn.LeakyReLU(.1), nn.Conv2d(hidden, hidden, 3, padding=1),
                                     nn.LeakyReLU(.1))
        self.delta128 = nn.Conv2d(hidden, 2, 3, padding=1)
        self.delta256 = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.zeros_(self.delta128.weight)
        nn.init.zeros_(self.delta128.bias)
        nn.init.zeros_(self.delta256.weight)
        nn.init.zeros_(self.delta256.bias)

    def gather_modality(self, feature128, feature256):
        low = self.project128(feature128)
        high = self.project256(feature256)
        low = low + self.fuse128(torch.cat((low, F.avg_pool2d(high, 2)), 1))
        high = high + self.fuse256(torch.cat((F.interpolate(
            self.project128(feature128), (256, 256), mode='bilinear',
            align_corners=False), high), 1))
        return low, high

    def forward(self, target128, source128, target256, source256, flow128):
        batch = flow128.shape[0]
        if (flow128.shape != (batch, 2, 128, 128) or
                target128.shape != source128.shape or
                target128.shape != (batch, 128, 128, 128) or
                target256.shape != source256.shape or
                target256.shape != (batch, 64, 256, 256)):
            raise ValueError('Expected 512px image VGG 128/256 grids and 512px flow')
        t128, t256 = self.gather_modality(target128, target256)
        s128, s256 = self.gather_modality(source128, source256)
        t128, s128 = self.dns128(t128, s128)
        t256, s256 = self.dns256(t256, s256)
        t128, s128 = self.scale128.self_attention(t128), self.scale128.self_attention(s128)
        t256, s256 = self.scale256.self_attention(t256), self.scale256.self_attention(s256)
        signal128, state128 = self.scale128.cross_attention(t128, s128, flow128)
        delta128 = self.delta128(self.body128(torch.cat((signal128, flow128 / 512), 1)))
        delta128 = delta128 * state128['valid_any'][:, None]
        updated128 = flow128 + delta128
        initial256 = F.interpolate(flow128, (256, 256), mode='bilinear',
                                    align_corners=False)
        flow256 = F.interpolate(updated128, (256, 256), mode='bilinear',
                                align_corners=False)
        signal256, state256 = self.scale256.cross_attention(t256, s256, flow256)
        delta256 = self.delta256(self.body256(torch.cat((signal256, flow256 / 512), 1)))
        delta256 = delta256 * state256['valid_any'][:, None]
        updated256 = flow256 + delta256
        initial512 = F.interpolate(flow128, (512, 512), mode='bilinear',
                                   align_corners=False)
        correction512 = F.interpolate(updated256 - initial256, (512, 512),
                                       mode='bilinear', align_corners=False)
        return {'flow128': updated128, 'flow256': updated256,
                'final512': initial512 + correction512,
                'valid128': state128['valid_any'],
                'valid256': state256['valid_any'],
                'delta128': delta128, 'delta256': delta256}
