"""One-pass 128/256 local structural guidance *inside* GLU-Net decoder1.

VI is the target, IR is the source, and flow is in 512-image-pixel units.
No deformable convolution or recurrent update is used here.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .coarse_dns import CoarseDNS
from .local_recurrent_128_256 import LocalScale


class LocalDecoderFusion(nn.Module):
    def __init__(self, corr_channels=85, width=16, hidden=32):
        super().__init__()
        self.corr_channels = corr_channels
        self.project128 = nn.Conv2d(128, width, 1)
        self.project256 = nn.Conv2d(64, width, 1)
        self.fuse128 = nn.Conv2d(2 * width, width, 3, padding=1)
        self.fuse256 = nn.Conv2d(2 * width, width, 3, padding=1)
        self.dns128 = CoarseDNS(width, hidden=width)
        self.dns256 = CoarseDNS(width, hidden=width)
        self.local128 = LocalScale(width, radius=2)
        self.local256 = LocalScale(width, radius=1)
        self.body = nn.Sequential(
            nn.Conv2d(2 * width + 4, hidden, 3, padding=1),
            nn.LeakyReLU(.1, inplace=False),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(.1, inplace=False))
        self.to_decoder_input = nn.Conv2d(hidden, corr_channels, 1)
        nn.init.zeros_(self.to_decoder_input.weight)
        nn.init.zeros_(self.to_decoder_input.bias)

    def _gather(self, feature128, feature256):
        low = self.project128(feature128)
        high = self.project256(feature256)
        fused128 = low + self.fuse128(torch.cat((low, F.avg_pool2d(high, 2)), 1))
        fused256 = high + self.fuse256(torch.cat((
            F.interpolate(low, size=high.shape[-2:], mode='bilinear',
                          align_corners=False), high), 1))
        return fused128, fused256

    def forward(self, target128, source128, target256, source256,
                current_flow512, decoder_channels):
        batch = target128.shape[0]
        if (target128.shape != (batch, 128, 128, 128) or
                source128.shape != target128.shape or
                target256.shape != (batch, 64, 256, 256) or
                source256.shape != target256.shape or
                current_flow512.shape != (batch, 2, 128, 128) or
                decoder_channels != self.corr_channels):
            raise ValueError('Expected 512-image 128/256 VGG grids and 512px flow')

        target128, target256 = self._gather(target128, target256)
        source128, source256 = self._gather(source128, source256)
        target128, source128 = self.dns128(target128, source128)
        target256, source256 = self.dns256(target256, source256)
        target128 = self.local128.self_attention(target128)
        source128 = self.local128.self_attention(source128)
        target256 = self.local256.self_attention(target256)
        source256 = self.local256.self_attention(source256)

        signal128, state128 = self.local128.cross_attention(
            target128, source128, current_flow512)
        flow256 = F.interpolate(current_flow512, (256, 256), mode='bilinear',
                                align_corners=False)
        signal256, state256 = self.local256.cross_attention(
            target256, source256, flow256)
        signal256 = F.avg_pool2d(signal256, 2)
        valid128 = state128['valid_any']
        valid256 = F.max_pool2d(state256['valid_any'][:, None].float(), 2)[:, 0].bool()
        features = torch.cat((signal128, signal256, current_flow512 / 512,
                              valid128[:, None].to(signal128.dtype),
                              valid256[:, None].to(signal128.dtype)), 1)
        # The frozen pretrained decoder sees exactly its old input at step 0.
        return self.to_decoder_input(self.body(features)) * (valid128 & valid256)[:, None]
