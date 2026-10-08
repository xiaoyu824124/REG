"""One bounded 256-grid update after a frozen GLU-Net 128-grid flow.

Target is visible and source is infrared. All stored flows use pixels of the
512x512 image; sampling on grid N uses N/512 times the flow. This module uses
3x3 local self/cross attention, never full-image attention at 256x256.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .coarse_dns import CoarseDNS


class LocalScaleAttention(nn.Module):
    def __init__(self, in_channels, width=16):
        super().__init__()
        self.project = nn.Conv2d(in_channels, width, 1)
        self.mix = nn.Sequential(nn.Conv2d(width * 4 + 2, width, 1),
                                 nn.LeakyReLU(.1, inplace=False))
        self.width = width

    @staticmethod
    def sample_neighbours(source, flow512):
        batch, _, height, width = source.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=source.device, dtype=source.dtype),
            torch.arange(width, device=source.device, dtype=source.dtype),
            indexing="ij")
        x = xx[None] + flow512[:, 0] * (width / 512.)
        y = yy[None] + flow512[:, 1] * (height / 512.)
        samples, valid = [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sx, sy = x + dx, y + dy
                good = (sx >= 0) & (sx <= width - 1) & (sy >= 0) & (sy <= height - 1)
                grid = torch.stack((2 * sx / (width - 1) - 1,
                                    2 * sy / (height - 1) - 1), dim=-1)
                samples.append(F.grid_sample(source, grid, align_corners=True))
                valid.append(good)
        return torch.stack(samples, dim=2), torch.stack(valid, dim=1)

    @staticmethod
    def attend(query, candidates, valid):
        scores = (query[:, :, None] * candidates).sum(dim=1) / query.shape[1] ** .5
        scores = scores.masked_fill(~valid, -1e4)
        weights = F.softmax(scores, dim=1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (candidates * weights[:, None]).sum(dim=2)

    def forward(self, target, source, flow512):
        target = self.project(target)
        source = self.project(source)
        batch, channels, height, width = target.shape
        # F.unfold is strictly local and avoids an N-by-N attention matrix.
        self_candidates = F.unfold(target, 3, padding=1).reshape(
            batch, channels, 9, height, width)
        self_valid = torch.ones(batch, 9, height, width,
                                device=target.device, dtype=torch.bool)
        self_context = self.attend(target, self_candidates, self_valid)
        cross_candidates, cross_valid = self.sample_neighbours(source, flow512)
        cross_context = self.attend(self_context, cross_candidates, cross_valid)
        difference = (self_context - cross_context).abs()
        signal = torch.cat((target, self_context, cross_context, difference,
                            cross_valid.any(dim=1, keepdim=True).to(target.dtype),
                            cross_valid[:, 4:5].to(target.dtype)), dim=1)
        return self.mix(signal), cross_valid[:, 4]


class LocalDNSAttentionFusion(nn.Module):
    def __init__(self, use_dns=True, use_attention=True, use_cross_scale=True):
        super().__init__()
        self.use_dns = use_dns
        self.use_attention = use_attention
        self.use_cross_scale = use_cross_scale
        self.dns128 = CoarseDNS(128, hidden=32) if use_dns else None
        self.dns256 = CoarseDNS(64, hidden=16) if use_dns else None
        self.scale128 = LocalScaleAttention(128)
        self.scale256 = LocalScaleAttention(64)
        # The no-attention ablation retains the same 3x3 aligned sampling.
        self.plain128 = nn.Conv2d(128, 16, 1)
        self.plain256 = nn.Conv2d(64, 16, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(16 * (2 if use_cross_scale else 1) + 2, 32, 3, padding=1),
            nn.LeakyReLU(.1, inplace=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(.1, inplace=False))
        self.head = nn.Conv2d(32, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def plain_signal(project, target, source, flow):
        target, source = project(target), project(source)
        candidates, valid = LocalScaleAttention.sample_neighbours(source, flow)
        aligned = candidates[:, :, 4]
        return target - aligned, valid[:, 4]

    def forward(self, target128, source128, target256, source256, flow128_512px):
        batch = target128.shape[0]
        if (target128.shape != (batch, 128, 128, 128) or
                source128.shape != target128.shape or
                target256.shape != (batch, 64, 256, 256) or
                source256.shape != target256.shape or
                flow128_512px.shape != (batch, 2, 128, 128)):
            raise ValueError("Expected paired 128/256 VGG grids and 128-grid XY flow")
        center256 = F.interpolate(flow128_512px, (256, 256),
                                  mode="bilinear", align_corners=False)
        if self.use_dns:
            target128, source128 = self.dns128(target128, source128)
            target256, source256 = self.dns256(target256, source256)
        if self.use_attention:
            signal256, valid256 = self.scale256(target256, source256, center256)
        else:
            signal256, valid256 = self.plain_signal(
                self.plain256, target256, source256, center256)
        if self.use_cross_scale:
            if self.use_attention:
                signal128, valid128 = self.scale128(target128, source128,
                                                    flow128_512px)
            else:
                signal128, valid128 = self.plain_signal(
                    self.plain128, target128, source128, flow128_512px)
            signal128 = F.interpolate(signal128, (256, 256),
                                      mode="bilinear", align_corners=False)
            valid128 = F.interpolate(valid128[:, None].float(), (256, 256),
                                     mode="nearest")[:, 0].bool()
            features = (signal256, signal128)
            valid = valid256 & valid128
        else:
            features = (signal256,)
            valid = valid256
        # Invalid source samples must not drive a residual.
        body = self.fuse(torch.cat((*features, center256 / 512.), dim=1))
        delta512 = 2. * torch.tanh(self.head(body)) * valid[:, None]
        baseline512 = F.interpolate(flow128_512px, (512, 512),
                                    mode="bilinear", align_corners=False)
        final512 = baseline512 + F.interpolate(delta512, (512, 512),
                                               mode="bilinear", align_corners=False)
        return final512, center256 + delta512, delta512, valid
