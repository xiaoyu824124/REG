"""Small matched single-frame / three-frame infrared-visible fusion ablation.

The temporal arm changes only aligned neighbour features. Registration is
external and frozen; this module never sees registration GT or video test data.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def luminance(rgb):
    return .299 * rgb[:, 0:1] + .587 * rgb[:, 1:2] + .114 * rgb[:, 2:3]


def rgb_with_luminance(visible, new_y):
    old_y = luminance(visible)
    return (visible + new_y - old_y).clamp(0, 1)


class ImageEncoder(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1), nn.LeakyReLU(.1),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.LeakyReLU(.1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1), nn.LeakyReLU(.1))

    def forward(self, image):
        return self.layers(image)


def warp_feature(neighbour, current_to_neighbour, valid):
    """Backward warp on the current feature grid; flow is in feature pixels."""
    batch, _, height, width = neighbour.shape
    if current_to_neighbour.shape != (batch, 2, height, width):
        raise ValueError('temporal flow must use current feature-grid XY pixels')
    if valid.shape != (batch, 1, height, width):
        raise ValueError('temporal validity mask shape mismatch')
    yy, xx = torch.meshgrid(torch.arange(height, device=neighbour.device),
                            torch.arange(width, device=neighbour.device),
                            indexing='ij')
    xy = torch.stack((xx, yy)).to(neighbour.dtype)[None]
    mapped = xy + current_to_neighbour
    inside = ((mapped[:, 0:1] >= 0) & (mapped[:, 0:1] <= width - 1) &
              (mapped[:, 1:2] >= 0) & (mapped[:, 1:2] <= height - 1))
    grid = torch.stack((2 * mapped[:, 0] / (width - 1) - 1,
                        2 * mapped[:, 1] / (height - 1) - 1), dim=-1)
    aligned = F.grid_sample(neighbour, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    mask = (valid > .5) & inside
    return aligned * mask.to(aligned.dtype), mask.to(aligned.dtype)


class TemporalUpdate(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels + 2, channels, 3, padding=1), nn.LeakyReLU(.1),
            nn.Conv2d(channels, channels, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, current, before, after, flow_before, flow_after,
                mask_before, mask_after):
        previous, previous_valid = warp_feature(before, flow_before, mask_before)
        following, following_valid = warp_feature(after, flow_after, mask_after)
        support = previous_valid + following_valid
        context = ((previous - current * previous_valid) +
                   (following - current * following_valid)) / support.clamp_min(1.)
        delta = self.net(torch.cat((current, context,
                                    previous_valid, following_valid), dim=1))
        return current + delta * (support > 0).to(delta.dtype)


class MinimalVideoFusion(nn.Module):
    """A: current pair; B: same core plus aligned temporal feature updates."""

    def __init__(self, temporal=False, channels=16):
        super().__init__()
        self.temporal = bool(temporal)
        self.encoder_ir = ImageEncoder(channels)
        self.encoder_vi = ImageEncoder(channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1), nn.LeakyReLU(.1),
            nn.Conv2d(channels, 1, 3, padding=1))
        if self.temporal:
            self.temporal_ir = TemporalUpdate(channels)
            self.temporal_vi = TemporalUpdate(channels)

    def forward(self, ir, visible, neighbours=None):
        if ir.shape != visible[:, :1].shape or visible.shape[1] != 3:
            raise ValueError('expected registered IR [B,1,H,W] and VI [B,3,H,W]')
        vi_y = luminance(visible)
        ir_feature = self.encoder_ir(ir)
        vi_feature = self.encoder_vi(vi_y)
        if self.temporal:
            if neighbours is None:
                raise ValueError('three-frame arm requires aligned neighbour inputs')
            ir_feature = self.temporal_ir(
                ir_feature, self.encoder_ir(neighbours['ir_before']),
                self.encoder_ir(neighbours['ir_after']),
                neighbours['flow_ir_before'], neighbours['flow_ir_after'],
                neighbours['mask_ir_before'], neighbours['mask_ir_after'])
            vi_feature = self.temporal_vi(
                vi_feature, self.encoder_vi(luminance(neighbours['vi_before'])),
                self.encoder_vi(luminance(neighbours['vi_after'])),
                neighbours['flow_vi_before'], neighbours['flow_vi_after'],
                neighbours['mask_vi_before'], neighbours['mask_vi_after'])
        residual = self.decoder(torch.cat((ir_feature, vi_feature), dim=1))
        residual = F.interpolate(residual, ir.shape[-2:], mode='bilinear',
                                 align_corners=False)
        fused_y = (.5 * (ir + vi_y) + .3 * torch.tanh(residual)).clamp(0, 1)
        return fused_y, rgb_with_luminance(visible, fused_y)


class SpatialFusionLoss(nn.Module):
    """Identical IR/VI intensity, edge and hot-target terms for both arms."""

    def __init__(self):
        super().__init__()
        self.register_buffer('sx', torch.tensor([[-1., 0., 1.], [-2., 0., 2.],
                                                  [-1., 0., 1.]])[None, None] / 8.)
        self.register_buffer('sy', torch.tensor([[-1., -2., -1.], [0., 0., 0.],
                                                  [1., 2., 1.]])[None, None] / 8.)

    def gradient(self, image):
        dx = F.conv2d(image, self.sx, padding=1)
        dy = F.conv2d(image, self.sy, padding=1)
        return torch.sqrt(dx.square() + dy.square() + 1e-6)

    def forward(self, fused_y, ir, visible, valid):
        vi_y = luminance(visible)
        if valid.shape != ir.shape or not (valid > .5).any():
            raise ValueError('fusion loss needs a nonempty current registration mask')
        mask = valid > .5
        inner = F.avg_pool2d(valid.float(), 3, stride=1, padding=1) > .999
        intensity = (fused_y - torch.maximum(ir, vi_y)).abs()[mask].mean()
        wanted_edge = torch.maximum(self.gradient(ir), self.gradient(vi_y))
        edge = (self.gradient(fused_y) - wanted_edge).abs()[inner].mean()
        hot = ((ir > vi_y + .1) & (ir > .6) & mask)
        heat = F.relu(ir - fused_y)[hot].mean() if hot.any() else fused_y.sum() * 0.
        parts = {'intensity': intensity, 'edge': edge, 'heat': heat}
        return intensity + 2. * edge + .5 * heat, parts
