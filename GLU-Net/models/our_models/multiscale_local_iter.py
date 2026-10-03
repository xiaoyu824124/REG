"""Optional multi-scale, recurrent local flow residual outside frozen GLU-Net.

All flows use 512-image-pixel units. The two feature grids are 64x64 and
128x128 for a 512x512 input; their displacements for grid_sample are scaled
by grid_size/512. Target is visible, source is infrared.
"""

import torch
from torch import nn
import torch.nn.functional as F


def warp_source_to_target(source_feature, target_to_source_flow_512px):
    """Sample source at target coordinate + flow, with align_corners=True."""
    batch, _, height, width = source_feature.shape
    if target_to_source_flow_512px.shape != (batch, 2, height, width):
        raise ValueError("Flow must be XY and on the source feature grid")
    y, x = torch.meshgrid(torch.arange(height, device=source_feature.device),
                          torch.arange(width, device=source_feature.device),
                          indexing="ij")
    sample_x = x[None] + target_to_source_flow_512px[:, 0] * width / 512.0
    sample_y = y[None] + target_to_source_flow_512px[:, 1] * height / 512.0
    valid = ((sample_x >= 0) & (sample_x <= width - 1) &
             (sample_y >= 0) & (sample_y <= height - 1))[:, None]
    grid = torch.stack((2 * sample_x / (width - 1) - 1,
                        2 * sample_y / (height - 1) - 1), dim=-1)
    warped = F.grid_sample(source_feature, grid, align_corners=True)
    return warped, valid


class MultiScaleLocalIter(nn.Module):
    """Two shared update rounds at each of 64 and 128 feature grids."""

    def __init__(self, channels64=256, channels128=128, width=32,
                 hidden=64, rounds=2, max_delta_grid=1.0):
        super().__init__()
        if rounds != 2:
            raise ValueError("The controlled experiment uses exactly two rounds")
        self.rounds = rounds
        self.max_delta_grid = float(max_delta_grid)
        self.project64 = nn.Conv2d(channels64, width, 1)
        self.project128 = nn.Conv2d(channels128, width, 1)
        self.fuse = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1),
                                  nn.LeakyReLU(.1, inplace=False))
        # Shared by both scales and both rounds. The final head is zero, so
        # every flow initially equals its frozen GLU-Net counterpart.
        self.update_body = nn.Sequential(
            nn.Conv2d(3 * width + 3, hidden, 3, padding=1),
            nn.LeakyReLU(.1, inplace=False),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(.1, inplace=False))
        self.update_head = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.zeros_(self.update_head.weight)
        nn.init.zeros_(self.update_head.bias)

    def fused_pair(self, target64, source64, target128, source128):
        if target64.shape[-2:] != (64, 64) or target128.shape[-2:] != (128, 128):
            raise ValueError("Expected actual full-image VGG grids 64 and 128")
        if (target64.shape[1] != self.project64.in_channels or
                target128.shape[1] != self.project128.in_channels):
            raise ValueError("Unexpected VGG feature channel count")
        target64 = self.project64(target64)
        source64 = self.project64(source64)
        target128 = self.project128(target128)
        source128 = self.project128(source128)

        def fuse(low, high):
            at64 = self.fuse(torch.cat((low, F.interpolate(
                high, size=(64, 64), mode="bilinear", align_corners=False)), 1))
            at128 = self.fuse(torch.cat((F.interpolate(
                low, size=(128, 128), mode="bilinear", align_corners=False),
                high), 1))
            return at64, at128

        target64, target128 = fuse(target64, target128)
        source64, source128 = fuse(source64, source128)
        return (target64, source64), (target128, source128)

    def one_update(self, target, source, flow_512px):
        height, width = target.shape[-2:]
        warped, valid = warp_source_to_target(source, flow_512px)
        validity = valid.to(target.dtype)
        grid_flow = torch.cat((flow_512px[:, :1] * width / 512.0,
                               flow_512px[:, 1:] * height / 512.0), dim=1)
        discrepancy = (target - warped).abs() * validity
        signal = torch.cat((target * validity, warped * validity,
                            discrepancy, grid_flow, validity), dim=1)
        delta_grid = self.max_delta_grid * torch.tanh(
            self.update_head(self.update_body(signal)))
        delta_512px = torch.cat((delta_grid[:, :1] * 512.0 / width,
                                  delta_grid[:, 1:] * 512.0 / height), dim=1)
        return flow_512px + delta_512px, {
            "valid_fraction": valid.float().mean(),
            "delta_512px": delta_512px}

    def forward(self, target64, source64, target128, source128,
                frozen_flow64_512px, frozen_flow128_512px):
        if (frozen_flow64_512px.shape[-2:] != (64, 64) or
                frozen_flow128_512px.shape[-2:] != (128, 128)):
            raise ValueError("Frozen flows must use 64/128 feature grids")
        (target64, source64), (target128, source128) = self.fused_pair(
            target64, source64, target128, source128)
        flow64 = frozen_flow64_512px
        iterations64, iterations128, diagnostics = [], [], []
        for _ in range(self.rounds):
            flow64, info = self.one_update(target64, source64, flow64)
            iterations64.append(flow64)
            diagnostics.append(info)
        # The 64-grid branch corrects the already-computed frozen 128-grid
        # flow. At zero initialization this is exactly the original flow128.
        correction64 = flow64 - frozen_flow64_512px
        flow128 = frozen_flow128_512px + F.interpolate(
            correction64, size=(128, 128), mode="bilinear",
            align_corners=False)
        for _ in range(self.rounds):
            flow128, info = self.one_update(target128, source128, flow128)
            iterations128.append(flow128)
            diagnostics.append(info)
        final512 = F.interpolate(flow128, size=(512, 512),
                                 mode="bilinear", align_corners=False)
        return {"flow64_rounds": iterations64,
                "flow128_rounds": iterations128,
                "final_512px": final512,
                "diagnostics": diagnostics}
