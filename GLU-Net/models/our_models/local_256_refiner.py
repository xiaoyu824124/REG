"""One 256-grid residual after the frozen GLU-Net 128-grid prediction.

Inputs and outputs are target-visible -> source-infrared XY flows in pixels of
the 512x512 image. VGG level_1 supplies the existing 256x256 feature grid.
"""

import torch
from torch import nn
import torch.nn.functional as F


class Local256Residual(nn.Module):
    def __init__(self, channels=64, radius=1, hidden=32):
        super().__init__()
        if radius != 1:
            raise ValueError("This controlled experiment fixes a 3x3 search")
        self.channels = channels
        self.radius = radius
        candidates = (2 * radius + 1) ** 2
        self.body = nn.Sequential(
            nn.Conv2d(2 * candidates + 2, hidden, 3, padding=1),
            nn.LeakyReLU(.1, inplace=False),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(.1, inplace=False),
        )
        self.head = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def local_correlation(target, source, flow_512px, radius=1):
        """Sample IR at visible xy + flow/2 + local offset on the 256 grid."""
        batch, channels, height, width = target.shape
        if (source.shape != target.shape or
                flow_512px.shape != (batch, 2, height, width) or
                (height, width) != (256, 256)):
            raise ValueError("Expected paired 256-grid features and XY flow")
        target = F.normalize(target, dim=1, eps=1e-6)
        source = F.normalize(source, dim=1, eps=1e-6)
        y, x = torch.meshgrid(
            torch.arange(height, device=target.device, dtype=target.dtype),
            torch.arange(width, device=target.device, dtype=target.dtype),
            indexing="ij")
        center_x = x[None] + flow_512px[:, 0] * (width / 512.0)
        center_y = y[None] + flow_512px[:, 1] * (height / 512.0)
        scores, masks = [], []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sx, sy = center_x + dx, center_y + dy
                valid = ((sx >= 0) & (sx <= width - 1) &
                         (sy >= 0) & (sy <= height - 1))
                grid = torch.stack((2 * sx / (width - 1) - 1,
                                    2 * sy / (height - 1) - 1), dim=-1)
                sampled = F.grid_sample(source, grid, align_corners=True)
                scores.append((target * sampled).sum(dim=1, keepdim=True) *
                              valid[:, None])
                masks.append(valid[:, None].to(target.dtype))
        return torch.cat(scores, dim=1), torch.cat(masks, dim=1)

    def forward(self, target256, source256, frozen_flow128_512px):
        if (target256.shape != source256.shape or
                target256.shape[1:] != (self.channels, 256, 256) or
                frozen_flow128_512px.shape !=
                (target256.shape[0], 2, 128, 128)):
            raise ValueError("Unexpected frozen feature or flow dimensions")
        # Interpolation changes the grid, not the image-pixel displacement.
        center256 = F.interpolate(frozen_flow128_512px, (256, 256),
                                  mode="bilinear", align_corners=False)
        with torch.no_grad():
            correlation, candidate_valid = self.local_correlation(
                target256.detach(), source256.detach(), center256.detach(),
                self.radius)
        signal = torch.cat((correlation, candidate_valid,
                            center256.detach() / 512.0), dim=1)
        delta_grid = torch.tanh(self.head(self.body(signal)))
        delta_512px = torch.cat((delta_grid[:, :1] * (512.0 / 256),
                                delta_grid[:, 1:] * (512.0 / 256)), dim=1)
        refined256 = center256 + delta_512px
        # One-step 128->512 bilinear is the existing evaluated baseline. Adding
        # only the new residual preserves it bit-for-bit at initialization.
        baseline512 = F.interpolate(frozen_flow128_512px, (512, 512),
                                    mode="bilinear", align_corners=False)
        refined512 = baseline512 + F.interpolate(
            delta_512px, (512, 512), mode="bilinear", align_corners=False)
        return refined512, refined256, delta_512px
