"""Zero-initialized deformable residual flow update at a GLU-Net local grid.

The input flow is in image-pixel units; torchvision deformable-convolution
offsets and the bounded output displacement are in *feature-grid* pixels.
This is a GLU-Net experiment inspired by recurrent flow updates, not CRFT's
Spatial Geometric Transform or discrepancy-guided attention implementation.
"""

import torch
from torch import nn
from torchvision.ops import DeformConv2d


class LocalDeformableFlowUpdate(nn.Module):
    def __init__(self, correlation_channels=81, hidden=64,
                 max_offset_grid=2.0, max_delta_grid=4.0):
        super().__init__()
        # Local correlation, current flow in grid pixels, mean absolute feature
        # discrepancy and current source-sampling validity.
        channels = correlation_channels + 4
        self.max_offset_grid = float(max_offset_grid)
        self.max_delta_grid = float(max_delta_grid)
        self.pre = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1),
                                 nn.LeakyReLU(0.1, inplace=False))
        self.offset = nn.Conv2d(hidden, 18, 3, padding=1)
        self.deform = DeformConv2d(hidden, hidden, 3, padding=1)
        self.post = nn.Sequential(nn.LeakyReLU(0.1, inplace=False),
                                  nn.Conv2d(hidden, hidden, 3, padding=1),
                                  nn.LeakyReLU(0.1, inplace=False))
        self.delta = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, correlation, target_feature, warped_source_feature,
                flow_image_pixels, image_width, image_height):
        if correlation.shape[1] != 81 or flow_image_pixels.shape[1] != 2:
            raise ValueError("Expected 9x9 local correlation and XY flow")
        height, width = target_feature.shape[-2:]
        if any(t.shape[-2:] != (height, width) for t in
               (correlation, warped_source_feature, flow_image_pixels)):
            raise ValueError("DCN inputs must share one feature grid")
        scale_x = width / float(image_width)
        scale_y = height / float(image_height)
        flow_grid = torch.cat((flow_image_pixels[:, 0:1] * scale_x,
                               flow_image_pixels[:, 1:2] * scale_y), dim=1)
        yy, xx = torch.meshgrid(torch.arange(height, device=flow_grid.device),
                                torch.arange(width, device=flow_grid.device),
                                indexing="ij")
        mapped_x = xx[None] + flow_grid[:, 0]
        mapped_y = yy[None] + flow_grid[:, 1]
        inside = ((mapped_x >= 0) & (mapped_x <= width - 1) &
                  (mapped_y >= 0) & (mapped_y <= height - 1))
        discrepancy = (target_feature - warped_source_feature).abs().mean(
            dim=1, keepdim=True)
        signal = torch.cat((correlation, flow_grid, discrepancy,
                            inside[:, None].to(correlation.dtype)), dim=1)
        hidden = self.pre(signal)
        offset = self.max_offset_grid * torch.tanh(self.offset(hidden))
        transformed = self.post(self.deform(hidden, offset))
        delta_grid = self.max_delta_grid * torch.tanh(self.delta(transformed))
        delta_image = torch.cat((delta_grid[:, 0:1] / scale_x,
                                 delta_grid[:, 1:2] / scale_y), dim=1)
        return flow_image_pixels + delta_image, {"offset_grid": offset,
            "delta_grid": delta_grid, "source_inside": inside}
