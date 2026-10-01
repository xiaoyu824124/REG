"""Fixed 2D MIND and two coarse-only adapters (not MIND-SSC or learned DNS).

Eight centre-to-neighbour patch SSDs; variance from the four axial neighbours.
Offsets are (dy, dx), ordered N,S,W,E,NW,NE,SW,SE. Patch weights are a fixed
3x3 Gaussian (sigma=1), offsets have radius 1, borders use replication.
M=exp(-(D-min_r D)/V), so max_r M=1. No parameters or dataset statistics are
learned by the descriptor. The original MIND paper is Heinrich et al., 2012,
doi:10.1016/j.media.2012.05.008; this is an explicit 2D neighbourhood choice.
"""

import torch
from torch import nn
import torch.nn.functional as F


class FixedMIND2D(nn.Module):
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1))

    def __init__(self):
        super().__init__()
        axis = torch.arange(-1, 2, dtype=torch.float32)
        kernel = torch.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / 2)
        kernel /= kernel.sum()
        self.register_buffer("kernel", kernel[None, None].repeat(8, 1, 1, 1))

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] not in (1, 3):
            raise ValueError("MIND expects BCHW grayscale or RGB in [0,1]")
        rgb = rgb.float()
        gray = (rgb * rgb.new_tensor((0.299, 0.587, 0.114))[None, :, None, None]).sum(
            dim=1, keepdim=True) if rgb.shape[1] == 3 else rgb
        height, width = gray.shape[-2:]
        padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        shifted = torch.cat([padded[..., 1 + dy:1 + dy + height,
                                    1 + dx:1 + dx + width]
                             for dy, dx in self.offsets], dim=1)
        distance = F.conv2d(F.pad((gray - shifted).square(), (1, 1, 1, 1),
                                 mode="replicate"), self.kernel, groups=8)
        variance = distance[:, :4].mean(dim=1, keepdim=True)
        # Per-image relative clamp: independent of batch composition and modality.
        mean_variance = variance.mean(dim=(-1, -2), keepdim=True)
        variance = torch.minimum(torch.maximum(variance, mean_variance * 0.001),
                                 mean_variance * 1000).clamp_min(1e-12)
        return torch.exp(-(distance - distance.amin(dim=1, keepdim=True)) / variance)


def rgb_from_glu_input(image):
    mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return image * std + mean


class CoarseMIND(nn.Module):
    """A: MIND->3-channel adapter->frozen encoder; B: gated feature residual.

    Both compute MIND on the same quantized 256px RGB input. B area-pools the
    descriptor to the actual encoder coarse grid; descriptor values are not
    flow vectors and are never multiplied by a resolution ratio.
    """
    def __init__(self, route, channels=512):
        super().__init__()
        if route not in ("a", "b"):
            raise ValueError("MIND route must be a or b")
        self.route = route
        self.descriptor = FixedMIND2D()
        if route == "a":
            self.input_projection = nn.Conv2d(8, 3, 1)
            nn.init.constant_(self.input_projection.weight, 1 / 8)
            nn.init.zeros_(self.input_projection.bias)
        else:
            self.feature_projection = nn.Sequential(nn.Conv2d(8, 64, 1),
                nn.LeakyReLU(0.1), nn.Conv2d(64, channels, 1))
            self.gate = nn.Parameter(torch.zeros(()))

    def encode_input(self, normalized_image, pyramid):
        mind = self.descriptor(rgb_from_glu_input(normalized_image))
        pseudo_rgb = self.input_projection(mind)
        mean = pseudo_rgb.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = pseudo_rgb.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        # Frozen encoder parameters still permit gradients to the input adapter.
        return pyramid((pseudo_rgb - mean) / std)[-3]

    def residual(self, feature, normalized_image):
        mind = self.descriptor(rgb_from_glu_input(normalized_image))
        mind = F.interpolate(mind, size=feature.shape[-2:], mode="area")
        return feature + self.gate.tanh() * self.feature_projection(2 * mind - 1)
