"""Fixed multi-scale phase-congruency maps and a coarse feature residual.

This is a small ACFR-Net-inspired structural prior, not the paper's PCEM or
its affine estimator. Eight unoriented-axis log-Gabor quadrature banks each
produce one continuous phase-congruency map. All filter choices are fixed
before inspecting validation results.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .mind import rgb_from_glu_input


class FixedPhaseCongruency2D(nn.Module):
    """Return BCHW phase-congruency maps in [0, 1], one per orientation.

    Input is RGB or grayscale in [0, 1] at 256x256. Reflection padding reduces
    periodic FFT border artifacts. For each orientation, PC is positive local
    quadrature energy after a fixed noise estimate, divided by summed response
    amplitude over scales. The threshold is 1.5 times the spatial median of
    the finest-scale amplitude, computed separately per image and orientation.
    """

    def __init__(self, image_size=256, padding=32, scales=3, orientations=8):
        super().__init__()
        if image_size <= padding or scales < 2 or orientations < 2:
            raise ValueError("Invalid phase-congruency filter bank")
        self.image_size = image_size
        self.padding = padding
        self.scales = scales
        self.orientations = orientations
        side = image_size + 2 * padding
        fy = torch.fft.fftfreq(side)[:, None]
        fx = torch.fft.fftfreq(side)[None, :]
        radius = torch.sqrt(fx.square() + fy.square())
        radius_safe = radius.clamp_min(1 / side)
        angle = torch.atan2(fy, fx)
        banks = []
        odd_banks = []
        for orientation in range(orientations):
            theta = orientation * math.pi / orientations
            # The angular envelope is pi-periodic, so the even kernel is
            # Hermitian. The signed Riesz projection supplies its odd mate.
            delta = .5 * torch.atan2(torch.sin(2 * (angle - theta)),
                                      torch.cos(2 * (angle - theta)))
            angular = torch.exp(-delta.square() /
                                (2 * (math.pi / orientations / 1.5) ** 2))
            signed_projection = (fx * math.cos(theta) + fy * math.sin(theta)) / radius_safe
            for scale in range(scales):
                centre_frequency = 1 / (3 * 2 ** scale)
                radial = torch.exp(-torch.log(radius_safe / centre_frequency).square() /
                                   (2 * math.log(.55) ** 2))
                filt = radial * angular
                filt[0, 0] = 0
                banks.append(filt)
                odd_banks.append((-1j * signed_projection * filt).to(torch.complex64))
        self.register_buffer("even_filters", torch.stack(banks), persistent=False)
        self.register_buffer("odd_filters", torch.stack(odd_banks), persistent=False)

    @torch.no_grad()
    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] not in (1, 3) or rgb.shape[-2:] != (
                self.image_size, self.image_size):
            raise ValueError("Phase congruency expects Bx(1 or 3)x256x256 images")
        rgb = rgb.float()
        gray = (rgb * rgb.new_tensor((.299, .587, .114))[None, :, None, None]).sum(
            dim=1, keepdim=True) if rgb.shape[1] == 3 else rgb
        gray = gray - gray.mean(dim=(-1, -2), keepdim=True)
        gray = F.pad(gray, (self.padding,) * 4, mode="reflect")[:, 0]
        spectrum = torch.fft.fft2(gray)
        even = torch.fft.ifft2(spectrum[:, None] * self.even_filters[None]).real
        odd = torch.fft.ifft2(spectrum[:, None] * self.odd_filters[None]).real
        batch, _, side, _ = even.shape
        even = even.reshape(batch, self.orientations, self.scales, side, side)
        odd = odd.reshape(batch, self.orientations, self.scales, side, side)
        amplitude = torch.sqrt(even.square() + odd.square() + 1e-12)
        local_energy = torch.sqrt(even.sum(dim=2).square() +
                                  odd.sum(dim=2).square() + 1e-12)
        finest = amplitude[:, :, 0].flatten(2)
        noise_floor = 1.5 * finest.median(dim=-1).values[..., None, None]
        congruency = ((local_energy - noise_floor).clamp_min(0) /
                      amplitude.sum(dim=2).clamp_min(1e-6))
        crop = self.padding
        return congruency[..., crop:-crop, crop:-crop].clamp(0, 1)


class CoarsePhaseGuide(nn.Module):
    """Zero-gated residual applied to each original 16x16 encoder feature."""

    def __init__(self, channels=512):
        super().__init__()
        self.descriptor = FixedPhaseCongruency2D()
        # Identical projection width and gate form to MIND route B.
        self.feature_projection = nn.Sequential(nn.Conv2d(8, 64, 1),
                                                nn.LeakyReLU(.1),
                                                nn.Conv2d(64, channels, 1))
        self.gate = nn.Parameter(torch.zeros(()))

    def maps(self, normalized_image):
        return self.descriptor(rgb_from_glu_input(normalized_image))

    def residual(self, feature, normalized_image):
        phase = F.interpolate(self.maps(normalized_image),
                              size=feature.shape[-2:], mode="area")
        return feature + self.gate.tanh() * self.feature_projection(2 * phase - 1)
