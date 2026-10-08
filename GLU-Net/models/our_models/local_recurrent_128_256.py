"""Independent 128/256-grid recurrent residual after frozen GLU-Net.

The target is VI, the sampled source is IR. Every flow tensor stores XY
displacements in 512-image-pixel units. At grid N, sampling moves by N/512
times that displacement. Only local 5x5/3x3 attention is constructed.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .coarse_dns import CoarseDNS


def sample_window(source, flow512, radius):
    """Sample IR at target xy + flow + local offsets; return candidates/mask."""
    batch, channels, height, width = source.shape
    if flow512.shape != (batch, 2, height, width):
        raise ValueError("Flow and source feature grid must agree")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing="ij")
    center_x = xx[None] + flow512[:, 0] * width / 512.
    center_y = yy[None] + flow512[:, 1] * height / 512.
    offsets = [(dy, dx) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)]
    grids, masks = [], []
    for dy, dx in offsets:
        sx, sy = center_x + dx, center_y + dy
        masks.append((sx >= 0) & (sx <= width - 1) &
                     (sy >= 0) & (sy <= height - 1))
        grids.append(torch.stack((2 * sx / (width - 1) - 1,
                                  2 * sy / (height - 1) - 1), dim=-1))
    k = len(offsets)
    grid = torch.stack(grids, dim=1).reshape(batch * k, height, width, 2)
    repeated = source[:, None].expand(batch, k, channels, height, width)
    repeated = repeated.reshape(batch * k, channels, height, width)
    sampled = F.grid_sample(repeated, grid, align_corners=True)
    sampled = sampled.reshape(batch, k, channels, height, width)
    candidates = sampled.permute(0, 2, 1, 3, 4).contiguous()
    valid = torch.stack(masks, dim=1)
    return candidates, valid


def masked_local_attention(query, candidates, valid):
    """Bounded K-candidate attention, safe when all candidates are invalid."""
    q = F.normalize(query, dim=1, eps=1e-6)
    c = F.normalize(candidates, dim=1, eps=1e-6)
    scores = (q[:, :, None] * c).sum(dim=1) * 4.
    scores = scores.masked_fill(~valid, -1e4)
    weights = F.softmax(scores, dim=1) * valid.to(scores.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    context = (candidates * weights[:, None]).sum(dim=2)
    return context, scores.masked_fill(~valid, 0), valid.any(dim=1)


class LocalScale(nn.Module):
    def __init__(self, width, radius):
        super().__init__()
        self.radius = radius
        self.compress = nn.Sequential(
            nn.Conv2d(width * 4 + 3, width, 1),
            nn.LeakyReLU(.1, inplace=False))

    def self_attention(self, features):
        batch, channels, height, width = features.shape
        k = (2 * self.radius + 1) ** 2
        gathered = F.unfold(features, 2 * self.radius + 1,
                            padding=self.radius).reshape(
            batch, channels, k, height, width)
        ones = torch.ones(batch, 1, height, width,
                          device=features.device, dtype=features.dtype)
        self_valid = F.unfold(ones, 2 * self.radius + 1,
                              padding=self.radius).reshape(
                                  batch, k, height, width).bool()
        context, _, _ = masked_local_attention(features, gathered, self_valid)
        return context

    def cross_attention(self, target_context, source_context, flow512):
        k = (2 * self.radius + 1) ** 2
        source_window, source_valid = sample_window(source_context, flow512,
                                                     self.radius)
        aligned_context, scores, any_valid = masked_local_attention(
            target_context, source_window, source_valid)
        discrepancy = (target_context - aligned_context).abs()
        center_valid = source_valid[:, k // 2]
        center_correlation = scores[:, k // 2] * center_valid
        signal = self.compress(torch.cat((target_context, aligned_context,
                                          discrepancy,
                                          target_context - aligned_context,
                                          scores.max(dim=1).values[:, None],
                                          center_correlation[:, None],
                                          any_valid[:, None].to(target_context.dtype)),
                                         dim=1))
        return signal, {"valid_any": any_valid,
                        "valid_center": center_valid,
                        "correlation_peak": scores.max(dim=1).values,
                        "discrepancy_mean": discrepancy.mean(dim=1)}


class LocalRecurrent128256(nn.Module):
    """Shared update CNN applied once or twice at the 256 grid."""

    def __init__(self, rounds=2, width=16, hidden=32, use_dns=True):
        super().__init__()
        if rounds not in (1, 2):
            raise ValueError("Only one or two shared rounds are controlled")
        self.rounds = rounds
        self.use_dns = bool(use_dns)
        self.project128 = nn.Conv2d(128, width, 1)
        self.project256 = nn.Conv2d(64, width, 1)
        self.fuse128 = nn.Conv2d(width * 2, width, 3, padding=1)
        self.fuse256 = nn.Conv2d(width * 2, width, 3, padding=1)
        self.dns128 = CoarseDNS(width, hidden=width) if use_dns else None
        self.dns256 = CoarseDNS(width, hidden=width) if use_dns else None
        self.scale128 = LocalScale(width, radius=2)  # 5x5 at 128
        self.scale256 = LocalScale(width, radius=1)   # 3x3 at 256
        self.update_body = nn.Sequential(
            nn.Conv2d(2 * width + 4, hidden, 3, padding=1),
            nn.LeakyReLU(.1, inplace=False),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(.1, inplace=False))
        self.update_head = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.zeros_(self.update_head.weight)
        nn.init.zeros_(self.update_head.bias)

    def gather_one_modality(self, features128, features256):
        """Fuse 1/4 and 1/2 VGG features within one modality only."""
        low = self.project128(features128)
        high = self.project256(features256)
        low_from_high = F.avg_pool2d(high, 2)
        high_from_low = F.interpolate(low, (256, 256), mode="bilinear",
                                      align_corners=False)
        fused128 = low + self.fuse128(torch.cat((low, low_from_high), dim=1))
        fused256 = high + self.fuse256(torch.cat((high_from_low, high), dim=1))
        return fused128, fused256

    def prepare_features(self, target128, source128, target256, source256):
        t128, t256 = self.gather_one_modality(target128, target256)
        s128, s256 = self.gather_one_modality(source128, source256)
        if self.use_dns:
            t128, s128 = self.dns128(t128, s128)
            t256, s256 = self.dns256(t256, s256)
        return ((self.scale128.self_attention(t128),
                 self.scale128.self_attention(s128)),
                (self.scale256.self_attention(t256),
                 self.scale256.self_attention(s256)))

    def one_round(self, prepared128, prepared256, current256_512px):
        current128 = F.interpolate(current256_512px, (128, 128),
                                   mode="bilinear", align_corners=False)
        signal128, state128 = self.scale128.cross_attention(
            *prepared128, current128)
        signal256, state256 = self.scale256.cross_attention(
            *prepared256, current256_512px)
        signal128 = F.interpolate(signal128, (256, 256), mode="bilinear",
                                  align_corners=False)
        valid128 = F.interpolate(state128["valid_any"][:, None].float(),
                                 (256, 256), mode="nearest")[:, 0].bool()
        update_valid = valid128 & state256["valid_any"]
        state = torch.cat((signal128, signal256,
                           current256_512px / 512.,
                           valid128[:, None].to(signal128.dtype),
                           state256["valid_any"][:, None].to(signal128.dtype)),
                          dim=1)
        # No tiny learned-displacement cap. Head is zero initialized and
        # gradients are clipped by the trainer for stability.
        delta512 = self.update_head(self.update_body(state)) * update_valid[:, None]
        return current256_512px + delta512, {
            "delta512": delta512, "update_valid": update_valid,
            "valid128_center": state128["valid_center"],
            "valid256_center": state256["valid_center"],
            "correlation128_peak": state128["correlation_peak"],
            "correlation256_peak": state256["correlation_peak"],
            "discrepancy128": state128["discrepancy_mean"],
            "discrepancy256": state256["discrepancy_mean"]}

    def forward(self, target128, source128, target256, source256,
                frozen_flow128_512px):
        batch = target128.shape[0]
        if (target128.shape != (batch, 128, 128, 128) or
                source128.shape != target128.shape or
                target256.shape != (batch, 64, 256, 256) or
                source256.shape != target256.shape or
                frozen_flow128_512px.shape != (batch, 2, 128, 128)):
            raise ValueError("Expected full-image VGG level_2/level_1 grids")
        initial256 = F.interpolate(frozen_flow128_512px, (256, 256),
                                   mode="bilinear", align_corners=False)
        initial512 = F.interpolate(frozen_flow128_512px, (512, 512),
                                   mode="bilinear", align_corners=False)
        prepared128, prepared256 = self.prepare_features(
            target128, source128, target256, source256)
        flow256 = initial256
        outputs = [initial512]
        diagnostics = []
        for _ in range(self.rounds):
            # Target/source SA is independent of flow; every recurrent round
            # rewarps source, remasks candidates and recomputes CA/differences.
            flow256, info = self.one_round(prepared128, prepared256, flow256)
            correction512 = F.interpolate(flow256 - initial256, (512, 512),
                                          mode="bilinear", align_corners=False)
            outputs.append(initial512 + correction512)
            diagnostics.append(info)
        return {"flows512": outputs, "flow256": flow256,
                "round_diagnostics": diagnostics}
