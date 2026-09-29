"""CRFT-inspired self/cross feature interaction at GLU-Net's 16x16 level."""

from torch import nn
import torch


class CoarseSelfCrossAttention(nn.Module):
    """Shared self attention followed by simultaneous bidirectional cross attention.

    The zero initialized gate makes a newly enabled module reproduce the original
    GLU-Net correlation before it is trained.
    """

    def __init__(self, channels=512, heads=8):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.self_norm = nn.LayerNorm(channels)
        self.cross_norm = nn.LayerNorm(channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.self_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, target, source):
        if target.shape != source.shape:
            raise ValueError("coarse target and source features must have equal shapes")
        batch, channels, height, width = target.shape
        target_tokens = target.flatten(2).transpose(1, 2)
        source_tokens = source.flatten(2).transpose(1, 2)

        def self_step(tokens):
            normalized = self.self_norm(tokens)
            update, _ = self.self_attention(normalized, normalized, normalized,
                                             need_weights=False)
            return tokens + update

        target_self = self_step(target_tokens)
        source_self = self_step(source_tokens)
        target_query = self.cross_norm(target_self)
        source_query = self.cross_norm(source_self)
        target_update, _ = self.cross_attention(target_query, source_query,
                                                source_query, need_weights=False)
        source_update, _ = self.cross_attention(source_query, target_query,
                                                target_query, need_weights=False)
        target_interacted = target_self + target_update
        source_interacted = source_self + source_update
        target_interacted = target_interacted + self.ffn(self.ffn_norm(target_interacted))
        source_interacted = source_interacted + self.ffn(self.ffn_norm(source_interacted))

        target_out = target_tokens + self.gate * (target_interacted - target_tokens)
        source_out = source_tokens + self.gate * (source_interacted - source_tokens)
        return (target_out.transpose(1, 2).reshape(batch, channels, height, width),
                source_out.transpose(1, 2).reshape(batch, channels, height, width))
