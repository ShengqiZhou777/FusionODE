from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        morph_dim: int,
        cnn_dim: int,
        out_dim: int,
        attn_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.morph_proj = nn.Linear(morph_dim, attn_dim)
        self.cnn_proj = nn.Linear(cnn_dim, attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(attn_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(attn_dim, out_dim),
            nn.Dropout(proj_dropout) if proj_dropout and proj_dropout > 0 else nn.Identity(),
        )

    def forward(self, morph: torch.Tensor, cnn: torch.Tensor) -> torch.Tensor:
        """
        morph: [B, W, Dm]
        cnn:   [B, W, Dc]
        return: [B, W, out_dim]
        """
        q = self.morph_proj(morph)
        k = self.cnn_proj(cnn)
        v = k
        attn_out, _ = self.attn(q, k, v)
        fused = self.norm(attn_out + q)
        return self.out_proj(fused)


class ConcatFusion(nn.Module):
    def __init__(self, morph_dim: int, cnn_dim: int) -> None:
        super().__init__()
        self.out_dim = morph_dim + cnn_dim

    def forward(self, morph: torch.Tensor, cnn: torch.Tensor) -> torch.Tensor:
        return torch.cat([morph, cnn], dim=-1)
