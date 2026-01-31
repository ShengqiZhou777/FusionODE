# src/models/encoders.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphMLP(nn.Module):
    """
    Morphological statistics encoder.
    Input : morph [B, W, Dm]
    Output: z_morph [B, W, Zm]
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 64,
        hidden_dims: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
        layernorm: bool = True,
    ):
        super().__init__()
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(out_dim) if layernorm else nn.Identity()

    def forward(self, morph: torch.Tensor) -> torch.Tensor:
        # morph: [B, W, Dm]
        B, W, Dm = morph.shape
        x = morph.reshape(B * W, Dm)
        z = self.net(x)
        z = self.ln(z)
        return z.reshape(B, W, -1)


class AttentionMIL(nn.Module):
    """
    Attention-based MIL pooling (Ilse et al. style).
    It aggregates a bag of instances into a single vector.

    Input:
      bags: [B, W, N, Din]
      mask: [B, W, N] (bool) True=valid, False=padding
    Output:
      z: [B, W, Dout]
      attn (optional): [B, W, N] attention weights (masked)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 64,
        attn_hidden: int = 128,
        dropout: float = 0.0,
        gated: bool = True,
        return_attention: bool = False,
        layernorm: bool = True,
    ):
        super().__init__()
        self.return_attention = return_attention
        self.gated = gated
        
        # input norm
        self.ln = nn.LayerNorm(in_dim) if layernorm else nn.Identity()

        # instance embedding before pooling
        self.phi = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

        # attention network
        self.attn_V = nn.Linear(out_dim, attn_hidden)
        self.attn_U = nn.Linear(out_dim, attn_hidden) if gated else None
        self.attn_w = nn.Linear(attn_hidden, 1)

    def forward(self, bags: torch.Tensor, mask: torch.Tensor):
        """
        bags: [B,W,N,Din]
        mask: [B,W,N] bool
        """
        if bags.dim() != 4:
            raise ValueError(f"bags must be [B,W,N,D], got {tuple(bags.shape)}")
        if mask.dim() != 3:
            raise ValueError(f"mask must be [B,W,N], got {tuple(mask.shape)}")

        B, W, N, Din = bags.shape
        BW = B * W

        x = bags.reshape(BW, N, Din)       # [BW,N,Din]
        m = mask.reshape(BW, N)            # [BW,N]
        
        # normalize
        x = self.ln(x)

        # embed instances
        h = self.phi(x)                    # [BW,N,Dout]

        # compute attention logits
        V = torch.tanh(self.attn_V(h))     # [BW,N,Ah]
        if self.gated:
            U = torch.sigmoid(self.attn_U(h))  # [BW,N,Ah]
            A = self.attn_w(V * U).squeeze(-1) # [BW,N]
        else:
            A = self.attn_w(V).squeeze(-1)     # [BW,N]

        # mask out padding: set logits to -inf
        A = A.masked_fill(~m, float("-inf"))

        # softmax along instances
        alpha = torch.softmax(A, dim=1)    # [BW,N]
        # for rows where all are masked (shouldn't happen), softmax gives NaN.
        # safeguard: replace NaN with uniform over valid (or zeros)
        if torch.isnan(alpha).any():
            valid_counts = m.sum(dim=1, keepdim=True).clamp(min=1)
            alpha = torch.where(
                torch.isnan(alpha),
                m.float() / valid_counts.float(),
                alpha
            )

        # weighted sum
        z = torch.einsum("bn,bnd->bd", alpha, h)  # [BW,Dout]
        z = z.reshape(B, W, -1)

        if self.return_attention:
            attn = alpha.reshape(B, W, N)
            return z, attn
        return z


class MeanPoolMIL(nn.Module):
    """
    Simple baseline: masked mean pooling over instances.
    Input : bags [B,W,N,D], mask [B,W,N]
    Output: z [B,W,D]
    """
    def __init__(self):
        super().__init__()

    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, W, N, D = bags.shape
        m = mask.unsqueeze(-1).float()          # [B,W,N,1]
        x = bags * m
        denom = m.sum(dim=2).clamp(min=1.0)     # [B,W,1]
        z = x.sum(dim=2) / denom                # [B,W,D]
        return z
