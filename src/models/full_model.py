# src/models/full_model.py
from __future__ import annotations
import torch
import torch.nn as nn

from src.models.encoders import MorphMLP, AttentionMIL
from src.models.decoder import RegressionHead
from src.models.fusion import CrossAttentionFusion


class FusionGRUModel(nn.Module):
    """
    Baseline model:
      morph stats -> MLP
      cnn bag -> AttentionMIL
      concat -> GRU
      last hidden -> regression head (4 targets)
    """
    def __init__(
        self,
        morph_dim: int,
        cnn_dim: int,
        z_morph: int = 64,
        z_cnn: int = 64,
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.morph_encoder = MorphMLP(in_dim=morph_dim, out_dim=z_morph, dropout=dropout)
        self.cnn_encoder = AttentionMIL(in_dim=cnn_dim, out_dim=z_cnn, attn_hidden=128, gated=True)

        fusion_dim = z_morph + z_cnn
        self.fusion = CrossAttentionFusion(
            morph_dim=z_morph,
            cnn_dim=z_cnn,
            out_dim=fusion_dim,
            attn_dim=64,
            num_heads=4,
            attn_dropout=dropout,
            proj_dropout=dropout,
        )

        # batch_first=True => input [B,W,F]
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )

        self.decoder = RegressionHead(in_dim=rnn_hidden, out_dim=4, hidden=128, dropout=dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch keys:
          morph: [B,W,Dm]
          bags: [B,W,N,Dc]
          bag_mask: [B,W,N]
          times: [B,W]  (baseline里不用，但保留接口不影响未来换 ODE-RNN)
        """
        morph = batch["morph"]
        bags = batch["bags"]
        mask = batch["bag_mask"]

        z_m = self.morph_encoder(morph)           # [B,W,Zm]
        z_c = self.cnn_encoder(bags, mask)        # [B,W,Zc]
        x = self.fusion(z_m, z_c)                 # [B,W,Z]

        h_seq, h_last = self.gru(x)               # h_last: [layers, B, H]
        h = h_last[-1]                            # [B,H] last layer

        y_hat = self.decoder(h)                   # [B,4]
        return y_hat


from src.models.odernn import ODERNN


class FusionODERNNModel(nn.Module):
    """
    ODE-RNN model:
      MorphMLP + AttentionMIL -> concat -> ODERNN (RK4+GRUCell) -> RegressionHead
    """
    def __init__(
        self,
        morph_dim: int,
        cnn_dim: int,
        z_morph: int = 64,
        z_cnn: int = 64,
        hidden_dim: int = 128,
        ode_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.morph_encoder = MorphMLP(in_dim=morph_dim, out_dim=z_morph, dropout=dropout)
        self.cnn_encoder = AttentionMIL(in_dim=cnn_dim, out_dim=z_cnn, attn_hidden=128, gated=True)

        fusion_dim = z_morph + z_cnn
        self.fusion = CrossAttentionFusion(
            morph_dim=z_morph,
            cnn_dim=z_cnn,
            out_dim=fusion_dim,
            attn_dim=64,
            num_heads=4,
            attn_dropout=dropout,
            proj_dropout=dropout,
        )
        self.temporal = ODERNN(
            input_dim=fusion_dim,
            hidden_dim=hidden_dim,
            ode_hidden=ode_hidden,
            dropout=0.0,
        )
        self.decoder = RegressionHead(in_dim=hidden_dim, out_dim=4, hidden=128, dropout=dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        morph = batch["morph"]        # [B,W,Dm]
        bags = batch["bags"]          # [B,W,N,Dc]
        mask = batch["bag_mask"]      # [B,W,N]
        times = batch["times"] / 72.0       # [B,W]
        z_m = self.morph_encoder(morph)        # [B,W,Zm]
        z_c = self.cnn_encoder(bags, mask)     # [B,W,Zc]
        x = self.fusion(z_m, z_c)             # [B,W,F]

        h_last = self.temporal(x, times)       # [B,H]
        y_hat = self.decoder(h_last)           # [B,4]
        return y_hat
