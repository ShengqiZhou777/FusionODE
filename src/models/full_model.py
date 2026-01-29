# src/models/full_model.py
from __future__ import annotations
import torch
import torch.nn as nn

from src.models.encoders import MorphMLP, AttentionMIL
from src.models.decoder import RegressionHead
from src.models.fusion import ConcatFusion, CrossAttentionFusion


def _build_time_features(
    times: torch.Tensor,
    time_scale: float,
    mode: str,
) -> torch.Tensor:
    """
    Build time features for RNN baselines.
    We keep this minimal and explicit because irregular sampling can otherwise
    confuse discrete RNNs. We normalize by a global scale to keep magnitudes
    stable for small-data training.
    """
    if mode == "none":
        return torch.zeros((*times.shape, 0), device=times.device, dtype=times.dtype)

    scale = max(float(time_scale), 1e-6)
    t_abs = (times / scale).unsqueeze(-1)
    if mode == "absolute":
        return t_abs

    if mode == "absolute+delta":
        dt = torch.zeros_like(times)
        dt[:, 1:] = times[:, 1:] - times[:, :-1]
        dt = (dt / scale).unsqueeze(-1)
        return torch.cat([t_abs, dt], dim=-1)

    raise ValueError(f"Unsupported time feature mode: {mode}")


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
        use_morph: bool = True,
        use_cnn: bool = True,
        fusion_type: str = "cross_attention",
        attn_dim: int = 64,
        attn_heads: int = 4,
        use_time: bool = True,
        time_features: str = "absolute",
        time_scale: float = 72.0,
    ):
        super().__init__()
        if not use_morph and not use_cnn:
            raise ValueError("At least one of use_morph/use_cnn must be True.")
        self.use_morph = use_morph
        self.use_cnn = use_cnn
        self.use_time = use_time
        self.time_features = time_features
        self.time_scale = time_scale
        self.morph_encoder = MorphMLP(in_dim=morph_dim, out_dim=z_morph, dropout=dropout)
        self.cnn_encoder = AttentionMIL(in_dim=cnn_dim, out_dim=z_cnn, attn_hidden=128, gated=True)

        fusion_dim = (z_morph if use_morph else 0) + (z_cnn if use_cnn else 0)
        if fusion_type == "cross_attention" and use_morph and use_cnn:
            self.fusion = CrossAttentionFusion(
                morph_dim=z_morph,
                cnn_dim=z_cnn,
                out_dim=fusion_dim,
                attn_dim=attn_dim,
                num_heads=attn_heads,
                attn_dropout=dropout,
                proj_dropout=dropout,
            )
        elif fusion_type == "concat" and use_morph and use_cnn:
            self.fusion = ConcatFusion(morph_dim=z_morph, cnn_dim=z_cnn)
        elif use_morph and not use_cnn:
            self.fusion = nn.Identity()
        elif use_cnn and not use_morph:
            self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        time_dim = 0
        if self.use_time:
            time_dim = 1 if time_features == "absolute" else 2

        # batch_first=True => input [B,W,F]
        self.gru = nn.GRU(
            input_size=fusion_dim + time_dim,
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
        times: [B,W]  (RNN baselines append normalized time to improve irregular sampling awareness)
        """
        morph = batch["morph"]
        bags = batch["bags"]
        mask = batch["bag_mask"]
        times = batch["times"]

        z_m = self.morph_encoder(morph) if self.use_morph else None  # [B,W,Zm]
        z_c = self.cnn_encoder(bags, mask) if self.use_cnn else None # [B,W,Zc]
        if self.use_morph and self.use_cnn:
            x = self.fusion(z_m, z_c)                 # [B,W,Z]
        elif self.use_morph:
            x = z_m
        else:
            x = z_c

        if self.use_time:
            t_feat = _build_time_features(times, self.time_scale, self.time_features)
            x = torch.cat([x, t_feat], dim=-1)

        h_seq, h_last = self.gru(x)               # h_last: [layers, B, H]
        h = h_last[-1]                            # [B,H] last layer

        y_hat = self.decoder(h)                   # [B,4]
        return y_hat


from src.models.odernn import ODERNN


class FusionLSTMModel(nn.Module):
    """
    LSTM baseline model:
      morph stats -> MLP
      cnn bag -> AttentionMIL
      fusion -> LSTM
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
        use_morph: bool = True,
        use_cnn: bool = True,
        fusion_type: str = "cross_attention",
        attn_dim: int = 64,
        attn_heads: int = 4,
        use_time: bool = True,
        time_features: str = "absolute",
        time_scale: float = 72.0,
    ):
        super().__init__()
        if not use_morph and not use_cnn:
            raise ValueError("At least one of use_morph/use_cnn must be True.")
        self.use_morph = use_morph
        self.use_cnn = use_cnn
        self.use_time = use_time
        self.time_features = time_features
        self.time_scale = time_scale
        self.morph_encoder = MorphMLP(in_dim=morph_dim, out_dim=z_morph, dropout=dropout)
        self.cnn_encoder = AttentionMIL(in_dim=cnn_dim, out_dim=z_cnn, attn_hidden=128, gated=True)

        fusion_dim = (z_morph if use_morph else 0) + (z_cnn if use_cnn else 0)
        if fusion_type == "cross_attention" and use_morph and use_cnn:
            self.fusion = CrossAttentionFusion(
                morph_dim=z_morph,
                cnn_dim=z_cnn,
                out_dim=fusion_dim,
                attn_dim=attn_dim,
                num_heads=attn_heads,
                attn_dropout=dropout,
                proj_dropout=dropout,
            )
        elif fusion_type == "concat" and use_morph and use_cnn:
            self.fusion = ConcatFusion(morph_dim=z_morph, cnn_dim=z_cnn)
        elif use_morph and not use_cnn:
            self.fusion = nn.Identity()
        elif use_cnn and not use_morph:
            self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        time_dim = 0
        if self.use_time:
            time_dim = 1 if time_features == "absolute" else 2

        self.lstm = nn.LSTM(
            input_size=fusion_dim + time_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )

        self.decoder = RegressionHead(in_dim=rnn_hidden, out_dim=4, hidden=128, dropout=dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        morph = batch["morph"]
        bags = batch["bags"]
        mask = batch["bag_mask"]
        times = batch["times"]

        z_m = self.morph_encoder(morph) if self.use_morph else None  # [B,W,Zm]
        z_c = self.cnn_encoder(bags, mask) if self.use_cnn else None # [B,W,Zc]
        if self.use_morph and self.use_cnn:
            x = self.fusion(z_m, z_c)                 # [B,W,Z]
        elif self.use_morph:
            x = z_m
        else:
            x = z_c

        if self.use_time:
            t_feat = _build_time_features(times, self.time_scale, self.time_features)
            x = torch.cat([x, t_feat], dim=-1)

        h_seq, (h_last, _) = self.lstm(x)             # h_last: [layers, B, H]
        h = h_last[-1]                                # [B,H]
        y_hat = self.decoder(h)                       # [B,4]
        return y_hat


class StaticMLPBaseline(nn.Module):
    """
    Static baseline:
      Encode morph + CNN -> fuse -> take last timepoint -> MLP head.
    This is a sanity-check baseline for small-data regimes where sequence
    models may overfit or collapse.
    """
    def __init__(
        self,
        morph_dim: int,
        cnn_dim: int,
        z_morph: int = 64,
        z_cnn: int = 64,
        dropout: float = 0.1,
        use_morph: bool = True,
        use_cnn: bool = True,
        fusion_type: str = "cross_attention",
        attn_dim: int = 64,
        attn_heads: int = 4,
        use_time: bool = True,
        time_features: str = "absolute",
        time_scale: float = 72.0,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if not use_morph and not use_cnn:
            raise ValueError("At least one of use_morph/use_cnn must be True.")
        self.use_morph = use_morph
        self.use_cnn = use_cnn
        self.use_time = use_time
        self.time_features = time_features
        self.time_scale = time_scale
        self.morph_encoder = MorphMLP(in_dim=morph_dim, out_dim=z_morph, dropout=dropout)
        self.cnn_encoder = AttentionMIL(in_dim=cnn_dim, out_dim=z_cnn, attn_hidden=128, gated=True)

        fusion_dim = (z_morph if use_morph else 0) + (z_cnn if use_cnn else 0)
        if fusion_type == "cross_attention" and use_morph and use_cnn:
            self.fusion = CrossAttentionFusion(
                morph_dim=z_morph,
                cnn_dim=z_cnn,
                out_dim=fusion_dim,
                attn_dim=attn_dim,
                num_heads=attn_heads,
                attn_dropout=dropout,
                proj_dropout=dropout,
            )
        elif fusion_type == "concat" and use_morph and use_cnn:
            self.fusion = ConcatFusion(morph_dim=z_morph, cnn_dim=z_cnn)
        elif use_morph and not use_cnn:
            self.fusion = nn.Identity()
        elif use_cnn and not use_morph:
            self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        time_dim = 0
        if self.use_time:
            time_dim = 1 if time_features == "absolute" else 2

        self.decoder = RegressionHead(in_dim=fusion_dim + time_dim, out_dim=4, hidden=hidden_dim, dropout=dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        morph = batch["morph"]
        bags = batch["bags"]
        mask = batch["bag_mask"]
        times = batch["times"]

        z_m = self.morph_encoder(morph) if self.use_morph else None
        z_c = self.cnn_encoder(bags, mask) if self.use_cnn else None
        if self.use_morph and self.use_cnn:
            x = self.fusion(z_m, z_c)
        elif self.use_morph:
            x = z_m
        else:
            x = z_c

        x_last = x[:, -1]
        if self.use_time:
            t_feat = _build_time_features(times, self.time_scale, self.time_features)
            x_last = torch.cat([x_last, t_feat[:, -1]], dim=-1)

        return self.decoder(x_last)


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
        use_morph: bool = True,
        use_cnn: bool = True,
        fusion_type: str = "cross_attention",
        attn_dim: int = 64,
        attn_heads: int = 4,
    ):
        super().__init__()
        if not use_morph and not use_cnn:
            raise ValueError("At least one of use_morph/use_cnn must be True.")
        self.use_morph = use_morph
        self.use_cnn = use_cnn
        self.morph_encoder = MorphMLP(in_dim=morph_dim, out_dim=z_morph, dropout=dropout)
        self.cnn_encoder = AttentionMIL(in_dim=cnn_dim, out_dim=z_cnn, attn_hidden=128, gated=True)

        fusion_dim = (z_morph if use_morph else 0) + (z_cnn if use_cnn else 0)
        if fusion_type == "cross_attention" and use_morph and use_cnn:
            self.fusion = CrossAttentionFusion(
                morph_dim=z_morph,
                cnn_dim=z_cnn,
                out_dim=fusion_dim,
                attn_dim=attn_dim,
                num_heads=attn_heads,
                attn_dropout=dropout,
                proj_dropout=dropout,
            )
        elif fusion_type == "concat" and use_morph and use_cnn:
            self.fusion = ConcatFusion(morph_dim=z_morph, cnn_dim=z_cnn)
        elif use_morph and not use_cnn:
            self.fusion = nn.Identity()
        elif use_cnn and not use_morph:
            self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
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
        z_m = self.morph_encoder(morph) if self.use_morph else None  # [B,W,Zm]
        z_c = self.cnn_encoder(bags, mask) if self.use_cnn else None # [B,W,Zc]
        if self.use_morph and self.use_cnn:
            x = self.fusion(z_m, z_c)             # [B,W,F]
        elif self.use_morph:
            x = z_m
        else:
            x = z_c

        h_last = self.temporal(x, times)       # [B,H]
        y_hat = self.decoder(h_last)           # [B,4]
        return y_hat
