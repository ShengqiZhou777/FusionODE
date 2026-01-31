
import torch
from typing import Any, Mapping

from src.models.full_model import (
    FusionGRUModel,
    FusionLSTMModel,
    FusionODERNNModel,
    StaticMLPBaseline,
)

def create_model(
    model_type: str,
    fusion_type: str,
    use_morph: bool,
    use_cnn: bool,
    input_dims: dict[str, int],
    time_scale: float,
    model_params: Mapping[str, Any],
) -> torch.nn.Module:
    """
    Factory to create fusion models.
    
    Args:
        model_type: "odernn", "gru", "lstm", "static"
        fusion_type: "cross_attention", "concat"
        use_morph: bool
        use_cnn: bool
        input_dims: dict with keys 'morph', 'cnn' (feature dimensions)
        time_scale: float
        model_params: dict containing hyperparameters like:
            hidden_dim, ode_hidden, rnn_hidden, rnn_layers,
            dropout, attn_dim, attn_heads, time_features
    """
    Dm = input_dims.get("morph", 0)
    Dc = input_dims.get("cnn", 0)
    
    # Common args for all models
    common_args = dict(
        morph_dim=Dm,
        cnn_dim=Dc,
        z_morph=model_params.get("z_morph", 64),
        z_cnn=model_params.get("z_cnn", 64),
        dropout=model_params.get("dropout", 0.1),
        use_morph=use_morph,
        use_cnn=use_cnn,
        fusion_type=fusion_type,
        attn_dim=model_params.get("attn_dim", 64),
        attn_heads=model_params.get("attn_heads", 4),
        use_time=model_params.get("time_features", "absolute+delta") != "none",
        time_features=model_params.get("time_features", "absolute+delta"),
        time_scale=time_scale,
    )

    if model_type == "odernn":
        return FusionODERNNModel(
            hidden_dim=model_params.get("hidden_dim", 128),
            ode_hidden=model_params.get("ode_hidden", 128),
            **common_args
        )
    elif model_type == "gru":
        return FusionGRUModel(
            rnn_hidden=model_params.get("rnn_hidden", 128),
            rnn_layers=model_params.get("rnn_layers", 1),
            **common_args
        )
    elif model_type == "lstm":
        return FusionLSTMModel(
            rnn_hidden=model_params.get("rnn_hidden", 128),
            rnn_layers=model_params.get("rnn_layers", 1),
            **common_args
        )
    elif model_type == "static":
        return StaticMLPBaseline(
            hidden_dim=model_params.get("hidden_dim", 128),
            **common_args
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
