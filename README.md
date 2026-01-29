# FusionODE

FusionODE is a research codebase for modeling irregularly sampled time‑lapse sequences and predicting four continuous regression targets from multimodal inputs.
It contains baseline sequence models (GRU/LSTM), ODE‑RNN variants for irregular sampling, and supporting training utilities (normalization, metrics, and early stopping).

## Project structure

```
configs/        Experiment configuration files.
data/           Datasets and processed artifacts (local only).
scripts/        Training and evaluation entrypoints.
src/            Core library code (models, training, utils).
main.py         Main executable entrypoint (if used by scripts).
```

## Quick start

1. Create and activate a Python environment (conda/venv).
2. Install dependencies (example):
   ```bash
   pip install -r requirements.txt
   ```
3. Run a training script with a config file:
   ```bash
   python scripts/train.py --config configs/example.json
   ```

You can override specific keys inline:
```bash
python scripts/train.py --config configs/example.json --set train.lr=3e-4 --set model.type=gru
```

> Note: YAML configs require PyYAML (`pip install pyyaml`).

## Models

* **GRU/LSTM baselines** with optional time features (absolute time or delta‑time).
* **ODE‑RNN** for irregularly sampled sequences.
* **Fusion models** that combine morphology features and CNN bag features.

All models output **4 regression targets** and are compatible with the shared training engine and metrics.

## Training & evaluation

The training loop supports:

* Target normalization (mean/std) with MSE computed in normalized space.
* Metrics reported in raw target space (RMSE and per‑target R²).
* Gradient clipping and early stopping.

See `src/train/` for details on the loss and metrics implementations.

## Diagnostic experiments (time-only vs. morph-only vs. image-only)

To sanity-check whether time features dominate interpolation performance, run the GRU
diagnostic configs below (all use the same training settings; only the input modality
changes). These compare:

* **Time-only** (no morph, no CNN features; time features only)
* **Morph-only**
* **Image-only** (CNN bag features)

```bash
python scripts/train.py --config configs/diagnostic/gru_time_only_dark.json
python scripts/train.py --config configs/diagnostic/gru_morph_only_dark.json
python scripts/train.py --config configs/diagnostic/gru_image_only_dark.json
```

You can run the same diagnostic suite with the ODE-RNN model to see whether its performance
is similarly dominated by time features:

```bash
python scripts/train.py --config configs/diagnostic/odernn_time_only_dark.json
python scripts/train.py --config configs/diagnostic/odernn_morph_only_dark.json
python scripts/train.py --config configs/diagnostic/odernn_image_only_dark.json
```

### Extrapolation split + target shuffle diagnostics

To evaluate true forecasting (extrapolation), split by timepoints instead of cells:

```bash
python scripts/train.py --config configs/ablation/step3_odernn_dark.json --set data.split_strategy=time
```

To test for time leakage directly, shuffle targets across timepoints **within** each condition:

```bash
python scripts/train.py --config configs/ablation/step3_odernn_dark.json --set data.target_shuffle=within_condition
```

## Reproducibility

To reproduce an experiment:

1. Store dataset paths and preprocessing settings in your config.
2. Fix random seeds and log the git commit hash.
3. Record hardware and software versions (CUDA, PyTorch).

## License

Internal research use only unless stated otherwise.
