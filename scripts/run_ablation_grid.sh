#!/usr/bin/env bash
set -euo pipefail

# Example grid runner for ablation experiments.
# Adjust arrays or add flags as needed.

MODEL_TYPES=("odernn" "gru" "lstm")
FUSION_TYPES=("concat" "cross_attention")
WINDOW_SIZES=(4 5)
CONDITIONS=("Light" "Dark")
MODALITIES=(
  ""            # both morph + cnn
  "--morph-only"
  "--cnn-only"
)

for condition in "${CONDITIONS[@]}"; do
  for window in "${WINDOW_SIZES[@]}"; do
    for model in "${MODEL_TYPES[@]}"; do
      for fusion in "${FUSION_TYPES[@]}"; do
        for modality in "${MODALITIES[@]}"; do
          python main.py \
            --model-type "${model}" \
            --fusion-type "${fusion}" \
            --window-size "${window}" \
            --target-condition "${condition}" \
            ${modality}
        done
      done
    done
  done
done
