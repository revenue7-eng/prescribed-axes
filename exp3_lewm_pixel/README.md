# Experiment 3: LeWM Pixel-Based — Prescribed vs Free CNN

## What this tests
Can a CNN encoder learning from raw pixels (96×96) match the prediction quality of prescribed state coordinates?

## Key result
**Prescribed 14.8× better than free CNN** (val loss 0.000884 vs 0.013046).
Prescribed: 20,038 params. CNN: 744,425 params (37× more).
CNN plateaus at epoch 7 and never improves across 43 remaining epochs.

## Setup
- **Environment:** Push-T (LeWM HDF5 dataset, 500 episodes)
- **Data source:** `quentinll/lewm-pusht` on HuggingFace (~13GB)
- **Prescribed encoder:** state[2:5] normalized
- **Free CNN encoder:** 4-layer CNN (3→32→64→128→256) + MLP → 3D
- **Epochs:** 50, Seed: 42
- **Platform:** CPU Windows (full run), Colab T4 (partial, same results)

## Results
| Mode | Best val loss | Best epoch | Parameters |
|------|-------------|------------|------------|
| prescribed | 0.000884 | 42 | 20,038 |
| free_cnn | 0.013046 | 7 | 744,425 |

## Files
```
code/lewm_pixels_v2.ipynb          — Compact Colab notebook
code/lewm_pixels_full.ipynb        — Full notebook with ablations
results/results.json               — Final numbers
results/results.png                — Training curves
results/history_prescribed.json    — Per-epoch prescribed
results/history_free_cnn.json      — Per-epoch free CNN
```

## How to reproduce
Upload `code/lewm_pixels_v2.ipynb` to Colab, select T4 GPU, Run All.
Data downloads automatically. Runtime: ~30 min on T4.
