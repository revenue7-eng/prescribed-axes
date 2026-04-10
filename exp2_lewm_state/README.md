# Experiment 2: LeWM State-Based — Prescribed vs Free

## What this tests
Can prescribed axes (block_x, block_y, block_angle) outperform a free learned embedding for next-state prediction in a world model? And how does SIGReg regularization interact with each?

## Key result
**Prescribed 38× better than free** (val loss 0.004 vs 0.157, averaged across 3 seeds).
SIGReg destroys free (0.037→0.156) but has no effect on prescribed.

## Setup
- **Environment:** Push-T (gym-pusht), real physics simulation
- **Task:** Next-state prediction from history of states and actions
- **Architecture:** Minimal Learned World Model with SIGReg regularization
- **Prescribed axes:** state[2:5] = (block_x, block_y, block_angle), normalized
- **Free baseline:** MLP encoder (5→64→64→3), learns its own embedding
- **Seeds:** 42, 123, 777
- **Epochs:** 50 per run
- **Platform:** Google Colab T4 GPU

## Results
| Condition | Seed 42 | Seed 123 | Seed 777 | Mean |
|-----------|---------|----------|----------|------|
| prescribed_block | 0.00366 | 0.00448 | 0.00413 | 0.00409 |
| prescribed_full | 0.00919 | 0.01147 | 0.00924 | 0.00997 |
| free | 0.15562 | 0.15349 | 0.16170 | 0.15694 |

Ratio free/prescribed_block = **38.4×**

## Files
```
code/lewm_pusht_experiment.py   — Standalone script
code/lewm_pusht_colab.ipynb     — Colab notebook
results/lewm_results.json       — All 9 conditions, full history
results/lewm_pusht_results.png  — Training curves
```

## How to reproduce
```bash
pip install gym-pusht torch numpy
python code/lewm_pusht_experiment.py --episodes 200 --epochs 50 --seed 42
```
Or use `--synthetic` for no-dependency mode.
