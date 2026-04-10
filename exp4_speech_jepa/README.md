# Experiment 4: Speech JEPA 2×2 — Frozen Structure vs Assignment Type

## What this tests
What stabilizes learning in speech JEPA: the method (GMM vs k-means),
the assignment type (soft vs hard), or the frozen structure itself?

## Key result
**Frozen structure matters, not the method or assignment type.**
- Soft ≈ Hard (Δ < 0.03%)
- GMM ≈ KM (+2.2pp, secondary)
- All prescribed: entropy +18-20pp vs Pure JEPA

## Results
| Condition | Entropy (%) | Active/512 |
|-----------|------------|------------|
| Pure JEPA | 53.2 | 138 |
| GMM+Soft | 71.5 | 128 |
| GMM+Hard | 71.5 | 128 |
| KM+Soft | 73.7 | 128 |
| KM+Hard | 73.7 | 128 |

Factorial: Soft vs Hard Δ=0.0%, GMM vs KM Δ=-2.2%, Anchored vs Pure Δ=+19.4%

## Setup
- **Modality:** Speech (log-mel features, LibriSpeech train-clean-100)
- **Architecture:** JEPA with clustering anchors (Ioannides et al., 2026)
- **Design:** 2×2 factorial: {GMM, k-means} × {soft, hard}, all frozen
- **Config:** K=512, 2000 steps, dim=256, τ=2.0
- **Platform:** Google Colab Pro T4

## Files
```
speech_jepa_2x2_v6.ipynb       — Complete notebook with execution outputs
results/final_results.pkl      — All metrics + training history
results/eval_results.pkl       — Per-condition metrics  
results/calibration.pkl        — Temperature calibration
results/cluster_metrics.png    — Entropy + active clusters + distribution
results/factorial_heatmap.png  — 2×2 factorial heatmap
results/training_curves.png    — Loss curves
README.md                      — This file
```

The notebook (v6) is the exact version that produced all results.
All outputs are preserved in the notebook.
