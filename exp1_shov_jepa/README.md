# Experiment 1: Shov-JEPA — Prescribed Axes for UI Understanding

## What this tests
Can prescribed semantic axes outperform free latent space in JEPA for UI understanding?

## Key results
| Condition | Accuracy | Source |
|-----------|---------|--------|
| Pixels baseline (224×224) | 68.8% | experiment_v4 |
| VH flat (prescribed, 12 features) | 72.5% | experiment_v4 |
| VH grid JEPA (prescribed + JEPA) | 73.8% | experiment_v4 |
| Prescribed 3D (Shov-JEPA) | 72.5% | experiment_v5 |
| Free 64D (Plain JEPA) + probe | 67.5% | experiment_v5 |

**Prescribed vs free: 72.5% vs 67.5% = +5%**
**Best overall: VH grid JEPA 73.8%**
JEPA loss reduction: 73-85%.

## Setup
- **Modality:** UI screenshots (vision)
- **Data:** Rico dataset, 398 mobile UI screenshots
- **Architecture:** Modified I-JEPA with prescribed axes (P, F, D)
- **Platform:** Google Colab T4 GPU

## Files
```
experiment_v4_vhgrid.ipynb              — 4 conditions: pixels, VH flat, VH grid JEPA (73.8%)
experiment_v5_prescribed_vs_free.ipynb  — Prescribed 3D vs Free 64D (72.5% vs 67.5%)
README.md                               — This file
```

Both notebooks are self-contained Colab notebooks with execution outputs preserved.
Upload to Colab, select T4 GPU, Run All.

## Significance
First empirical confirmation that the structure of the representation space
matters more than pixel input quality. Prescribed axes (+5% over free) and
JEPA structure (+1.3% over flat) both contribute independently.

## Reports
```
shov-jepa-report.docx     — Full experiment report (English)
shov-jepa-report-ru.docx  — Full experiment report (Russian)
```
