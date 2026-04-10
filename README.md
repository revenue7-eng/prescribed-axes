# The Space Matters More Than the Loss: JEPA Collapse as a Problem of Structure, Not Optimization

**Author:** Andrey Lazarev | Independent Researcher | April 2026

## Summary

When a JEPA model trains in a free latent space, it can collapse — map all inputs to a single point. Existing solutions address this through the loss function: adding regularizers (VICReg, SIGReg) or a generative decoder (PAN). We show that the problem is not the loss but the structure of the space.

If the axes of the latent space are defined before training — from physical coordinates, semantic features, or cluster centroids — collapse becomes structurally impossible, and regularization is unnecessary. We call this approach **prescribed axes**.

Four experiments across different modalities confirm this:

| Experiment | Modality | Prescribed | Free | Gap | Metric |
|---|---|---|---|---|---|
| Shov-JEPA | Vision (UI) | 72.5% | 67.5% | +5% | Accuracy |
| LeWM State | States | 0.004 | 0.157 | 38× | Val loss (↓) |
| LeWM Pixel | Pixels | 0.0009 | 0.013 | 14.8× | Val loss (↓) |
| Speech JEPA | Speech | 73.7% | 53.2% | +20pp | Entropy (↑) |

SIGReg degrades free representations and has no effect on prescribed ones — regularization treats what prescribed axes prevent.

## Repository Structure

```
prescribed-axes/
├── README.md
├── exp1_shov_jepa/           # Experiment 1: Vision (UI understanding)
│   ├── README.md
│   ├── experiment_v4_vhgrid.ipynb
│   ├── experiment_v5_prescribed_vs_free.ipynb
│   └── shov-jepa-report.docx
├── exp2_lewm_state/          # Experiment 2: State prediction (Push-T)
│   ├── README.md
│   ├── code/
│   │   ├── lewm_pusht_colab.ipynb
│   │   └── lewm_pusht_experiment.py
│   └── results/
│       ├── lewm_results.json
│       └── lewm_pusht_results.png
├── exp3_lewm_pixel/          # Experiment 3: Pixel prediction (CNN vs prescribed)
│   ├── README.md
│   ├── code/
│   │   ├── lewm_pixels_full.ipynb
│   │   └── lewm_pixels_v2.ipynb
│   ├── results/
│   │   ├── history_free_cnn.json
│   │   ├── history_prescribed.json
│   │   ├── results.json
│   │   └── results.png
│   └── lewm-pixel-report.docx
└── exp4_speech_jepa/         # Experiment 4: Speech (clustering anchors)
    ├── README.md
    ├── speech_jepa_2x2_v6.ipynb
    └── results/
        ├── eval_results.pkl
        ├── final_results.pkl
        ├── calibration.pkl
        ├── cluster_metrics.png
        ├── training_curves.png
        └── factorial_heatmap.png
```

## Experiments

### Experiment 1: Shov-JEPA (Vision)
Prescribed axes (position, functionality, depth) from View Hierarchy vs. free 64D JEPA encoder on Rico dataset (398 mobile UI screenshots). 3 prescribed dimensions outperform 64 free dimensions: 72.5% vs. 67.5%.

**Run:** Open `experiment_v5_prescribed_vs_free.ipynb` in Google Colab (T4 GPU).

### Experiment 2: LeWM State (Robotic Control)
Prescribed physical coordinates (x, y, θ) vs. free MLP encoder on Push-T environment. 3 seeds, 50 epochs. Prescribed: 0.004 val loss. Free: 0.157. Gap: 38×. SIGReg ablation shows regularization hurts free (4.2× worse) and has zero effect on prescribed.

**Run:** Open `lewm_pusht_colab.ipynb` in Google Colab, or run `lewm_pusht_experiment.py --mode synthetic` locally without dependencies.

### Experiment 3: LeWM Pixel (CNN Encoder)
Prescribed state coordinates (20K params) vs. CNN encoder on raw 96×96 pixels (744K params). Prescribed: 14.8× lower error. CNN plateaus at epoch 7 out of 50.

**Run:** Open `lewm_pixels_full.ipynb` in Google Colab, or run locally on CPU.

### Experiment 4: Speech JEPA (Clustering)
2×2 factorial design: {GMM, k-means} × {soft, hard} with frozen cluster anchors vs. Pure JEPA on LibriSpeech. All prescribed conditions outperform Pure JEPA by +18–20pp entropy. Dominant factor: frozen structure itself, not clustering method or assignment type.

**Run:** Open `speech_jepa_2x2_v6.ipynb` in Google Colab (T4 GPU).

## Key Finding

Representational collapse in JEAs is not primarily a loss function problem but a structural one, resolvable by defining the right axes of variation before learning begins. Prescribed axes represent a third approach to the collapse problem, distinct from both regularization-based (VICReg, SIGReg) and generative (PAN) methods.

## Citation

```
@article{lazarev2026prescribed,
  title={The Space Matters More Than the Loss: JEPA Collapse as a Problem of Structure, Not Optimization},
  author={Lazarev, Andrey},
  year={2026}
}
```

## License

MIT
