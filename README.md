# The Space Matters More Than the Loss: JEPA Collapse as a Problem of Structure, Not Optimization

**Author:** Andrey Lazarev | Independent Researcher | April 2026

📄 **[Read the paper (PDF)](paper.pdf)**

## Summary

When a JEPA model trains in a free latent space, it can collapse — map all inputs to a single point. Existing solutions address this through the loss function: adding regularizers (VICReg, SIGReg) or a generative decoder (PAN). We show that the problem is not the loss but the structure of the space.

If the axes of the latent space are defined before training — from physical coordinates, semantic features, or cluster centroids — collapse becomes structurally impossible, and regularization is unnecessary. We call this approach **prescribed axes**.

Four experiments across different modalities confirm this:

| Experiment | Modality | Prescribed | Free | Gap | Metric |
|---|---|---|---|---|---|
| Speech JEPA | Speech | 73.7% | 53.2% | +20pp | Entropy (↑) |
| Shov-JEPA | Vision (UI) | 72.5% | 67.5% | +5% | Accuracy |
| LeWM State | States | 0.004 | 0.157 | 38× | Val loss (↓) |
| LeWM Pixel | Pixels | 0.0009 | 0.013 | 14.8× | Val loss (↓) |

Experiments are ordered by increasing evidential depth: from pilot studies (1–2) to controlled experiments with ablation (3–4).

SIGReg degrades free representations and has no effect on prescribed ones — regularization treats what prescribed axes prevent.

## Repository Structure

```
prescribed-axes/
├── README.md
├── exp1_shov_jepa/           # Experiment 2 in paper: Vision (UI understanding)
│   ├── README.md
│   ├── experiment_v4_vhgrid.ipynb
│   ├── experiment_v5_prescribed_vs_free.ipynb
│   └── shov-jepa-report.docx
├── exp2_lewm_state/          # Experiment 3 in paper: State prediction (Push-T)
│   ├── README.md
│   ├── code/
│   │   ├── lewm_pusht_colab.ipynb
│   │   └── lewm_pusht_experiment.py
│   └── results/
│       ├── lewm_results.json
│       └── lewm_pusht_results.png
├── exp3_lewm_pixel/          # Experiment 4 in paper: Pixel prediction (CNN vs prescribed)
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
└── exp4_speech_jepa/         # Experiment 1 in paper: Speech (clustering anchors)
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

**Note:** Folder names (`exp1_`–`exp4_`) reflect the chronological order in which experiments were conducted. The paper orders experiments by evidential depth: Speech JEPA (pilot) → Shov-JEPA (pilot) → LeWM State (controlled) → LeWM Pixel (controlled). Folder names are preserved for backward compatibility.

## Experiments

### Experiment 1 (paper): Speech JEPA (Clustering) — `exp4_speech_jepa/`
2×2 factorial design: {GMM, k-means} × {soft, hard} with frozen cluster anchors vs. Pure JEPA on LibriSpeech. All prescribed conditions outperform Pure JEPA by +18–20pp entropy. Dominant factor: frozen structure itself, not clustering method or assignment type. Pilot study: entropy measures codebook utilization, not downstream quality.

**Run:** Open `speech_jepa_2x2_v6.ipynb` in Google Colab (T4 GPU).

### Experiment 2 (paper): Shov-JEPA (Vision) — `exp1_shov_jepa/`
Prescribed axes (position, functionality, depth) from View Hierarchy vs. free 64D JEPA encoder on Rico dataset (398 mobile UI screenshots). 3 prescribed dimensions outperform 64 free dimensions: 72.5% vs. 67.5%. Classification accuracy is a downstream evaluation. Pilot study (398 samples, single seed, +5% gap).

**Run:** Open `experiment_v5_prescribed_vs_free.ipynb` in Google Colab (T4 GPU).

### Experiment 3 (paper): LeWM State (Robotic Control) — `exp2_lewm_state/`
Prescribed physical coordinates (x, y, θ) vs. free MLP encoder on Push-T environment. 3 seeds, 50 epochs. Prescribed: 0.004 val loss. Free: 0.157. Gap: 38×. SIGReg ablation: free without SIGReg achieves 0.037, free with SIGReg degrades to 0.156 (4.2× worse). SIGReg on prescribed: zero effect.

**Run:** Open `lewm_pusht_colab.ipynb` in Google Colab, or run `lewm_pusht_experiment.py --mode synthetic` locally without dependencies.

### Experiment 4 (paper): LeWM Pixel (CNN Encoder) — `exp3_lewm_pixel/`
Prescribed state coordinates (20K params) vs. CNN encoder on raw 96×96 pixels (744K params). Prescribed: 14.8× lower error. CNN plateaus at epoch 7 out of 50. The advantage is separation of concerns: the free encoder must simultaneously learn what to represent and how to predict; prescribed axes remove the first task entirely.

**Run:** Open `lewm_pixels_full.ipynb` in Google Colab, or run locally on CPU.

## Key Finding

Prescribed axes reframe the collapse problem. The current debate asks: "how do we prevent collapse in a free space?" — through regularization (VICReg, SIGReg) or generative supervision (PAN). Prescribed axes ask a different question: "why does the encoder define the space at all?"

This is not a competing method but a change in problem formulation. The SIGReg diagnostic confirms this empirically: SIGReg is neither helpful nor harmful under prescribed axes — it is irrelevant. Prescribed axes operate at a level where regularization is not yet defined.

## Continuation: Paper 2

**→ [Semantic Drift, Not Rank Collapse](https://github.com/revenue7-eng/prescribed-axes-drift)** — identifies *why* prescribed axes work: the free encoder's coordinate system drifts so rapidly that downstream modules cannot learn in it.


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
