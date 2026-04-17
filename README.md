# The Space Matters More Than the Loss: JEPA Collapse as a Problem of Structure, Not Optimization

**Author:** Andrey Lazarev | Independent Researcher | April 2026

📄 **[Read the paper (PDF)](paper.pdf)**

## Summary

When a JEPA model trains in a free latent space, it can collapse — map all inputs to a single point. Existing solutions address this through the loss function: adding regularizers (VICReg, SIGReg) or a generative decoder (PAN). We show that in settings where domain-relevant coordinates are available, the problem is not the loss but the structure of the space.

If the axes of the latent space are defined before training — from physical coordinates, semantic features, or cluster centroids — collapse becomes structurally impossible, and regularization is unnecessary. We call this approach **prescribed axes**.

| Experiment | Modality | Prescribed | Free | Gap | Metric |
|---|---|---|---|---|---|
| Speech JEPA | Speech | 73.7% | 53.2% | +20pp | Entropy (↑) |
| Shov-JEPA | Vision (UI) | 72.5% | 67.5% | +5% | Accuracy |
| LeWM State | States | 0.004 | 0.157 | 38× | Val loss (↓) |
| LeWM Pixel | Pixels | 0.0009 | 0.013 | 14.8× | Val loss (↓) |
| Pendulum (negative control) | States (2D) | 0.000463 | 0.000403 | 0.87× | Val loss (↓) |

Experiments are ordered by increasing evidential depth: from pilot studies (1–2) to controlled experiments with ablation (3–4). The pendulum is a negative control establishing boundary conditions.

## Paper v10 — What's New

- **Section 4.7 (Mechanism: Why Fixation Works)** — explains the dual-task elimination mechanism. A free encoder solves two coupled tasks simultaneously (organize space + predict within it); prescribed axes break this loop by removing task 1.
- **Section 6.3 (Boundary Conditions)** — adds simple pendulum negative control. When state_dim = latent_dim (no subspace selection) and data is sufficient, free encoder wins. Establishes when prescribed axes apply.
- **Updated Abstract** — incorporates pendulum boundary condition and sample efficiency framing.

## Key Controls

**Equal-input control:** A free encoder receiving the same normalized (x, y, θ) as prescribed is 7.6× worse — the advantage is from coordinate fixation, not information access.

**SIGReg ablation:** Removing SIGReg improves the free encoder by 1.9× (0.006 vs 0.012). SIGReg on prescribed: 0.6% effect. Regularization treats what prescribed axes prevent.

**Random axes control:** Tested at two data scales. At 200 episodes, any fixed coordinate system (meaningful or random) outperforms a learned one by 4–7×. At 500 episodes, the relationship inverts: the free encoder converges to ~10⁻⁹ while fixed conditions plateau at ~8.5×10⁻⁴. The advantage of prescribed axes is sample efficiency — eliminating the dual-task problem at the cost of a fixed accuracy ceiling.

**Pendulum negative control:** On a simple pendulum (2 DOF, no subspace selection), the free encoder matches or surpasses prescribed at all dimensions, with normalization reducing the gap by 72–74%. Establishes that prescribed axes require either a subspace selection advantage (state_dim > latent_dim) or insufficient data for the free encoder to converge.

## Repository Structure

```
prescribed-axes/
├── README.md
├── LICENSE
├── paper.pdf                          # Paper v10
├── requirements.txt
├── reviewer_response_experiments.py   # Equal-input control + SIGReg ablation (5 conditions × 3 seeds)
│
├── exp1_shov_jepa/                    # Experiment 2 in paper: Vision (UI understanding)
│   ├── README.md
│   ├── experiment_v4_vhgrid.ipynb
│   ├── experiment_v5_prescribed_vs_free.ipynb
│   ├── shov-jepa-report.docx
│   └── shov_jepa_results.png
│
├── exp2_lewm_state/                   # Experiment 3 in paper: State prediction (Push-T)
│   ├── README.md
│   ├── code/
│   │   ├── lewm_pusht_colab.ipynb
│   │   └── lewm_pusht_experiment.py
│   ├── results/
│   │   ├── lewm_results.json
│   │   └── lewm_pusht_results.png
│   └── lewm-state-report.docx
│
├── exp3_lewm_pixel/                   # Experiment 4 in paper: Pixel prediction (CNN vs prescribed)
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
│
├── exp4_speech_jepa/                  # Experiment 1 in paper: Speech (clustering anchors)
│   ├── README.md
│   ├── speech_jepa_2x2_v6.ipynb
│   ├── results/
│   │   ├── eval_results.pkl
│   │   ├── final_results.pkl
│   │   ├── calibration.pkl
│   │   ├── cluster_metrics.png
│   │   ├── training_curves.png
│   │   └── factorial_heatmap.png
│   └── speech-jepa-report.docx
│
├── exp15_pendulum/                    # Section 6.3: Pendulum negative control (boundary condition)
│   ├── README.md
│   ├── code/
│   │   ├── run_e15.py                 # Current: 200ep, 30ep, 4 modes × 5 dims × 3 seeds
│   │   └── run_pendulum.py            # Original (superseded)
│   └── results/
│       ├── e15_results.json           # 60 conditions, per-seed
│       └── pendulum_results.json      # Original (superseded)
│
├── random_axes_control/               # Section 4.6: Random axes + scaling analysis
│   ├── RESULTS.md
│   ├── run_random_axes_control.py     # 200ep experiment
│   ├── run_isotropic_control.py       # 200ep isotropic normalization control
│   ├── all_results_500ep.json         # Full 500ep results (18 runs with training curves)
│   ├── random_axes_results_500ep.png  # Training curves + bar charts for 500ep
│   └── pusht_data_synthetic_500ep.npz # Synthetic Push-T dataset (500 episodes)
│
└── reviewer_results/                  # Section 4.4: Equal-input control + SIGReg ablation results
    ├── full_results.json
    ├── summary.json
    └── history_*.json                 # 15 history files (5 conditions × 3 seeds)
```

**Note:** Folder names (`exp1_`–`exp4_`) reflect the chronological order in which experiments were conducted. The paper orders experiments by evidential depth: Speech JEPA (pilot) → Shov-JEPA (pilot) → LeWM State (controlled) → LeWM Pixel (controlled). Folder names are preserved for backward compatibility. The pendulum experiment (`exp15_pendulum/`) was conducted later (April 2026) as a boundary condition test.

## Experiments

### Experiment 1 (paper): Speech JEPA (Clustering) — `exp4_speech_jepa/`
2×2 factorial design: {GMM, k-means} × {soft, hard} with frozen cluster anchors vs. Pure JEPA on LibriSpeech. All prescribed conditions outperform Pure JEPA by +18–20pp entropy. Dominant factor: frozen structure itself, not clustering method or assignment type. Pilot study: entropy measures codebook utilization, not downstream quality.

**Run:** Open `speech_jepa_2x2_v6.ipynb` in Google Colab (T4 GPU).

### Experiment 2 (paper): Shov-JEPA (Vision) — `exp1_shov_jepa/`
Prescribed axes (position, functionality, depth) from View Hierarchy vs. free 64D JEPA encoder on Rico dataset (398 mobile UI screenshots). 3 prescribed dimensions outperform 64 free dimensions: 72.5% vs. 67.5%. Classification accuracy is a downstream evaluation. Pilot study (398 samples, single seed, +5% gap).

**Run:** Open `experiment_v5_prescribed_vs_free.ipynb` in Google Colab (T4 GPU).

### Experiment 3 (paper): LeWM State (Robotic Control) — `exp2_lewm_state/`
Prescribed physical coordinates (x, y, θ) vs. free MLP encoder on Push-T environment. 3 seeds, 50 epochs. Prescribed: 0.004 val loss. Free: 0.157. Gap: 38×. SIGReg ablation: removing SIGReg improves free by 1.9×; SIGReg on prescribed: 0.6% effect.

**Run:** Open `lewm_pusht_colab.ipynb` in Google Colab, or run `lewm_pusht_experiment.py --mode synthetic` locally without dependencies.

### Experiment 4 (paper): LeWM Pixel (CNN Encoder) — `exp3_lewm_pixel/`
Prescribed state coordinates (20K params) vs. CNN encoder on raw 96×96 pixels (744K params). Prescribed: 14.8× lower error. CNN plateaus at epoch 7 out of 50. The advantage is separation of concerns: the free encoder must simultaneously learn what to represent and how to predict; prescribed axes remove the first task entirely.

**Run:** Open `lewm_pixels_full.ipynb` in Google Colab, or run locally on CPU.

### Section 6.3: Pendulum Negative Control — `exp15_pendulum/`
2-DOF system (θ, θ̇), 200 episodes, 30 epochs, 3 seeds, 60 runs total. Tests prescribed vs free when both conditions receive the same 2D state (no subspace selection advantage). Two normalization variants: raw (θ/π, θ̇/5) and min-max [0,1]. Free wins at all dimensions under both normalizations. Normalization improves prescribed by 72–74%. At dim=2 with normalization, gap narrows to 0.87× (near parity), but free has 63× higher variance across seeds.

**Run:** `python exp15_pendulum/code/run_e15.py` (~4 minutes on CPU, resumes from checkpoint).

### Equal-Input Control + SIGReg Ablation — `reviewer_response_experiments.py`
Five conditions × 3 seeds, isolating fixation from information access:

| Condition | Mean val loss | vs Prescribed |
|---|---|---|
| Prescribed | 0.000472 | 1.0× |
| Free (5D input) | 0.011614 | 24.6× |
| Free 3D (same input as prescribed) | 0.003570 | 7.6× |
| Free no SIGReg | 0.005987 | 12.7× |
| Prescribed no SIGReg | 0.000469 | 1.0× |

Free_3d receives identical normalized (x, y, θ) input — same information, but learned projection instead of fixed identity. 7.6× worse confirms the advantage is from fixation, not data access.

**Run:** `python reviewer_response_experiments.py --synthetic --seeds 42 123 777`

### Random Axes Control — `random_axes_control/`

Isolating fixation from axis semantics at two data scales:

**Low-data regime (200 episodes, 30 epochs, 3 seeds):**

| Condition | Mean val loss | vs Prescribed |
|---|---|---|
| prescribed | 0.000673 | 1.0× |
| random_fixed | 0.000408 | 0.61× (better) |
| free_3d | 0.003007 | 4.47× (worse) |

**High-data regime (500 episodes, 50 epochs, 3–9 runs per condition):**

| Condition | Mean val loss | vs Prescribed |
|---|---|---|
| prescribed | 8.51×10⁻⁴ | 1.0× |
| random_fixed | 8.51×10⁻⁴ | 1.00× |
| free_3d | 1.22×10⁻⁹ | ~700,000× better |
| free_5d | 2.92×10⁻⁹ | ~290,000× better |

At low data, fixation wins. At high data, flexibility wins — the free encoder surpasses fixed conditions by six orders of magnitude. Prescribed axes provide sample efficiency, not absolute superiority.

**Run:** `python random_axes_control/run_random_axes_control.py`

## Key Finding

Prescribed axes reframe the collapse problem. The current debate asks: "how do we prevent collapse in a free space?" — through regularization (VICReg, SIGReg) or generative supervision (PAN). Prescribed axes ask a different question: "why does the encoder define the space at all?"

This is not a competing method but a change in problem formulation. The SIGReg diagnostic confirms this empirically: SIGReg is neither helpful nor harmful under prescribed axes — it is irrelevant. The equal-input control confirms the advantage is from coordinate fixation, not privileged information.

The random axes control reveals both the mechanism and its limits: prescribed axes eliminate the dual-task problem (simultaneously learning the space and predicting in it), but at the cost of a fixed accuracy ceiling. In the low-data regime, fixation dominates; in the high-data regime, the flexibility of a learned space dominates. The advantage is sample efficiency — guaranteed coordinate stability from the first epoch, without requiring the free encoder to discover it.

The pendulum negative control establishes a boundary condition: when there is no subspace selection advantage (prescribed and free receive the same dimensions) and the free encoder has sufficient data, fixation provides stability but not accuracy. Prescribed axes are not universally better — they are better in a specific regime defined by data scarcity, the need for information selection, and the cost of coordinate instability.

## Installation

```bash
git clone https://github.com/revenue7-eng/prescribed-axes.git
cd prescribed-axes
pip install -r requirements.txt
```

For Push-T experiments with real physics: install `gymnasium`, `gym-pusht`, and `pymunk` (commented in `requirements.txt`). For dependency-free synthetic mode, base requirements suffice.

## Continuation: Paper 2

**→ [Semantic Drift, Not Rank Collapse](https://github.com/revenue7-eng/prescribed-axes-drift)** — investigates the dynamics of the free encoder's representation space that prescribed axes bypass: coordinate drift, its two phases, and why standard remedies (LR tuning, EMA, PCA alignment) do not resolve it.

## Citation

```
@article{lazarev2026prescribed,
  title={The Space Matters More Than the Loss: JEPA Collapse as a Problem of Structure, Not Optimization},
  author={Lazarev, Andrey},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE)
