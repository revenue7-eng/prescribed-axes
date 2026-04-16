---
title: "The Space Matters More Than the Loss: JEPA Collapse as a Problem of Structure, Not Optimization"
author: "Andrey Lazarev — Independent Researcher"
date: "April 2026"
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
---

# Abstract

When a JEPA model trains in a free latent space, it can collapse — map all inputs to a single point. Existing solutions address this through the loss function: adding regularizers (VICReg, SIGReg) or a generative decoder (PAN). We show that in settings where domain-relevant coordinates are available, the problem is not the loss but the structure of the space. If the axes of the latent space are defined before training — from physical coordinates, semantic features, or cluster centroids — collapse becomes structurally impossible, and regularization is unnecessary.

Four experiments across different modalities confirm this: a pre-defined latent space structure yields +5% accuracy (vision), 38× lower prediction error (states), 14.8× lower error with 37× fewer parameters (pixels), and +18--20 percentage points in representation entropy (speech). An equal-input control confirms the advantage stems from coordinate fixation, not privileged information access: a free encoder receiving identical input is 7.6× worse than prescribed. A random axes control reveals both the mechanism and its limits: in the low-data regime (200 episodes), any fixed coordinate system outperforms a learned one by 4--7×; in the high-data regime (500 episodes), the free encoder surpasses fixed conditions by six orders of magnitude. The advantage of prescribed axes is sample efficiency — eliminating the dual-task problem of simultaneously learning the space and predicting within it — not absolute superiority. SIGReg degrades free representations by 1.9× and has no effect on pre-defined ones — regularization treats what the right space structure prevents.

A negative control on a simple pendulum (2 DOF, no subspace selection) shows the free encoder matching or surpassing prescribed at all dimensions, with normalization reducing the gap by 72--74%. This establishes a boundary condition: prescribed axes require either a subspace selection advantage or insufficient data for the free encoder to converge.

These results reframe the collapse problem in structured domains: prescribed axes offer a structurally guaranteed, data-efficient alternative to learned spaces, with clear boundary conditions on when they apply.

# 1. Introduction

JEPA architectures predict latent representations of masked inputs from visible context, without reconstructing pixels (LeCun, 2022; Assran et al., 2023). This formulation has a structural vulnerability: if the encoder can minimize loss by mapping everything to a single point rather than learning to distinguish inputs, it will. This is the problem of representational collapse.

Two main directions have emerged in the current discussion around this problem:

1. Constrain the space through the loss function: VICReg (Bardes et al., 2022) penalizes axis degeneration, SIGReg (Balestriero & LeCun, 2025) forces the embedding distribution toward an isotropic Gaussian.

2. Supplement prediction with reconstruction: PAN (Ozkara et al., 2025) adds a generative decoder, tying latent dynamics to observable pixels.

Both solutions assume the problem is *how* the model trains. We propose something different: the problem is *where* the model trains — in which space. If the axes of the latent space are defined before training — from physical coordinates, semantic features, or cluster centroids — then collapse is structurally impossible, and regularization is unnecessary. We call this approach *prescribed axes*.

We test this approach on four tasks: speech (clustering anchors), vision (UI screenshots), state prediction (Push-T robot), and pixel prediction (CNN vs. pre-defined structure). Different modalities, different architectures, different metrics — same result.

# 2. Method

## 2.1 Problem Setup

Consider a JEPA with an encoder $f_\theta$, a predictor $g_\phi$, and a target encoder (typically an EMA copy of $f_\theta$). For input $x$ with context $x_c$ and target $x_t$, the standard training objective is:

$$\mathcal{L} = \| g_\phi(f_\theta(x_c)) - \text{sg}(\bar{f}(x_t)) \|^2$$

where sg denotes stop-gradient. The collapse problem arises because the trivial solution $f_\theta(x) = c$ for all $x$ achieves $\mathcal{L} = 0$ when $g_\phi$ learns the identity mapping. Regularization adds penalty terms to $\mathcal{L}$ to prevent this degenerate case.

## 2.2 Prescribed Axes

We replace the learned encoder $f_\theta$ with a fixed, pre-defined mapping $h: X \to \mathbb{R}^d$ that extracts semantically meaningful coordinates. The modified objective becomes:

$$\mathcal{L}_p = \| g_\phi(h(x_c)) - h(x_t) \|^2$$

Three differences from the standard formulation: (1) $h$ has no learnable parameters — it is defined by domain knowledge before training begins; (2) no stop-gradient is needed because $h$ is fixed; (3) no EMA target encoder is needed because the target representation $h(x_t)$ is deterministic.

Collapse is impossible by construction: $h$ is fixed and distinguishes inputs, so any constant prediction $g_\phi(z) = c$ yields $\mathcal{L}_p > 0$. The only path to zero loss is perfect prediction. A context-independent predictor can reduce loss by learning the marginal mean $\mathbb{E}[h(x_t)]$, but cannot reach zero — and in deterministic environments, the mean is a poor approximation when context is informative.

This guarantee is deliberately minimal: it shows that prescribed axes eliminate collapse, not that they produce good predictions. The empirical question — whether the predictor learns useful structure within the prescribed space — is addressed by Experiments 1--4 and the equal-input control (Section 4.4).

Since $h$ is fixed and has no learnable parameters, EMA and stop-gradient are unnecessary — not as a methodological advantage, but as a direct consequence of the formulation. The structure of the space makes regularization unnecessary.

The prescribed mapping uses domain information unavailable to the free encoder. Experiment 4 tests whether a more powerful encoder can compensate, and the equal-input control (Section 4.4) isolates fixation from information access.

## 2.3 Instantiations Across Experiments

**Vision (Shov-JEPA):** $h(x) = [P(x), F(x), D(x)]$ — position, functionality, and depth from the View Hierarchy XML, producing a 3D or 12D feature vector per UI element.

**State prediction (LeWM State):** $h(s) = \text{normalize}(s[2:5])$ — normalized block coordinates $(x, y, \theta)$ from the full 5D state vector. Normalization is min-max to $[0,1]$ using dataset-wide constants: $x$ and $y$ are divided by 512 (workspace range), $\theta$ by $2\pi$. These constants are fixed before training and do not depend on the training data distribution.

A second prescribed condition, prescribed\_full, uses all five state coordinates with the same normalization. Its worse performance (2.4× vs prescribed\_block; see Table 3) suggests that including irrelevant dimensions (agent position) dilutes the prediction signal.

**Pixel prediction (LeWM Pixel):** $h$ is the same state extraction from ground-truth, while the free baseline uses a 4-layer CNN $f_\theta: \mathbb{R}^{3\times96\times96} \to \mathbb{R}^3$.

**Speech (Speech JEPA):** $h(x) = \text{softmax}(-\|x - \mu_k\|^2/\tau)$ — projection onto frozen cluster centroids (GMM or k-means, $K=512$, dim$=256$) in a fixed codebook space. Cluster centroids were fitted on the same LibriSpeech train-clean-100 split used for JEPA training. This means the prescribed encoder has implicit access to training data statistics through the clustering step. We note this as a potential confound.

## 2.4 Free Baseline

In all experiments, the free baseline uses a learned encoder $f_\theta$ with matching output dimensionality. For state prediction: MLP ($5→64→64→3$). For pixel prediction: CNN ($3→32→64→128→256$ channels + MLP $\to 3$). The free encoder is trained end-to-end with the predictor. Where applicable, SIGReg regularization is tested as an ablation on both conditions.

## 2.5 Shared Components

Predictor architecture is identical across all conditions: an MLP mapping $H×2×d$ inputs to $d$ outputs through two hidden layers of 128 units each, with LayerNorm and GELU activations. Action encoder: MLP ($2→32→d$) with GELU. Both are trained end-to-end with the encoder (or with the predictor only, when the encoder is fixed).

# 3. Experiments

We evaluate prescribed axes across four domains with different modalities, architectures, and metrics. All experiments compare prescribed representations against free baselines under matched conditions.

## 3.1 Summary of Results

| Experiment | Modality | Prescribed | Free | Gap | Metric |
|---|---|---|---|---|---|
| Speech JEPA | Speech | 73.7% | 53.2% | +20pp | Entropy (↑) |
| Shov-JEPA | Vision (UI) | 72.5% | 67.5% | +5% | Accuracy |
| LeWM State | States | 0.004 | 0.157 | 38× | Val loss (↓) |
| LeWM Pixel | Pixels | 0.0009 | 0.013 | 14.8× | Val loss (↓) |

Table 1: Prescribed vs. free performance across four experiments. Experiments are ordered by increasing evidential depth: from pilot studies (1--2) to controlled experiments with ablation (3--4). Metrics are domain-specific and not directly comparable across experiments.

## 3.2 Experiment 1: Speech JEPA (Clustering)

We begin with speech: a JEPA architecture with clustering-based anchors, inspired by Ioannides et al. (2026). A 2×2 factorial design crossing clustering method (GMM vs. k-means) with assignment type (soft vs. hard). All four conditions use frozen (prescribed) cluster anchors ($K=512$, dim$=256$). Baseline: Pure JEPA without anchors. Data: LibriSpeech train-clean-100, 2000 training steps.

| Condition | Entropy (%) | Active / 512 | Δ vs Pure JEPA |
|---|---|---|---|
| Pure JEPA | 53.2 | 138 | — |
| GMM + Soft | 71.5 | 128 | +18.3pp |
| GMM + Hard | 71.5 | 128 | +18.3pp |
| KM + Soft | 73.7 | 128 | +20.5pp |
| KM + Hard | 73.7 | 128 | +20.5pp |

Table 2: Representation quality measured by codebook entropy.

The factorial analysis reveals a clear hierarchy of effects. The dominant factor is frozen structure itself (+19.4pp averaged across conditions). The clustering method is secondary (k-means +2.2pp over GMM). Assignment type (soft vs. hard) has no measurable effect ($\Delta < 0.03$pp).

This is a pilot study: entropy measures codebook utilization, not downstream representation quality. The pattern, however, is unambiguous — what matters is that the space is prescribed, not how.

## 3.3 Experiment 2: Shov-JEPA (Vision)

Prescribed axes for UI understanding: Rico dataset (Deka et al., 2017), 398 mobile screenshots. The prescribed representation uses three semantic axes from the View Hierarchy: position (P), functionality (F), depth (D). The free baseline is a 64-dimensional JEPA encoder trained end-to-end.

3 prescribed dimensions yield 72.5%, 64 free dimensions yield 67.5%. The free encoder uses 21× more dimensions — and loses. More dimensions do not compensate for wrong structure — they add capacity where none is needed. A VH-grid JEPA variant combining prescribed features with spatial structure reaches 73.8%, the best result overall. JEPA loss reduction during training: 73--85%, confirming that the predictor learns effectively within the prescribed space.

Unlike Experiment 1, the metric here is classification accuracy — a downstream evaluation: representations trained in prescribed space are used for a separate classification task. This is a pilot study (398 samples, single seed, +5% gap), but it shows the same pattern on a different modality.

## 3.4 Experiment 3: LeWM State (Robotic Control)

Push-T environment (Maes et al., 2026): next-state prediction from history. The prescribed representation uses the block's physical coordinates $(x, y, \theta)$, normalized to $[0,1]$. The free baseline is an MLP encoder ($5→64→64→3$) learning its own 3D embedding. Three seeds (42, 123, 777), 50 epochs.

| Condition | Seed 42 | Seed 123 | Seed 777 | Mean ± std |
|---|---|---|---|---|
| Prescribed (block) | 0.00366 | 0.00448 | 0.00413 | 0.00409 ± 0.00041 |
| Prescribed (full) | 0.00919 | 0.01147 | 0.00924 | 0.00997 ± 0.00130 |
| Free | 0.15562 | 0.15349 | 0.16170 | 0.15694 ± 0.00421 |

Table 3: Validation loss (MSE) across seeds.

For a world model, next-state prediction is not an intermediate metric but the target task: the model exists to predict. A 38× gap means the prescribed world model is qualitatively more accurate than the free one.

The free encoder reaches its best validation loss at epochs 19--22 (depending on seed) and plateaus for the remaining epochs. Prescribed continues improving through epoch 50. This is consistent with representational instability in the free encoder limiting predictor convergence.

**SIGReg ablation.** Removing SIGReg improves the free encoder by 1.9× (mean val loss 0.00599 without vs 0.01161 with, 3 seeds). SIGReg on prescribed: 0.6% effect (0.000469 vs 0.000472). SIGReg and prescribed axes address the same problem (collapse) through different mechanisms. SIGReg constrains what the encoder can do; prescribed axes remove the need to do it. When applied to a space that is already structured, SIGReg has nothing to act on. When applied to a free encoder that has found its own organization, SIGReg disrupts it.

## 3.5 Experiment 4: LeWM Pixel (CNN Encoder)

One might object: prescribed axes win in Experiment 3 because they receive privileged information — ready-made block coordinates that the free encoder must extract on its own. This is a fair objection. Experiment 4 tests one aspect: whether a more powerful encoder with full visual access can compensate. A direct test — giving the free encoder the same three coordinates — is reported in Section 4.4.

The CNN sees pixels — the entire scene, all objects, all positions. Prescribed axes see three numbers: $x, y, \theta$ of the block. CNN: 744K parameters. Prescribed: 20K. Prescribed is 14.8× more accurate (0.000884 vs. 0.01305). The CNN stalls at epoch 7 and shows no improvement across the remaining 43 epochs. Prescribed continues learning through epoch 42.

The CNN uses lr$=$3e-4, batch size 64, no data augmentation, cosine annealing schedule. No hyperparameter search was performed. A better-tuned CNN or a different architecture (e.g., ResNet) might narrow the gap. The 14.8× result should be read as evidence from a specific configuration, not an optimized comparison.

The advantage of prescribed here is not privileged data access but separation of concerns. The free encoder solves two tasks simultaneously: learn what to represent and learn how to predict. Prescribed axes remove the first task entirely. 744K parameters split across both tasks lose to 20K parameters focused on one — not because the CNN lacks information, but because it must simultaneously organize the space and work within it.

# 4. Analysis

## 4.1 The SIGReg Diagnostic

The SIGReg asymmetry is not just another comparison. It is a diagnostic test. If prescribed axes and SIGReg solve the same problem (collapse), then under prescribed axes SIGReg should change nothing — and it does not (0.6% effect). If under free space SIGReg should help — but it hurts (1.9×), then the free encoder has already found its own spatial organization, and SIGReg breaks it.

Prescribed axes do not conflict with regularization — they make it unnecessary.

## 4.2 Parameter Efficiency

A free encoder solves two tasks simultaneously: (1) extract useful features and (2) organize the space to prevent collapse. Prescribed axes remove task (2). All capacity goes to (1). This is why 20K prescribed parameters outperform 744K free parameters — most of the free encoder's capacity is spent on a task that prescribed axes solve for free.

## 4.3 Cross-Modal Consistency

Four modalities, three metrics, different architectures — prescribed wins everywhere. This is not a property of one task. In tasks where meaningful coordinates exist a priori, the structure of the space in which prediction occurs matters more than the power of the mechanism that populates it.

## 4.4 Equal-Input Control

A natural objection to Experiments 3 and 4 is that prescribed axes receive privileged information: ready-made coordinates that the free encoder must extract from raw state or pixels. To isolate the effect of fixation from information access, we tested a free encoder receiving the same normalized $(x, y, \theta)$ input as prescribed — identical information, but with a learned projection (MLP $3→64→64→3$) instead of a fixed identity mapping.

| Condition | Seed 42 | Seed 123 | Seed 777 | Mean ± std |
|---|---|---|---|---|
| Prescribed | 0.000131 | 0.000637 | 0.000646 | 0.000472 ± 0.000241 |
| Free (5D input) | 0.011794 | 0.011520 | 0.011527 | 0.011614 ± 0.000128 |
| Free 3D (same input) | 0.000467 | 0.005024 | 0.005219 | 0.003570 ± 0.002195 |
| Free no SIGReg | 0.004151 | 0.006252 | 0.007558 | 0.005987 ± 0.001403 |
| Prescribed no SIGReg | 0.000135 | 0.000619 | 0.000653 | 0.000469 ± 0.000236 |

Table 4: Equal-input control and SIGReg ablation. Synthetic Push-T data, 3 seeds, 50 epochs.

Two effects are visible. First, input selection: free\_3d (0.00357) is 3.3× better than free\_5d (0.01161), confirming that receiving the right coordinates helps even without fixation. Second, fixation: prescribed (0.000472) is 7.6× better than free\_3d (0.00357), despite receiving identical information. The advantage of prescribed axes is not from privileged data access but from coordinate fixation.

The free\_3d encoder has high variance across seeds (std $= 0.00220$ vs prescribed std $= 0.000241$), consistent with the instability of learned representations even when input information is correct.

## 4.5 Per-Axis Analysis

In the LeWM State experiment, the prescribed advantage is not uniform across axes: position axes ($x, y$) show 53--63× improvement, the angle axis ($\theta$) shows 25×. Linear quantities are easier to prescribe than cyclical ones. We did not test alternative angle encodings such as $(\sin\theta, \cos\theta)$. The 25× gap on the angle axis may partly reflect the linear encoding of a cyclical quantity rather than an intrinsic difficulty. This is a limitation of the current analysis.

## 4.6 Random Axes Control

The equal-input control (Section 4.4) established that the advantage of prescribed axes comes from coordinate fixation, not information access. A natural follow-up: does the advantage come from meaningful coordinates, or from fixation alone? And does the answer depend on data scale?

We introduce a third condition: random\_fixed, which applies a frozen random orthogonal matrix $W$ to the same normalized input as prescribed. The three conditions receive identical information — normalized $(x, y, \theta)$ — and differ only in how coordinates are assigned. We test at two data scales: 200 episodes (low-data) and 500 episodes.

| Condition | Encoder | Fixed? | Meaningful? |
|---|---|---|---|
| prescribed | $h(s) = \text{normalize}(s[2:5])$ | Yes | Yes |
| random\_fixed | $h(s) = \text{normalize}(s[2:5]) \cdot W$ | Yes | No |
| free\_3d | MLP(normalize($s[2:5]$)) $\to$ 3D | No | Learned |

$W$ is a frozen orthonormal matrix, identical across all training seeds. No SIGReg in any condition. At 500 episodes: 3 rotation seeds × 3 training seeds $= 9$ runs for random\_fixed; 3 training seeds for prescribed and free conditions.

**Results, 200 episodes** (30 epochs, 3 seeds):

| Condition | Mean val loss | Std | vs prescribed |
|---|---|---|---|
| prescribed | 0.000673 | 0.000243 | 1.0× |
| random\_fixed | 0.000408 | 0.000134 | 0.61× |
| free\_3d | 0.003007 | 0.003029 | 4.47× |

Table 5: Random axes control, low-data regime. 200 episodes, 30 epochs, 3 seeds.

**Results, 500 episodes** (50 epochs; random\_fixed: 9 runs, others: 3 seeds):

| Condition | Mean val loss | Std | vs prescribed |
|---|---|---|---|
| prescribed | $8.51 \times 10^{-4}$ | $1.0 \times 10^{-6}$ | 1.0× |
| random\_fixed | $8.51 \times 10^{-4}$ | $2.2 \times 10^{-6}$ | 1.00× |
| free\_3d | $1.22 \times 10^{-9}$ | $4.5 \times 10^{-10}$ | ${\sim}700{,}000\times$ better |
| free\_5d | $2.92 \times 10^{-9}$ | $9.6 \times 10^{-10}$ | ${\sim}290{,}000\times$ better |

Table 6: Random axes control, high-data regime. 500 episodes, 50 epochs.

The results invert. At 200 episodes, prescribed and random\_fixed outperform free by 4--7×. At 500 episodes, the free encoder converges to ${\sim}10^{-9}$ — effectively zero — while prescribed and random\_fixed plateau at ${\sim}8.5 \times 10^{-4}$ and do not improve.

**Interpretation.** The prescribed/random\_fixed plateau is the irreducible error of a fixed mapping: min-max $[0,1]$ normalization introduces quantization error that no predictor can eliminate. The free encoder, by contrast, learns its space end-to-end, allowing it to reach arbitrarily low error given sufficient data — including compensation for nonlinear dependencies inaccessible to fixed linear mappings.

The critical transition: at low data (200 episodes), the dual-task problem (simultaneously learning the space and the predictor) dominates, and the free encoder is unstable (std $= 0.003$). At sufficient data (500 episodes), the dual-task problem is solved, the free encoder stabilizes (std $= 4.5 \times 10^{-10}$), and its flexibility wins.

**Conclusion for prescribed axes:** the advantage of prescribed axes is sample efficiency, not absolute superiority. Prescribed axes eliminate the dual-task problem at the cost of a fixed accuracy ceiling. In the low-data regime, fixation is the dominant factor. In the high-data regime, the flexibility of the free encoder dominates. The boundary between regimes lies between 200 and 500 episodes in this experiment.

**Isotropic normalization control** (200 episodes). Standardization to zero-mean unit-variance degrades both fixed conditions by ${\sim}15\times$ (prescribed\_iso: 0.010250; random\_fixed\_iso: 0.007019), while preserving the random/prescribed ratio (${\sim}0.7\times$). The gap is not explained by scale isotropy.

## 4.7 Mechanism: Why Fixation Works

The controls in Sections 4.4--4.6 progressively isolate the mechanism. The equal-input control (Section 4.4) shows that information access is not the explanation: a free encoder receiving the same $(x, y, \theta)$ is 7.6× worse. The random axes control (Section 4.6) shows that axis semantics are not the explanation: a random frozen projection matches prescribed performance. What remains is fixation itself.

A free encoder solves two coupled tasks simultaneously: (1) organize the representation space and (2) predict the next state within it. These tasks interact destructively. The predictor trains on a coordinate system that changes every gradient step. The encoder adapts its space in response to prediction error, but this adaptation moves the target for the predictor. Neither module converges independently — they co-evolve in a feedback loop where the predictor chases a moving coordinate system.

Prescribed axes break this loop by removing task (1). The coordinate system is fixed from epoch 0. The predictor sees a stable target from the first gradient step and can converge without interference. This is why 20K prescribed parameters outperform 744K free parameters (Experiment 4): the free encoder's capacity is split across both tasks, while the prescribed system concentrates all capacity on prediction.

Three lines of evidence support this account:

First, convergence dynamics. In Experiment 3, prescribed continues improving through epoch 50. The free encoder reaches its best validation loss at epochs 19--22 and plateaus — consistent with the predictor failing to track coordinate drift. In Experiment 4, the CNN encoder stalls at epoch 7 and shows no improvement for 43 epochs.

Second, variance across seeds. Free\_3d in the equal-input control has std $= 0.00220$ vs prescribed std $= 0.000241$. In the pendulum experiment (Section 6.3), free variance is 63× higher than prescribed at matched dimensionality. The free encoder's trajectory through representation space is seed-dependent; prescribed axes produce the same trajectory every time.

Third, the data-scale transition. At 200 episodes, the dual-task problem dominates and fixation wins by 4--7×. At 500 episodes, the free encoder has sufficient data to solve both tasks and surpasses prescribed by six orders of magnitude. The transition point marks where the cost of co-learning drops below the cost of a fixed ceiling.

The mechanism is not that prescribed axes provide better information, nor that they find the right coordinate system. The mechanism is that any fixed coordinate system eliminates the instability of co-learning, and in the low-data regime this instability — not representation quality — is the binding constraint.

# 5. Related Work

**Regularization-based approaches.** VICReg (Bardes et al., 2022) adds variance, invariance, and covariance penalties. SIGReg (Balestriero & LeCun, 2025) uses sketched isotropic Gaussian regularization. Both treat collapse as an optimization problem. Our SIGReg experiment (Section 4.1) shows that regularization is not merely unnecessary under pre-defined structure — it is harmful to free encoders.

**Architectural approaches.** I-JEPA (Assran et al., 2023) uses an EMA target encoder for stable targets. V-JEPA (Bardes et al., 2024) extends this to video. V-JEPA 2.1 (Mur-Labadia et al., 2026) introduces dense predictive loss and hierarchical self-supervision for spatial grounding. These approaches stabilize training but retain the free latent space paradigm.

**Generative approaches.** PAN (Ozkara et al., 2025) couples latent dynamics with a generative video decoder, arguing that latent prediction without reconstruction is insufficient for grounded world modeling. This represents the second pole of the current debate. Our approach avoids generation entirely by fixing the prediction target space.

**Contrastive and non-contrastive approaches.** BYOL (Grill et al., 2020) and SimSiam (Chen & He, 2021) prevent collapse through asymmetric architectures without negative pairs. Barlow Twins (Zbontar et al., 2021) decorrelates embedding dimensions. DINO (Caron et al., 2021) uses self-distillation with centering. These methods address collapse more broadly in self-supervised learning. Our work focuses on JEPA architectures; the prescribed axes principle may apply to these frameworks as well, but this remains untested.

**Structured representations.** Speech JEPA with clustering anchors (Ioannides et al., 2026) is closest to our Experiment 1, using frozen cluster centroids as structured targets. Our factorial analysis extends their work by isolating the contribution of frozen structure from the clustering method. STP (Huang, LeCun, & Balestriero, 2026) straightens representation trajectories but does not prescribe the space itself.

# 6. Discussion

## 6.1 Prescribed Axes as a Reframing of the Problem

The current debate in JEPA research is framed as a choice: regularization (LeCun/Meta FAIR) or generative supervision (PAN/MBZUAI). Both directions answer the same question: "how do we prevent collapse in a free space?" They accept the free latent space as given and look for ways to cope with it.

Prescribed axes do not answer this question. They cancel it. The question prescribed axes ask: "why does the encoder define the space at all?" This is a different level — not a solution method, but a problem formulation.

An analogy from physics: choosing coordinates is not a third method of solving an equation alongside two others. It is a decision made before the choice of method, one that determines how effective any method will be. The analogy has a limit: in physics, coordinate changes are invertible — no information is lost. Prescribed axes are not invertible: $h(x)$ projects into a subspace, and information not encoded in $h$ is discarded. The analogy holds for the principle — a decision made before the method, affecting all downstream performance — but not for information preservation.

The SIGReg diagnostic confirms this empirically. If prescribed axes were an alternative to SIGReg at the same level, SIGReg should either conflict with or complement them. It does neither — it is irrelevant. Prescribed axes operate at a level where regularization is not yet defined.

The random axes control (Section 4.6) reveals a more nuanced picture than expected. At 200 episodes, any fixed coordinate system (meaningful or random) outperforms a learned one by 4--7×. At 500 episodes, the relationship inverts: the free encoder converges to ${\sim}10^{-9}$ while fixed conditions plateau at ${\sim}8.5 \times 10^{-4}$. Prescribed axes eliminate the dual-task problem — but at the cost of a fixed accuracy ceiling. The advantage is sample efficiency: in the low-data regime, fixation dominates; in the high-data regime, the flexibility of a learned space dominates. This reframes prescribed axes from an absolute solution to a regime-dependent one, with the boundary determined by whether the free encoder has sufficient data to solve the dual-task problem.

This claim is partially falsified by our own data: at 500 episodes, a free encoder with the same input consistently surpasses prescribed performance by six orders of magnitude. The dual-task problem is real — but solvable with sufficient data. What prescribed axes provide is not a better solution but a faster path: the guarantee that coordinate stability holds from the first epoch, without requiring the free encoder to discover it.

## 6.2 Limitations

The main limitation of prescribed axes is the requirement for domain knowledge. In our experiments, the axes are straightforward: physical robot coordinates, semantic UI features, acoustic clusters. But what happens when the right axes are not obvious?

We see three paths. First: prescribed axes from pilot studies — train a small free model, analyze what it learned, and freeze the discovered axes. Second: hierarchical prescribing — define coarse axes (position, scale, type) and leave fine structure free. Third: domain knowledge engineering — in robotics, medicine, and finance, meaningful coordinates are already known to practitioners. None of these paths is tested in this work.

Our experiments are small: 3-dimensional state spaces, hundreds to thousands of samples. We do not know whether the effect holds at 128 or 768 dimensions. In high-dimensional spaces, random projections preserve distances (Johnson--Lindenstrauss), and gauge freedom may be less catastrophic. The scaling question is open.

We compare prescribed axes against free MLP and CNN baselines, not against established self-supervised methods (I-JEPA, V-JEPA, BYOL, DINO). Our free baselines may be weaker than state-of-the-art. The comparison isolates the structural effect (fixed vs learned space), not benchmark performance. Whether prescribed axes improve upon strong free methods is an open question.

Prescribed JEPA is deliberately closer to supervised regression with hand-crafted features than to self-supervised representation learning. We do not claim novelty in the method. The contribution is diagnostic: when we remove the free encoder and replace it with fixed coordinates, system performance improves dramatically. This tells us something about the cost of free encoding, not about the sophistication of prescribed encoding.

All experiments use deterministic environments (or deterministic mappings from state to representation). In stochastic environments, the predictor must approximate a conditional distribution rather than a point, and the advantage of prescribed axes may change. This has not been tested.

Additional confounds we did not isolate: the action encoder must learn in a drifting space for free conditions; weight decay (1e-3) applies differently when the encoder has no learnable parameters; gradient clipping budget is shared differently; the SIGReg implementation generates random projections on each forward pass without fixing the seed. We used the same cosine annealing schedule for all conditions; the free encoder may benefit from alternative schedules.

The pendulum experiment (Section 6.3) reveals that prescribed axes do not generalize to all dynamical systems. When prescribed and free encoders receive the same dimensions — no subspace selection — the free encoder wins. This suggests the Push-T advantage has two components: coordinate stability and information selection (choosing 3 relevant dimensions from 5). Disentangling these two components further — for instance, testing prescribed axes with all 5 Push-T dimensions against free with all 5 — would clarify the relative contribution of each.

Normalization is a confound we did not fully control. On the pendulum, switching from ad-hoc normalization ($\theta/\pi$, $\dot\theta/5$) to min-max $[0,1]$ reduces prescribed error by 72--74%. Push-T uses min-max normalization by design, so this confound does not affect the main results. But it suggests that reported gaps in other prescribed-axes applications may be partly attributable to normalization quality rather than the structural advantage itself.

A fifth experiment is currently underway on the EB-JEPA codebase (Bar et al., 2025), testing whether the same structural effect reproduces on Meta FAIR's architecture with planning success rate as the metric. This is the subject of a separate study.

## 6.3 Boundary Conditions: Where Prescribed Axes Fail

The preceding experiments test prescribed axes on Push-T — a 3-DOF system with known physical coordinates. An additional experiment tests whether the advantage generalizes to a different dynamical system.

**Simple pendulum (E15).** A 2-DOF system $(\theta, \dot\theta)$ with 200 episodes, 30 epochs, 3 seeds. Unlike Push-T, prescribed and free encoders receive the same 2D state — there is no subspace selection advantage. At all dimensions (1--5) and under both normalization schemes, the free encoder outperforms prescribed.

| Dim | Prescr. (raw) | Free (raw) | Prescr. (norm) | Free (norm) |
|-----|---------------|------------|----------------|-------------|
| 1 | 0.005682 | 0.004206 | 0.001462 | 0.000787 |
| 2 | 0.001754 | 0.000278 | 0.000463 | 0.000403 |
| 3 | 0.001187 | 0.000304 | 0.000332 | 0.000160 |
| 4 | 0.000898 | 0.000186 | 0.000240 | 0.000169 |
| 5 | 0.000727 | 0.000207 | 0.000204 | 0.000144 |

Table 7: Simple pendulum results. "Raw" $= \theta/\pi, \dot\theta/5$. "Norm" $=$ min-max $[0,1]$. 3 seeds, 200 episodes, 30 epochs.

Two patterns are visible. First, normalization reduces prescribed error by 72--74% across all dimensions. The raw prescribed axes ($\theta/\pi, \dot\theta/5$) have an ad-hoc normalization that creates a ${\sim}3\times$ scale mismatch between axes. Min-max $[0,1]$ eliminates this mismatch and closes much of the gap. Second, even after normalization, free wins — but the gap narrows from 2--6× (raw) to 1.1--2× (norm). At dim$=$2 with normalization, the gap is only 0.87× (prescribed 0.000463 vs free 0.000403), approaching parity.

Critically, the free encoder at dim$=$2 norm shows high variance across seeds: seed 42 achieves 0.000032, seed 777 achieves 0.001118 — a 35× spread. Prescribed variance is 63× lower (std 0.000008 vs std 0.000506). The prescribed encoder is more stable but less accurate.

**Interpretation.** The pendulum result does not contradict the Push-T findings. In Push-T, prescribed axes select a relevant 3D subspace from a 5D state — the free encoder must discover which dimensions matter. In the pendulum, there is no subspace selection: both conditions receive the full 2D state. The prescribed advantage on Push-T comes from two factors: coordinate stability and information selection. The pendulum isolates coordinate stability alone, and in this 2-DOF system with 200 episodes, the free encoder can solve the dual-task problem sufficiently well.

This establishes a boundary condition: prescribed axes require either (a) a subspace selection advantage (state\_dim $>$ latent\_dim) or (b) insufficient data for the free encoder to solve the dual-task problem. When neither condition holds, fixation provides stability but not accuracy.

## 6.4 Implications

Representation learning assumes by default that representations should be unconstrained — let the model find its own structure. Our results complicate this assumption. In the low-data regime, a free encoder with 37× more capacity loses to prescribed by 14.8×. An equal-input free encoder with identical information loses by 7.6×. This gap arises because the free encoder must simultaneously organize the representation space and learn to predict within it — two tasks that prescribed axes separate. But the random axes control shows that this gap closes with sufficient data: at 500 episodes, the free encoder surpasses prescribed by six orders of magnitude. The dual-task problem is not unsolvable — it is expensive, and prescribed axes eliminate that cost.

# 7. Conclusion

Prescribed axes stabilize the latent space across four modalities — vision, states, pixels, speech — and under different metrics: accuracy, prediction loss, entropy. Regularization that is necessary for free spaces becomes unnecessary — and potentially harmful — under prescribed ones. An equal-input control confirms the advantage is from coordinate fixation (7.6×), not information access. A random axes control reveals the mechanism and its limits: in the low-data regime (200 episodes), any fixed coordinate system outperforms a learned one by 4--7×. In the high-data regime (500 episodes), the free encoder surpasses fixed conditions by six orders of magnitude, converging to near-zero error while prescribed axes plateau. The advantage of prescribed axes is sample efficiency — eliminating the dual-task problem at the cost of a fixed accuracy ceiling.

A negative control on a simple pendulum (2 DOF) establishes a boundary condition: when there is no subspace selection advantage and the free encoder has sufficient data, fixation provides stability but not accuracy. Prescribed axes are not universally better — they are better in a specific regime defined by data scarcity, the need for information selection, and the cost of coordinate instability.

Prescribed axes are not a universal solution, and they are not unconditionally better than learned spaces. But they show that in structured domains with limited data, collapse is a problem of structure, not optimization. The solution lies not in making the loss more complex, but in making the task simpler — with the understanding that this simplification has a cost when data is abundant.

# 8. Reproducibility

All four experiments are implemented as self-contained notebooks (Google Colab compatible) with preserved execution outputs. Code, data, and results are publicly available at https://github.com/revenue7-eng/prescribed-axes.

Experiment 3 includes a standalone Python script with a synthetic data mode for dependency-free verification of the training pipeline. The standalone script's synthetic mode uses simplified physics and does not reproduce the exact numbers reported in the paper, which were obtained using gym-pusht. The equal-input control and SIGReg ablation are implemented in reviewer\_response\_experiments.py with full result archiving. The pendulum negative control is implemented in exp15\_pendulum/code/run\_e15.py with per-seed JSON output.

Experiment 3 (LeWM State) was run with three seeds; Experiments 1, 2, and 4 with a single seed. For Experiment 2 (Shov-JEPA), a +5% gap with a single seed represents pilot-level evidence. For Experiment 4 (LeWM Pixel), a 14.8× gap makes single-seed less critical, but additional seeds would strengthen the result.

# References

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR 2023*.

Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. *arXiv:2511.08544*.

Bar, A., Terver, G., Garrido, Q., Balestriero, R., Rabbat, M., & LeCun, Y. (2025). EB-JEPA: Joint-Embedding Predictive Architecture with Energy-Based Models. *GitHub: facebookresearch/eb\_jepa*.

Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*.

Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., & Ballas, N. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video. *arXiv:2404.08471*.

Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV 2021*.

Chen, X., & He, K. (2021). Exploring Simple Siamese Representation Learning. *CVPR 2021*.

Deka, B., Huang, Z., Franzen, C., Hibschman, J., Afergan, D., Li, Y., Nichols, J., & Kumar, R. (2017). Rico: A Mobile App Dataset for Building Data-Driven Design Applications. *UIST 2017*.

Grill, J.-B., et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. *NeurIPS 2020*.

Huang, H., LeCun, Y., & Balestriero, R. (2026). Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA. *arXiv:2602.22617*.

Ioannides, G., et al. (2026). Soft Clustering Anchors for Self-Supervised Speech Representation Learning in Joint Embedding Prediction Architectures. *arXiv:2602.09040*.

LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *Technical Report, Meta AI*.

Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. *arXiv:2603.19312*.

Mur-Labadia, L., et al. (2026). V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning. *arXiv:2603.14482*.

Ozkara, B., et al. (2025). PAN: A World Model for General, Interactable, and Long-Horizon World Simulation. *arXiv:2511.09057*.

Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. *ICML 2021*.
