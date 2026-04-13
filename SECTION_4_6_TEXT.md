# Section 4.6: Random Axes Control

## Text for insertion into the paper (after Section 4.5 or current Section 4.4)

---

### 4.6 Random Axes Control

The equal-input control (Section 4.4) established that the advantage of prescribed axes comes from coordinate fixation, not information access. A natural follow-up question arises: does the advantage come from *meaningful* coordinates, or from fixation alone?

To answer this, we introduce a third condition: **random_fixed**, which applies a frozen random orthogonal matrix W to the same normalized input as prescribed. The three conditions receive identical information — normalized (x, y, θ) — and differ only in how coordinates are assigned:

| Condition | Encoder | Fixed? | Meaningful? |
|-----------|---------|--------|-------------|
| prescribed | h(s) = normalize(s[2:5]) | Yes | Yes |
| random_fixed | h(s) = normalize(s[2:5]) · W | Yes | No |
| free_3d | MLP(normalize(s[2:5])) → 3D | No | Learned |

W is a frozen orthonormal matrix (seed=9999), identical across all training seeds. No SIGReg in any condition.

**Results** (200 episodes, 30 epochs, 3 seeds):

| Condition | Mean val loss | Std | vs prescribed |
|-----------|--------------|-----|---------------|
| prescribed | 0.000673 | 0.000243 | 1.0× |
| random_fixed | 0.000408 | 0.000134 | 0.61× |
| free_3d | 0.003007 | 0.003029 | 4.47× |

The random fixed projection performs comparably to prescribed — and slightly better (0.61×). The free encoder with identical input but learned projection is 4.5× worse and highly unstable (std 0.003 vs 0.0001–0.0002). Seed 777 under free_3d produces 7.29×10⁻³ — nine times worse than the best seed — while both fixed conditions show low variance across seeds.

**Isotropic normalization control.** One hypothesis for the random_fixed advantage is that the random rotation mixes the unequal axis scales (σ_θ/σ_x = 1.82×), producing a more isotropic MSE landscape. To test this, we standardized prescribed axes to zero-mean unit-variance before training. Standardization degraded both conditions by ~15× (prescribed_iso: 0.010250; random_fixed_iso: 0.007019), but the random/prescribed ratio was preserved (~0.7×). The gap is not explained by scale isotropy. Isotropic variants were trained for 20 epochs vs. 30 for the main conditions; the 15× degradation far exceeds any convergence difference from 10 additional epochs.

**Interpretation.** The advantage of prescribed axes is primarily from coordinate stability, not axis semantics. The predictor adapts to any fixed coordinate system; what it cannot tolerate is a drifting target space. In this domain, the random orthogonal projection preserves all distances and is invertible — the predictor compensates for the rotation.

This does not diminish the value of meaningful coordinates. In domains where axis semantics are available, they provide interpretability, transferability across tasks, and composability with domain knowledge. But the primary mechanism is fixation: removing the dual-task problem where the model must simultaneously learn the representation space and predict within it.

---

## Notes for Discussion section (suggested addition to 6.1)

Add after the current text about prescribed axes as reframing:

"The random axes control (Section 4.6) further clarifies the mechanism. Prescribed axes work not because domain-meaningful coordinates are inherently superior, but because any fixed coordinate system eliminates the dual-task problem: the predictor need not track a moving target space. In domains where meaningful coordinates are available, they provide the additional benefit of interpretability and transferability — but the primary mechanism is fixation. Whether meaningful coordinates provide additional advantages in higher-dimensional or more complex domains remains an open question."
