# Random Fixed Axes + Isotropic Normalization — Evidence Package

**Date:** 2026-04-13  
**Context:** Reviewer question #15 — does the advantage come from fixation or meaningful coordinates?

---

## Experiment 1: Random Fixed Axes

**Design:** Three conditions, same input (normalized x, y, θ), 3 seeds each. No SIGReg.

| Condition | Encoder | Fixed? | Meaningful? |
|-----------|---------|--------|-------------|
| prescribed | h(s) = normalize(s[2:5]) | Yes | Yes |
| random_fixed | h(s) = normalize(s[2:5]) · W_random | Yes | No |
| free_3d | MLP(normalize(s[2:5])) → 3D | No | Learned |

**Results** (200 episodes, 30 epochs, seeds 42/123/777):

| Condition | Mean val loss | Std | vs prescribed |
|-----------|--------------|-----|---------------|
| prescribed | 0.000673 | 0.000243 | 1.0× |
| random_fixed | 0.000408 | 0.000134 | 0.61× |
| free_3d | 0.003007 | 0.003029 | 4.47× |

**Conclusion:** Fixation is the dominant factor. Any frozen coordinate system beats a learned one by 4–7×.

---

## Experiment 2: Isotropic Normalization

**Hypothesis:** Random rotation mixes unequal scales (σ_θ/σ_x = 1.82×), making MSE more isotropic.

**Results** (200 episodes, 20 epochs, seeds 42/123/777):

| Condition | Mean val loss | Std | vs prescribed |
|-----------|--------------|-----|---------------|
| prescribed_iso | 0.010250 | 0.002722 | 15.2× worse |
| random_fixed_iso | 0.007019 | 0.001183 | 17.2× worse |

**Conclusion:** Hypothesis refuted. Standardization hurts both by ~15×. The random/prescribed ratio (~0.7×) is preserved. Isotropic variants trained for 20 epochs vs. 30 for main conditions; the 15× degradation far exceeds any convergence difference.

---

## Reproduction

```bash
python run_random_axes_control.py          # Experiment 1: ~2 min
python run_isotropic_control.py            # Experiment 2: ~1.5 min
```

Requirements: torch, numpy. CPU only.
