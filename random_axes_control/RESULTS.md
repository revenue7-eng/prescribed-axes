# Random Fixed Axes — Evidence Package (v2)

**Date:** 2026-04-14
**Context:** Reviewer question #15 + scaling analysis (200 vs 500 episodes)

---

## Design

**Question:** Does the advantage of prescribed axes come from fixation or from meaningful coordinates? Does the answer depend on data scale?

| Condition | Encoder | Fixed? | Meaningful? |
|-----------|---------|--------|-------------|
| prescribed | h(s) = normalize(s[2:5]) with [1/512, 1/512, 1/2π] | Yes | Yes |
| random_fixed | h(s) = normalize(s[2:5]) @ W_random | Yes | No |
| free_3d | MLP(normalize(s[2:5])) → 3D | No | Learned |
| free_5d | MLP(s[0:5]) → 3D | No | Learned |

No SIGReg. Pure MSE prediction loss.

---

## Experiment 1: Low-Data Regime (200 episodes)

W_random: frozen orthonormal matrix, seed=9999, same across all training seeds.
30 epochs, seeds 42/123/777.

| Condition | Seed 42 | Seed 123 | Seed 777 | Mean | Std |
|-----------|---------|----------|----------|------|-----|
| prescribed | 0.000333 | 0.000883 | 0.000804 | **0.000673** | 0.000243 |
| random_fixed | 0.000219 | 0.000502 | 0.000502 | **0.000408** | 0.000134 |
| free_3d | 0.000801 | 0.000929 | 0.007290 | **0.003007** | 0.003029 |

**Key ratios:**
- random_fixed / prescribed = **0.61×** (random is slightly better)
- free_3d / prescribed = **4.47×** (learned is worse)

**Conclusion at this scale:** Fixation is the dominant factor. Any frozen coordinate system beats a learned one by 4–7×.

---

## Experiment 2: High-Data Regime (500 episodes)

50 epochs. 3 training seeds (42/123/777). random_fixed: 3 rotation seeds × 3 training seeds = 9 runs.

| Condition | Mean val loss | Std | vs prescribed |
|-----------|--------------|-----|---------------|
| prescribed | 8.51×10⁻⁴ | 1.0×10⁻⁶ | 1.0× |
| random_fixed | 8.51×10⁻⁴ | 2.2×10⁻⁶ | 1.00× |
| free_3d | 1.22×10⁻⁹ | 4.5×10⁻¹⁰ | ~700,000× better |
| free_5d | 2.92×10⁻⁹ | 9.6×10⁻¹⁰ | ~290,000× better |

**The results invert.** At 500 episodes, the free encoder converges to ~10⁻⁹ — effectively zero — while prescribed and random_fixed plateau at ~8.5×10⁻⁴.

**Training curves** (seed=42): see `random_axes_results_500ep.png`. prescribed/random_fixed flat at ~10⁻³; free_3d/free_5d monotonically decrease through all 50 epochs.

**Full per-run data:** `all_results_500ep.json` (18 runs total: 3 prescribed, 9 random_fixed, 3 free_3d, 3 free_5d).

---

## Experiment 3: Isotropic Normalization Control (200 episodes)

**Hypothesis:** random_fixed is better at 200ep because random rotation mixes x/y/θ scales, making MSE more isotropic.

Axis statistics after min-max normalization:
```
mean: [0.5094, 0.5047, 0.5109]
std:  [0.1604, 0.1724, 0.2921]
std ratio θ/x: 1.82×
```

| Condition | Mean | Std |
|-----------|------|-----|
| prescribed | 0.000673 | 0.000243 |
| prescribed_iso | 0.010250 | 0.002722 |
| random_fixed | 0.000408 | 0.000134 |
| random_fixed_iso | 0.007019 | 0.001183 |

Note: prescribed/random_fixed from Experiment 1 (30 epochs). Iso variants ran 20 epochs.

**Hypothesis refuted.** Standardization makes both conditions ~15× worse. The random/prescribed ratio is preserved (~0.6–0.7×). The gap is NOT explained by scale isotropy.

---

## Combined Interpretation

1. **At low data (200 episodes): fixation dominates.** Any fixed coordinate system (meaningful or random) beats a learned one by 4–7×. The dual-task problem (simultaneously learning the space and predicting in it) is the bottleneck.

2. **At high data (500 episodes): flexibility dominates.** The free encoder converges to ~10⁻⁹ while fixed conditions plateau at ~8.5×10⁻⁴. The irreducible error of fixed mappings (quantization from min-max [0,1] normalization) cannot be eliminated by any predictor.

3. **Prescribed axes = sample efficiency, not absolute superiority.** They eliminate the dual-task problem at the cost of a fixed accuracy ceiling.

4. **Axis semantics are secondary.** prescribed ≈ random_fixed at both scales. The predictor adapts to any fixed coordinate system.

5. **Min-max [0,1] normalization matters.** Zero-mean unit-variance hurts convergence by 15×.

---

## Reproduction

```bash
# Low-data: prescribed + random_fixed (200 ep, 30 epochs) ~2 min
python run_random_axes_control.py

# Low-data: isotropic variants (200 ep, 20 epochs) ~1.5 min
python run_isotropic_control.py

# High-data: 500 ep experiment — requires pusht_data_synthetic_500ep.npz
# Script: see exp5_random_axes_500ep.py (generates all 18 conditions)
```

---

## Files

| File | Description |
|------|-------------|
| `RESULTS.md` | This file |
| `run_random_axes_control.py` | 200ep experiment (prescribed + random_fixed + free_3d) |
| `run_isotropic_control.py` | 200ep isotropic normalization control |
| `all_results_500ep.json` | Full 500ep results: 18 runs with training curves |
| `random_axes_results_500ep.png` | Training curves + bar charts for 500ep |
| `pusht_data_synthetic_500ep.npz` | Synthetic Push-T dataset (500 episodes) |
