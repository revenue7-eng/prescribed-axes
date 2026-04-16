# Реестр экспериментов: Prescribed Axes

Автор: Andrey Lazarev | Дата начала: март 2026
Последнее обновление: 16 апреля 2026

---

## Нумерация

- **E01–E05**: Paper 1 (prescribed-axes)
- **E06–E12**: Paper 2 (prescribed-axes-drift)
- **E13–E18**: Dim sweep (prescribed-axes-dim-sweep)
- **E19–E21**: Tier 1 critical tests
- **E22–E24**: Tier 2 confound tests
- **E25–E27**: Tier 3 generalization tests

---

## Paper 1: The Space Matters More Than the Loss

### E01. Speech JEPA: prescribed cluster anchors vs free
- **Среда:** LibriSpeech
- **Условия:** 2×2 factorial {GMM, k-means} × {soft, hard} vs Pure JEPA
- **Метрика:** Cluster entropy (codebook utilization)
- **Результат:** +18–20pp entropy для prescribed. Soft ≈ hard (Δ<0.03%). Frozen structure — доминирующий фактор.
- **Параметры:** Пилотное
- **Факты:** Ф3
- **Код:** prescribed-axes repo
- **Данные:** —

### E02. Shov-JEPA: 3 prescribed axes vs 64 free (Rico UI, Vision)
- **Среда:** Rico dataset, 398 UI screenshots
- **Условия:** ShovJEPA (3 axes: position, functionality, depth) vs Free 64D
- **Метрика:** Validation accuracy
- **Результат:** 72.5% vs 67.5% (+5%)
- **Параметры:** 398 samples, single seed, пилотное
- **Факты:** Ф4
- **Код:** prescribed-axes repo (shov-jepa)
- **Данные:** shov-jepa-report-ru.docx

### E03. LeWM State: prescribed 3D vs free 3D (Push-T)
- **Среда:** Push-T (gym-pusht, pymunk physics)
- **Условия:** Prescribed = normalize(x_b, y_b, θ_b) vs Free MLP 5→3 + SIGReg
- **Метрика:** Val prediction loss
- **Результат:** Prescribed 0.004, Free 0.157 = **38×**. Per-axis: x 53×, y 63×, θ 25×
- **Параметры:** 3 seeds, 50 epochs, 200 episodes, SIGReg block
- **Факты:** Ф1
- **Код:** prescribed-axes repo (lewm_state)
- **Данные:** lewm_state_results/

### E04. LeWM Pixel: prescribed 3D vs free CNN (Push-T from pixels)
- **Среда:** Push-T (96×96 pixel observations)
- **Условия:** Prescribed 3D (20K params) vs Free CNN (744K params)
- **Метрика:** Val prediction loss
- **Результат:** Prescribed **14.8×** лучше, **37× fewer parameters**. CNN плато на epoch 7.
- **Параметры:** 50 epochs
- **Факты:** Ф2
- **Код:** prescribed-axes repo (lewm_pixels)
- **Данные:** lewm_pixels_results/

### E05. Controls: random fixed, equal-input, SIGReg ablation (Push-T)
- **Среда:** Push-T
- **Условия:** Random fixed 3D, free 3D same input, ±SIGReg
- **Метрика:** Val prediction loss
- **Результат:**
  - Random fixed ≈ prescribed (0.61×) → фиксация > семантика (Ф5)
  - Equal-input free 7.6× хуже prescribed → не доступ к информации (Ф6)
  - SIGReg removal improves free 1.9× (Ф7)
- **Параметры:** 3 seeds, 50 epochs, 200 episodes
- **Факты:** Ф5, Ф6, Ф7
- **Код:** prescribed-axes repo (reviewer_response_experiments.py)
- **Данные:** reviewer_results/summary.json, reviewer_results/full_results.json

### E05a. Random axes scaling: 200ep vs 500ep (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed, random_fixed, free_3d, free_5d at 200 and 500 episodes
- **Метрика:** Val prediction loss
- **Результат:**
  - 200 ep: random 0.61× prescribed, free 4.47× worse (Ф39)
  - 500 ep: random 1.00× prescribed, **free 695,000× BETTER** (Ф38)
  - Fixed encoders plateau at ~8.5×10⁻⁴, free → 10⁻⁹
  - Prescribed = sample efficiency, not absolute superiority
  - Isotropic normalization 15× worse (Ф40)
- **Параметры:** 200 ep: 3 seeds, 30 epochs. 500 ep: 3-9 runs, 50 epochs. No SIGReg.
- **Факты:** Ф38, Ф39, Ф40
- **Код:** random_axes_control/run_random_axes_control.py, run_isotropic_control.py
- **Данные:** exp5_random_axes/all_results.json (18 runs), random_fixed_results/results.json

### E05b. Gauge fixing free encoder (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed, free, gauge_fixed_free, linear_free
- **Метрика:** Val prediction loss
- **Результат:**
  - gauge_fixed_free 1.08× ≈ free — gauge fixing не помогает (Ф37)
  - linear_free: 16009 (взрыв)
- **Параметры:** Data seed 42, training seeds [42, 123, 777], 50 epochs, synthetic
- **Факты:** Ф37
- **Код:** (gauge_fix experiment script)
- **Данные:** gauge_fix_results/results.json

---

## Paper 2: Semantic Drift, Not Rank Collapse

### E06. Covariance + Drift analysis (Push-T, gym-pusht)
- **Среда:** Push-T (gym-pusht, real pymunk physics)
- **Условия:** Prescribed vs Free, covariance at sampled epochs, drift metrics
- **Метрика:** Effective rank, isotropy, raw/aligned drift, R² transfer
- **Результат:**
  - Free: rank 2.99, isotropy 0.86 → проигрывает prescribed 233× (Ф9)
  - R² transfer epoch 0→1: −16.9 / −62.2 / −25.4 (Ф10)
  - 80% drift structural after Procrustes
  - SIGReg вредит free: 4.2× хуже (Ф8)
- **Параметры:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes, SIGReg λ=0.09
- **Факты:** Ф8, Ф9, Ф10
- **Код:** paper2_full_analysis.py, drift_analysis_standalone.py, covariance_analysis_standalone.py
- **Данные:** all_results.json (142KB)

### E07. Freeze test (Push-T, gym-pusht)
- **Среда:** Push-T (gym-pusht)
- **Условия:** Free encoder frozen at epoch T = {1, 2, 3, 5, 7, 10}
- **Метрика:** Best val loss
- **Результат:** Freeze@1: +20%, Freeze@10: −1.1%. Causal evidence for drift harm. (Ф11)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф11
- **Код:** freeze_test_standalone.py
- **Данные:** all_results.json

### E08. Random fixed encoder control (Push-T, synthetic)
- **Среда:** Push-T (synthetic physics)
- **Условия:** Prescribed, rotated prescribed, random fixed 5→3, free
- **Метрика:** Best val loss
- **Результат:**
  - Random fixed 17× лучше free (Ф12)
  - Prescribed 13× лучше random fixed (Ф13)
  - Rotated ≈ prescribed 1.09× (Ф14)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф12, Ф13, Ф14
- **Код:** random_fixed_encoder.py
- **Данные:** random_fixed_v2_results.json (39KB)

### E09. Aligned-but-drifting + 2×2 factorial (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed, random fixed, aligned-drifting (linear + MLP), free
- **Метрика:** Best val loss
- **Результат:**
  - Aligned-drifting ≈ free or worse (Ф15)
  - 2×2 factorial: stability × alignment interaction 19× (Ф16)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф15, Ф16
- **Код:** paper2_aligned_drifting_colab.ipynb
- **Данные:** aligned_drifting_results.json (49KB)

### E10. LR sweep + EMA baseline (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Free LR={1e-4, 3e-4, 1e-3, 3e-3}, free+EMA (decay=0.996), prescribed
- **Метрика:** Best val loss, R²(0→1)
- **Результат:** Prescribed wins at every LR (4.3–7.0×). EMA 6.1× worse than prescribed.
- **Параметры:** Seed 42, 50 epochs
- **Факты:** (Paper 2, Section 5.6–5.7)
- **Код:** lr_sweep_ema_baseline.ipynb
- **Данные:** lr_sweep_results.json (104KB)

### E11. Rico UI drift analysis (Vision)
- **Среда:** Rico dataset, 398 UI screenshots
- **Условия:** Free 3D + SIGReg vs ShovJEPA prescribed 3D
- **Метрика:** R² transfer, effective rank, condition number
- **Результат:** Drift in vision weaker (R² 0.93 vs 0.78 in Push-T). Cross-modal confirmation.
- **Параметры:** Seed 42, 100 epochs
- **Факты:** (Paper 2, Section 5.8)
- **Код:** rico_drift_v2.ipynb
- **Данные:** rico_drift_v2_results.json (53KB)

---

## Dim Sweep Experiments

### E12. 11 prescribed axes (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed_3, prescribed_11, free_3, free_11, random_fixed_11
- **Метрика:** Val loss
- **Результат:** Prescribed_11 20× хуже prescribed_3. Free_11 beats prescribed_11 by 6×. (Ф17)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф17
- **Код:** dim-sweep/exp1_11axes/run_11axes.py
- **Данные:** dim-sweep/exp1_11axes/results.json

### E13. Dimension sweep 3–15 (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed vs free at dim = 3, 4, 5, 6, 7, 9, 11, 15
- **Метрика:** Val loss
- **Результат:** Crossover at dim=3→4. Prescribed wins only dim ≤ 3. (Ф18)
- **Параметры:** 3 seeds, 20 epochs, 100 episodes (предварительные)
- **Факты:** Ф18
- **Код:** dim-sweep/exp2_sweep/run_sweep.py
- **Данные:** dim-sweep/exp2_sweep/sweep_results.json

### E14. Lower boundary dim 1–3 (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed vs free at dim = 1, 2, 3
- **Метрика:** Val loss
- **Результат:** dim=1: 78×, dim=2: 12×, dim=3: 1.5×. Max advantage at min dim.
- **Параметры:** 2–3 seeds, 15–20 epochs, 50–100 episodes
- **Факты:** (включено в Ф18)
- **Код:** dim-sweep/exp3_lower/run_lower.py
- **Данные:** dim-sweep/exp3_lower/output.txt

### E15. Simple pendulum sweep + normalization test (2 DOF)
- **Среда:** Simple pendulum (θ, θ̇), synthetic Euler integration
- **Условия:** 4 modes × 5 dims × 3 seeds = 60 runs:
  - prescribed_raw (θ/π, θ̇/5), prescribed_norm ([0,1] min-max)
  - free_raw, free_norm
  - dim = 1, 2, 3, 4, 5
- **Метрика:** Val loss (MSE)
- **Результат:**
  - Free wins at ALL dims under BOTH normalizations (Ф19)
  - Raw: free 1.4–6.3× better (dim=2: 6.3×, dim=1: 1.4×)
  - Norm: free 1.1–1.9× better (dim=2: 1.1×, dim=3: 2.1×)
  - Normalization improves prescribed by 72–74% at all dims (Ф41)
  - Prescribed 63× more stable than free by variance (dim=2 norm) (Ф42)
- **Параметры:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes (160/40 split)
- **Факты:** Ф19, Ф41, Ф42
- **Код:** exp15_pendulum/code/run_e15.py (new), exp15_pendulum/code/run_pendulum.py (old)
- **Данные:** exp15_pendulum/results/e15_results.json (60 conditions, per-seed)
- **Примечание:** Re-run 16.04.2026. Supersedes original (100ep, 20ep, no per-seed, no norm test)

### E16. Double pendulum sweep (4 DOF)
- **Среда:** Double pendulum (θ₁, ω₁, θ₂, ω₂), synthetic
- **Условия:** Prescribed vs free at dim = 1, 2, 4, 8, identical input
- **Метрика:** Val loss
- **Результат:** Prescribed wins only dim=1 (2.1×). Free wins dim ≥ 2. (Ф20)
- **Параметры:** 3 seeds, 20 epochs, 100 episodes
- **Факты:** Ф20
- **Код:** dim-sweep/exp5_double_pendulum/run_double_pendulum.py
- **Данные:** dim-sweep/exp5_double_pendulum/results/results.json

### E17. Fragility test: 4th axis types (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed_3, +sin(θ), +agent_x, +distance, +noise
- **Метрика:** Val loss
- **Результат:** Noise: 1106×. Agent_x: 7.9×. Distance: 8.2×. Sin(θ): 4.8×. (Ф21–Ф23)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф21, Ф22, Ф23
- **Код:** dim-sweep/exp6_fragility/run_fragility.py
- **Данные:** dim-sweep/exp6_fragility/results/results.json

---

## Tier 1: Critical Hypothesis Tests

### E18. MLP decoder transfer (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Free encoder, each epoch transition: train linear + MLP decoder on epoch t, evaluate on epoch t+1
- **Метрика:** R² transfer (linear vs MLP)
- **Результат:**
  - Epoch 0→1: MLP xfer = −283, linear xfer = −71 → information destroyed (Ф24)
  - Epoch 2+: MLP xfer ≈ 0.81, linear xfer ≈ 0.69 → info preserved, linear readability lost (Ф25)
  - Two-phase drift model confirmed
- **Параметры:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes
- **Факты:** Ф24, Ф25
- **Код:** tier1_all_tests.py (T1 section)
- **Данные:** tier1_results.json (T1 key)

### E19. Update ratio + differential LR (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed; Free K=1,3,5 (predictor steps per encoder step); diffLR 10×, 100×
- **Метрика:** Best val loss
- **Результат:**
  - DiffLR 100×: gap 62× (vs 222× baseline) → 72% improvement, but 62× remains (Ф26)
  - K=3: WORSE than K=1 (−26%)
  - NOT pure optimization lag
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф26
- **Код:** tier1_all_tests.py (T2 section)
- **Данные:** tier1_results.json (T2 key)

### E20. PCA canonicalization (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Free encoder, each epoch: PCA-align embeddings, measure R² transfer in canonical vs raw space
- **Метрика:** R² transfer (raw vs PCA-canonical)
- **Результат:** PCA worsens R² transfer at most epochs. Drift is nonlinear, not rotation/scaling. (Ф27)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф27
- **Код:** tier1_all_tests.py (T3 section)
- **Данные:** tier1_results.json (T3 key)

---

## Tier 2: Confound Tests

### E21. Aligned-drifting ± SIGReg (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Aligned-linear ± SIGReg, free ± SIGReg, prescribed
- **Метрика:** Best val loss
- **Результат:**
  - SIGReg stabilizes aligned-linear (prevents divergence on seed 123) (Ф29)
  - Neither with nor without SIGReg approaches prescribed (356× / 12601×)
  - Free without SIGReg slightly better (0.007 vs 0.008)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф29
- **Код:** tier2_confound_tests.py (T4 section)
- **Данные:** tier2_results.json (T4 key)

### E22. Optimizer state preservation in freeze test (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Freeze@1 and @3 with new optimizer vs preserving optimizer state
- **Метрика:** Best val loss
- **Результат:**
  - freeze@1: new_opt 0.008785, keep_state 0.008701 → difference 1.0% (Ф30)
  - freeze@3: difference 2.9%
  - Optimizer reset is NOT a confound
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф30
- **Код:** tier2_confound_tests.py (T5 section)
- **Данные:** tier2_results.json (T5 key)

### E23. Random projection 3D vs 5D subspace (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed, rotated_prescribed, random_fixed_3d (block coords), random_fixed_5d (all coords), free
- **Метрика:** Best val loss
- **Результат:**
  - random_fixed_3d ≈ prescribed ≈ rotated_prescribed (all ~0.000037) (Ф31)
  - random_fixed_5d EXPLODES (376,053 mean) (Ф32)
  - Alignment within subspace irrelevant; subspace selection + normalization + freeze = the mechanism
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф31, Ф32
- **Код:** tier2_confound_tests.py (T7 section)
- **Данные:** tier2_results.json (T7 key)

---

## Tier 3: Generalization Tests

### E24. Baseline 3D comparison (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed 3D vs Free 3D (same architecture as Tier 1-2)
- **Метрика:** Best val loss, drift_01, R² transfer
- **Результат:** Gap 169×. Drift 1.53. R² transfer −70.
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** (included in Ф33)
- **Код:** tier3_highdim.py (baseline section)
- **Данные:** tier3_results.json (baseline_3d key)

### E25. 5D latent space (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed 5D (all 5 coords normalized), Free MLP 5→5, Random fixed orthogonal 5→5
- **Метрика:** Best val loss, drift_01, R² transfer
- **Результат:**
  - Gap prescribed/free: 66× (Ф33)
  - random_fixed ≈ prescribed (0.92×) (Ф34)
  - Drift 1.91, R² transfer −65
  - Prescribed without subspace selection still works (Ф36)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф33, Ф34, Ф36
- **Код:** tier3_highdim.py (T9a section)
- **Данные:** tier3_results.json (T9a_5d key)

### E26. 16D latent space (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed 16D (engineered nonlinear features), Free MLP 5→16, Random fixed 5→16
- **Метрика:** Best val loss, drift_01, R² transfer
- **Результат:**
  - Gap prescribed/free: 50× (Ф33)
  - random_fixed / prescribed: 1.53× — alignment begins to matter at high dim (Ф34)
  - Drift 3.58, R² transfer −596 — drift amplifies with dimension (Ф35)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф33, Ф34, Ф35
- **Код:** tier3_highdim.py (T9b section)
- **Данные:** tier3_results.json (T9b_16d key)

---

## Вспомогательный: Drift rate correlation

### E27. T8: Drift rate vs downstream quality (Push-T, gym-pusht)
- **Среда:** Push-T (gym-pusht, real physics)
- **Условия:** Existing data from E06 — no new training
- **Метрика:** Pearson/Spearman correlation drift rate × val loss, phase analysis
- **Результат:**
  - Pearson = 0.95, Spearman = 0.51 (nonlinear relationship) (Ф28)
  - Two regimes: catastrophe (drift > 0.3) and saturation (drift < 0.1)
  - R² ceiling ≈ 0.75 (linear decoder on free encoder)
  - Early drift 8–14× larger than late drift
- **Параметры:** 3 seeds, 30 epochs, 200 episodes (data from E06)
- **Факты:** Ф28
- **Код:** analysis script (in-conversation)
- **Данные:** all_results.json (from E06)

---

## П2 Resolution

### E28. Dim sweep full parameters (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed vs free at dim = 1, 2, 3, 4, 5, 7, 11. Predictor hidden = max(128, dim×8).
- **Метрика:** Best val loss
- **Результат:**
  - **NO CROSSOVER.** Prescribed wins at ALL dimensions 1–11.
  - dim=1: 60×, dim=2: 1820×, dim=3: 228×, dim=4: 114×, dim=5: 66×, dim=7: 57×, dim=11: 42×
  - Gap monotonically decreases with dim but never reaches 1×
  - dim=5 matches Tier 3 E25 exactly (66.3× vs 66.2×)
  - **Ф18 (crossover at dim=4) REFUTED** — was underpowered artifact of E13
  - **Ф17 updated:** prescribed_11 now beats free_11 (42×) with proper predictor capacity
- **Параметры:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes, predictor max(128, dim×8)
- **Факты:** Ф18 (refuted), Ф17 (updated)
- **Код:** p2_dim_sweep_full.py
- **Данные:** p2_dim_sweep_results.json

---

## Сводная таблица

| ID | Название | Среда | Данные | Seeds | Epochs | Episodes | Ключевой результат |
|---|---|---|---|---|---|---|---|
| E01 | Speech JEPA | LibriSpeech | — | pilot | — | — | +18–20pp entropy |
| E02 | Shov-JEPA Vision | Rico UI | — | 1 | — | 398 | +5% accuracy |
| E03 | LeWM State | Push-T gym | gym | 3 | 50 | 200 | 38× |
| E04 | LeWM Pixel | Push-T pixel | gym | — | 50 | — | 14.8× |
| E05 | Controls (Paper 1) | Push-T | gym | 3 | 50 | 200 | random≈prescribed |
| E06 | Cov + Drift | Push-T gym | gym | 3 | 30 | 200 | rank 2.99 → 233× worse |
| E07 | Freeze test | Push-T gym | gym | 3 | 30 | 200 | freeze@1 +20% |
| E08 | Random fixed | Push-T syn | syn | 3 | 30 | 200 | 17× stability |
| E09 | Aligned-drifting | Push-T syn | syn | 3 | 30 | 200 | aligned≈free |
| E10 | LR sweep + EMA | Push-T syn | syn | 1 | 50 | — | 4.3–7.0× all LR |
| E11 | Rico drift | Rico UI | — | 1 | 100 | 398 | R²=0.93 |
| E12 | 11 axes | Push-T syn | syn | 3 | 30 | 200 | 20× worse |
| E13 | Dim sweep 3–15 | Push-T syn | syn | 3 | 20 | 100 | crossover 3→4 |
| E14 | Lower boundary | Push-T syn | syn | 2–3 | 15–20 | 50–100 | dim=1: 78× |
| E15 | Pendulum + norm | Pendulum syn | syn | 3 | 30 | 200 | free wins all, norm +72–74% |
| E16 | Double pendulum | Dbl pend syn | syn | 3 | 20 | 100 | prescribed@dim=1 only |
| E17 | Fragility | Push-T syn | syn | 3 | 30 | 200 | noise: 1106× |
| E18 | MLP decoder xfer | Push-T syn | syn | 3 | 30 | 200 | ep0→1: info destroyed |
| E19 | Update ratio | Push-T syn | syn | 3 | 30 | 200 | 62× gap remains |
| E20 | PCA canonical | Push-T syn | syn | 3 | 30 | 200 | PCA worsens |
| E21 | ±SIGReg aligned | Push-T syn | syn | 3 | 30 | 200 | SIGReg stabilizes |
| E22 | Optimizer freeze | Push-T syn | syn | 3 | 30 | 200 | confound absent |
| E23 | Random 3D vs 5D | Push-T syn | syn | 3 | 30 | 200 | random_3d≈prescribed |
| E24 | Baseline 3D | Push-T syn | syn | 3 | 30 | 200 | 169× |
| E25 | 5D latent | Push-T syn | syn | 3 | 30 | 200 | 66×, random≈prescribed |
| E26 | 16D latent | Push-T syn | syn | 3 | 30 | 200 | 50×, align emerges 1.5× |
| E27 | Drift correlation | Push-T gym | gym | 3 | 30 | 200 | Pearson=0.95 |
| E28 | Dim sweep full | Push-T syn | syn | 3 | 30 | 200 | NO crossover, prescribed wins 1–11 |

---

## Файлы данных

| Файл | Эксперименты | Размер | Источник |
|---|---|---|---|
| all_results.json | E06, E07, E27 | 143KB | prescribed-axes-drift repo |
| random_fixed_v2_results.json | E08 | 39KB | prescribed-axes-drift repo |
| aligned_drifting_results.json | E09 | 49KB | prescribed-axes-drift repo |
| lr_sweep_results.json | E10 | 104KB | prescribed-axes-drift repo |
| rico_drift_v2_results.json | E11 | 53KB | prescribed-axes-drift repo |
| dim-sweep/exp1_11axes/results.json | E12 | 2KB | dim-sweep archive |
| dim-sweep/exp2_sweep/sweep_results.json | E13 | 3KB | dim-sweep archive |
| dim-sweep/exp3_lower/output.txt | E14 | <1KB | dim-sweep archive |
| exp15_pendulum/results/e15_results.json | E15 | 8KB | E15 re-run (16.04.2026) |
| dim-sweep/exp4_pendulum/results/ | E15 (old) | <1KB | dim-sweep archive (SUPERSEDED) |
| dim-sweep/exp5_double_pendulum/results/results.json | E16 | 1KB | dim-sweep archive |
| dim-sweep/exp6_fragility/results/results.json | E17 | 1KB | dim-sweep archive |
| tier1_results.json | E18, E19, E20 | 30KB | Tier 1 script |
| tier2_results.json | E21, E22, E23 | 3KB | Tier 2 script |
| tier3_results.json | E24, E25, E26 | 5KB | Tier 3 script |
| p2_dim_sweep_results.json | E28 | 3KB | П2 resolution |

---

## Скрипты

| Файл | Эксперименты | Источник |
|---|---|---|
| paper2_full_analysis.py | E06, E07 | prescribed-axes-drift repo |
| random_fixed_encoder.py | E08 | prescribed-axes-drift repo |
| paper2_aligned_drifting_colab.ipynb | E09 | prescribed-axes-drift repo |
| lr_sweep_ema_baseline.ipynb | E10 | prescribed-axes-drift repo |
| rico_drift_v2.ipynb | E11 | prescribed-axes-drift repo |
| dim-sweep/exp1_11axes/run_11axes.py | E12 | dim-sweep archive |
| dim-sweep/exp2_sweep/run_sweep.py | E13 | dim-sweep archive (SUPERSEDED by E28) |
| dim-sweep/exp3_lower/run_lower.py | E14 | dim-sweep archive |
| exp15_pendulum/code/run_e15.py | E15 | E15 re-run (16.04.2026) |
| dim-sweep/exp4_pendulum/run_pendulum.py | E15 (old) | dim-sweep archive (SUPERSEDED by run_e15.py) |
| dim-sweep/exp5_double_pendulum/run_double_pendulum.py | E16 | dim-sweep archive |
| dim-sweep/exp6_fragility/run_fragility.py | E17 | dim-sweep archive |
| tier1_all_tests.py | E18, E19, E20 | Tier 1 (15.04.2026) |
| tier2_confound_tests.py | E21, E22, E23 | Tier 2 (15.04.2026) |
| tier3_highdim.py | E24, E25, E26 | Tier 3 (15.04.2026) |
| p2_dim_sweep_full.py | E28 | П2 resolution (15.04.2026) |

---

## Противоречия между экспериментами

**П1. E25 (5D prescribed works, 66×) vs E15/E16 (маятники prescribed не работает) — PARTIALLY CLOSED**
- Push-T 5D prescribed нормализует все координаты → работает
- E16 (double pendulum): с нормализацией prescribed побеждает все dim 1–8 (Ф20) → CLOSED
- E15 (simple pendulum): с нормализацией gap сужается до 1.1–1.9×, но free по-прежнему побеждает (Ф19, Ф41)
- Причина residual gap: pendulum state_dim = latent_dim (2 DOF), нет subspace selection
- Push-T/double pendulum: state_dim > latent_dim → subspace selection + normalization = win
- Prescribed advantage = normalization + fixation + subspace selection (when available)

**П2. ~~E13 (dim sweep: crossover at dim=4) vs E25 (prescribed_5d wins 66×)~~ CLOSED**
- E28 (full parameters) confirms: NO crossover. Prescribed wins dim 1–11.
- E13 was underpowered (100 ep, 20 epochs, 2 seeds, predictor hidden=128).
- dim=5 in E28 matches E25 exactly (66.3× vs 66.2×).

**П3. E08 (random_fixed_5d = 17× vs free) vs E23 (random_fixed_5d explodes)**
- E08: random_fixed проецирует 5→3, с bias, конкретные seeds
- E23: random_fixed проецирует 5→3, без нормализации входа, другие seeds
- Различие в реализации → разный результат. Original E08 result unreliable.
