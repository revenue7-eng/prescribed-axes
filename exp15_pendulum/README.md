# Experiment 15: Simple Pendulum Sweep (2 DOF) + Normalization Test

## What this tests
Does prescribed advantage hold in a different dynamical system (simple pendulum, θ and θ̇)?
Does normalization affect the result? (П1 test)

## Key results
- Free wins at ALL dimensions (1–5) under both normalizations (Ф19 confirmed)
- No crossover point exists
- Prescribed and free receive identical 2D input — no information selection advantage
- Min-max [0,1] normalization improves prescribed by **72–74%** (Ф41)
- At dim=2 + norm: near parity (ratio 0.87×), free variance 63× higher than prescribed (Ф42)

## Setup
- **Environment:** Simple pendulum (θ, θ̇), synthetic Euler integration
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200 (160 train, 40 val)
- **Conditions:** 4 modes × 5 dims × 3 seeds = 60 runs
- **Platform:** CPU (~4 minutes total)
- **Date:** 16 April 2026 (re-run with proper parameters)

## Results

### Raw normalization (θ/π, θ̇/5)
| Dim | Prescribed | Free    | Ratio | Winner |
|-----|-----------|---------|-------|--------|
| 1   | 0.005682  | 0.004206| 0.74× | FREE   |
| 2   | 0.001754  | 0.000278| 0.16× | FREE   |
| 3   | 0.001187  | 0.000304| 0.26× | FREE   |
| 4   | 0.000898  | 0.000186| 0.21× | FREE   |
| 5   | 0.000727  | 0.000207| 0.28× | FREE   |

### Min-max normalization ([0,1])
| Dim | Prescribed | Free    | Ratio | Winner |
|-----|-----------|---------|-------|--------|
| 1   | 0.001462  | 0.000787| 0.54× | FREE   |
| 2   | 0.000463  | 0.000403| 0.87× | FREE   |
| 3   | 0.000332  | 0.000160| 0.48× | FREE   |
| 4   | 0.000240  | 0.000169| 0.70× | FREE   |
| 5   | 0.000204  | 0.000144| 0.70× | FREE   |

### Normalization effect on prescribed
| Dim | Raw      | Norm     | Improvement |
|-----|----------|----------|-------------|
| 1   | 0.005682 | 0.001462 | +74.3%      |
| 2   | 0.001754 | 0.000463 | +73.6%      |
| 3   | 0.001187 | 0.000332 | +72.0%      |
| 4   | 0.000898 | 0.000240 | +73.2%      |
| 5   | 0.000727 | 0.000204 | +71.9%      |

### Variance (dim=2, norm)
| Condition | Seed 42  | Seed 123 | Seed 777 | Mean     | Std      |
|-----------|----------|----------|----------|----------|----------|
| prescribed| 0.000458 | 0.000474 | 0.000458 | 0.000463 | 0.000008 |
| free      | 0.000032 | 0.000059 | 0.001118 | 0.000403 | 0.000506 |

Free std is 63× higher than prescribed.

## Files
```
code/run_e15.py               — Full experiment script (4 modes × 5 dims × 3 seeds, resume support)
results/e15_results.json      — All 60 conditions with per-seed data
results/pendulum_results.json — OLD: reconstructed from original run (100ep, 20ep, no per-seed)
code/run_pendulum.py          — OLD: original script (100 episodes, 20 epochs)
```

## Data provenance
- **Old data** (pendulum_results.json): 100 episodes, 20 epochs, reconstructed from console output, no per-seed. SUPERSEDED.
- **New data** (e15_results.json): 200 episodes, 30 epochs, per-seed JSON, includes normalization variant. PRIMARY.

## Facts
- Ф19: Free wins at all dims on simple pendulum (confirmed with larger data)
- Ф41: Min-max normalization improves prescribed by 72–74%
- Ф42: At dim=2+norm, near parity (0.87×) but free variance 63× higher

## Significance
First negative result for prescribed axes. Establishes boundary condition: prescribed advantage requires either (a) subspace selection (state_dim > latent_dim, as in Push-T) or (b) insufficient data for free encoder to converge. When neither holds, fixation provides stability but not accuracy.

Normalization is a confound: 72–74% of prescribed error on pendulum comes from scale mismatch, not from the structural limitation.

## Reproduction
```bash
python code/run_e15.py
# ~4 minutes on CPU, resumes from checkpoint
# Output: e15_results.json (same directory)
```
