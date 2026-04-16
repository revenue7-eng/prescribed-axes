#!/usr/bin/env python3
"""
E15: Simple Pendulum — Prescribed vs Free with Normalization Test
=================================================================
Environment: simple pendulum (θ, θ̇), 2 DOF
Episodes: 200, Epochs: 30, Seeds: [42, 123, 777]

Conditions:
  prescribed_raw   — θ/π, θ̇/5 (current normalization)
  prescribed_norm  — min-max to [0,1] per coordinate
  free_raw         — encoder receives [θ/π, θ̇/5]
  free_norm        — encoder receives min-max [0,1]

Dims: 1, 2, 3, 4, 5

Output: e15_results.json with per-seed data
"""
import json, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

SEEDS = [42, 123, 777]
EPISODES = 200
EPOCHS = 30
BS = 128
LR = 3e-4
HIDDEN_PRED = 128
HIDDEN_ENC = 64

# ================================================================
# Data generation
# ================================================================

def generate_pendulum(n_episodes, steps=50, seed=0):
    rng = np.random.RandomState(seed)
    g_over_L = 9.81
    damping = 0.1
    dt = 0.05
    episodes = []
    for _ in range(n_episodes):
        theta = rng.uniform(-np.pi, np.pi)
        theta_dot = rng.uniform(-3, 3)
        states = []
        for t in range(steps):
            states.append([theta, theta_dot])
            theta_ddot = -g_over_L * np.sin(theta) - damping * theta_dot
            theta_dot += theta_ddot * dt
            theta += theta_dot * dt
            theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        episodes.append(np.array(states, dtype=np.float32))
    return episodes


def compute_prescribed(states, dim):
    """Prescribed axes, cumulative:
    1: θ/π
    2: θ̇/5
    3: sin(θ)
    4: cos(θ)
    5: θ̈ approximation (normalized)
    """
    theta = states[:, 0]
    theta_dot = states[:, 1]
    axes = [theta / np.pi]
    if dim >= 2:
        axes.append(theta_dot / 5.0)
    if dim >= 3:
        axes.append(np.sin(theta))
    if dim >= 4:
        axes.append(np.cos(theta))
    if dim >= 5:
        # Approximate angular acceleration
        theta_ddot = -9.81 * np.sin(theta) - 0.1 * theta_dot
        axes.append(theta_ddot / 15.0)  # normalize
    return np.stack(axes[:dim], axis=-1).astype(np.float32)


def make_pairs(episodes, dim, mode='prescribed_raw', stats=None):
    """
    mode: prescribed_raw, prescribed_norm, free_raw, free_norm
    Returns (ctx, tgt) tensors + stats dict for normalization
    """
    contexts, targets = [], []

    if mode.startswith('prescribed'):
        for ep in episodes:
            for t in range(len(ep) - 1):
                c = compute_prescribed(ep[t:t+1], dim)
                tg = compute_prescribed(ep[t+1:t+2], dim)
                contexts.append(c[0])
                targets.append(tg[0])
    else:  # free
        for ep in episodes:
            normed = ep.copy()
            normed[:, 0] /= np.pi
            normed[:, 1] /= 5.0
            for t in range(len(ep) - 1):
                contexts.append(normed[t])
                targets.append(normed[t + 1])

    ctx = np.array(contexts, dtype=np.float32)
    tgt = np.array(targets, dtype=np.float32)

    # Apply min-max normalization for _norm variants
    if mode.endswith('_norm'):
        if stats is None:
            # Compute from training data
            all_data = np.concatenate([ctx, tgt], axis=0)
            mins = all_data.min(axis=0)
            maxs = all_data.max(axis=0)
            ranges = maxs - mins
            ranges[ranges < 1e-8] = 1.0  # avoid division by zero
            stats = {'mins': mins, 'maxs': maxs, 'ranges': ranges}
        ctx = (ctx - stats['mins']) / stats['ranges']
        tgt = (tgt - stats['mins']) / stats['ranges']

    return (torch.tensor(ctx), torch.tensor(tgt), stats)


# ================================================================
# Models
# ================================================================

class Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, HIDDEN_PRED), nn.LayerNorm(HIDDEN_PRED), nn.ReLU(),
            nn.Linear(HIDDEN_PRED, HIDDEN_PRED), nn.LayerNorm(HIDDEN_PRED), nn.ReLU(),
            nn.Linear(HIDDEN_PRED, dim))
    def forward(self, x):
        return self.net(x)


class FreeEncoder(nn.Module):
    def __init__(self, in_d=2, out_d=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, HIDDEN_ENC), nn.LayerNorm(HIDDEN_ENC), nn.ReLU(),
            nn.Linear(HIDDEN_ENC, HIDDEN_ENC), nn.LayerNorm(HIDDEN_ENC), nn.ReLU(),
            nn.Linear(HIDDEN_ENC, out_d))
    def forward(self, x):
        return self.net(x)


# ================================================================
# Training
# ================================================================

def train_one(ctx_tr, tgt_tr, ctx_va, tgt_va, dim, free_enc=None):
    pred = Predictor(dim)
    params = list(pred.parameters())
    if free_enc is not None:
        params += list(free_enc.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val = float('inf')
    best_ep = 0
    n = len(ctx_tr)

    for epoch in range(EPOCHS):
        # Train
        pred.train()
        if free_enc: free_enc.train()
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            c, t = ctx_tr[idx], tgt_tr[idx]
            if free_enc:
                c, t = free_enc(c), free_enc(t)
            loss = F.mse_loss(pred(c), t.detach() if free_enc else t)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sch.step()

        # Validate
        pred.eval()
        if free_enc: free_enc.eval()
        with torch.no_grad():
            c, t = ctx_va, tgt_va
            if free_enc:
                c, t = free_enc(c), free_enc(t)
            val_loss = F.mse_loss(pred(c), t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_ep = epoch

    return best_val, best_ep


# ================================================================
# Main
# ================================================================

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    total_t0 = time.time()
    save_path = Path('e15_results.json')

    # Resume support
    if save_path.exists():
        with open(save_path) as f:
            results = json.load(f)
        print(f"Resuming from {save_path} ({len([k for k in results if k != 'config'])} entries)")
    else:
        results = {
            'config': {
                'seeds': SEEDS,
                'episodes': EPISODES,
                'epochs': EPOCHS,
                'environment': 'simple_pendulum',
                'state_dim': 2,
                'dims': [1, 2, 3, 4, 5],
                'modes': ['prescribed_raw', 'prescribed_norm', 'free_raw', 'free_norm'],
                'note': 'Normalization test for П1. raw=θ/π,θ̇/5; norm=min-max [0,1]'
            }
        }

    print("=" * 65)
    print("E15: SIMPLE PENDULUM — PRESCRIBED vs FREE + NORMALIZATION")
    print(f"Episodes: {EPISODES}, Epochs: {EPOCHS}, Seeds: {SEEDS}")
    print("=" * 65)

    # Generate data once
    all_episodes = generate_pendulum(EPISODES, seed=0)
    n_tr = int(0.8 * EPISODES)
    tr_eps = all_episodes[:n_tr]
    va_eps = all_episodes[n_tr:]

    dims = [1, 2, 3, 4, 5]
    modes = ['prescribed_raw', 'prescribed_norm', 'free_raw', 'free_norm']

    for dim in dims:
        for mode in modes:
            for seed in SEEDS:
                key = f"{mode}_dim{dim}_seed{seed}"
                if key in results:
                    r = results[key]
                    print(f"  {key}: cached ({r['best_val_loss']:.6f})")
                    continue

                t0 = time.time()
                torch.manual_seed(seed)
                np.random.seed(seed)

                is_free = mode.startswith('free')

                if is_free:
                    # Free encoder: input is always 2D raw state
                    raw_mode = 'free_raw' if mode == 'free_raw' else 'free_norm'
                    ctx_tr, tgt_tr, stats = make_pairs(tr_eps, dim, raw_mode)
                    ctx_va, tgt_va, _ = make_pairs(va_eps, dim, raw_mode, stats=stats)
                    free_enc = FreeEncoder(in_d=ctx_tr.shape[1], out_d=dim)
                    best_val, best_ep = train_one(ctx_tr, tgt_tr, ctx_va, tgt_va, dim, free_enc=free_enc)
                else:
                    # Prescribed: input = prescribed axes
                    ctx_tr, tgt_tr, stats = make_pairs(tr_eps, dim, mode)
                    ctx_va, tgt_va, _ = make_pairs(va_eps, dim, mode, stats=stats)
                    best_val, best_ep = train_one(ctx_tr, tgt_tr, ctx_va, tgt_va, dim)

                dt = time.time() - t0
                results[key] = {
                    'best_val_loss': best_val,
                    'best_epoch': best_ep,
                    'time': round(dt, 1)
                }
                print(f"  {key}: {best_val:.6f} (ep {best_ep}, {dt:.0f}s)")
                save_json(results, save_path)

    # Summary tables
    print(f"\n{'=' * 65}")
    print("SUMMARY: RAW NORMALIZATION (θ/π, θ̇/5)")
    print(f"{'Dim':>4} {'Prescr_raw':>12} {'Free_raw':>12} {'Ratio':>8} {'Winner':>10}")
    print("-" * 50)
    for dim in dims:
        p_vals = [results[f'prescribed_raw_dim{dim}_seed{s}']['best_val_loss'] for s in SEEDS]
        f_vals = [results[f'free_raw_dim{dim}_seed{s}']['best_val_loss'] for s in SEEDS]
        pm, fm = np.mean(p_vals), np.mean(f_vals)
        ratio = fm / pm if pm > 0 else 0
        winner = "PRESCR" if pm < fm else "FREE"
        print(f"{dim:>4} {pm:>12.6f} {fm:>12.6f} {ratio:>7.2f}× {winner:>10}")

    print(f"\nSUMMARY: MIN-MAX NORMALIZATION ([0,1])")
    print(f"{'Dim':>4} {'Prescr_norm':>12} {'Free_norm':>12} {'Ratio':>8} {'Winner':>10}")
    print("-" * 50)
    for dim in dims:
        p_vals = [results[f'prescribed_norm_dim{dim}_seed{s}']['best_val_loss'] for s in SEEDS]
        f_vals = [results[f'free_norm_dim{dim}_seed{s}']['best_val_loss'] for s in SEEDS]
        pm, fm = np.mean(p_vals), np.mean(f_vals)
        ratio = fm / pm if pm > 0 else 0
        winner = "PRESCR" if pm < fm else "FREE"
        print(f"{dim:>4} {pm:>12.6f} {fm:>12.6f} {ratio:>7.2f}× {winner:>10}")

    print(f"\nNORMALIZATION EFFECT (prescribed_norm vs prescribed_raw):")
    print(f"{'Dim':>4} {'Raw':>12} {'Norm':>12} {'Improvement':>12}")
    print("-" * 45)
    for dim in dims:
        raw_vals = [results[f'prescribed_raw_dim{dim}_seed{s}']['best_val_loss'] for s in SEEDS]
        norm_vals = [results[f'prescribed_norm_dim{dim}_seed{s}']['best_val_loss'] for s in SEEDS]
        rm, nm = np.mean(raw_vals), np.mean(norm_vals)
        pct = (rm - nm) / rm * 100 if rm > 0 else 0
        print(f"{dim:>4} {rm:>12.6f} {nm:>12.6f} {pct:>+10.1f}%")

    print(f"\nTotal time: {time.time() - total_t0:.0f}s")
    print(f"Results saved to {save_path}")


if __name__ == '__main__':
    main()
