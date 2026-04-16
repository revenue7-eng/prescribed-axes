"""
Prescribed Axes Dimension Sweep: Simple Pendulum
=================================================
Internal dimensionality = 2 (theta, theta_dot)

Prescribed axes added cumulatively:
  1: θ
  2: θ̇ (angular velocity)
  3: + sinθ (redundant)
  4: + cosθ (redundant)
  5: + θ̈ (angular acceleration, redundant)
  6: + energy (redundant: 0.5*θ̇² - cosθ)

Hypothesis: crossover at dim=2→3

Usage: python run_pendulum.py
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path


def generate_pendulum_episodes(n_episodes=100, steps=50, seed=0):
    """
    Simple pendulum: θ'' = -g/L * sin(θ) - damping * θ'
    State: [θ, θ̇]
    Euler integration, dt=0.05
    """
    rng = np.random.RandomState(seed)
    g_over_L = 9.81  # g/L = 1m
    damping = 0.1
    dt = 0.05
    episodes = []

    for _ in range(n_episodes):
        theta = rng.uniform(-np.pi, np.pi)
        theta_dot = rng.uniform(-3, 3)
        states = []
        for t in range(steps):
            states.append([theta, theta_dot])
            # Physics
            theta_ddot = -g_over_L * np.sin(theta) - damping * theta_dot
            theta_dot = theta_dot + theta_ddot * dt
            theta = theta + theta_dot * dt
            # Keep theta in [-pi, pi]
            theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        episodes.append(np.array(states, dtype=np.float32))
    return episodes


def compute_prescribed(states_t, states_prev, dim):
    """
    Compute prescribed axes for pendulum.
      1: θ / π  (normalized to [-1, 1])
      2: θ̇ / 5  (normalized, max ~5 rad/s)
      3: sin(θ)
      4: cos(θ)
      5: θ̈ (approx from consecutive θ̇)
      6: energy = 0.5*θ̇² - cos(θ), normalized
    """
    theta = states_t[:, 0]
    theta_dot = states_t[:, 1]

    axes = [theta / np.pi]  # 0

    if dim >= 2:
        axes.append(theta_dot / 5.0)  # 1

    if dim >= 3:
        axes.append(np.sin(theta))  # 2 (redundant with θ)

    if dim >= 4:
        axes.append(np.cos(theta))  # 3 (redundant with θ)

    if dim >= 5:
        # Angular acceleration from consecutive states
        if states_prev is not None:
            theta_ddot = (theta_dot - states_prev[:, 1]) / 0.05 / 50.0
        else:
            theta_ddot = np.zeros_like(theta)
        axes.append(theta_ddot)  # 4 (redundant with θ, θ̇)

    if dim >= 6:
        # Mechanical energy (normalized)
        energy = (0.5 * theta_dot**2 - np.cos(theta)) / 10.0
        axes.append(energy)  # 5 (redundant)

    return np.stack(axes[:dim], axis=-1).astype(np.float32)


def make_pairs_prescribed(episodes, dim):
    contexts, targets = [], []
    for ep in episodes:
        start = 1 if dim >= 5 else 0
        for t in range(max(start, 1), len(ep) - 1):
            prev = ep[t-1:t] if t > 0 else ep[t:t+1]
            ctx = compute_prescribed(ep[t:t+1], prev, dim)
            tgt = compute_prescribed(ep[t+1:t+2], ep[t:t+1], dim)
            contexts.append(ctx[0])
            targets.append(tgt[0])
    return (torch.tensor(np.array(contexts), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


def make_pairs_raw(episodes):
    """Raw state normalized: θ/π, θ̇/5"""
    contexts, targets = [], []
    for ep in episodes:
        normed = ep.copy()
        normed[:, 0] /= np.pi
        normed[:, 1] /= 5.0
        for t in range(len(ep) - 1):
            contexts.append(normed[t])
            targets.append(normed[t + 1])
    return (torch.tensor(np.array(contexts), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


class Predictor(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, dim))
    def forward(self, x): return self.net(x)


class FreeEncoder(nn.Module):
    def __init__(self, in_d=2, out_d=2, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)


def train(ctx_tr, tgt_tr, ctx_va, tgt_va, dim, free_enc=None,
          epochs=20, bs=128, lr=3e-4):
    pred = Predictor(dim)
    params = list(pred.parameters())
    if free_enc: params += list(free_enc.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = float("inf")
    n = len(ctx_tr)

    for ep in range(epochs):
        pred.train()
        if free_enc: free_enc.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            c, t = ctx_tr[idx], tgt_tr[idx]
            if free_enc: c, t = free_enc(c), free_enc(t)
            loss = nn.functional.mse_loss(pred(c), t.detach() if free_enc else t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sch.step()
        pred.eval()
        if free_enc: free_enc.eval()
        with torch.no_grad():
            c, t = ctx_va, tgt_va
            if free_enc: c, t = free_enc(c), free_enc(t)
            vl = nn.functional.mse_loss(pred(c), t).item()
        if vl < best: best = vl
    return best


def main():
    print("Simple Pendulum — Dimension Sweep")
    print("Internal dimensionality = 2 (θ, θ̇)")
    print("=" * 55)

    episodes = generate_pendulum_episodes(100)
    n_tr = 80
    dims = [1, 2, 3, 4, 5, 6]
    seeds = [42, 123, 777]

    results = {}
    print(f"\n{'Dim':>4} {'Prescribed':>12} {'Free':>12} {'Ratio':>8} {'Winner':>12}")
    print("-" * 55)

    crossover = None
    for dim in dims:
        p_vals, f_vals = [], []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            ctx_tr, tgt_tr = make_pairs_prescribed(episodes[:n_tr], dim)
            ctx_va, tgt_va = make_pairs_prescribed(episodes[n_tr:], dim)
            p = train(ctx_tr, tgt_tr, ctx_va, tgt_va, dim, epochs=20)
            p_vals.append(p)

            torch.manual_seed(seed); np.random.seed(seed)
            cr, tr = make_pairs_raw(episodes[:n_tr])
            cv, tv = make_pairs_raw(episodes[n_tr:])
            fe = FreeEncoder(2, dim)
            f = train(cr, tr, cv, tv, dim, free_enc=fe, epochs=20)
            f_vals.append(f)

        pm, fm = np.mean(p_vals), np.mean(f_vals)
        ratio = fm / pm if pm > 0 else 0
        winner = "PRESCRIBED" if pm < fm else "FREE"
        print(f"{dim:>4} {pm:>12.6f} {fm:>12.6f} {ratio:>7.1f}× {winner:>12}")

        results[dim] = {"prescribed": pm, "free": fm, "ratio": ratio, "winner": winner}

        if pm > fm and crossover is None:
            crossover = dim

    print(f"\n{'='*55}")
    if crossover:
        print(f"Crossover at dim={crossover}")
        print(f"Predicted internal dimensionality: {crossover - 1}")
        print(f"Actual internal dimensionality: 2")
        print(f"Match: {'YES' if crossover - 1 == 2 else 'NO'}")
    else:
        print("No crossover found — prescribed wins at all dimensions")

    # Save
    out = Path("results_pendulum")
    out.mkdir(exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved to {out}/results.json")


if __name__ == "__main__":
    main()
