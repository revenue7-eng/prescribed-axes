"""
Isotropic Normalization Control
Tests whether the random_fixed advantage over prescribed is explained by scale isotropy.

Hypothesis: random rotation mixes unequal axis scales (σ_θ/σ_x = 1.82×),
making MSE more isotropic. If we standardize to unit variance, the gap should close.

Result: Hypothesis refuted. Standardization makes both 15× worse.
The random/prescribed ratio is preserved (~0.7×).

Usage:
  python run_isotropic_control.py [--episodes 200] [--epochs 20] [--seeds 42 123 777]

Author: Andrey Lazarev
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
import time
from pathlib import Path
from run_random_axes_control import (
    generate_pusht_episodes, normalize_state, random_orthogonal_matrix,
    Predictor, make_pairs
)


def standardize(data):
    """Zero-mean, unit-variance per axis."""
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    return (data - mean) / (std + 1e-8), mean, std


def train_iso_condition(condition, episodes, seed, W_random=None,
                        epochs=20, batch_size=128, lr=3e-4, wd=1e-3):
    """Train with isotropic normalization."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_train = int(0.8 * len(episodes))
    train_eps = episodes[:n_train]
    val_eps = episodes[n_train:]

    ctx_train, tgt_train = make_pairs(train_eps)
    ctx_val, tgt_val = make_pairs(val_eps)

    # Print axis statistics (once)
    if seed == 42:
        print(f"  Axis mean: [{ctx_train[:, 0].mean():.4f}, {ctx_train[:, 1].mean():.4f}, {ctx_train[:, 2].mean():.4f}]")
        print(f"  Axis std:  [{ctx_train[:, 0].std():.4f}, {ctx_train[:, 1].std():.4f}, {ctx_train[:, 2].std():.4f}]")
        print(f"  Std ratio θ/x: {ctx_train[:, 2].std() / ctx_train[:, 0].std():.2f}×")

    # Standardize
    all_data = torch.cat([ctx_train, tgt_train], dim=0)
    mean = all_data.mean(dim=0, keepdim=True)
    std = all_data.std(dim=0, keepdim=True) + 1e-8

    ctx_train = (ctx_train - mean) / std
    tgt_train = (tgt_train - mean) / std
    ctx_val = (ctx_val - mean) / std
    tgt_val = (tgt_val - mean) / std

    W_tensor = None
    if "random" in condition and W_random is not None:
        W_tensor = torch.tensor(W_random, dtype=torch.float32)
        ctx_train = ctx_train @ W_tensor
        tgt_train = tgt_train @ W_tensor
        ctx_val = ctx_val @ W_tensor
        tgt_val = tgt_val @ W_tensor

    predictor = Predictor(3)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    n = len(ctx_train)

    for epoch in range(epochs):
        predictor.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            pred = predictor(ctx_train[idx])
            loss = nn.functional.mse_loss(pred, tgt_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(predictor(ctx_val), tgt_val).item()
        if val_loss < best_val:
            best_val = val_loss

    return best_val


def main():
    parser = argparse.ArgumentParser(description="Isotropic Normalization Control")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument("--output", type=str, default="random_axes_results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    episodes = generate_pusht_episodes(n_episodes=args.episodes)
    W = random_orthogonal_matrix(3, seed=9999)

    conditions = ["prescribed_iso", "random_fixed_iso"]
    results = {}

    for cond in conditions:
        results[cond] = {}
        for seed in args.seeds:
            t0 = time.time()
            val_loss = train_iso_condition(
                cond, episodes, seed,
                W_random=W if "random" in cond else None,
                epochs=args.epochs,
            )
            elapsed = time.time() - t0
            results[cond][f"seed_{seed}"] = val_loss
            print(f"{cond} s={seed}: {val_loss:.6f} ({elapsed:.1f}s)")

        vals = [v for k, v in results[cond].items() if k.startswith("seed")]
        mean = np.mean(vals)
        std = np.std(vals)
        results[cond]["mean"] = mean
        results[cond]["std"] = std
        print(f"→ {mean:.6f} ± {std:.6f}\n")

    # Save
    with open(output_dir / "isotropic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}/isotropic_results.json")


if __name__ == "__main__":
    main()
