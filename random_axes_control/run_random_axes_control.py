"""
Random Fixed Axes Experiment
Tests whether the advantage of prescribed axes comes from meaningful coordinates or fixation alone.

Three conditions, same input (normalized x, y, θ):
  - prescribed: identity mapping (meaningful, fixed)
  - random_fixed: frozen random orthogonal rotation (meaningless, fixed)
  - free_3d: learned MLP projection (learned, drifting)

Usage:
  python run_random_axes_control.py [--episodes 200] [--epochs 30] [--seeds 42 123 777]

Author: Andrey Lazarev
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
import time
from pathlib import Path


def generate_pusht_episodes(n_episodes=200, steps_per_episode=50, seed=0):
    """Generate synthetic Push-T episodes with (x, y, θ) block state."""
    rng = np.random.RandomState(seed)
    episodes = []
    for _ in range(n_episodes):
        x = rng.uniform(50, 462, size=(steps_per_episode,))
        y = rng.uniform(50, 462, size=(steps_per_episode,))
        theta = rng.uniform(0, 2 * np.pi, size=(steps_per_episode,))
        # Smooth trajectories
        for t in range(1, steps_per_episode):
            x[t] = 0.9 * x[t-1] + 0.1 * x[t]
            y[t] = 0.9 * y[t-1] + 0.1 * y[t]
            theta[t] = theta[t-1] + 0.1 * (theta[t] - theta[t-1])
        states = np.stack([x, y, theta], axis=-1)
        episodes.append(states)
    return episodes


def normalize_state(states):
    """Normalize to [0, 1]: x,y by 512, θ by 2π."""
    normed = states.copy()
    normed[:, 0] /= 512.0
    normed[:, 1] /= 512.0
    normed[:, 2] /= (2 * np.pi)
    return normed


def random_orthogonal_matrix(dim, seed=9999):
    """Generate a random orthogonal matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    # Ensure proper rotation (det = +1)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q.astype(np.float32)


class Predictor(nn.Module):
    """MLP predictor: context → target prediction."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


class FreeEncoder(nn.Module):
    """Learned MLP encoder: 3D → 3D."""
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_pairs(episodes, normalize=True):
    """Create (context, target) pairs from consecutive timesteps."""
    contexts, targets = [], []
    for ep in episodes:
        if normalize:
            ep = normalize_state(ep)
        for t in range(len(ep) - 1):
            contexts.append(ep[t])
            targets.append(ep[t + 1])
    return (torch.tensor(np.array(contexts), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


def train_condition(condition, episodes, seed, W_random=None,
                    epochs=30, batch_size=128, lr=3e-4, wd=1e-3):
    """Train one condition and return best validation loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split episodes
    n_train = int(0.8 * len(episodes))
    train_eps = episodes[:n_train]
    val_eps = episodes[n_train:]

    ctx_train, tgt_train = make_pairs(train_eps)
    ctx_val, tgt_val = make_pairs(val_eps)

    W_tensor = None
    if W_random is not None:
        W_tensor = torch.tensor(W_random, dtype=torch.float32)

    free_encoder = None
    dim = 3

    if condition == "prescribed":
        # Identity: targets are the normalized coordinates themselves
        pass
    elif condition == "random_fixed":
        # Apply frozen random rotation to targets
        tgt_train = tgt_train @ W_tensor
        tgt_val = tgt_val @ W_tensor
        ctx_train = ctx_train @ W_tensor
        ctx_val = ctx_val @ W_tensor
    elif condition == "free_3d":
        # Learned encoder
        free_encoder = FreeEncoder(3, 3)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    predictor = Predictor(dim)

    if free_encoder is not None:
        params = list(predictor.parameters()) + list(free_encoder.parameters())
    else:
        params = list(predictor.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    n_train_samples = len(ctx_train)

    for epoch in range(epochs):
        # Train
        predictor.train()
        if free_encoder:
            free_encoder.train()

        perm = torch.randperm(n_train_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train_samples, batch_size):
            idx = perm[i:i+batch_size]
            ctx_b = ctx_train[idx]
            tgt_b = tgt_train[idx]

            if free_encoder:
                ctx_b = free_encoder(ctx_b)
                tgt_b = free_encoder(tgt_b)

            pred = predictor(ctx_b)
            loss = nn.functional.mse_loss(pred, tgt_b.detach() if free_encoder else tgt_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        predictor.eval()
        if free_encoder:
            free_encoder.eval()

        with torch.no_grad():
            ctx_v = ctx_val
            tgt_v = tgt_val
            if free_encoder:
                ctx_v = free_encoder(ctx_v)
                tgt_v = free_encoder(tgt_v)
            pred_v = predictor(ctx_v)
            val_loss = nn.functional.mse_loss(pred_v, tgt_v).item()

        if val_loss < best_val:
            best_val = val_loss

    return best_val


def main():
    parser = argparse.ArgumentParser(description="Random Fixed Axes Control Experiment")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument("--output", type=str, default="random_axes_results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Generate data
    episodes = generate_pusht_episodes(n_episodes=args.episodes)

    # Random orthogonal matrix (frozen, same for all seeds)
    W = random_orthogonal_matrix(3, seed=9999)

    conditions = ["prescribed", "random_fixed", "free_3d"]
    results = {}

    for cond in conditions:
        results[cond] = {}
        for seed in args.seeds:
            t0 = time.time()
            val_loss = train_condition(
                cond, episodes, seed,
                W_random=W if cond == "random_fixed" else None,
                epochs=args.epochs,
            )
            elapsed = time.time() - t0
            results[cond][f"seed_{seed}"] = val_loss
            print(f"{cond} s={seed}: {val_loss:.6f} ({elapsed:.1f}s)")

        vals = list(results[cond].values())
        mean = np.mean(vals)
        std = np.std(vals)
        results[cond]["mean"] = mean
        results[cond]["std"] = std
        print(f"→ {mean:.6f} ± {std:.6f}\n")

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    p_mean = results["prescribed"]["mean"]
    r_mean = results["random_fixed"]["mean"]
    f_mean = results["free_3d"]["mean"]
    print(f"prescribed:   {p_mean:.6f}")
    print(f"random_fixed: {r_mean:.6f} ({r_mean/p_mean:.2f}× vs prescribed)")
    print(f"free_3d:      {f_mean:.6f} ({f_mean/p_mean:.2f}× vs prescribed)")
    print(f"free/random:  {f_mean/r_mean:.2f}×")

    # Save
    with open(output_dir / "random_axes_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/random_axes_results.json")


if __name__ == "__main__":
    main()
