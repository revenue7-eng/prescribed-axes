"""
11 Prescribed Axes Experiment
=============================
Tests prescribed axes at dimensionality 11 on Push-T.

Three conditions:
  - prescribed_11: 11 physically derived axes (x, y, θ, ẋ, ẏ, θ̇, d_target, sinθ, cosθ, x_agent, y_agent)
  - random_fixed_11: frozen random orthogonal 5→11 (meaningless, fixed)
  - free_11: learned MLP 5→11 (learned, drifting)

Plus baselines:
  - prescribed_3: original 3 axes (x, y, θ) for comparison
  - free_3: learned MLP 5→3

Usage:
  python run_11axes.py [--episodes 200] [--epochs 30] [--seeds 42 123 777]

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
    """Generate synthetic Push-T episodes with full 5D state: (agent_x, agent_y, block_x, block_y, block_angle)."""
    rng = np.random.RandomState(seed)
    episodes = []
    for _ in range(n_episodes):
        # Agent position
        ax = rng.uniform(50, 462)
        ay = rng.uniform(50, 462)
        # Block position
        bx = rng.uniform(100, 412)
        by = rng.uniform(100, 412)
        btheta = rng.uniform(0, 2 * np.pi)
        # Target for agent movement
        tx, ty = rng.uniform(50, 462), rng.uniform(50, 462)

        states = []
        for t in range(steps_per_episode):
            if t % 20 == 0:
                tx, ty = rng.uniform(50, 462), rng.uniform(50, 462)

            # Agent moves toward target
            dx, dy = tx - ax, ty - ay
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                ax += dx * min(1.0, 20.0 / dist) + rng.normal(0, 2)
                ay += dy * min(1.0, 20.0 / dist) + rng.normal(0, 2)
            ax = np.clip(ax, 0, 512)
            ay = np.clip(ay, 0, 512)

            # Block physics: agent pushes block if close
            tb = np.sqrt((bx - ax)**2 + (by - ay)**2)
            if 0 < tb < 30:
                force = (30 - tb) / 30 * 5
                bx += (bx - ax) / tb * force
                by += (by - ay) / tb * force
                btheta += rng.normal(0, 0.05) * force
            bx = np.clip(bx, 0, 512)
            by = np.clip(by, 0, 512)
            btheta = btheta % (2 * np.pi)

            states.append([ax, ay, bx, by, btheta])

        episodes.append(np.array(states, dtype=np.float32))
    return episodes


def compute_11_axes(states_t, states_prev=None, target_pos=None):
    """
    Compute 11 prescribed axes from raw state.
    State: [agent_x, agent_y, block_x, block_y, block_angle]

    Axes:
      0: block_x / 512
      1: block_y / 512
      2: block_angle / 2π
      3: block_vx (velocity x, from consecutive states)
      4: block_vy (velocity y)
      5: block_vtheta (angular velocity)
      6: distance agent-to-block / 512
      7: sin(block_angle)
      8: cos(block_angle)
      9: agent_x / 512
     10: agent_y / 512
    """
    ax, ay, bx, by, btheta = states_t[:, 0], states_t[:, 1], states_t[:, 2], states_t[:, 3], states_t[:, 4]

    # Velocities (if previous state available)
    if states_prev is not None:
        vx = (bx - states_prev[:, 2]) / 512.0
        vy = (by - states_prev[:, 3]) / 512.0
        vtheta = (btheta - states_prev[:, 4]) / (2 * np.pi)
    else:
        vx = np.zeros_like(bx)
        vy = np.zeros_like(by)
        vtheta = np.zeros_like(btheta)

    # Distance agent to block
    d_ab = np.sqrt((ax - bx)**2 + (ay - by)**2) / 512.0

    result = np.stack([
        bx / 512.0,          # 0: block x
        by / 512.0,          # 1: block y
        btheta / (2*np.pi),  # 2: block angle
        vx,                  # 3: velocity x
        vy,                  # 4: velocity y
        vtheta,              # 5: angular velocity
        d_ab,                # 6: distance agent-block
        np.sin(btheta),      # 7: sin(angle)
        np.cos(btheta),      # 8: cos(angle)
        ax / 512.0,          # 9: agent x
        ay / 512.0,          # 10: agent y
    ], axis=-1)

    return result.astype(np.float32)


def compute_3_axes(states_t):
    """Original 3 prescribed axes: block (x, y, θ) normalized."""
    bx, by, btheta = states_t[:, 2], states_t[:, 3], states_t[:, 4]
    return np.stack([bx/512.0, by/512.0, btheta/(2*np.pi)], axis=-1).astype(np.float32)


def make_pairs_prescribed(episodes, dim=11):
    """Create (context, target) pairs with prescribed axes."""
    contexts, targets = [], []
    for ep in episodes:
        for t in range(1, len(ep) - 1):
            if dim == 11:
                ctx = compute_11_axes(ep[t:t+1], ep[t-1:t])
                tgt = compute_11_axes(ep[t+1:t+2], ep[t:t+1])
            else:
                ctx = compute_3_axes(ep[t:t+1])
                tgt = compute_3_axes(ep[t+1:t+2])
            contexts.append(ctx[0])
            targets.append(tgt[0])
    return (torch.tensor(np.array(contexts), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


def make_pairs_raw(episodes):
    """Create (context, target) pairs with raw 5D state normalized."""
    contexts, targets = [], []
    for ep in episodes:
        normed = ep.copy()
        normed[:, 0] /= 512.0
        normed[:, 1] /= 512.0
        normed[:, 2] /= 512.0
        normed[:, 3] /= 512.0
        normed[:, 4] /= (2 * np.pi)
        for t in range(len(ep) - 1):
            contexts.append(normed[t])
            targets.append(normed[t + 1])
    return (torch.tensor(np.array(contexts), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


def random_orthogonal_matrix(input_dim, output_dim, seed=9999):
    """Generate a random semi-orthogonal matrix input_dim → output_dim."""
    rng = np.random.RandomState(seed)
    H = rng.randn(output_dim, input_dim)
    U, _, Vt = np.linalg.svd(H, full_matrices=False)
    if output_dim <= input_dim:
        return Vt[:output_dim].T.astype(np.float32)  # input_dim × output_dim
    else:
        return U[:, :input_dim].astype(np.float32).T  # input_dim × output_dim


class Predictor(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class FreeEncoder(nn.Module):
    def __init__(self, input_dim=5, output_dim=11, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_condition(condition, episodes, seed, dim, epochs=30, batch_size=128, lr=3e-4, wd=1e-3):
    """Train one condition, return best val loss and training history."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_train = int(0.8 * len(episodes))
    train_eps = episodes[:n_train]
    val_eps = episodes[n_train:]

    free_encoder = None
    W_random = None

    if condition == "prescribed_11":
        ctx_train, tgt_train = make_pairs_prescribed(train_eps, dim=11)
        ctx_val, tgt_val = make_pairs_prescribed(val_eps, dim=11)
        effective_dim = 11
    elif condition == "prescribed_3":
        ctx_train, tgt_train = make_pairs_prescribed(train_eps, dim=3)
        ctx_val, tgt_val = make_pairs_prescribed(val_eps, dim=3)
        effective_dim = 3
    elif condition == "random_fixed_11":
        ctx_raw_train, tgt_raw_train = make_pairs_raw(train_eps)
        ctx_raw_val, tgt_raw_val = make_pairs_raw(val_eps)
        W = torch.tensor(random_orthogonal_matrix(5, 11, seed=9999), dtype=torch.float32)
        ctx_train = ctx_raw_train @ W
        tgt_train = tgt_raw_train @ W
        ctx_val = ctx_raw_val @ W
        tgt_val = tgt_raw_val @ W
        effective_dim = 11
    elif condition == "free_11":
        ctx_train, tgt_train = make_pairs_raw(train_eps)
        ctx_val, tgt_val = make_pairs_raw(val_eps)
        free_encoder = FreeEncoder(5, 11)
        effective_dim = 11
    elif condition == "free_3":
        ctx_train, tgt_train = make_pairs_raw(train_eps)
        ctx_val, tgt_val = make_pairs_raw(val_eps)
        free_encoder = FreeEncoder(5, 3)
        effective_dim = 3
    else:
        raise ValueError(f"Unknown condition: {condition}")

    predictor = Predictor(effective_dim)

    if free_encoder is not None:
        params = list(predictor.parameters()) + list(free_encoder.parameters())
    else:
        params = list(predictor.parameters())

    n_params = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    history = []
    n_train_samples = len(ctx_train)

    for epoch in range(epochs):
        # Train
        if free_encoder:
            free_encoder.train()
        predictor.train()

        perm = torch.randperm(n_train_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train_samples, batch_size):
            idx = perm[i:i+batch_size]
            ctx_b = ctx_train[idx]
            tgt_b = tgt_train[idx]

            if free_encoder:
                ctx_b = free_encoder(ctx_b)
                tgt_b_enc = free_encoder(tgt_b)
            else:
                tgt_b_enc = tgt_b

            pred = predictor(ctx_b)
            loss = nn.functional.mse_loss(pred, tgt_b_enc.detach() if free_encoder else tgt_b_enc)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # Validate
        if free_encoder:
            free_encoder.eval()
        predictor.eval()

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

        history.append({"epoch": epoch + 1, "train": train_loss, "val": val_loss})

    return best_val, n_params, history


def main():
    parser = argparse.ArgumentParser(description="11 Prescribed Axes Experiment")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument("--output", type=str, default="results_11axes")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print("Generating episodes...")
    episodes = generate_pusht_episodes(n_episodes=args.episodes)
    print(f"Generated {len(episodes)} episodes, {len(episodes[0])} steps each\n")

    conditions = ["prescribed_3", "prescribed_11", "random_fixed_11", "free_3", "free_11"]
    results = {}

    for cond in conditions:
        dim = 11 if "11" in cond else 3
        results[cond] = {"seeds": {}}
        print(f"{'='*50}")
        print(f"  {cond.upper()} (dim={dim})")
        print(f"{'='*50}")

        for seed in args.seeds:
            t0 = time.time()
            val_loss, n_params, history = train_condition(
                cond, episodes, seed, dim=dim, epochs=args.epochs
            )
            elapsed = time.time() - t0
            results[cond]["seeds"][str(seed)] = {
                "val_loss": val_loss,
                "n_params": n_params,
                "time": round(elapsed, 1)
            }
            print(f"  seed={seed}: val_loss={val_loss:.6f} params={n_params:,} ({elapsed:.1f}s)")

        vals = [v["val_loss"] for v in results[cond]["seeds"].values()]
        results[cond]["mean"] = float(np.mean(vals))
        results[cond]["std"] = float(np.std(vals))
        results[cond]["dim"] = dim
        print(f"  → mean={results[cond]['mean']:.6f} ± {results[cond]['std']:.6f}\n")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<20} {'Dim':>4} {'Mean val loss':>14} {'vs presc_3':>12}")
    print("-" * 60)

    p3 = results["prescribed_3"]["mean"]
    for cond in conditions:
        m = results[cond]["mean"]
        ratio = m / p3 if p3 > 0 else float("inf")
        marker = " ★" if cond == "prescribed_11" else ""
        print(f"{cond:<20} {results[cond]['dim']:>4} {m:>14.6f} {ratio:>11.2f}×{marker}")

    # Key comparisons
    print(f"\n--- Key comparisons ---")
    p11 = results["prescribed_11"]["mean"]
    r11 = results["random_fixed_11"]["mean"]
    f11 = results["free_11"]["mean"]
    f3 = results["free_3"]["mean"]

    print(f"prescribed_11 vs prescribed_3:  {p11/p3:.2f}× ({'better' if p11 < p3 else 'worse'})")
    print(f"prescribed_11 vs free_11:       {f11/p11:.2f}× gap")
    print(f"random_fixed_11 vs free_11:     {f11/r11:.2f}× gap")
    print(f"prescribed_11 vs random_fixed:  {r11/p11:.2f}× gap")
    print(f"free_11 vs free_3:              {f11/f3:.2f}×")

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
