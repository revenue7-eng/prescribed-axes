"""
Prescribed Axes Dimension Sweep
================================
Tests prescribed axes at dimensions 3, 4, 5, 6, 7, 9, 11, 15.

Each dimension adds one more derived axis:
  3: x, y, θ                          (core)
  4: + d_agent_block                   (distance)
  5: + x_agent                         (agent position)
  6: + y_agent
  7: + ẋ                               (velocity)
  8: + ẏ
  9: + θ̇                              (angular velocity)
  11: + sinθ, cosθ                     (redundant with θ)
  15: + ẍ, ÿ, θ̈, d²                  (second derivatives + redundant distance)

For each dimension: prescribed vs free encoder, 3 seeds.
Result: two curves showing where prescribed stops winning.

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
    rng = np.random.RandomState(seed)
    episodes = []
    for _ in range(n_episodes):
        ax, ay = rng.uniform(50, 462), rng.uniform(50, 462)
        bx, by = rng.uniform(100, 412), rng.uniform(100, 412)
        btheta = rng.uniform(0, 2 * np.pi)
        tx, ty = rng.uniform(50, 462), rng.uniform(50, 462)

        states = []
        for t in range(steps_per_episode):
            if t % 20 == 0:
                tx, ty = rng.uniform(50, 462), rng.uniform(50, 462)
            dx, dy = tx - ax, ty - ay
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                ax += dx * min(1.0, 20.0 / dist) + rng.normal(0, 2)
                ay += dy * min(1.0, 20.0 / dist) + rng.normal(0, 2)
            ax, ay = np.clip(ax, 0, 512), np.clip(ay, 0, 512)
            tb = np.sqrt((bx - ax)**2 + (by - ay)**2)
            if 0 < tb < 30:
                force = (30 - tb) / 30 * 5
                bx += (bx - ax) / tb * force
                by += (by - ay) / tb * force
                btheta += rng.normal(0, 0.05) * force
            bx, by = np.clip(bx, 0, 512), np.clip(by, 0, 512)
            btheta = btheta % (2 * np.pi)
            states.append([ax, ay, bx, by, btheta])
        episodes.append(np.array(states, dtype=np.float32))
    return episodes


def compute_axes(states_t, states_prev, states_prev2, dim):
    """
    Compute prescribed axes up to requested dimensionality.
    Each dimension is cumulative — dim=5 includes everything from dim=3,4.
    
    Order chosen by information value (non-redundant first):
      3: bx, by, bθ
      4: + d_agent_block
      5: + ax
      6: + ay
      7: + vx (block velocity)
      8: + vy
      9: + vθ
     11: + sinθ, cosθ         (redundant with θ)
     15: + accx, accy, accθ, v_agent  (second derivatives)
    """
    ax = states_t[:, 0]
    ay = states_t[:, 1]
    bx = states_t[:, 2]
    by = states_t[:, 3]
    btheta = states_t[:, 4]

    axes = [
        bx / 512.0,                                                    # 0
        by / 512.0,                                                    # 1
        btheta / (2 * np.pi),                                         # 2
    ]

    if dim >= 4:
        d_ab = np.sqrt((ax - bx)**2 + (ay - by)**2) / 512.0
        axes.append(d_ab)                                              # 3

    if dim >= 5:
        axes.append(ax / 512.0)                                        # 4

    if dim >= 6:
        axes.append(ay / 512.0)                                        # 5

    if dim >= 7:
        vx = (bx - states_prev[:, 2]) / 512.0
        axes.append(vx)                                                # 6

    if dim >= 8:
        vy = (by - states_prev[:, 3]) / 512.0
        axes.append(vy)                                                # 7

    if dim >= 9:
        vtheta = (btheta - states_prev[:, 4]) / (2 * np.pi)
        axes.append(vtheta)                                            # 8

    if dim >= 11:
        axes.append(np.sin(btheta))                                    # 9
        axes.append(np.cos(btheta))                                    # 10

    if dim >= 15:
        # Second derivatives (accelerations)
        vx_now = (bx - states_prev[:, 2])
        vx_prev = (states_prev[:, 2] - states_prev2[:, 2])
        acc_x = (vx_now - vx_prev) / 512.0
        axes.append(acc_x)                                             # 11

        vy_now = (by - states_prev[:, 3])
        vy_prev = (states_prev[:, 3] - states_prev2[:, 3])
        acc_y = (vy_now - vy_prev) / 512.0
        axes.append(acc_y)                                             # 12

        vt_now = (btheta - states_prev[:, 4])
        vt_prev = (states_prev[:, 4] - states_prev2[:, 4])
        acc_t = (vt_now - vt_prev) / (2 * np.pi)
        axes.append(acc_t)                                             # 13

        # Agent velocity magnitude
        v_agent = np.sqrt((ax - states_prev[:, 0])**2 + (ay - states_prev[:, 1])**2) / 512.0
        axes.append(v_agent)                                           # 14

    return np.stack(axes[:dim], axis=-1).astype(np.float32)


def make_pairs_prescribed(episodes, dim):
    contexts, targets = [], []
    for ep in episodes:
        # Need t-1 for velocities (dim>=7), t-2 for accelerations (dim>=15)
        start = 2 if dim >= 15 else 1 if dim >= 7 else 1
        for t in range(start, len(ep) - 1):
            prev = ep[t-1:t]
            prev2 = ep[t-2:t-1] if t >= 2 else ep[t-1:t]
            ctx = compute_axes(ep[t:t+1], prev, prev2, dim)
            
            prev_next = ep[t:t+1]
            prev2_next = ep[t-1:t]
            tgt = compute_axes(ep[t+1:t+2], prev_next, prev2_next, dim)
            
            contexts.append(ctx[0])
            targets.append(tgt[0])
    if len(contexts) == 0:
        raise ValueError(f"No pairs generated for dim={dim}, episodes={len(episodes)}")
    return (torch.tensor(np.array(contexts), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


def make_pairs_raw(episodes):
    contexts, targets = [], []
    for ep in episodes:
        normed = ep.copy()
        normed[:, :4] /= 512.0
        normed[:, 4] /= (2 * np.pi)
        for t in range(len(ep) - 1):
            contexts.append(normed[t])
            targets.append(normed[t + 1])
    return (torch.tensor(np.array(contexts), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


class Predictor(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class FreeEncoder(nn.Module):
    def __init__(self, input_dim=5, output_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


def train_one(condition, ctx_train, tgt_train, ctx_val, tgt_val, dim,
              free_encoder=None, epochs=30, batch_size=128, lr=3e-4, wd=1e-3):
    predictor = Predictor(dim)
    params = list(predictor.parameters())
    if free_encoder:
        params += list(free_encoder.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val = float("inf")
    n = len(ctx_train)

    for epoch in range(epochs):
        if free_encoder: free_encoder.train()
        predictor.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            c, t = ctx_train[idx], tgt_train[idx]
            if free_encoder:
                c, t = free_encoder(c), free_encoder(t)
            pred = predictor(c)
            loss = nn.functional.mse_loss(pred, t.detach() if free_encoder else t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        scheduler.step()

        if free_encoder: free_encoder.eval()
        predictor.eval()
        with torch.no_grad():
            c, t = ctx_val, tgt_val
            if free_encoder:
                c, t = free_encoder(c), free_encoder(t)
            val_loss = nn.functional.mse_loss(predictor(c), t).item()
        if val_loss < best_val:
            best_val = val_loss

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument("--output", type=str, default="results_sweep")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print("Generating episodes...")
    episodes = generate_pusht_episodes(n_episodes=args.episodes)
    print(f"Generated {len(episodes)} episodes\n")

    dims = [3, 4, 5, 6, 7, 9, 11, 15]
    results = {}

    for dim in dims:
        results[dim] = {"prescribed": {}, "free": {}}
        print(f"\n{'='*50}")
        print(f"  DIM = {dim}")
        print(f"{'='*50}")

        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Prescribed
            n_train = int(0.8 * len(episodes))
            ctx_tr, tgt_tr = make_pairs_prescribed(episodes[:n_train], dim)
            ctx_va, tgt_va = make_pairs_prescribed(episodes[n_train:], dim)
            t0 = time.time()
            p_loss = train_one("prescribed", ctx_tr, tgt_tr, ctx_va, tgt_va, dim,
                               epochs=args.epochs)
            p_time = time.time() - t0

            # Free
            torch.manual_seed(seed)
            ctx_tr_r, tgt_tr_r = make_pairs_raw(episodes[:n_train])
            ctx_va_r, tgt_va_r = make_pairs_raw(episodes[n_train:])
            free_enc = FreeEncoder(5, dim)
            t0 = time.time()
            f_loss = train_one("free", ctx_tr_r, tgt_tr_r, ctx_va_r, tgt_va_r, dim,
                               free_encoder=free_enc, epochs=args.epochs)
            f_time = time.time() - t0

            results[dim]["prescribed"][str(seed)] = p_loss
            results[dim]["free"][str(seed)] = f_loss

            ratio = f_loss / p_loss if p_loss > 0 else 0
            winner = "PRESCRIBED" if p_loss < f_loss else "FREE"
            print(f"  seed={seed}: prescribed={p_loss:.6f} free={f_loss:.6f} ratio={ratio:.1f}× [{winner}]")

        p_vals = list(results[dim]["prescribed"].values())
        f_vals = list(results[dim]["free"].values())
        results[dim]["prescribed_mean"] = float(np.mean(p_vals))
        results[dim]["free_mean"] = float(np.mean(f_vals))
        results[dim]["prescribed_std"] = float(np.std(p_vals))
        results[dim]["free_std"] = float(np.std(f_vals))

        pm, fm = results[dim]["prescribed_mean"], results[dim]["free_mean"]
        ratio = fm / pm if pm > 0 else 0
        winner = "PRESCRIBED" if pm < fm else "FREE"
        print(f"  MEAN: prescribed={pm:.6f} free={fm:.6f} ratio={ratio:.1f}× [{winner}]")

    # Final summary
    print(f"\n{'='*60}")
    print("DIMENSION SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dim':>4} {'Prescribed':>12} {'Free':>12} {'Ratio':>8} {'Winner':>12}")
    print("-" * 60)

    crossover = None
    for dim in dims:
        pm = results[dim]["prescribed_mean"]
        fm = results[dim]["free_mean"]
        ratio = fm / pm if pm > 0 else 0
        winner = "PRESCRIBED" if pm < fm else "FREE"
        print(f"{dim:>4} {pm:>12.6f} {fm:>12.6f} {ratio:>7.1f}× {winner:>12}")
        if pm > fm and crossover is None:
            crossover = dim

    if crossover:
        print(f"\n→ Crossover at dim={crossover}: free encoder starts winning")
    else:
        print(f"\n→ Prescribed wins at all tested dimensions")

    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_dir}/sweep_results.json")


if __name__ == "__main__":
    main()
