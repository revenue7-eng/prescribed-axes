"""
Prescribed Axes Fragility Test
================================
Why does dim=3→4 kill prescribed advantage on Push-T?

Three hypotheses:
  A. Prediction conflict: 4th axis is redundant, creates inconsistency
  B. Gradient dilution: extra axis wastes predictor capacity
  C. Subspace violation: 4th axis breaks the block-only subspace selection

Test: 5 conditions, all dim=4, different 4th axis:
  1. prescribed_3:     baseline (dim=3, x,y,θ)
  2. prescribed_4_dist: x,y,θ + distance_agent_block (redundant, from different subspace)
  3. prescribed_4_sin:  x,y,θ + sin(θ) (redundant, same subspace)
  4. prescribed_4_noise: x,y,θ + frozen random noise (independent, no information)
  5. prescribed_4_agent: x,y,θ + agent_x (independent, different subspace)
  6. free_3:            free encoder baseline

If 4_noise ≈ 4_dist ≈ 4_sin → mechanism B (any extra axis hurts)
If 4_noise < 4_dist and 4_noise < 4_sin → mechanism A (conflict from redundancy)
If 4_agent ≈ 4_dist but 4_sin better → mechanism C (subspace violation)

Author: Andrey Lazarev
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path


def generate_pusht_episodes(n_episodes=200, steps=50, seed=0):
    rng = np.random.RandomState(seed)
    episodes = []
    for _ in range(n_episodes):
        ax, ay = rng.uniform(50, 462), rng.uniform(50, 462)
        bx, by = rng.uniform(100, 412), rng.uniform(100, 412)
        btheta = rng.uniform(0, 2 * np.pi)
        tx, ty = rng.uniform(50, 462), rng.uniform(50, 462)
        states = []
        for t in range(steps):
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


def make_prescribed_pairs(episodes, condition, noise_per_episode=None):
    """
    Build (context, target) pairs for different 4th-axis conditions.
    All conditions share first 3 axes: bx/512, by/512, bθ/2π
    """
    contexts, targets = [], []
    for ep_idx, ep in enumerate(episodes):
        for t in range(len(ep) - 1):
            s_now = ep[t]
            s_next = ep[t + 1]
            ax, ay, bx, by, btheta = s_now
            ax2, ay2, bx2, by2, btheta2 = s_next

            base_now = [bx/512, by/512, btheta/(2*np.pi)]
            base_next = [bx2/512, by2/512, btheta2/(2*np.pi)]

            if condition == "prescribed_3":
                contexts.append(base_now)
                targets.append(base_next)

            elif condition == "prescribed_4_dist":
                d_now = np.sqrt((ax-bx)**2 + (ay-by)**2) / 512
                d_next = np.sqrt((ax2-bx2)**2 + (ay2-by2)**2) / 512
                contexts.append(base_now + [d_now])
                targets.append(base_next + [d_next])

            elif condition == "prescribed_4_sin":
                contexts.append(base_now + [np.sin(btheta)])
                targets.append(base_next + [np.sin(btheta2)])

            elif condition == "prescribed_4_noise":
                # Frozen noise: same random value for same (episode, timestep)
                n_now = noise_per_episode[ep_idx][t]
                n_next = noise_per_episode[ep_idx][t + 1]
                contexts.append(base_now + [n_now])
                targets.append(base_next + [n_next])

            elif condition == "prescribed_4_agent":
                contexts.append(base_now + [ax/512])
                targets.append(base_next + [ax2/512])

    dim = 3 if condition == "prescribed_3" else 4
    return (torch.tensor(contexts, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            dim)


def make_free_pairs(episodes):
    contexts, targets = [], []
    for ep in episodes:
        normed = ep.copy()
        normed[:, :4] /= 512
        normed[:, 4] /= (2 * np.pi)
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
    def __init__(self, in_d=5, out_d=3, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)


def train(ctx_tr, tgt_tr, ctx_va, tgt_va, dim, free_enc=None,
          epochs=30, bs=128, lr=3e-4):
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
    print("Prescribed Axes Fragility Test")
    print("Why does dim=3→4 break prescribed?")
    print("=" * 65)

    episodes = generate_pusht_episodes(200)

    # Generate frozen noise for noise condition
    rng_noise = np.random.RandomState(12345)
    noise_per_episode = []
    for ep in episodes:
        noise_per_episode.append(rng_noise.uniform(0, 1, size=len(ep)).astype(np.float32))

    n_tr = 160
    train_eps = episodes[:n_tr]
    val_eps = episodes[n_tr:]
    noise_tr = noise_per_episode[:n_tr]
    noise_va = noise_per_episode[n_tr:]

    conditions = [
        "prescribed_3",
        "prescribed_4_dist",
        "prescribed_4_sin",
        "prescribed_4_noise",
        "prescribed_4_agent",
    ]

    seeds = [42, 123, 777]
    results = {}

    for cond in conditions:
        results[cond] = {}
        print(f"\n{'='*50}")
        print(f"  {cond.upper()}")
        print(f"{'='*50}")

        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)

            noise_arg_tr = noise_tr if cond == "prescribed_4_noise" else None
            noise_arg_va = noise_va if cond == "prescribed_4_noise" else None

            ctx_tr, tgt_tr, dim = make_prescribed_pairs(train_eps, cond, noise_arg_tr)
            ctx_va, tgt_va, _ = make_prescribed_pairs(val_eps, cond, noise_arg_va)

            val_loss = train(ctx_tr, tgt_tr, ctx_va, tgt_va, dim, epochs=30)
            results[cond][str(seed)] = val_loss
            print(f"  seed={seed}: {val_loss:.6f}")

        vals = list(results[cond].values())
        results[cond]["mean"] = float(np.mean(vals))
        results[cond]["std"] = float(np.std(vals))
        print(f"  → mean={results[cond]['mean']:.6f} ± {results[cond]['std']:.6f}")

    # Free baseline
    print(f"\n{'='*50}")
    print(f"  FREE_3")
    print(f"{'='*50}")
    results["free_3"] = {}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        ctx_tr, tgt_tr = make_free_pairs(train_eps)
        ctx_va, tgt_va = make_free_pairs(val_eps)
        fe = FreeEncoder(5, 3)
        val_loss = train(ctx_tr, tgt_tr, ctx_va, tgt_va, 3, free_enc=fe, epochs=30)
        results["free_3"][str(seed)] = val_loss
        print(f"  seed={seed}: {val_loss:.6f}")
    vals = list(v for k, v in results["free_3"].items() if k != "mean" and k != "std")
    results["free_3"]["mean"] = float(np.mean(vals))
    results["free_3"]["std"] = float(np.std(vals))
    print(f"  → mean={results['free_3']['mean']:.6f}")

    # Summary
    print(f"\n{'='*65}")
    print("FRAGILITY TEST SUMMARY")
    print(f"{'='*65}")
    p3 = results["prescribed_3"]["mean"]
    print(f"{'Condition':<25} {'Mean':>10} {'vs p3':>8} {'4th axis type':<20}")
    print("-" * 65)
    descriptions = {
        "prescribed_3": "baseline (no 4th)",
        "prescribed_4_dist": "redundant, cross-subspace",
        "prescribed_4_sin": "redundant, same subspace",
        "prescribed_4_noise": "independent, no info",
        "prescribed_4_agent": "independent, cross-subspace",
        "free_3": "learned (free encoder)",
    }
    for cond in list(conditions) + ["free_3"]:
        m = results[cond]["mean"]
        ratio = m / p3 if p3 > 0 else 0
        print(f"{cond:<25} {m:>10.6f} {ratio:>7.1f}× {descriptions[cond]:<20}")

    # Diagnosis
    print(f"\n--- Diagnosis ---")
    d_dist = results["prescribed_4_dist"]["mean"]
    d_sin = results["prescribed_4_sin"]["mean"]
    d_noise = results["prescribed_4_noise"]["mean"]
    d_agent = results["prescribed_4_agent"]["mean"]

    print(f"noise vs dist:  {d_noise/d_dist:.2f}×", end="")
    if abs(d_noise - d_dist) / max(d_dist, 1e-9) < 0.3:
        print(" (similar → mechanism B: any extra axis hurts)")
    elif d_noise < d_dist:
        print(" (noise better → mechanism A: redundancy conflict)")
    else:
        print(" (noise worse → unexpected)")

    print(f"noise vs sin:   {d_noise/d_sin:.2f}×", end="")
    if abs(d_noise - d_sin) / max(d_sin, 1e-9) < 0.3:
        print(" (similar → mechanism B)")
    elif d_noise < d_sin:
        print(" (noise better → mechanism A)")
    else:
        print(" (noise worse → unexpected)")

    print(f"sin vs dist:    {d_sin/d_dist:.2f}×", end="")
    if d_sin < d_dist * 0.7:
        print(" (sin much better → mechanism C: subspace matters)")
    else:
        print(" (similar → subspace not the key factor)")

    print(f"agent vs dist:  {d_agent/d_dist:.2f}×")

    # Save
    out = Path("results_fragility")
    out.mkdir(exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}/results.json")


if __name__ == "__main__":
    main()
