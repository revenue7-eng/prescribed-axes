"""
Prescribed Axes: Lower Boundary Test
Tests dim=1, 2, 3 to find if dim=3 is optimal or just the lowest tested.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time

def generate_pusht_episodes(n_episodes=100, steps_per_episode=50, seed=0):
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


def compute_axes(states, dim):
    bx, by, btheta = states[:, 2], states[:, 3], states[:, 4]
    if dim == 1:
        return np.stack([bx / 512.0], axis=-1).astype(np.float32)
    elif dim == 2:
        return np.stack([bx / 512.0, by / 512.0], axis=-1).astype(np.float32)
    else:  # dim == 3
        return np.stack([bx / 512.0, by / 512.0, btheta / (2*np.pi)], axis=-1).astype(np.float32)


def make_pairs_prescribed(episodes, dim):
    contexts, targets = [], []
    for ep in episodes:
        for t in range(len(ep) - 1):
            ctx = compute_axes(ep[t:t+1], dim)
            tgt = compute_axes(ep[t+1:t+2], dim)
            contexts.append(ctx[0])
            targets.append(tgt[0])
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


def train(ctx_tr, tgt_tr, ctx_va, tgt_va, dim, free_enc=None, epochs=20, bs=128, lr=3e-4):
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
    episodes = generate_pusht_episodes(100)
    n_tr = 80
    dims = [1, 2, 3]
    seeds = [42, 123, 777]

    print(f"{'Dim':>4} {'Prescribed':>12} {'Free':>12} {'Ratio':>8} {'Winner':>10}")
    print("-" * 52)

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
            fe = FreeEncoder(5, dim)
            f = train(cr, tr, cv, tv, dim, free_enc=fe, epochs=20)
            f_vals.append(f)

        pm, fm = np.mean(p_vals), np.mean(f_vals)
        ratio = fm / pm if pm > 0 else 0
        winner = "PRESCRIBED" if pm < fm else "FREE"
        print(f"{dim:>4} {pm:>12.6f} {fm:>12.6f} {ratio:>7.1f}× {winner:>10}")

if __name__ == "__main__":
    main()
