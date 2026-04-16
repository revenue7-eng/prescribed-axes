#!/usr/bin/env python3
"""
Fix Missing Data: E14 and E16
==============================
E14: Push-T lower boundary (dim=1,2,3), 200 episodes, 30 epochs, 3 seeds
E16: Double pendulum (dim=1,2,3,4,5,6,7,8), 200 episodes, 30 epochs, 3 seeds

Run: python fix_e14_e16.py
Results: fix_e14_results.json, fix_e16_results.json
"""
import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

SEEDS = [42, 123, 777]
EPISODES = 200
EPOCHS = 30

# ================================================================
# Infrastructure
# ================================================================

class SIGReg(nn.Module):
    def __init__(self):
        super().__init__()
        t = torch.linspace(0, 3, 17); dt = 3 / 16
        w = torch.full((17,), 2 * dt); w[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', phi)
        self.register_buffer('weights', w * phi)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 512, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x = (proj @ A).unsqueeze(-1) * self.t
        err = (x.cos().mean(-3) - self.phi).square() + x.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


class SeqDS(Dataset):
    def __init__(self, episodes, H=3):
        self.w = []
        for e in episodes:
            st, a = e['s'], e['a']
            for t in range(len(a) - H):
                self.w.append((st[t:t + H + 2].astype(np.float32),
                               a[t:t + H + 1].astype(np.float32)))
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        st, a = self.w[i]
        return torch.from_numpy(st), torch.from_numpy(a)


class FreeEncoder(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, d_out))
    def forward(self, x): return self.net(x)


class ActionEncoder(nn.Module):
    def __init__(self, d_act, d_lat):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_act, 32), nn.GELU(), nn.Linear(32, d_lat))
    def forward(self, a): return self.net(a)


class Predictor(nn.Module):
    def __init__(self, d_lat, H=3):
        super().__init__()
        h = max(128, d_lat * 8)
        self.net = nn.Sequential(
            nn.Linear(H * 2 * d_lat, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, d_lat))
    def forward(self, e, ae):
        return self.net(torch.cat([e, ae], -1).reshape(e.size(0), -1))


class WorldModel(nn.Module):
    def __init__(self, enc, d_lat, d_act, H=3):
        super().__init__()
        self.enc = enc
        self.ae = ActionEncoder(d_act, d_lat)
        self.pr = Predictor(d_lat, H)
        self.sig = SIGReg()
        self.H = H
    def forward(self, st, a):
        emb = self.enc(st)
        ctx, tgt = emb[:, :self.H], emb[:, self.H]
        aem = self.ae(a[:, :self.H])
        p = self.pr(ctx, aem)
        return {
            'pl': F.mse_loss(p, tgt.detach()),
            'sl': self.sig(emb.transpose(0, 1)),
        }


@torch.no_grad()
def val_loss(mdl, vl):
    mdl.eval(); tp = 0; n = 0
    for s, a in vl:
        o = mdl(s, a); tp += o['pl'].item() * s.size(0); n += s.size(0)
    return tp / n


def make_loaders(episodes, seed, batch_size=64):
    ds = SeqDS(episodes, 3)
    nt = int(len(ds) * 0.9); nv = len(ds) - nt
    tr, va = random_split(ds, [nt, nv],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=batch_size)
    return tl, vl


def train_one(enc, d_lat, d_act, tl, vl, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    mdl = WorldModel(enc, d_lat, d_act)
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)
    best = float('inf'); best_ep = 0
    for ep in range(1, epochs + 1):
        mdl.train()
        for s, a in tl:
            o = mdl(s, a); l = o['pl'] + 0.09 * o['sl']
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step()
        vl_ = val_loss(mdl, vl)
        if vl_ < best: best = vl_; best_ep = ep
    return best, best_ep


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [Saved to {path}]")


# ================================================================
# Data generators
# ================================================================

def synth_pusht(n, seed=42):
    rng = np.random.default_rng(seed); eps = []
    for _ in range(n):
        ag = rng.uniform(50, 462, 2).astype(np.float32)
        bp = rng.uniform(100, 412, 2).astype(np.float32)
        ba = np.float32(rng.uniform(0, 2 * np.pi))
        st = np.array([ag[0], ag[1], bp[0], bp[1], ba], dtype=np.float32)
        ss, aa = [st.copy()], []
        tgt = rng.uniform(50, 462, 2).astype(np.float32)
        for step in range(300):
            if step % 20 == 0:
                tgt = rng.uniform(50, 462, 2).astype(np.float32)
            act = np.clip(tgt + rng.normal(0, 10, 2), 0, 512).astype(np.float32)
            d = act - ag; dn = np.linalg.norm(d)
            if dn > 0: ag += d * min(1., 20. / dn)
            ag = np.clip(ag, 0, 512)
            tb = bp - ag; cd = np.linalg.norm(tb)
            if 0 < cd < 30:
                f = (30 - cd) / 30 * 5; bp += (tb / cd) * f
                ba = (ba + rng.normal(0, .05) * f) % (2 * np.pi)
            bp = np.clip(bp, 0, 512)
            st = np.array([ag[0], ag[1], bp[0], bp[1], ba], dtype=np.float32)
            if (step + 1) % 5 == 0:
                ss.append(st.copy()); aa.append(act)
        if len(aa) >= 4:
            eps.append({'s': np.array(ss[:len(aa) + 1]), 'a': np.array(aa)})
    return eps


def synth_double_pendulum(n, seed=42):
    rng = np.random.default_rng(seed)
    g = 9.81; dt_phys = 0.02
    eps = []
    for _ in range(n):
        t1 = np.float32(rng.uniform(-np.pi, np.pi))
        w1 = np.float32(rng.uniform(-1, 1))
        t2 = np.float32(rng.uniform(-np.pi, np.pi))
        w2 = np.float32(rng.uniform(-1, 1))
        st = np.array([t1, w1, t2, w2], dtype=np.float32)
        ss, aa = [st.copy()], []
        for step in range(300):
            tau1 = np.float32(rng.uniform(-1, 1))
            tau2 = np.float32(rng.uniform(-1, 1))
            dt_ = t1 - t2
            den = 1.0 + 1.0 * np.sin(dt_) ** 2
            a1 = (-1.0 * w1**2 * np.sin(dt_) * np.cos(dt_)
                   - 1.0 * w2**2 * np.sin(dt_)
                   - 2.0 * g * np.sin(t1)
                   + 1.0 * g * np.sin(t2) * np.cos(dt_)
                   + tau1) / den
            a2 = (1.0 * w2**2 * np.sin(dt_) * np.cos(dt_)
                  + 2.0 * w1**2 * np.sin(dt_)
                  + 2.0 * g * np.sin(t1) * np.cos(dt_)
                  - 2.0 * g * np.sin(t2)
                  + tau2) / den
            w1 = np.float32(np.clip(w1 + a1 * dt_phys, -10, 10))
            w2 = np.float32(np.clip(w2 + a2 * dt_phys, -10, 10))
            t1 = np.float32((t1 + w1 * dt_phys + np.pi) % (2 * np.pi) - np.pi)
            t2 = np.float32((t2 + w2 * dt_phys + np.pi) % (2 * np.pi) - np.pi)
            st = np.array([t1, w1, t2, w2], dtype=np.float32)
            if (step + 1) % 5 == 0:
                ss.append(st.copy())
                aa.append(np.array([tau1, tau2], dtype=np.float32))
        if len(aa) >= 4:
            eps.append({'s': np.array(ss[:len(aa) + 1]), 'a': np.array(aa)})
    return eps


# ================================================================
# Prescribed encoders
# ================================================================

class PushTPrescribed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        scales = [1/512, 1/512, 1/(2*np.pi)]
        self.register_buffer('sc', torch.tensor(scales[:dim]))
    def forward(self, x):
        indices = [2, 3, 4][:self.dim]
        return x[..., indices] * self.sc


class DoublePendulumPrescribed(nn.Module):
    def __init__(self, dim, normalize=False):
        super().__init__()
        self.dim = dim
        self.normalize = normalize
        if normalize:
            # t1,t2 in [-pi,pi] -> [0,1], w1,w2 in [-10,10] -> [0,1]
            lows = [-np.pi, -10.0, -np.pi, -10.0]
            highs = [np.pi, 10.0, np.pi, 10.0]
            self.register_buffer('low', torch.tensor(lows[:dim], dtype=torch.float32))
            self.register_buffer('rng', torch.tensor([h - l for h, l in zip(highs[:dim], lows[:dim])], dtype=torch.float32))

    def forward(self, x):
        out = x[..., :self.dim]
        if self.normalize:
            out = (out - self.low) / self.rng
        return out


# ================================================================
# E14: Push-T Lower Boundary
# ================================================================

def run_e14():
    print("\n" + "#" * 65)
    print("# E14: PUSH-T LOWER BOUNDARY (dim=1,2,3)")
    print("#" * 65)

    save_path = Path("fix_e14_results.json")
    if save_path.exists():
        with open(save_path) as f:
            results = json.load(f)
        print(f"Resuming from {save_path}")
    else:
        results = {'config': {'seeds': SEEDS, 'episodes': EPISODES, 'epochs': EPOCHS,
                               'environment': 'push-t', 'dims': [1, 2, 3]}}

    for dim in [1, 2, 3]:
        for seed in SEEDS:
            for enc_type in ['prescribed', 'free']:
                key = f"{enc_type}_dim{dim}_seed{seed}"
                if key in results and key != 'config':
                    print(f"  {key}: cached ({results[key]['best_val_loss']:.6f})")
                    continue

                t0 = time.time()
                torch.manual_seed(seed); np.random.seed(seed)
                episodes = synth_pusht(EPISODES, seed)
                tl, vl = make_loaders(episodes, seed)

                if enc_type == 'prescribed':
                    enc = PushTPrescribed(dim)
                else:
                    enc = FreeEncoder(5, dim)

                best, best_ep = train_one(enc, dim, 2, tl, vl, seed)
                dt = time.time() - t0
                results[key] = {'best_val_loss': best, 'best_epoch': best_ep, 'time': round(dt, 1)}
                print(f"  {key}: {best:.6f} (ep {best_ep}, {dt:.0f}s)")
                save_json(results, save_path)

    print("\n  --- E14 Summary ---")
    print(f"  {'Dim':>5} {'Prescribed':>12} {'Free':>12} {'Ratio':>8}")
    for dim in [1, 2, 3]:
        p = [results[f'prescribed_dim{dim}_seed{s}']['best_val_loss'] for s in SEEDS]
        f = [results[f'free_dim{dim}_seed{s}']['best_val_loss'] for s in SEEDS]
        print(f"  {dim:>5} {np.mean(p):>12.6f} {np.mean(f):>12.6f} {np.mean(f)/np.mean(p):>7.1f}x")


# ================================================================
# E16: Double Pendulum
# ================================================================

def run_e16():
    print("\n" + "#" * 65)
    print("# E16: DOUBLE PENDULUM (dim=1,2,3,4,5,6,7,8)")
    print("#" * 65)

    save_path = Path("fix_e16_results.json")
    if save_path.exists():
        with open(save_path) as f:
            results = json.load(f)
        print(f"Resuming from {save_path}")
    else:
        results = {'config': {'seeds': SEEDS, 'episodes': EPISODES, 'epochs': EPOCHS,
                               'environment': 'double_pendulum', 'state_dim': 4,
                               'dims': [1, 2, 3, 4, 5, 6, 7, 8]}}

    for dim in [1, 2, 3, 4, 5, 6, 7, 8]:
        for seed in SEEDS:
            for enc_type in ['prescribed', 'prescribed_norm', 'free']:
                key = f"{enc_type}_dim{dim}_seed{seed}"
                if key in results and key != 'config':
                    print(f"  {key}: cached ({results[key]['best_val_loss']:.6f})")
                    continue

                t0 = time.time()
                torch.manual_seed(seed); np.random.seed(seed)
                episodes = synth_double_pendulum(EPISODES, seed)
                tl, vl = make_loaders(episodes, seed)

                d_state = 4; d_act = 2
                if enc_type == 'prescribed':
                    actual_dim = min(dim, d_state)
                    enc = DoublePendulumPrescribed(actual_dim, normalize=False)
                    lat_dim = actual_dim
                elif enc_type == 'prescribed_norm':
                    actual_dim = min(dim, d_state)
                    enc = DoublePendulumPrescribed(actual_dim, normalize=True)
                    lat_dim = actual_dim
                else:
                    enc = FreeEncoder(d_state, dim)
                    lat_dim = dim

                best, best_ep = train_one(enc, lat_dim, d_act, tl, vl, seed)
                dt = time.time() - t0
                results[key] = {'best_val_loss': best, 'best_epoch': best_ep, 'time': round(dt, 1)}
                print(f"  {key}: {best:.6f} (ep {best_ep}, {dt:.0f}s)")
                save_json(results, save_path)

    print("\n  --- E16 Summary ---")
    print(f"  {'Dim':>5} {'Prescribed':>12} {'Presc_Norm':>12} {'Free':>12} {'P/F':>7} {'PN/F':>7}")
    for dim in [1, 2, 3, 4, 5, 6, 7, 8]:
        actual = min(dim, 4)
        p = [results.get(f'prescribed_dim{dim}_seed{s}', {}).get('best_val_loss') for s in SEEDS]
        pn = [results.get(f'prescribed_norm_dim{dim}_seed{s}', {}).get('best_val_loss') for s in SEEDS]
        f = [results.get(f'free_dim{dim}_seed{s}', {}).get('best_val_loss') for s in SEEDS]
        p = [v for v in p if v is not None]
        pn = [v for v in pn if v is not None]
        f = [v for v in f if v is not None]
        if p and f:
            pm, fm = np.mean(p), np.mean(f)
            r_pf = fm / pm if pm > 0 else 0
            cap = f"({actual})" if actual < dim else ""
            if pn:
                pnm = np.mean(pn)
                r_pnf = fm / pnm if pnm > 0 else 0
                print(f"  {dim:>5} {pm:>12.6f} {pnm:>12.6f} {fm:>12.6f} {r_pf:>6.2f}x {r_pnf:>6.2f}x {cap}")
            else:
                print(f"  {dim:>5} {pm:>12.6f} {'—':>12} {fm:>12.6f} {r_pf:>6.2f}x {'—':>7} {cap}")

    # П1 test
    print("\n  --- П1: Does normalization help? ---")
    for dim in [1, 2, 4]:
        p = [results.get(f'prescribed_dim{dim}_seed{s}', {}).get('best_val_loss') for s in SEEDS]
        pn = [results.get(f'prescribed_norm_dim{dim}_seed{s}', {}).get('best_val_loss') for s in SEEDS]
        p = [v for v in p if v is not None]
        pn = [v for v in pn if v is not None]
        if p and pn:
            pm, pnm = np.mean(p), np.mean(pn)
            print(f"  dim={dim}: raw={pm:.6f}, norm={pnm:.6f}, norm/raw={pnm/pm:.3f}x")


# ================================================================
# Main
# ================================================================

def main():
    total_t0 = time.time()
    print("=" * 65)
    print("FIX MISSING DATA: E14 + E16")
    print(f"Seeds: {SEEDS}, Episodes: {EPISODES}, Epochs: {EPOCHS}")
    print("=" * 65)

    run_e14()
    run_e16()

    print(f"\nTotal time: {time.time() - total_t0:.0f}s")
    print(f"Results: fix_e14_results.json, fix_e16_results.json")


if __name__ == '__main__':
    main()
