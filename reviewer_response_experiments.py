#!/usr/bin/env python3
"""
Reviewer Response Experiments — Push-T
=======================================
Closes reviewer questions #1, #8, #18, #19, #20, #33.

Five conditions x 3 seeds:
  1. prescribed          — fixed h(s)=normalize(s[2:5]), with SIGReg (lam=0.09)
  2. free                — MLP 5->64->64->3, with SIGReg (lam=0.09)
  3. free_3d             — MLP 3->64->64->3, same (x,y,theta) input as prescribed, with SIGReg
                           Equal-input control: isolates fixation vs learning at equal information.
  4. free_no_sigreg      — MLP 5->64->64->3, SIGReg disabled (lam=0)
  5. prescribed_no_sigreg — fixed h(s)=normalize(s[2:5]), SIGReg disabled (lam=0)

Run:
  python reviewer_response_experiments.py --synthetic --seeds 42 123 777
  # Or with gym-pusht:
  python reviewer_response_experiments.py --seeds 42 123 777

Author: Andrey Lazarev
"""

import os, time, json, argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# --- SIGReg ---
class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=512):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3/(knots-1)
        w = torch.full((knots,), 2*dt); w[[0,-1]] = dt
        phi = torch.exp(-t.square()/2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w*phi)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3)-self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights)*proj.size(-2)).mean()


class ZeroReg(nn.Module):
    """Returns 0. Replaces SIGReg when disabled."""
    def forward(self, proj):
        return torch.tensor(0.0, device=proj.device)


# --- Data ---
def collect_gym_data(n_ep=200, max_steps=300, fs=5, seed=42):
    try:
        import gymnasium as gym
        import gym_pusht
    except ImportError:
        print("gym-pusht not found -> synthetic")
        return collect_synthetic_data(n_ep, max_steps, fs, seed)
    rng = np.random.default_rng(seed)
    eps = []
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    for i in range(n_ep):
        obs, _ = env.reset(seed=int(rng.integers(0,100000)))
        ss, aa = [obs.copy()], []
        for step in range(max_steps):
            if step==0 or rng.random()<0.3:
                a = env.action_space.sample()
            else:
                a = aa[-1]+rng.normal(0,30,2).astype(np.float32)
                a = np.clip(a,0,512)
            obs,_,d,tr,_ = env.step(a)
            if (step+1)%fs==0: ss.append(obs.copy()); aa.append(a.copy())
            if d or tr: break
        if len(aa)>=4:
            eps.append({"s":np.array(ss[:len(aa)+1]),"a":np.array(aa)})
        if (i+1)%50==0: print(f"  {i+1}/{n_ep}")
    env.close()
    print(f"Gym: {len(eps)} eps, avg {np.mean([len(e['a']) for e in eps]):.0f}")
    return eps

def collect_synthetic_data(n_ep=200, max_steps=300, fs=5, seed=42):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_ep):
        ag = rng.uniform(50,462,2).astype(np.float32)
        bp = rng.uniform(100,412,2).astype(np.float32)
        ba = np.float32(rng.uniform(0,2*np.pi))
        st = np.array([ag[0],ag[1],bp[0],bp[1],ba],dtype=np.float32)
        ss, aa = [st.copy()], []
        tgt = rng.uniform(50,462,2).astype(np.float32)
        for step in range(max_steps):
            if step%20==0: tgt = rng.uniform(50,462,2).astype(np.float32)
            act = np.clip(tgt+rng.normal(0,10,2),0,512).astype(np.float32)
            d = act-ag; dn = np.linalg.norm(d)
            if dn>0: ag += d*min(1.0,20.0/dn)
            ag = np.clip(ag,0,512)
            tb = bp-ag; cd = np.linalg.norm(tb)
            if 0<cd<30:
                f = (30-cd)/30*5; bp += (tb/cd)*f
                ba = (ba+rng.normal(0,0.05)*f)%(2*np.pi)
            bp = np.clip(bp,0,512)
            st = np.array([ag[0],ag[1],bp[0],bp[1],ba],dtype=np.float32)
            if (step+1)%fs==0: ss.append(st.copy()); aa.append(act)
        if len(aa)>=4:
            eps.append({"s":np.array(ss[:len(aa)+1]),"a":np.array(aa)})
    print(f"Synth: {len(eps)} eps, avg {np.mean([len(e['a']) for e in eps]):.0f}")
    return eps

class SeqDS(Dataset):
    def __init__(self, eps, H=3):
        self.w = []
        for e in eps:
            s,a = e["s"],e["a"]
            for t in range(len(a)-H):
                self.w.append((s[t:t+H+2].astype(np.float32), a[t:t+H+1].astype(np.float32)))
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        s,a = self.w[i]
        return torch.from_numpy(s), torch.from_numpy(a)


# --- Models ---
class PrescEnc(nn.Module):
    """Fixed encoder: extracts and normalizes block coordinates (x, y, theta)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("sc", torch.tensor([1/512,1/512,1/(2*np.pi)]))
    def forward(self, s): return s[...,2:5]*self.sc

class FreeEnc(nn.Module):
    """Learned encoder: MLP di->h->h->do."""
    def __init__(self, di=5, do=3, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,do))
    def forward(self, s): return self.net(s)

class Free3DEnc(nn.Module):
    """Learned encoder with SAME input as prescribed: only (x, y, theta).
    Input: extracts s[...,2:5], normalizes, then passes through MLP.
    This is the equal-input control."""
    def __init__(self, do=3, h=64):
        super().__init__()
        self.register_buffer("sc", torch.tensor([1/512,1/512,1/(2*np.pi)]))
        self.net = nn.Sequential(nn.Linear(3,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,do))
    def forward(self, s):
        x = s[...,2:5]*self.sc  # same extraction + normalization as prescribed
        return self.net(x)

class ActEnc(nn.Module):
    def __init__(self, di=2, do=3, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di,h),nn.GELU(),nn.Linear(h,do))
    def forward(self, a): return self.net(a)

class Pred(nn.Module):
    def __init__(self, d=3, H=3, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H*2*d,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,d))
    def forward(self, e, ae):
        return self.net(torch.cat([e,ae],dim=-1).reshape(e.size(0),-1))

class Model(nn.Module):
    def __init__(self, enc, aenc, pred, reg, H=3):
        super().__init__()
        self.enc,self.aenc,self.pred,self.reg,self.H = enc,aenc,pred,reg,H
    def forward(self, s, a):
        H = self.H
        emb = self.enc(s)
        ctx, tgt = emb[:,:H], emb[:,H]
        ae = self.aenc(a[:,:H])
        p = self.pred(ctx, ae)
        return {"pl": F.mse_loss(p,tgt.detach()),
                "sl": self.reg(emb.transpose(0,1)),
                "p": p.detach(), "t": tgt.detach()}


# --- Training ---
def train_ep(m, dl, opt, lam, dev):
    m.train(); tp,ts,n = 0,0,0
    for s,a in dl:
        s,a = s.to(dev),a.to(dev)
        o = m(s,a); l = o["pl"]+lam*o["sl"]
        opt.zero_grad(); l.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        b=s.size(0); tp+=o["pl"].item()*b; ts+=o["sl"].item()*b; n+=b
    return tp/n,ts/n

@torch.no_grad()
def val_ep(m, dl, dev):
    m.eval(); tp,ts,n = 0,0,0; ps,ts_ = [],[]
    for s,a in dl:
        s,a = s.to(dev),a.to(dev); o = m(s,a)
        b=s.size(0); tp+=o["pl"].item()*b; ts+=o["sl"].item()*b; n+=b
        ps.append(o["p"].cpu()); ts_.append(o["t"].cpu())
    p,t = torch.cat(ps),torch.cat(ts_)
    return tp/n, ts/n, (p-t).pow(2).mean(0).tolist()


# --- Condition definitions ---
CONDITIONS = {
    "prescribed": {
        "desc": "Fixed encoder, SIGReg on (lam=0.09)",
        "make_enc": lambda dim: PrescEnc(),
        "use_sigreg": True,
    },
    "free": {
        "desc": "Learned MLP 5->64->64->3, SIGReg on (lam=0.09)",
        "make_enc": lambda dim: FreeEnc(5, dim, 64),
        "use_sigreg": True,
    },
    "free_3d": {
        "desc": "Learned MLP 3->64->64->3, SAME input as prescribed, SIGReg on (lam=0.09)",
        "make_enc": lambda dim: Free3DEnc(dim, 64),
        "use_sigreg": True,
    },
    "free_no_sigreg": {
        "desc": "Learned MLP 5->64->64->3, SIGReg OFF (lam=0)",
        "make_enc": lambda dim: FreeEnc(5, dim, 64),
        "use_sigreg": False,
    },
    "prescribed_no_sigreg": {
        "desc": "Fixed encoder, SIGReg OFF (lam=0)",
        "make_enc": lambda dim: PrescEnc(),
        "use_sigreg": False,
    },
}


def run_condition(cond_name, cond_cfg, eps, seed, dim=3, H=3, hid=128,
                  epochs=50, bs=128, lr=3e-4, wd=1e-3, lam=0.09, split=0.9, dev="cpu"):
    print(f"\n{'='*60}")
    print(f"  {cond_name.upper()} | seed={seed}")
    print(f"  {cond_cfg['desc']}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = SeqDS(eps, H)
    nt = int(len(ds)*split); nv = len(ds)-nt
    tr,va = random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr,batch_size=bs,shuffle=True,drop_last=True)
    vl = DataLoader(va,batch_size=bs)
    print(f"  train={nt} val={nv}")

    enc = cond_cfg["make_enc"](dim)
    reg = SIGReg(17,512) if cond_cfg["use_sigreg"] else ZeroReg()
    effective_lam = lam if cond_cfg["use_sigreg"] else 0.0

    mdl = Model(enc, ActEnc(2,dim,32), Pred(dim,H,hid), reg, H).to(dev)
    np_ = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  params: {np_:,}  lam={effective_lam}")

    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                            lr=lr,weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)

    best,bep,hist = float("inf"),0,[]
    for ep in range(1,epochs+1):
        t0 = time.time()
        tp,ts = train_ep(mdl,tl,opt,effective_lam,dev)
        vp,vs,axes = val_ep(mdl,vl,dev)
        sch.step()
        if vp<best: best,bep = vp,ep
        hist.append({"ep":ep,"tp":tp,"ts":ts,"vp":vp,"vs":vs,"ax":axes})
        if ep%10==0 or ep==1:
            ax = ", ".join(f"{v:.6f}" for v in axes)
            print(f"  ep {ep:3d} | tr {tp:.6f} | val {vp:.6f} | sig {vs:.4f} | [{ax}] | {time.time()-t0:.1f}s")

    print(f"  Best: {best:.6f} (ep {bep})")
    return {"cond": cond_name, "seed": seed, "params": np_,
            "best_vp": best, "best_ep": bep,
            "lam": effective_lam, "use_sigreg": cond_cfg["use_sigreg"],
            "desc": cond_cfg["desc"], "history": hist}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes",type=int,default=200)
    pa.add_argument("--epochs",type=int,default=50)
    pa.add_argument("--batch-size",type=int,default=128)
    pa.add_argument("--seeds",type=int,nargs="+",default=[42,123,777])
    pa.add_argument("--embed-dim",type=int,default=3)
    pa.add_argument("--output-dir",type=str,default="reviewer_results")
    pa.add_argument("--synthetic",action="store_true")
    args = pa.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    print(f"Seeds: {args.seeds}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Total runs: {len(CONDITIONS) * len(args.seeds)}")

    # Collect data once (same data for all conditions and seeds)
    print(f"\n{'='*60}\n  DATA COLLECTION\n{'='*60}")
    eps = (collect_synthetic_data if args.synthetic else collect_gym_data)(
        args.episodes, 300, 5, 42)  # data seed always 42 for consistency

    # Run all conditions
    all_results = {}
    for seed in args.seeds:
        for cond_name, cond_cfg in CONDITIONS.items():
            key = f"{seed}_{cond_name}"
            result = run_condition(
                cond_name, cond_cfg, eps, seed,
                dim=args.embed_dim, epochs=args.epochs,
                bs=args.batch_size, dev=dev)
            all_results[key] = result

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<25} {'Seed 42':>10} {'Seed 123':>10} {'Seed 777':>10} {'Mean':>10}")
    print("-"*70)

    for cond_name in CONDITIONS:
        vals = []
        parts = []
        for seed in args.seeds:
            key = f"{seed}_{cond_name}"
            v = all_results[key]["best_vp"]
            vals.append(v)
            parts.append(f"{v:>10.6f}")
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"{cond_name:<25} {' '.join(parts)} {mean:>10.6f} +/- {std:.6f}")

    # Key comparisons
    print(f"\n--- Key Comparisons ---")
    def mean_vp(cond):
        return np.mean([all_results[f"{s}_{cond}"]["best_vp"] for s in args.seeds])

    p = mean_vp("prescribed")
    f = mean_vp("free")
    f3 = mean_vp("free_3d")
    fns = mean_vp("free_no_sigreg")
    pns = mean_vp("prescribed_no_sigreg")

    print(f"  prescribed:             {p:.6f}")
    print(f"  free:                   {f:.6f}  (ratio vs prescribed: {f/p:.1f}x)")
    print(f"  free_3d (equal input):  {f3:.6f}  (ratio vs prescribed: {f3/p:.1f}x)")
    print(f"  free_no_sigreg:         {fns:.6f}  (ratio vs prescribed: {fns/p:.1f}x)")
    print(f"  prescribed_no_sigreg:   {pns:.6f}  (ratio vs prescribed: {pns/p:.1f}x)")

    print(f"\n--- Reviewer Questions ---")
    print(f"  #33 (equal input):  free_3d/prescribed = {f3/p:.2f}x")
    if f3/p > 2:
        print(f"       -> Free encoder with SAME input is {f3/p:.1f}x worse.")
        print(f"       -> Advantage is from FIXATION, not information access.")
    elif f3/p < 1.5:
        print(f"       -> Free encoder with same input is close to prescribed.")
        print(f"       -> Advantage may be partly from information access.")

    print(f"  #18-20 (SIGReg ablation):")
    print(f"       free WITH SIGReg:    {f:.6f}")
    print(f"       free WITHOUT SIGReg: {fns:.6f}")
    if fns < f:
        print(f"       -> SIGReg HURTS free by {f/fns:.1f}x (removing it helps)")
    else:
        print(f"       -> SIGReg HELPS free by {fns/f:.1f}x")
    print(f"       prescribed WITH SIGReg:    {p:.6f}")
    print(f"       prescribed WITHOUT SIGReg: {pns:.6f}")
    print(f"       -> SIGReg effect on prescribed: {abs(pns-p)/p*100:.1f}%")

    # Save
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    # Full results with histories
    with open(out/"full_results.json","w") as fh:
        json.dump(all_results, fh, indent=2)

    # Summary table (compact, no histories)
    summary = {}
    for key, r in all_results.items():
        summary[key] = {k:v for k,v in r.items() if k != "history"}
    with open(out/"summary.json","w") as fh:
        json.dump(summary, fh, indent=2)

    # Per-condition histories for plotting
    for cond_name in CONDITIONS:
        for seed in args.seeds:
            key = f"{seed}_{cond_name}"
            with open(out/f"history_{key}.json","w") as fh:
                json.dump(all_results[key]["history"], fh)

    print(f"\n  Results saved to {out}/")
    print(f"  Files: full_results.json, summary.json, history_*.json")


if __name__ == "__main__":
    main()
