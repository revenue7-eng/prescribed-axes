#!/usr/bin/env python3
"""
LeWM Prescribed vs Free Space — Push-T Experiment
===================================================
Prescribed axes (block_x, block_y, block_angle) vs free (learned) embedding
for next-state prediction. Minimal LeWM architecture.

State = [agent_x, agent_y, block_x, block_y, block_angle]
Prescribed = state[2:5] normalized

Run:
  pip install gym-pusht torch numpy
  python lewm_pusht_experiment.py
  # Or without gym-pusht:
  python lewm_pusht_experiment.py --synthetic

Author: Andrey Lazarev
"""

import os, time, json, argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# --- SIGReg (from LeWM module.py) ---
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
    def __init__(self):
        super().__init__()
        self.register_buffer("sc", torch.tensor([1/512,1/512,1/(2*np.pi)]))
    def forward(self, s): return s[...,2:5]*self.sc

class FreeEnc(nn.Module):
    def __init__(self, di=5, do=3, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,do))
    def forward(self, s): return self.net(s)

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
    def __init__(self, enc, aenc, pred, sig, H=3):
        super().__init__()
        self.enc,self.aenc,self.pred,self.sig,self.H = enc,aenc,pred,sig,H
    def forward(self, s, a):
        H = self.H
        emb = self.enc(s)
        ctx, tgt = emb[:,:H], emb[:,H]
        ae = self.aenc(a[:,:H])
        p = self.pred(ctx, ae)
        return {"pl": F.mse_loss(p,tgt.detach()),
                "sl": self.sig(emb.transpose(0,1)),
                "p": p.detach(), "t": tgt.detach()}


# --- Training ---
@dataclass
class Cfg:
    n_ep:int=200; max_steps:int=300; fs:int=5; seed:int=42
    dim:int=3; hid:int=128; H:int=3
    epochs:int=50; bs:int=128; lr:float=3e-4; wd:float=1e-3
    lam:float=0.09; split:float=0.9; out:str="lewm_results"

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

def run(mode, eps, cfg, dev):
    print(f"\n{'='*50}\n  {mode.upper()}\n{'='*50}")
    ds = SeqDS(eps, cfg.H)
    nt = int(len(ds)*cfg.split); nv = len(ds)-nt
    tr,va = random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(cfg.seed))
    tl = DataLoader(tr,batch_size=cfg.bs,shuffle=True,drop_last=True)
    vl = DataLoader(va,batch_size=cfg.bs)
    print(f"  train={nt} val={nv}")

    enc = PrescEnc() if mode=="prescribed" else FreeEnc(5,cfg.dim,64)
    mdl = Model(enc, ActEnc(2,cfg.dim,32), Pred(cfg.dim,cfg.H,cfg.hid),
                SIGReg(17,512), cfg.H).to(dev)
    np_ = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  params: {np_:,}")

    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                            lr=cfg.lr,weight_decay=cfg.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.epochs)

    best,bep,hist = float("inf"),0,[]
    for ep in range(1,cfg.epochs+1):
        t0 = time.time()
        tp,ts = train_ep(mdl,tl,opt,cfg.lam,dev)
        vp,vs,axes = val_ep(mdl,vl,dev)
        sch.step()
        if vp<best: best,bep = vp,ep
        hist.append({"ep":ep,"tp":tp,"ts":ts,"vp":vp,"vs":vs,"ax":axes})
        if ep%10==0 or ep==1:
            ax = ", ".join(f"{v:.6f}" for v in axes)
            print(f"  ep {ep:3d} | tr {tp:.6f} | val {vp:.6f} | sig {vs:.4f} | [{ax}] | {time.time()-t0:.1f}s")
    print(f"  Best: {best:.6f} (ep {bep})")
    return {"mode":mode,"params":np_,"best":best,"bep":bep,"hist":hist}

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes",type=int,default=200)
    pa.add_argument("--epochs",type=int,default=50)
    pa.add_argument("--batch-size",type=int,default=128)
    pa.add_argument("--seed",type=int,default=42)
    pa.add_argument("--embed-dim",type=int,default=3)
    pa.add_argument("--output-dir",type=str,default="lewm_results")
    pa.add_argument("--synthetic",action="store_true")
    args = pa.parse_args()

    cfg = Cfg(n_ep=args.episodes,epochs=args.epochs,bs=args.batch_size,
              seed=args.seed,dim=args.embed_dim,out=args.output_dir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}\n{json.dumps(asdict(cfg),indent=2)}")

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    print(f"\n{'='*50}\n  DATA\n{'='*50}")
    eps = (collect_synthetic_data if args.synthetic else collect_gym_data)(
        cfg.n_ep,cfg.max_steps,cfg.fs,cfg.seed)

    res = {}
    for mode in ["prescribed","free"]:
        torch.manual_seed(cfg.seed)
        res[mode] = run(mode, eps, cfg, dev)

    print(f"\n{'='*50}\n  RESULTS\n{'='*50}")
    p,f = res["prescribed"]["best"], res["free"]["best"]
    d = (f-p)/p*100 if p>0 else 0
    print(f"  Prescribed: {p:.6f}  ({res['prescribed']['params']:,} params)")
    print(f"  Free:       {f:.6f}  ({res['free']['params']:,} params)")
    print(f"  Delta: {d:+.1f}% ({'prescribed лучше' if d>0 else 'free лучше'})")

    out = Path(cfg.out); out.mkdir(parents=True,exist_ok=True)
    with open(out/"results.json","w") as fh:
        json.dump({"cfg":asdict(cfg),
                   "prescribed":{"vp":p,"ep":res["prescribed"]["bep"],"params":res["prescribed"]["params"]},
                   "free":{"vp":f,"ep":res["free"]["bep"],"params":res["free"]["params"]},
                   "delta_pct":d}, fh, indent=2)
    for m in res:
        with open(out/f"history_{m}.json","w") as fh:
            json.dump(res[m]["hist"], fh)
    print(f"  -> {out}/")

if __name__ == "__main__":
    main()
