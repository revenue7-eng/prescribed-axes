"""
Random Axes Ablation: Fixation vs. Meaningful Coordinates
Experiment 5 for prescribed-axes repository.

Standalone script for local execution (Windows/Linux, CPU).
Run: python random_axes_ablation.py

Results saved to: ./exp5_random_axes/

Resume: re-run the script — completed runs are skipped automatically.

Author: Andrey Lazarev | Independent Researcher | April 2026
Repository: https://github.com/revenue7-eng/prescribed-axes
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIGURATION
# ============================================================

# Data mode: True = gym-pusht (realistic, matches paper), False = synthetic (pipeline test)
USE_GYM_PUSHT = False

N_EPISODES = 500
N_STEPS = 200
DATA_SEED = 0

TRAINING_SEEDS = [42, 123, 777]
ROTATION_SEEDS = [0, 1, 2]
N_EPOCHS = 50
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-3
H = 5  # history length
D = 3  # latent dimension

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp5_random_axes')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATA GENERATION
# ============================================================

def generate_gym_pusht_dataset(n_episodes, n_steps, seed):
    """gym-pusht: realistic Box2D physics matching the paper."""
    import gymnasium as gym
    env = gym.make('gym_pusht/PushT-v0', obs_type='state', render_mode=None)
    rng = np.random.default_rng(seed)
    all_states, all_actions = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        states, actions = [obs.copy()], []

        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            states.append(obs.copy())
            actions.append(action.copy())
            if terminated or truncated:
                break

        all_states.append(np.array(states))
        all_actions.append(np.array(actions))

        if (ep + 1) % 100 == 0:
            print(f'  Generated {ep+1}/{n_episodes} episodes')

    env.close()
    return all_states, all_actions


def generate_synthetic_dataset(n_episodes, n_steps, seed):
    """Simplified physics — pipeline verification only."""
    rng = np.random.default_rng(seed)
    all_states, all_actions = [], []

    for i in range(n_episodes):
        agent = rng.uniform(50, 462, size=2)
        block = rng.uniform(100, 412, size=2)
        angle = rng.uniform(0, 2 * np.pi)
        states, actions = [], []

        for _ in range(n_steps):
            states.append(np.array([agent[0], agent[1], block[0], block[1], angle]))
            action = rng.normal(0, 30, size=2)
            actions.append(action)
            agent = np.clip(agent + action, 0, 512)
            dist = np.linalg.norm(agent - block)
            if dist < 60:
                push_dir = (block - agent) / (np.linalg.norm(block - agent) + 1e-8)
                block = np.clip(block + push_dir * max(0, 1 - dist / 60) * 15, 0, 512)
                offset = agent - block
                angle = (angle + (offset[0] * action[1] - offset[1] * action[0]) * 0.0003) % (2 * np.pi)

        states.append(np.array([agent[0], agent[1], block[0], block[1], angle]))
        all_states.append(np.array(states))
        all_actions.append(np.array(actions))

        if (i + 1) % 100 == 0:
            print(f'  Generated {i+1}/{n_episodes} episodes')

    return all_states, all_actions


class PushTSequenceDataset(Dataset):
    def __init__(self, episodes_states, episodes_actions, H=5):
        self.samples = []
        for states, actions in zip(episodes_states, episodes_actions):
            for t in range(H, len(actions)):
                self.samples.append((states[t-H:t], actions[t-H:t], states[t]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist_s, hist_a, next_s = self.samples[idx]
        return (
            torch.tensor(hist_s, dtype=torch.float32),
            torch.tensor(hist_a, dtype=torch.float32),
            torch.tensor(next_s, dtype=torch.float32)
        )


# ============================================================
# MODEL
# ============================================================

def normalize_block_coords(state_5d):
    block = state_5d[..., 2:5].clone()
    block[..., 0] /= 512.0
    block[..., 1] /= 512.0
    block[..., 2] /= (2 * np.pi)
    return block


def generate_random_orthogonal(d=3, seed=0):
    """Random orthogonal matrix via QR (Stewart 1980, Mezzadri 2007)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(Z)
    D = np.diag(np.sign(np.diag(R)))
    Q = Q @ D
    return torch.tensor(Q, dtype=torch.float32)


class ActionEncoder(nn.Module):
    def __init__(self, d=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, d))

    def forward(self, a):
        return self.net(a)


class FreeEncoder(nn.Module):
    def __init__(self, input_dim=5, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, H=5, d=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(H * 2 * d, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, d)
        )

    def forward(self, x):
        return self.net(x)


class LeWMModel(nn.Module):
    def __init__(self, mode='prescribed', H=5, d=3, rotation_matrix=None):
        super().__init__()
        self.mode = mode
        self.H = H
        self.d = d

        if mode == 'prescribed':
            self.encoder = None
        elif mode == 'random_fixed':
            assert rotation_matrix is not None
            self.register_buffer('R', rotation_matrix)
            self.encoder = None
        elif mode == 'free_3d':
            self.encoder = FreeEncoder(input_dim=3, output_dim=d)
        elif mode == 'free_5d':
            self.encoder = FreeEncoder(input_dim=5, output_dim=d)

        self.action_encoder = ActionEncoder(d=d)
        self.predictor = Predictor(H=H, d=d)

    def encode(self, state_5d):
        if self.mode == 'prescribed':
            return normalize_block_coords(state_5d)
        elif self.mode == 'random_fixed':
            return normalize_block_coords(state_5d) @ self.R.T
        elif self.mode == 'free_3d':
            return self.encoder(normalize_block_coords(state_5d))
        elif self.mode == 'free_5d':
            s = state_5d.clone()
            s[..., 0] /= 512.0
            s[..., 1] /= 512.0
            s[..., 2] /= 512.0
            s[..., 3] /= 512.0
            s[..., 4] /= (2 * np.pi)
            return self.encoder(s)

    def forward(self, hist_states, hist_actions, target_state):
        B, H, _ = hist_states.shape
        encoded_states = []
        encoded_actions = []
        for t in range(H):
            encoded_states.append(self.encode(hist_states[:, t, :]))
            encoded_actions.append(self.action_encoder(hist_actions[:, t, :]))

        interleaved = []
        for t in range(H):
            interleaved.append(encoded_states[t])
            interleaved.append(encoded_actions[t])

        pred = self.predictor(torch.cat(interleaved, dim=-1))
        target = self.encode(target_state)
        if self.mode in ('prescribed', 'random_fixed'):
            target = target.detach()

        loss = nn.functional.mse_loss(pred, target)
        return loss, pred, target


# ============================================================
# TRAINING
# ============================================================

def train_one_run(model, train_loader, val_loader, n_epochs, lr, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': -1
    }

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for hist_s, hist_a, next_s in train_loader:
            optimizer.zero_grad()
            loss, _, _ = model(hist_s, hist_a, next_s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for hist_s, hist_a, next_s in val_loader:
                loss, _, _ = model(hist_s, hist_a, next_s)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch

        if (epoch + 1) % 10 == 0:
            print(f'    epoch {epoch+1:3d}/{n_epochs}: train={train_loss:.6f} val={val_loss:.6f}')

    return history


# ============================================================
# MAIN
# ============================================================

def main():
    print('=' * 60)
    print('Random Axes Ablation: Fixation vs. Meaningful Coordinates')
    print('=' * 60)

    # --- Data ---
    data_tag = 'gym' if USE_GYM_PUSHT else 'synthetic'
    data_cache = os.path.join(OUTPUT_DIR, f'pusht_data_{data_tag}_{N_EPISODES}ep.npz')

    if os.path.exists(data_cache):
        print(f'\nLoading cached data: {data_cache}')
        loaded = np.load(data_cache, allow_pickle=True)
        all_states = list(loaded['states'])
        all_actions = list(loaded['actions'])
    else:
        if USE_GYM_PUSHT:
            print(f'\nGenerating {N_EPISODES} episodes with gym-pusht...')
            all_states, all_actions = generate_gym_pusht_dataset(N_EPISODES, N_STEPS, DATA_SEED)
        else:
            print(f'\nGenerating {N_EPISODES} episodes with synthetic physics...')
            print('  WARNING: synthetic mode is for pipeline testing only.')
            print('  Set USE_GYM_PUSHT=True for results matching the paper.')
            all_states, all_actions = generate_synthetic_dataset(N_EPISODES, N_STEPS, DATA_SEED)

        np.savez(data_cache,
                 states=np.array(all_states, dtype=object),
                 actions=np.array(all_actions, dtype=object))
        print(f'Data cached: {data_cache}')

    n_train = int(len(all_states) * 0.8)
    train_dataset = PushTSequenceDataset(all_states[:n_train], all_actions[:n_train], H=H)
    val_dataset = PushTSequenceDataset(all_states[n_train:], all_actions[n_train:], H=H)

    print(f'Data mode: {data_tag}')
    print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Rotation matrices ---
    rotation_matrices = {}
    print(f'\nRotation matrices:')
    for rs in ROTATION_SEEDS:
        R = generate_random_orthogonal(D, seed=rs)
        rotation_matrices[rs] = R
        print(f'  seed={rs}: det={torch.det(R).item():.4f}, '
              f'orthogonality_err={(R @ R.T - torch.eye(D)).abs().max().item():.2e}')

    # --- Define runs ---
    all_runs = []
    for ts in TRAINING_SEEDS:
        all_runs.append(('prescribed', ts, None))
        all_runs.append(('free_3d', ts, None))
        all_runs.append(('free_5d', ts, None))
        for rs in ROTATION_SEEDS:
            all_runs.append(('random_fixed', ts, rs))

    print(f'\nTotal runs: {len(all_runs)}')

    # --- Load existing results (resume) ---
    results_path = os.path.join(OUTPUT_DIR, 'all_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        print(f'Loaded {len(all_results)} completed runs from {results_path}')
    else:
        all_results = {}

    # --- Run experiments ---
    total_start = time.time()

    for i, (mode, ts, rs) in enumerate(all_runs):
        tag = f'{mode}_ts{ts}' + (f'_rs{rs}' if rs is not None else '')

        if tag in all_results:
            v = all_results[tag]['best_val_loss']
            print(f'[{i+1:2d}/{len(all_runs)}] {tag:30s} — done (val={v:.6f}), skip')
            continue

        print(f'[{i+1:2d}/{len(all_runs)}] {tag:30s} — training...')
        run_start = time.time()

        torch.manual_seed(ts)
        np.random.seed(ts)

        R = rotation_matrices[rs] if rs is not None else None
        model = LeWMModel(mode=mode, H=H, d=D, rotation_matrix=R)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'    params: {n_params:,}')

        history = train_one_run(model, train_loader, val_loader,
                                N_EPOCHS, LR, WEIGHT_DECAY)

        run_time = time.time() - run_start

        all_results[tag] = {
            'mode': mode,
            'training_seed': ts,
            'rotation_seed': rs,
            'best_val_loss': history['best_val_loss'],
            'best_epoch': history['best_epoch'],
            'final_val_loss': history['val_loss'][-1],
            'final_train_loss': history['train_loss'][-1],
            'train_loss_curve': history['train_loss'],
            'val_loss_curve': history['val_loss'],
            'run_time_seconds': run_time
        }

        # Autosave after each run
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f'    best_val={history["best_val_loss"]:.6f} (ep {history["best_epoch"]}), '
              f'time={run_time:.1f}s')

    total_time = time.time() - total_start
    print(f'\nAll runs complete. Total: {total_time:.1f}s ({total_time/60:.1f} min)')

    # --- Results table ---
    print('\n' + '=' * 75)
    print(f'{"Condition":20s} | {"Mean":>12s} | {"Std":>12s} | {"N":>3s} | {"vs Prescribed":>14s}')
    print('-' * 75)

    from collections import defaultdict
    cond_results = defaultdict(list)
    for tag, res in all_results.items():
        mode = res['mode']
        if mode == 'random_fixed':
            cond_results['random_fixed'].append(res['best_val_loss'])
        else:
            cond_results[mode].append(res['best_val_loss'])

    p_mean = np.mean(cond_results['prescribed'])

    for cond in ['prescribed', 'random_fixed', 'free_3d', 'free_5d']:
        vals = cond_results[cond]
        mean = np.mean(vals)
        std = np.std(vals)
        ratio = mean / p_mean if p_mean > 0 else float('inf')
        ratio_str = f'{ratio:.1f}x' if cond != 'prescribed' else '—'
        print(f'{cond:20s} | {mean:12.6f} | {std:12.6f} | {len(vals):3d} | {ratio_str:>14s}')

    print('=' * 75)

    # Per-rotation-seed
    print('\n--- Random Fixed: per rotation seed ---')
    for rs in ROTATION_SEEDS:
        vals = [all_results[f'random_fixed_ts{ts}_rs{rs}']['best_val_loss']
                for ts in TRAINING_SEEDS
                if f'random_fixed_ts{ts}_rs{rs}' in all_results]
        if vals:
            print(f'  R(seed={rs}): {np.mean(vals):.6f} +/- {np.std(vals):.6f} '
                  f'({np.mean(vals)/p_mean:.1f}x vs prescribed)')

    # Per-training-seed
    print(f'\n{"Seed":>6s} | {"Prescribed":>12s} | {"Random(mean)":>14s} | {"Free 3D":>12s} | {"Free 5D":>12s}')
    print('-' * 70)
    for ts in TRAINING_SEEDS:
        p = all_results.get(f'prescribed_ts{ts}', {}).get('best_val_loss', float('nan'))
        f3 = all_results.get(f'free_3d_ts{ts}', {}).get('best_val_loss', float('nan'))
        f5 = all_results.get(f'free_5d_ts{ts}', {}).get('best_val_loss', float('nan'))
        r_vals = [all_results[f'random_fixed_ts{ts}_rs{rs}']['best_val_loss']
                  for rs in ROTATION_SEEDS
                  if f'random_fixed_ts{ts}_rs{rs}' in all_results]
        r_mean = np.mean(r_vals) if r_vals else float('nan')
        print(f'{ts:6d} | {p:12.6f} | {r_mean:14.6f} | {f3:12.6f} | {f5:12.6f}')

    # --- Interpretation ---
    r_mean = np.mean(cond_results['random_fixed'])
    f3_mean = np.mean(cond_results['free_3d'])
    r_vs_p = r_mean / p_mean

    print('\n' + '=' * 60)
    print('INTERPRETATION')
    print('=' * 60)
    print(f'Prescribed:   {p_mean:.6f}')
    print(f'Random fixed: {r_mean:.6f} ({r_vs_p:.1f}x)')
    print(f'Free 3D:      {f3_mean:.6f} ({f3_mean/p_mean:.1f}x)')

    if r_vs_p < 2.0:
        print('\n-> random_fixed ~ prescribed')
        print('-> Fixation is the dominant factor. Semantic alignment is secondary.')
    elif r_vs_p > 0.5 * (f3_mean / p_mean):
        fixation_pct = (f3_mean - r_mean) / (f3_mean - p_mean) * 100 if f3_mean != p_mean else 0
        semantic_pct = (r_mean - p_mean) / (f3_mean - p_mean) * 100 if f3_mean != p_mean else 0
        print(f'\n-> prescribed > random_fixed > free_3d')
        print(f'-> Fixation: ~{fixation_pct:.0f}%, Semantics: ~{semantic_pct:.0f}%')
    else:
        print('\n-> random_fixed ~ free_3d')
        print('-> Meaningful coordinates are critical. Fixation alone is not enough.')

    # --- Save figure ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes_arr = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Training curves (seed 42)
        ax = axes_arr[0]
        ts = 42
        colors = {'prescribed': '#2ecc71', 'random_fixed': '#3498db',
                  'free_3d': '#e67e22', 'free_5d': '#e74c3c'}

        for mode in ['prescribed', 'free_3d', 'free_5d']:
            tag = f'{mode}_ts{ts}'
            if tag in all_results:
                ax.semilogy(all_results[tag]['val_loss_curve'],
                            color=colors[mode], label=mode, linewidth=2)

        for j, rs in enumerate(ROTATION_SEEDS):
            tag = f'random_fixed_ts{ts}_rs{rs}'
            if tag in all_results:
                label = 'random_fixed' if j == 0 else None
                ax.semilogy(all_results[tag]['val_loss_curve'],
                            color=colors['random_fixed'], alpha=0.5, linewidth=1.5, label=label)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Loss (log)')
        ax.set_title(f'Training Curves (seed={ts})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 2: Bar chart
        ax = axes_arr[1]
        conds = ['prescribed', 'random_fixed', 'free_3d', 'free_5d']
        means = [np.mean(cond_results[c]) for c in conds]
        stds = [np.std(cond_results[c]) for c in conds]
        bar_colors = [colors[c] for c in conds]

        bars = ax.bar(conds, means, yerr=stds, color=bar_colors, capsize=5,
                      edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Best Val Loss (MSE)')
        ax.set_title('Mean Best Val Loss')
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{m:.5f}', ha='center', va='bottom', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Ratio
        ax = axes_arr[2]
        ratios = [m / p_mean for m in means]
        bars = ax.bar(conds, ratios, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Ratio (x vs prescribed)')
        ax.set_title('Gap vs Prescribed')
        for bar, r in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{r:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, 'random_axes_results.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f'\nFigure saved: {fig_path}')
        plt.close()

    except Exception as e:
        print(f'\nCould not generate figure: {e}')

    print(f'\nAll results: {results_path}')
    print('Done.')


if __name__ == '__main__':
    main()
