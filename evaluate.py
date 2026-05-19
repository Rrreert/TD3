"""
Evaluation & Visualization Script
Supports both TD3 and ATL-TD3 models.
Plots trajectory and rudder angle figures matching Fig.9 / Fig.10 of the paper.

Usage:
  # Evaluate TD3
  python evaluate.py --algo td3 --model checkpoints/td3_best.pt

  # Evaluate ATL-TD3
  python evaluate.py --algo atl_td3 --model checkpoints/atl_td3_best.pt

  # Single case
  python evaluate.py --algo atl_td3 --model checkpoints/atl_td3_best.pt --case 11

  # Save figures
  python evaluate.py --algo atl_td3 --model checkpoints/atl_td3_best.pt --save-fig

  # Compare both algorithms side by side on a single case
  python evaluate.py --compare \
      --td3-model   checkpoints/td3_best.pt \
      --atltd3-model checkpoints/atl_td3_best.pt \
      --case 11
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from environment     import USVEnv
from td3_agent       import TD3
from atl_td3_agent   import ATLTD3, EpisodeWindow
from imazu_scenarios import IMAZU_CASES


# ─────────────────────────────────────────────
# Agent factory helpers
# ─────────────────────────────────────────────

SEQ_LEN = 8   # must match training


def load_td3(model_path, device):
    agent = TD3(state_dim=USVEnv.STATE_DIM, action_dim=USVEnv.ACTION_DIM,
                device=device)
    if model_path:
        agent.load(model_path)
    return agent


def load_atl_td3(model_path, device):
    agent = ATLTD3(state_dim=USVEnv.STATE_DIM, action_dim=USVEnv.ACTION_DIM,
                   seq_len=SEQ_LEN, lstm_hidden=128, num_heads=4,
                   hidden_dim=256, device=device)
    if model_path:
        agent.load(model_path)
    return agent


# ─────────────────────────────────────────────
# Run one episode and collect trajectory
# ─────────────────────────────────────────────

def run_case(agent, algo, case_id, max_steps=2000):
    """
    Execute one episode on the given Imazu case.
    Returns a result dict with trajectory data.
    """
    cfg = IMAZU_CASES[case_id]
    env = USVEnv(target_ships_config=cfg['targets'],
                 goal=cfg['goal'], os_init=cfg['os'],
                 dt=1.0, wind_wave=False)
    obs  = env.reset()
    done = False

    os_traj  = [env.eta[:2].copy()]
    ts_trajs = [[ts.pos] for ts in env.targets]
    deltas   = [0.0]
    info_log = {}

    if algo == 'atl_td3':
        win = EpisodeWindow(USVEnv.STATE_DIM, SEQ_LEN)
        win.reset(obs)

    while not done and len(os_traj) <= max_steps:
        if algo == 'atl_td3':
            action = agent.select_action(win.get())
        else:
            action = agent.select_action(obs)

        obs, rew, done, info = env.step(action)

        if algo == 'atl_td3':
            win.push(obs)

        os_traj.append(env.eta[:2].copy())
        for i, ts in enumerate(env.targets):
            ts_trajs[i].append(ts.pos)
        deltas.append(env.delta)
        info_log = info

    return {
        'case_id':     case_id,
        'algo':        algo,
        'desc':        cfg['desc'],
        'os_traj':     np.array(os_traj),
        'ts_trajs':    [np.array(t) for t in ts_trajs],
        'deltas':      np.array(deltas),
        'success':     info_log.get('arrived',   False),
        'collision':   info_log.get('collision', False),
        'steps':       len(os_traj) - 1,
        'goal':        np.array(cfg['goal']),
        'os_init':     cfg['os'],
        'targets_cfg': cfg['targets'],
    }


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

TS_COLORS = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a']
ALGO_COLORS = {
    'td3':     '#1d3557',
    'atl_td3': '#e76f51',
}
ALGO_LABELS = {
    'td3':     'TD3',
    'atl_td3': 'ATL-TD3',
}


def _draw_arrow(ax, x, y, psi, color, length=0.4):
    dx = length * np.cos(psi)
    dy = length * np.sin(psi)
    ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.2, mutation_scale=10))


def plot_case(ax, result, show_title=True):
    """Plot trajectory for one result on given axes."""
    os_traj = result['os_traj']
    goal    = result['goal']
    algo    = result['algo']
    color   = ALGO_COLORS.get(algo, '#333333')
    label   = ALGO_LABELS.get(algo, algo)

    # OS trajectory
    ax.plot(os_traj[:, 0], os_traj[:, 1],
            color=color, linewidth=1.8, label=label, zorder=5)
    ax.plot(os_traj[0, 0], os_traj[0, 1], 'o',
            color=color, markersize=6, zorder=6)

    # Goal
    ax.plot(goal[0], goal[1], '^', color='green',
            markersize=8, zorder=6, label='Goal')
    ax.plot(goal[0], goal[1], 'o', color='green',
            markersize=12, alpha=0.2, zorder=5)

    # TS trajectories
    for i, ts_traj in enumerate(result['ts_trajs']):
        c = TS_COLORS[i % len(TS_COLORS)]
        ax.plot(ts_traj[:, 0], ts_traj[:, 1],
                color=c, linewidth=1.2, linestyle='--',
                label=f'TS-{i+1:02d}', zorder=4)
        ax.plot(ts_traj[0, 0], ts_traj[0, 1], 's', color=c,
                markersize=5, zorder=5)
        ts_cfg = result['targets_cfg'][i]
        _draw_arrow(ax, ts_traj[0, 0], ts_traj[0, 1],
                    ts_cfg['psi'], c, length=0.3)

    # OS heading arrow
    _draw_arrow(ax, os_traj[0, 0], os_traj[0, 1],
                result['os_init']['psi'], color, length=0.4)

    # Status badge
    if result['success']:
        fc, txt = '#c8f7c5', 'SUCCESS'
    elif result['collision']:
        fc, txt = '#ffc9c9', 'COLLISION'
    else:
        fc, txt = '#fff3cd', 'TIMEOUT'

    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            fontsize=7, va='top', ha='left',
            bbox=dict(facecolor=fc, edgecolor='none', alpha=0.85, pad=2))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=6)
    ax.set_xlabel('X / (n mile)', fontsize=6)
    ax.set_ylabel('Y / (n mile)', fontsize=6)

    if show_title:
        ax.set_title(f"Case ({result['case_id']})", fontsize=7, pad=3)

    ax.legend(loc='upper right', fontsize=5, framealpha=0.7,
              handlelength=1.5, borderpad=0.4)


def plot_rudder(ax, result):
    t     = np.arange(len(result['deltas']))
    color = ALGO_COLORS.get(result['algo'], '#333333')
    ax.plot(t, result['deltas'], color=color, linewidth=0.8)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylim(-25, 25)
    ax.set_ylabel('δ (°)', fontsize=6)
    ax.set_xlabel('Time / s', fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_title(f"Case ({result['case_id']})", fontsize=7, pad=2)
    ax.grid(True, alpha=0.2)


def plot_all_trajectories(results, algo_label='', save_path=None):
    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    axes = axes.flatten()
    for i, cid in enumerate(sorted(results)):
        plot_case(axes[i], results[cid])
    fig.suptitle(f'Navigation Trajectories – 20 Imazu Cases ({algo_label})',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


def plot_all_rudders(results, algo_label='', save_path=None):
    fig, axes = plt.subplots(4, 5, figsize=(18, 8))
    axes = axes.flatten()
    for i, cid in enumerate(sorted(results)):
        plot_rudder(axes[i], results[cid])
    fig.suptitle(f'Rudder Angles – 20 Imazu Cases ({algo_label})',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


def plot_comparison(res_td3, res_atl, case_id, save_path=None):
    """
    Side-by-side trajectory + rudder comparison for a single case.
    Left col = TD3, right col = ATL-TD3.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_case(axes[0, 0],  res_td3)
    plot_case(axes[0, 1],  res_atl)
    plot_rudder(axes[1, 0], res_td3)
    plot_rudder(axes[1, 1], res_atl)
    axes[0, 0].set_title(f"TD3 – Case {case_id}", fontsize=9)
    axes[0, 1].set_title(f"ATL-TD3 – Case {case_id}", fontsize=9)
    fig.suptitle(
        f"Case {case_id}: {IMAZU_CASES[case_id]['desc']}", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────

def print_summary(results, label=''):
    n_s = sum(1 for r in results.values() if r['success'])
    n_c = sum(1 for r in results.values() if r['collision'])
    n_t = sum(1 for r in results.values()
              if not r['success'] and not r['collision'])
    n   = len(results)
    tag = f"[{label}] " if label else ''
    print(f"\n{'='*62}")
    print(f"  {tag}Imazu Test Results")
    print(f"{'='*62}")
    print(f"  {'Case':6s} {'Result':14s} {'Steps':6s}  Description")
    print(f"  {'-'*58}")
    for cid in sorted(results):
        r = results[cid]
        if r['success']:
            res = 'SUCCESS ✓'
        elif r['collision']:
            res = 'COLLISION ✗'
        else:
            res = 'TIMEOUT ~'
        print(f"  {cid:6d} {res:14s} {r['steps']:6d}  {IMAZU_CASES[cid]['desc']}")
    print(f"  {'-'*58}")
    print(f"  Success:   {n_s}/{n}   Collision: {n_c}/{n}   Timeout: {n_t}/{n}")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Comparison mode ──
    if args.compare:
        if not args.case:
            parser.error("--compare requires --case <id>")
        td3_agent  = load_td3(args.td3_model,    device)
        atl_agent  = load_atl_td3(args.atltd3_model, device)
        res_td3 = run_case(td3_agent,  'td3',     args.case)
        res_atl = run_case(atl_agent,  'atl_td3', args.case)
        save = f"compare_case{args.case:02d}.png" if args.save_fig else None
        plot_comparison(res_td3, res_atl, args.case, save_path=save)
        return

    # ── Single-algo mode ──
    if args.algo == 'td3':
        agent      = load_td3(args.model, device)
        algo_label = 'TD3'
    else:
        agent      = load_atl_td3(args.model, device)
        algo_label = 'ATL-TD3'

    case_ids = [args.case] if args.case else list(range(1, 21))

    results = {}
    for cid in case_ids:
        print(f"  Running case {cid:2d} [{algo_label}] ...", end=' ', flush=True)
        r = run_case(agent, args.algo, cid)
        results[cid] = r
        print('SUCCESS' if r['success'] else
              ('COLLISION' if r['collision'] else 'TIMEOUT'))

    print_summary(results, algo_label)

    if args.case:
        # Detailed single-case plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_case(ax1, results[args.case])
        plot_rudder(ax2, results[args.case])
        plt.suptitle(f"{algo_label} – Case {args.case}: "
                     f"{IMAZU_CASES[args.case]['desc']}")
        plt.tight_layout()
        if args.save_fig:
            p = f"{args.algo}_case{args.case:02d}.png"
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f"Saved → {p}")
        plt.show()
    else:
        suffix     = args.algo
        traj_path  = f"{suffix}_trajectories.png" if args.save_fig else None
        ruddr_path = f"{suffix}_rudders.png"      if args.save_fig else None
        plot_all_trajectories(results, algo_label, traj_path)
        plot_all_rudders(results, algo_label, ruddr_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate TD3 / ATL-TD3 on Imazu scenarios')

    # Single-algo mode
    parser.add_argument('--algo',   type=str, default='td3',
                        choices=['td3', 'atl_td3'],
                        help='Algorithm to evaluate')
    parser.add_argument('--model',  type=str, default=None,
                        help='Checkpoint path')
    parser.add_argument('--case',   type=int, default=None,
                        help='Run only this case (1-20)')
    parser.add_argument('--save-fig', action='store_true')

    # Comparison mode
    parser.add_argument('--compare',       action='store_true',
                        help='Compare TD3 vs ATL-TD3 side by side (requires --case)')
    parser.add_argument('--td3-model',     type=str, default=None)
    parser.add_argument('--atltd3-model',  type=str, default=None)

    args = parser.parse_args()
    main(args)
