"""
Evaluation & Visualization Script
Runs a trained TD3 model on all 20 Imazu cases and plots trajectories
matching the style of Fig.9 in the paper.

Usage:
  python evaluate.py --model checkpoints/best_model.pt
  python evaluate.py --model checkpoints/best_model.pt --case 1
  python evaluate.py --model checkpoints/best_model.pt --save-fig
"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

import torch
from environment     import USVEnv
from td3_agent       import TD3
from imazu_scenarios import IMAZU_CASES, OS_GOAL


# ─────────────────────────────────────────────
# Run one episode and collect trajectory
# ─────────────────────────────────────────────

def run_case(agent, case_id, max_steps=2000):
    cfg = IMAZU_CASES[case_id]
    env = USVEnv(
        target_ships_config=cfg['targets'],
        goal=cfg['goal'],
        os_init=cfg['os'],
        dt=1.0,
        wind_wave=False
    )
    obs  = env.reset()
    done = False

    os_traj   = [env.eta[:2].copy()]
    ts_trajs  = [[ts.pos] for ts in env.targets]
    deltas    = [0.0]
    info_log  = {}

    while not done:
        action        = agent.select_action(obs)
        obs, rew, done, info = env.step(action)

        os_traj.append(env.eta[:2].copy())
        for i, ts in enumerate(env.targets):
            ts_trajs[i].append(ts.pos)
        deltas.append(env.delta)

        if len(os_traj) >= max_steps:
            break

        info_log = info

    result = {
        'case_id':   case_id,
        'desc':      cfg['desc'],
        'os_traj':   np.array(os_traj),
        'ts_trajs':  [np.array(t) for t in ts_trajs],
        'deltas':    np.array(deltas),
        'success':   info_log.get('arrived',   False),
        'collision': info_log.get('collision', False),
        'steps':     len(os_traj) - 1,
        'goal':      np.array(cfg['goal']),
        'os_init':   cfg['os'],
        'targets_cfg': cfg['targets'],
    }
    return result


# ─────────────────────────────────────────────
# Plot single case trajectory
# ─────────────────────────────────────────────

TS_COLORS = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a']
OS_COLOR  = '#1d3557'


def plot_case(ax, result, show_title=True):
    """Plot trajectory for one Imazu case on given axes."""
    os_traj = result['os_traj']
    goal    = result['goal']
    os_init = result['os_init']

    # OS trajectory
    ax.plot(os_traj[:, 0], os_traj[:, 1],
            color=OS_COLOR, linewidth=1.8, label='USV', zorder=5)

    # OS start marker
    ax.plot(os_traj[0, 0], os_traj[0, 1], 'o',
            color=OS_COLOR, markersize=6, zorder=6)

    # Goal marker
    ax.plot(goal[0], goal[1], '^',
            color='green', markersize=8, zorder=6, label='Goal')
    ax.plot(goal[0], goal[1], 'o',
            color='green', markersize=12, alpha=0.2, zorder=5)

    # Target ship trajectories
    for i, ts_traj in enumerate(result['ts_trajs']):
        color = TS_COLORS[i % len(TS_COLORS)]
        ax.plot(ts_traj[:, 0], ts_traj[:, 1],
                color=color, linewidth=1.2, linestyle='--',
                label=f'TS-{i+1:02d}', zorder=4)
        # TS start marker
        ax.plot(ts_traj[0, 0], ts_traj[0, 1], 's',
                color=color, markersize=5, zorder=5)
        # TS heading arrow
        cfg_ts = result['targets_cfg'][i]
        _draw_arrow(ax, ts_traj[0, 0], ts_traj[0, 1],
                    cfg_ts['psi'], color, length=0.3)

    # OS heading arrow
    _draw_arrow(ax, os_traj[0, 0], os_traj[0, 1],
                os_init['psi'], OS_COLOR, length=0.4)

    # Result text
    if result['success']:
        status_str = 'SUCCESS'
        fc = '#c8f7c5'
    elif result['collision']:
        status_str = 'COLLISION'
        fc = '#ffc9c9'
    else:
        status_str = 'TIMEOUT'
        fc = '#fff3cd'

    ax.text(0.03, 0.97, status_str, transform=ax.transAxes,
            fontsize=7, va='top', ha='left',
            bbox=dict(facecolor=fc, edgecolor='none', alpha=0.85, pad=2))

    # Axes
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


def _draw_arrow(ax, x, y, psi, color, length=0.4):
    """Draw a heading arrow in world frame."""
    dx = length * np.cos(psi)
    dy = length * np.sin(psi)
    ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.2, mutation_scale=10))


# ─────────────────────────────────────────────
# Plot rudder angle history
# ─────────────────────────────────────────────

def plot_rudder(ax, result):
    t = np.arange(len(result['deltas']))
    ax.plot(t, result['deltas'], color=OS_COLOR, linewidth=0.8)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylim(-25, 25)
    ax.set_ylabel('δ (°)', fontsize=6)
    ax.set_xlabel('Time / s', fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_title(f"Case ({result['case_id']})", fontsize=7, pad=2)
    ax.grid(True, alpha=0.2)


# ─────────────────────────────────────────────
# Plot all 20 cases (4 rows × 5 cols)
# ─────────────────────────────────────────────

def plot_all_trajectories(results, save_path=None):
    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    axes = axes.flatten()

    for i, cid in enumerate(sorted(results.keys())):
        plot_case(axes[i], results[cid])

    fig.suptitle('Navigation Trajectories – 20 Imazu Cases (TD3)',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory figure → {save_path}")
    plt.show()


def plot_all_rudders(results, save_path=None):
    fig, axes = plt.subplots(4, 5, figsize=(18, 8))
    axes = axes.flatten()

    for i, cid in enumerate(sorted(results.keys())):
        plot_rudder(axes[i], results[cid])

    fig.suptitle('Rudder Angles – 20 Imazu Cases (TD3)',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved rudder figure → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────

def print_summary(results):
    n_success   = sum(1 for r in results.values() if r['success'])
    n_collision = sum(1 for r in results.values() if r['collision'])
    n_timeout   = sum(1 for r in results.values() if r.get('timeout', False))
    n_total     = len(results)

    print(f"\n{'='*60}")
    print(f"  Imazu Test Results – TD3 Agent")
    print(f"{'='*60}")
    print(f"  {'Case':6s} {'Result':12s} {'Steps':6s}  Description")
    print(f"  {'-'*56}")
    for cid in sorted(results):
        r = results[cid]
        if r['success']:
            res = 'SUCCESS ✓'
        elif r['collision']:
            res = 'COLLISION ✗'
        else:
            res = 'TIMEOUT ~'
        desc = IMAZU_CASES[cid]['desc']
        print(f"  {cid:6d} {res:12s} {r['steps']:6d}  {desc}")
    print(f"  {'-'*56}")
    print(f"  Success:   {n_success:2d}/{n_total}")
    print(f"  Collision: {n_collision:2d}/{n_total}")
    print(f"  Timeout:   {n_timeout:2d}/{n_total}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from environment import USVEnv
    agent = TD3(
        state_dim  = USVEnv.STATE_DIM,
        action_dim = USVEnv.ACTION_DIM,
        device     = device
    )

    if args.model:
        agent.load(args.model)
    else:
        print("WARNING: No model specified – running with random policy.")

    # Select cases
    if args.case:
        case_ids = [args.case]
    else:
        case_ids = list(range(1, 21))

    # Run
    results = {}
    for cid in case_ids:
        print(f"  Running case {cid:2d}...", end=' ', flush=True)
        r = run_case(agent, cid)
        results[cid] = r
        status = 'SUCCESS' if r['success'] else \
                 ('COLLISION' if r['collision'] else 'TIMEOUT')
        print(status)

    # Summary
    print_summary(results)

    # Plots
    if args.case:
        # Single case: detailed plot
        fig, (ax_traj, ax_rud) = plt.subplots(1, 2, figsize=(12, 5))
        plot_case(ax_traj, results[args.case])
        plot_rudder(ax_rud, results[args.case])
        plt.suptitle(f"Case {args.case}: {IMAZU_CASES[args.case]['desc']}")
        plt.tight_layout()
        if args.save_fig:
            path = f"case_{args.case:02d}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved → {path}")
        plt.show()
    else:
        # All 20 cases
        traj_path = 'imazu_trajectories.png' if args.save_fig else None
        rudd_path = 'imazu_rudders.png'      if args.save_fig else None
        plot_all_trajectories(results, save_path=traj_path)
        plot_all_rudders(results,      save_path=rudd_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate TD3 on Imazu scenarios')
    parser.add_argument('--model',    type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--case',     type=int, default=None,
                        help='Run only this case (1-20)')
    parser.add_argument('--save-fig', action='store_true',
                        help='Save trajectory/rudder plots to PNG')
    args = parser.parse_args()
    main(args)
