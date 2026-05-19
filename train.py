"""
Training Script – TD3 / ATL-TD3 for USV Autonomous Collision Avoidance
Section 4.1-4.2 of the paper.

Usage:
  # Train standard TD3 (default)
  python train.py --algo td3

  # Train ATL-TD3
  python train.py --algo atl_td3

  # Resume from checkpoint
  python train.py --algo atl_td3 --resume checkpoints/atl_td3_best.pt

  # With wind/wave disturbance
  python train.py --algo atl_td3 --wind-wave --episodes 5000
"""

import argparse
import os
import random
import numpy as np
import torch

from environment     import USVEnv
from td3_agent       import TD3, ReplayBuffer
from atl_td3_agent   import ATLTD3, SequenceReplayBuffer, EpisodeWindow
from imazu_scenarios import IMAZU_CASES


# ─────────────────────────────────────────────
# Scenario randomisation (shared by both algos)
# ─────────────────────────────────────────────

def random_os_init():
    x   = np.random.uniform(-0.5, 0.5)
    y   = np.random.uniform(-4.5, -3.5)
    psi = np.random.uniform(np.pi/2 - 0.15, np.pi/2 + 0.15)
    spd = np.random.uniform(0.35, 0.45)
    return {'x': x, 'y': y, 'psi': psi, 'speed': spd}


def random_target_ship(n_ships=None):
    if n_ships is None:
        n_ships = random.choice([1, 2, 3])
    targets = []
    for _ in range(n_ships):
        r     = np.random.uniform(2.5, 4.0)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        toward = np.arctan2(-y, -x)
        psi = toward + np.random.uniform(-np.pi/3, np.pi/3)
        spd = np.random.uniform(0.15, 0.35)
        targets.append({
            'x': float(x),
            'y': float(y + (-3.0 + r * np.sin(theta)) / 2),
            'psi': float(psi), 'speed': float(spd), 'length': 0.15
        })
    return targets


def random_goal(os_init):
    x = os_init['x'] + np.random.uniform(-0.5, 0.5)
    y = np.random.uniform(3.5, 5.0)
    return [float(x), float(y)]


def make_env(wind_wave=False):
    os_cfg = random_os_init()
    ts_cfg = random_target_ship()
    goal   = random_goal(os_cfg)
    return USVEnv(target_ships_config=ts_cfg, goal=goal,
                  os_init=os_cfg, dt=1.0, wind_wave=wind_wave)


# ─────────────────────────────────────────────
# Evaluation (works for both TD3 and ATL-TD3)
# ─────────────────────────────────────────────

def evaluate_imazu(agent, algo='td3', case_ids=None,
                   max_steps=2000, seq_len=8):
    """
    Run agent on all (or selected) Imazu cases without exploration noise.
    Returns dict: {case_id: {'success', 'collision', 'steps', 'timeout'}}
    """
    if case_ids is None:
        case_ids = list(range(1, 21))

    results = {}
    for cid in case_ids:
        cfg = IMAZU_CASES[cid]
        env = USVEnv(target_ships_config=cfg['targets'],
                     goal=cfg['goal'], os_init=cfg['os'],
                     dt=1.0, wind_wave=False)
        obs  = env.reset()
        done = False
        t    = 0
        collision = False
        success   = False

        if algo == 'atl_td3':
            win = EpisodeWindow(USVEnv.STATE_DIM, seq_len)
            win.reset(obs)

        while not done and t < max_steps:
            if algo == 'atl_td3':
                action = agent.select_action(win.get())
            else:
                action = agent.select_action(obs)

            obs, reward, done, info = env.step(action)
            t += 1

            if algo == 'atl_td3':
                win.push(obs)

            if info.get('collision'):
                collision = True
            if info.get('arrived'):
                success = True

        results[cid] = {
            'success':   success,
            'collision': collision,
            'steps':     t,
            'timeout':   (t >= max_steps and not success and not collision)
        }
    return results


def print_eval_results(results, algo_label=''):
    n_success = sum(1 for r in results.values() if r['success'])
    n_total   = len(results)
    tag = f"[{algo_label}] " if algo_label else ''
    print(f"\n{'='*52}")
    print(f"{tag}Imazu Evaluation: {n_success}/{n_total} passed")
    print(f"{'='*52}")
    for cid in sorted(results):
        r = results[cid]
        status = '✓ SUCCESS' if r['success'] else \
                 ('✗ COLLISION' if r['collision'] else '~ TIMEOUT')
        print(f"  Case {cid:2d}: {status}  (steps={r['steps']})")
    print(f"{'='*52}\n")
    return n_success


# ─────────────────────────────────────────────
# TD3 training loop
# ─────────────────────────────────────────────

def train_td3(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TD3] Device: {device}")

    S = USVEnv.STATE_DIM
    A = USVEnv.ACTION_DIM

    agent = TD3(state_dim=S, action_dim=A, hidden_dim=256,
                actor_lr=3e-4, critic_lr=3e-4, discount=0.87,
                tau=0.005, policy_noise=0.2, noise_clip=0.5,
                policy_delay=2, device=device)

    if args.resume:
        agent.load(args.resume)

    buf = ReplayBuffer(S, A, max_size=int(1e5))
    os.makedirs(args.save_dir, exist_ok=True)

    BATCH       = 256
    WARMUP      = 1000
    NOISE       = 0.15
    EVAL_FREQ   = 200
    MAX_EP_STEP = USVEnv.MAX_STEPS

    total_steps  = 0
    best_success = 0
    ep_rewards   = []

    print(f"[TD3] Training for {args.episodes} episodes ...\n")

    for ep in range(1, args.episodes + 1):
        env = make_env(args.wind_wave)
        obs = env.reset()
        ep_r = 0.0
        done = False
        t    = 0

        while not done and t < MAX_EP_STEP:
            total_steps += 1
            t           += 1

            if total_steps < WARMUP:
                action = np.random.uniform(-1.0, 1.0, (A,))
            else:
                action = agent.select_action_with_noise(obs, NOISE)

            next_obs, reward, done, info = env.step(action)
            ep_r += reward

            mask = 0.0 if (done and not info.get('timeout', False)) else 1.0
            buf.add(obs, action, next_obs, reward, 1.0 - mask)
            obs = next_obs

            if total_steps >= WARMUP and len(buf) >= BATCH:
                agent.train(buf, BATCH)

        ep_rewards.append(ep_r)

        if ep % 50 == 0:
            avg = np.mean(ep_rewards[-50:])
            print(f"[TD3] Ep {ep:5d} | Steps {total_steps:7d} | "
                  f"AvgR(50)={avg:8.1f}")

        if ep % EVAL_FREQ == 0:
            res  = evaluate_imazu(agent, algo='td3')
            n_ok = print_eval_results(res, 'TD3')
            if n_ok >= best_success:
                best_success = n_ok
                path = os.path.join(args.save_dir, 'td3_best.pt')
                agent.save(path)
                print(f"  → Best TD3: {best_success}/20  → {path}")

    print("\n[TD3] Final evaluation:")
    res = evaluate_imazu(agent, algo='td3')
    print_eval_results(res, 'TD3')
    final = os.path.join(args.save_dir, 'td3_final.pt')
    agent.save(final)
    print(f"[TD3] Done. Final model: {final}")


# ─────────────────────────────────────────────
# ATL-TD3 training loop
# ─────────────────────────────────────────────

def train_atl_td3(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ATL-TD3] Device: {device}")

    S       = USVEnv.STATE_DIM
    A       = USVEnv.ACTION_DIM
    SEQ_LEN = 8    # history window length T

    agent = ATLTD3(
        state_dim=S, action_dim=A,
        seq_len=SEQ_LEN, lstm_hidden=128, num_heads=4, hidden_dim=256,
        actor_lr=3e-4, critic_lr=3e-4, discount=0.87,
        tau=0.005, policy_noise=0.2, noise_clip=0.5,
        policy_delay=2, device=device
    )

    if args.resume:
        agent.load(args.resume)

    buf = SequenceReplayBuffer(S, A, seq_len=SEQ_LEN, max_size=int(1e5))
    os.makedirs(args.save_dir, exist_ok=True)

    BATCH       = 256
    WARMUP      = 1500    # slightly longer warmup for LSTM stability
    NOISE       = 0.15
    EVAL_FREQ   = 200
    MAX_EP_STEP = USVEnv.MAX_STEPS

    total_steps  = 0
    best_success = 0
    ep_rewards   = []

    print(f"[ATL-TD3] Training for {args.episodes} episodes ...\n")

    for ep in range(1, args.episodes + 1):
        env = make_env(args.wind_wave)
        obs = env.reset()
        ep_r = 0.0
        done = False
        t    = 0

        # Episode-level sliding window
        win = EpisodeWindow(S, SEQ_LEN)
        win.reset(obs)

        while not done and t < MAX_EP_STEP:
            total_steps += 1
            t           += 1

            seq = win.get()   # (SEQ_LEN, S)

            if total_steps < WARMUP:
                action = np.random.uniform(-1.0, 1.0, (A,))
            else:
                action = agent.select_action_with_noise(seq, NOISE)

            next_obs, reward, done, info = env.step(action)
            ep_r += reward

            # Push next obs into window, then snapshot for storage
            win.push(next_obs)
            next_seq = win.get()   # (SEQ_LEN, S)

            mask = 0.0 if (done and not info.get('timeout', False)) else 1.0
            buf.add(seq, action, next_seq, reward, 1.0 - mask)

            obs = next_obs

            if total_steps >= WARMUP and len(buf) >= BATCH:
                agent.train(buf, BATCH)

        ep_rewards.append(ep_r)

        if ep % 50 == 0:
            avg = np.mean(ep_rewards[-50:])
            print(f"[ATL-TD3] Ep {ep:5d} | Steps {total_steps:7d} | "
                  f"AvgR(50)={avg:8.1f}")

        if ep % EVAL_FREQ == 0:
            res  = evaluate_imazu(agent, algo='atl_td3', seq_len=SEQ_LEN)
            n_ok = print_eval_results(res, 'ATL-TD3')
            if n_ok >= best_success:
                best_success = n_ok
                path = os.path.join(args.save_dir, 'atl_td3_best.pt')
                agent.save(path)
                print(f"  → Best ATL-TD3: {best_success}/20  → {path}")

    print("\n[ATL-TD3] Final evaluation:")
    res = evaluate_imazu(agent, algo='atl_td3', seq_len=SEQ_LEN)
    print_eval_results(res, 'ATL-TD3')
    final = os.path.join(args.save_dir, 'atl_td3_final.pt')
    agent.save(final)
    print(f"[ATL-TD3] Done. Final model: {final}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train TD3 or ATL-TD3 for USV collision avoidance')
    parser.add_argument('--algo',      type=str, default='td3',
                        choices=['td3', 'atl_td3'],
                        help='Algorithm: td3 | atl_td3  (default: td3)')
    parser.add_argument('--episodes',  type=int, default=5000)
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--resume',    type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--save-dir',  type=str, default='./checkpoints')
    parser.add_argument('--wind-wave', action='store_true',
                        help='Enable wind/wave disturbance')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.algo == 'td3':
        train_td3(args)
    else:
        train_atl_td3(args)
