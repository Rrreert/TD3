"""
Training Script – TD3 for USV Autonomous Collision Avoidance
Follows the training setup described in Section 4.1-4.2 of the paper.

Training env:
  - Mixed scenarios: random 2/3/4-ship encounters + static obstacles
  - Episodes: up to 5000 (paper), we use 5000 as default
  - Max steps per episode: 2000

Usage:
  python train.py                      # train from scratch
  python train.py --resume model.pt    # resume training
  python train.py --episodes 3000      # custom episode count
"""

import argparse
import os
import random
import numpy as np
import torch

from environment    import USVEnv
from td3_agent      import TD3, ReplayBuffer
from imazu_scenarios import IMAZU_CASES, OS_INIT, OS_GOAL


# ─────────────────────────────────────────────
# Random training scenario generator
# ─────────────────────────────────────────────

def random_os_init():
    """Randomise OS start position slightly for generalization."""
    x   = np.random.uniform(-0.5, 0.5)
    y   = np.random.uniform(-4.5, -3.5)
    psi = np.random.uniform(np.pi/2 - 0.15, np.pi/2 + 0.15)  # ~North
    spd = np.random.uniform(0.35, 0.45)
    return {'x': x, 'y': y, 'psi': psi, 'speed': spd}


def random_target_ship(n_ships=None):
    """Generate random target ships for training diversity."""
    if n_ships is None:
        n_ships = random.choice([1, 2, 3])

    targets = []
    for _ in range(n_ships):
        # Random position in encounter zone
        r     = np.random.uniform(2.5, 4.0)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Random heading (converging toward OS area)
        # Bias toward approaching headings for more training signal
        toward_angle = np.arctan2(-y, -x)
        psi = toward_angle + np.random.uniform(-np.pi/3, np.pi/3)

        spd = np.random.uniform(0.15, 0.35)
        targets.append({
            'x': float(x), 'y': float(y + (-3.0 + r * np.sin(theta))/2),
            'psi': float(psi), 'speed': float(spd), 'length': 0.15
        })
    return targets


def random_goal(os_init):
    """Random goal roughly ahead of OS."""
    x = os_init['x'] + np.random.uniform(-0.5, 0.5)
    y = np.random.uniform(3.5, 5.0)
    return [float(x), float(y)]


def make_training_env(wind_wave=False):
    """Create an environment with randomised scenario."""
    os_cfg  = random_os_init()
    ts_cfg  = random_target_ship()
    goal    = random_goal(os_cfg)
    return USVEnv(
        target_ships_config=ts_cfg,
        goal=goal,
        os_init=os_cfg,
        dt=1.0,
        wind_wave=wind_wave
    )


# ─────────────────────────────────────────────
# Evaluation on Imazu scenarios
# ─────────────────────────────────────────────

def evaluate_imazu(agent, case_ids=None, max_steps=2000, render=False):
    """
    Run agent on specified Imazu cases (no exploration noise).
    Returns dict: {case_id: {'success': bool, 'steps': int, 'collision': bool}}
    """
    from environment import USVEnv
    if case_ids is None:
        case_ids = list(range(1, 21))

    results = {}
    for cid in case_ids:
        cfg = IMAZU_CASES[cid]
        env = USVEnv(
            target_ships_config=cfg['targets'],
            goal=cfg['goal'],
            os_init=cfg['os'],
            dt=1.0,
            wind_wave=False
        )
        obs  = env.reset()
        done = False
        t    = 0
        collision = False
        success   = False

        while not done and t < max_steps:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            t += 1
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


def print_eval_results(results):
    n_success = sum(1 for r in results.values() if r['success'])
    n_total   = len(results)
    print(f"\n{'='*50}")
    print(f"Imazu Evaluation: {n_success}/{n_total} passed")
    print(f"{'='*50}")
    for cid in sorted(results):
        r = results[cid]
        status = '✓ SUCCESS' if r['success'] else \
                 ('✗ COLLISION' if r['collision'] else '~ TIMEOUT')
        print(f"  Case {cid:2d}: {status}  (steps={r['steps']})")
    print(f"{'='*50}\n")
    return n_success


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def train(args):
    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Agent
    STATE_DIM  = USVEnv.STATE_DIM
    ACTION_DIM = USVEnv.ACTION_DIM

    agent = TD3(
        state_dim    = STATE_DIM,
        action_dim   = ACTION_DIM,
        hidden_dim   = 256,
        actor_lr     = 3e-4,
        critic_lr    = 3e-4,
        discount     = 0.87,      # from Table 1
        tau          = 0.005,
        policy_noise = 0.2,
        noise_clip   = 0.5,
        policy_delay = 2,
        device       = device
    )

    if args.resume:
        agent.load(args.resume)

    replay_buffer = ReplayBuffer(
        state_dim  = STATE_DIM,
        action_dim = ACTION_DIM,
        max_size   = int(1e5)     # larger than paper's 2000 for stability
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Training hyper-parameters
    BATCH_SIZE        = 256
    WARMUP_STEPS      = 1000    # random actions before training starts
    EXPLORE_NOISE     = 0.15    # Gaussian std during training
    TRAIN_FREQ        = 1       # train every N env steps
    EVAL_FREQ         = 200     # evaluate every N episodes
    MAX_STEPS_EP      = USVEnv.MAX_STEPS

    total_steps    = 0
    best_success   = 0
    ep_rewards     = []

    print(f"\nStarting training for {args.episodes} episodes...")
    print(f"Warmup: {WARMUP_STEPS} steps\n")

    for episode in range(1, args.episodes + 1):

        env = make_training_env(wind_wave=args.wind_wave)
        obs = env.reset()
        ep_reward = 0.0
        done = False
        t    = 0

        while not done and t < MAX_STEPS_EP:
            total_steps += 1
            t           += 1

            # Action selection
            if total_steps < WARMUP_STEPS:
                action = np.random.uniform(-1.0, 1.0, size=(ACTION_DIM,))
            else:
                action = agent.select_action_with_noise(obs, EXPLORE_NOISE)

            next_obs, reward, done, info = env.step(action)
            ep_reward += reward

            # Store transition (mask done on timeout)
            not_done_mask = 0.0 if (done and not info.get('timeout', False)) else 1.0
            replay_buffer.add(obs, action, next_obs, reward, 1.0 - not_done_mask)

            obs = next_obs

            # Train
            if total_steps >= WARMUP_STEPS and len(replay_buffer) >= BATCH_SIZE:
                if total_steps % TRAIN_FREQ == 0:
                    agent.train(replay_buffer, BATCH_SIZE)

        ep_rewards.append(ep_reward)

        # Logging
        if episode % 50 == 0:
            avg_r = np.mean(ep_rewards[-50:])
            print(f"Episode {episode:5d} | Steps {total_steps:7d} | "
                  f"AvgReward(50) {avg_r:8.1f}")

        # Periodic evaluation
        if episode % EVAL_FREQ == 0:
            results = evaluate_imazu(agent)
            n_ok    = print_eval_results(results)
            if n_ok >= best_success:
                best_success = n_ok
                ckpt = os.path.join(args.save_dir, 'best_model.pt')
                agent.save(ckpt)
                print(f"  → New best: {best_success}/20  saved to {ckpt}")

    # Final evaluation
    print("\n=== Final Evaluation on all 20 Imazu cases ===")
    results = evaluate_imazu(agent)
    print_eval_results(results)

    # Save final model
    final_path = os.path.join(args.save_dir, 'final_model.pt')
    agent.save(final_path)
    print(f"Training complete. Final model: {final_path}")


# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train TD3 for USV collision avoidance')
    parser.add_argument('--episodes',  type=int,   default=5000,
                        help='Number of training episodes (default: 5000)')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--resume',    type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save-dir',  type=str,   default='./checkpoints',
                        help='Directory to save models')
    parser.add_argument('--wind-wave', action='store_true',
                        help='Enable wind/wave disturbance during training')
    args = parser.parse_args()
    train(args)
