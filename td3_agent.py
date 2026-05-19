"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
Standard implementation following Fujimoto et al. (2018).
Network parameters from Table 1 of the paper.

Architecture (from Fig.7):
  Actor:  Input(52) → Hidden(256) → Hidden(256) → Output(1)  [tanh]
  Critic: Input(52+1) → Hidden(256) → Hidden(256) → Q-value  (×2)

Training params (Table 1):
  Actor LR  = 0.0003
  Critic LR = 0.0003
  Discount   = 0.87
  Buffer     = 2000 (paper uses small buffer; we use 100000 for stability)
  Policy delay = 2 (standard TD3)
  Target smoothing noise std = 0.2, clip = 0.5
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr      = 0
        self.size     = 0

        self.state      = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward     = np.zeros((max_size, 1),         dtype=np.float32)
        self.done       = np.zeros((max_size, 1),         dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr]      = state
        self.action[self.ptr]     = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr]     = reward
        self.done[self.ptr]       = done

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[idx]).to(device),
            torch.FloatTensor(self.action[idx]).to(device),
            torch.FloatTensor(self.next_state[idx]).to(device),
            torch.FloatTensor(self.reward[idx]).to(device),
            torch.FloatTensor(self.done[idx]).to(device)
        )

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────
# Actor Network
# ─────────────────────────────────────────────

class Actor(nn.Module):
    """
    Policy network: state → action ∈ [-1, 1]
    Architecture: FC(256) → ReLU → FC(256) → ReLU → FC(action_dim) → tanh
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        # Last layer small init for stable early actions
        last_linear = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(last_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(last_linear.bias,   -3e-3, 3e-3)

    def forward(self, state):
        return self.net(state)


# ─────────────────────────────────────────────
# Critic Network (Double Q)
# ─────────────────────────────────────────────

class Critic(nn.Module):
    """
    Two Q-networks in a single module.
    Input: (state, action) → Q1, Q2
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for net in (self.q1, self.q2):
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


# ─────────────────────────────────────────────
# TD3 Agent
# ─────────────────────────────────────────────

class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient.

    Key TD3 tricks:
      1. Clipped double Q-learning  (Eq.7)
      2. Delayed policy updates     (Eq.8,10)
      3. Target policy smoothing    (Eq.9)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim      = 256,
        actor_lr        = 3e-4,    # Table 1
        critic_lr       = 3e-4,    # Table 1
        discount        = 0.87,    # Table 1 (gamma)
        tau             = 0.005,   # soft update coefficient
        policy_noise    = 0.2,     # target smoothing noise std
        noise_clip      = 0.5,     # c in Eq.9
        policy_delay    = 2,       # update actor every N critic updates
        device          = None
    ):
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )

        # Actor
        self.actor        = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic
        self.critic        = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt    = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.discount      = discount
        self.tau           = tau
        self.policy_noise  = policy_noise
        self.noise_clip    = noise_clip
        self.policy_delay  = policy_delay

        self.total_it      = 0   # training iterations counter

    # ─────────────── action selection ───────────────

    @torch.no_grad()
    def select_action(self, state):
        """Deterministic action for evaluation."""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor(s).cpu().numpy().flatten()

    def select_action_with_noise(self, state, noise_std=0.1):
        """Action + Gaussian exploration noise for training."""
        action = self.select_action(state)
        noise  = np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action + noise, -1.0, 1.0)

    # ─────────────── training step ───────────────

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample batch
        state, action, next_state, reward, done = \
            replay_buffer.sample(batch_size, self.device)

        # ── Critic update ──
        with torch.no_grad():
            # Target policy smoothing (Eq.9)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            # Clipped double Q (Eq.7)
            tq1, tq2    = self.critic_target(next_state, next_action)
            target_Q    = reward + (1 - done) * self.discount * torch.min(tq1, tq2)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Delayed actor update (Eq.10) ──
        actor_loss_val = None
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            actor_loss_val = actor_loss.item()

            # Soft target update
            self._soft_update(self.actor,  self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss':  actor_loss_val
        }

    def _soft_update(self, net, target_net):
        """Polyak averaging: θ' ← τ·θ + (1-τ)·θ'"""
        for p, pt in zip(net.parameters(), target_net.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

    # ─────────────── save / load ───────────────

    def save(self, path):
        torch.save({
            'actor':         self.actor.state_dict(),
            'actor_target':  self.actor_target.state_dict(),
            'critic':        self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'total_it':      self.total_it
        }, path)
        print(f"[TD3] Saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.total_it = ckpt.get('total_it', 0)
        print(f"[TD3] Loaded from {path}  (iter={self.total_it})")
