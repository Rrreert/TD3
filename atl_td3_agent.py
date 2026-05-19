"""
ATL-TD3: Attention Long Short-Term Memory Twin Delayed Deep Deterministic
Policy Gradient — full implementation from Section 3.1 of the paper.

Key additions over standard TD3:
  1. LSTM layer captures historical state sequences (Eq.11-15)
  2. Multi-Head Self-Attention (MHA) on top of LSTM output (Eq.16-18)
     – 4 attention heads (h=4, Table 1)
     – Causal mask so each step only attends to past steps
  3. Sequence-aware ReplayBuffer storing fixed-length state windows
  4. Same TD3 training mechanics: clipped double-Q (Eq.7),
     delayed actor update (Eq.8,10), target smoothing (Eq.9)

Network architecture (Fig.7 of paper):
  Actor:
    Input(52) → MHA-LSTM layer → FC(256,ReLU) → FC(256,ReLU) → FC(1,Tanh)
  Critic (×2):
    Input(52+1) → MHA-LSTM layer → FC(256,ReLU) → FC(256,ReLU) → FC(1)

MHA-LSTM layer detail:
  – LSTM(input_dim, lstm_hidden=128)   produces hidden sequence H
  – MultiheadAttention(embed=128, heads=4) over H  → context vector C
  – Output = concat(H_last, C_last)  → dim 256  (feeds into FC layers)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import deque


# ─────────────────────────────────────────────────────────────
# Sequence Replay Buffer
# Stores the last SEQ_LEN states as a single transition unit so
# the LSTM sees a history window at each training step.
# ─────────────────────────────────────────────────────────────

class SequenceReplayBuffer:
    """
    Replay buffer that stores fixed-length observation sequences.

    Each stored item is a tuple:
      (state_seq, action, next_state_seq, reward, done)

    state_seq      : (SEQ_LEN, state_dim)  — history window ending at s_t
    next_state_seq : (SEQ_LEN, state_dim)  — history window ending at s_{t+1}

    During an episode a sliding window of length SEQ_LEN is maintained.
    Transitions are only stored once the window is full.
    """

    def __init__(self, state_dim, action_dim,
                 seq_len=8, max_size=int(1e5)):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.seq_len    = seq_len
        self.max_size   = max_size
        self.ptr        = 0
        self.size       = 0

        self.state_seq      = np.zeros(
            (max_size, seq_len, state_dim), dtype=np.float32)
        self.action         = np.zeros(
            (max_size, action_dim),         dtype=np.float32)
        self.next_state_seq = np.zeros(
            (max_size, seq_len, state_dim), dtype=np.float32)
        self.reward         = np.zeros((max_size, 1), dtype=np.float32)
        self.done           = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state_seq, action, next_state_seq, reward, done):
        """
        state_seq / next_state_seq: numpy arrays of shape (seq_len, state_dim)
        """
        self.state_seq[self.ptr]      = state_seq
        self.action[self.ptr]         = action
        self.next_state_seq[self.ptr] = next_state_seq
        self.reward[self.ptr]         = reward
        self.done[self.ptr]           = done

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state_seq[idx]).to(device),       # (B,T,S)
            torch.FloatTensor(self.action[idx]).to(device),           # (B,A)
            torch.FloatTensor(self.next_state_seq[idx]).to(device),   # (B,T,S)
            torch.FloatTensor(self.reward[idx]).to(device),           # (B,1)
            torch.FloatTensor(self.done[idx]).to(device)              # (B,1)
        )

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────
# Episode Sequence Window
# Helper used during environment interaction to maintain
# the sliding observation window fed to the network.
# ─────────────────────────────────────────────────────────────

class EpisodeWindow:
    """Maintains a fixed-length deque of recent observations."""

    def __init__(self, state_dim, seq_len=8):
        self.seq_len   = seq_len
        self.state_dim = state_dim
        self._buf      = deque(maxlen=seq_len)
        self.reset()

    def reset(self, init_obs=None):
        self._buf.clear()
        pad = np.zeros(self.state_dim, dtype=np.float32)
        for _ in range(self.seq_len):
            self._buf.append(pad.copy())
        if init_obs is not None:
            self._buf.append(np.array(init_obs, dtype=np.float32))

    def push(self, obs):
        self._buf.append(np.array(obs, dtype=np.float32))

    def get(self):
        """Returns numpy array of shape (seq_len, state_dim)."""
        return np.stack(list(self._buf), axis=0)


# ─────────────────────────────────────────────────────────────
# MHA-LSTM Block  (Fig.5, Fig.6, Fig.7 of paper)
# ─────────────────────────────────────────────────────────────

class MHALSTMBlock(nn.Module):
    """
    Core ATL block: LSTM → Multi-Head Self-Attention → fused output.

    Input shape:  (batch, seq_len, input_dim)
    Output shape: (batch, 2 * lstm_hidden)   — last hidden + attended context

    LSTM equations (Eq.11-15):
      z^i = σ(W_xi·x + W_hi·h + b_i)
      z^f = σ(W_xf·x + W_hf·h + b_f)
      z^o = σ(W_xo·x + W_ho·h + b_o)
      c   = z^f ⊙ c_prev + z^i ⊙ tanh(W_xg·x + W_hg·h + b_g)
      h   = z^o ⊙ tanh(c)

    Attention (Eq.16-18):
      Attention(Q,K,V) = softmax(mask(QK^T)/√d_k) V
      Head_i = Attention(QW^q_i, KW^k_i, VW^v_i)
      MultiHead = concat([Head_i]) W^h
    """

    def __init__(self, input_dim, lstm_hidden=128, num_heads=4, dropout=0.0):
        super().__init__()
        assert lstm_hidden % num_heads == 0, \
            f"lstm_hidden ({lstm_hidden}) must be divisible by num_heads ({num_heads})"

        self.lstm_hidden = lstm_hidden

        # LSTM (Eq.11-15) — PyTorch's built-in LSTM implements the paper's gates
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,   # (B, T, input_dim)
            dropout=dropout
        )

        # Multi-Head Self-Attention (Eq.16-18)
        # Q, K, V all come from LSTM output sequence H
        self.mha = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True    # (B, T, embed_dim)
        )

        # Layer norm for stability (common practice with attention)
        self.ln_lstm = nn.LayerNorm(lstm_hidden)
        self.ln_mha  = nn.LayerNorm(lstm_hidden)

        # Output projection: concat(h_last, context_last) → 2*lstm_hidden
        # This feeds directly into the downstream FC layers
        self.out_dim = 2 * lstm_hidden

        self._init_lstm()

    def _init_lstm(self):
        for name, p in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)
                # Forget gate bias = 1 (helps remember by default)
                n = p.size(0)
                p.data[n//4 : n//2].fill_(1.0)

    def _causal_mask(self, seq_len, device):
        """
        Causal (lower-triangular) attention mask.
        Positions where mask=True are IGNORED by nn.MultiheadAttention.
        So we set upper triangle to True (future positions are masked out).
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, x, hidden=None):
        """
        x:      (B, T, input_dim)
        hidden: optional (h_0, c_0) tuple for stateful inference
        Returns:
          out:    (B, 2*lstm_hidden)  — fused representation at last timestep
          hidden: updated (h_n, c_n) for stateful use
        """
        B, T, _ = x.shape

        # ── LSTM pass ──
        H, hidden = self.lstm(x, hidden)   # H: (B, T, lstm_hidden)
        H = self.ln_lstm(H)

        # ── Multi-Head Self-Attention (causal) ──
        # Q = K = V = H  (self-attention over the LSTM output sequence)
        mask = self._causal_mask(T, x.device)
        C, _ = self.mha(H, H, H, attn_mask=mask)   # C: (B, T, lstm_hidden)
        C = self.ln_mha(C)

        # Residual connection: blend attended context with LSTM output
        C = H + C   # (B, T, lstm_hidden)

        # Take last timestep for both streams
        h_last = H[:, -1, :]   # (B, lstm_hidden)
        c_last = C[:, -1, :]   # (B, lstm_hidden)

        out = torch.cat([h_last, c_last], dim=-1)   # (B, 2*lstm_hidden)
        return out, hidden


# ─────────────────────────────────────────────────────────────
# ATL Actor Network
# ─────────────────────────────────────────────────────────────

class ATLActor(nn.Module):
    """
    ATL Actor:  seq_input → MHALSTMBlock → FC(256,ReLU) → FC(256,ReLU)
                         → FC(action_dim, Tanh)

    Input:  (B, T, state_dim)   or   (B, state_dim) for single-step inference
    Output: (B, action_dim)
    """

    def __init__(self, state_dim, action_dim,
                 lstm_hidden=128, num_heads=4, hidden_dim=256):
        super().__init__()

        self.atl_block = MHALSTMBlock(
            input_dim=state_dim,
            lstm_hidden=lstm_hidden,
            num_heads=num_heads
        )
        atl_out_dim = self.atl_block.out_dim   # 2 * lstm_hidden = 256

        self.fc = nn.Sequential(
            nn.Linear(atl_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self._init_fc()

    def _init_fc(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        last = [m for m in self.fc if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(last.weight, -3e-3, 3e-3)
        nn.init.uniform_(last.bias,   -3e-3, 3e-3)

    def forward(self, state_seq, hidden=None):
        """
        state_seq: (B, T, state_dim)  — sequence of observations
        Returns action: (B, action_dim)
        """
        feat, hidden = self.atl_block(state_seq, hidden)
        return self.fc(feat), hidden

    def forward_single(self, state_seq):
        """Convenience wrapper returning only action (no hidden state)."""
        action, _ = self.forward(state_seq)
        return action


# ─────────────────────────────────────────────────────────────
# ATL Critic Network (Double Q)
# ─────────────────────────────────────────────────────────────

class ATLCritic(nn.Module):
    """
    ATL Critic: two Q-networks, each with its own MHALSTMBlock.
    Input:  state_seq (B,T,S)  +  action (B,A)
    Output: Q1, Q2   both (B,1)

    Action is concatenated after the ATL block (not fed into LSTM),
    following the common actor-critic practice for continuous control.
    """

    def __init__(self, state_dim, action_dim,
                 lstm_hidden=128, num_heads=4, hidden_dim=256):
        super().__init__()

        atl_out_dim = 2 * lstm_hidden

        # Q1
        self.atl1 = MHALSTMBlock(state_dim, lstm_hidden, num_heads)
        self.q1   = nn.Sequential(
            nn.Linear(atl_out_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2
        self.atl2 = MHALSTMBlock(state_dim, lstm_hidden, num_heads)
        self.q2   = nn.Sequential(
            nn.Linear(atl_out_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_fc()

    def _init_fc(self):
        for net in (self.q1, self.q2):
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, state_seq, action):
        """
        state_seq: (B, T, state_dim)
        action:    (B, action_dim)
        """
        feat1, _ = self.atl1(state_seq)
        feat2, _ = self.atl2(state_seq)

        sa1 = torch.cat([feat1, action], dim=-1)
        sa2 = torch.cat([feat2, action], dim=-1)
        return self.q1(sa1), self.q2(sa2)

    def Q1(self, state_seq, action):
        feat1, _ = self.atl1(state_seq)
        sa1 = torch.cat([feat1, action], dim=-1)
        return self.q1(sa1)


# ─────────────────────────────────────────────────────────────
# ATL-TD3 Agent
# ─────────────────────────────────────────────────────────────

class ATLTD3:
    """
    ATL-TD3: Attention LSTM Twin Delayed Deep Deterministic Policy Gradient.

    Identical training mechanics to TD3 (Eq.7-10) but with ATLActor /
    ATLCritic replacing the plain FC networks.

    Hyper-parameters from Table 1:
      Actor LR   = 0.0003
      Critic LR  = 0.0003
      Discount γ = 0.87
      MHA heads  = 4
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        seq_len         = 8,       # history window length fed to LSTM
        lstm_hidden     = 128,     # LSTM hidden size (→ 256 after concat)
        num_heads       = 4,       # MHA heads (Table 1: h=4)
        hidden_dim      = 256,     # FC hidden size (Fig.7)
        actor_lr        = 3e-4,
        critic_lr       = 3e-4,
        discount        = 0.87,    # γ, Table 1
        tau             = 0.005,
        policy_noise    = 0.2,
        noise_clip      = 0.5,
        policy_delay    = 2,
        device          = None
    ):
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        self.seq_len    = seq_len
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # ── Networks ──
        self.actor = ATLActor(
            state_dim, action_dim, lstm_hidden, num_heads, hidden_dim
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)

        self.critic = ATLCritic(
            state_dim, action_dim, lstm_hidden, num_heads, hidden_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt    = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)

        # ── TD3 hyper-parameters ──
        self.discount      = discount
        self.tau           = tau
        self.policy_noise  = policy_noise
        self.noise_clip    = noise_clip
        self.policy_delay  = policy_delay
        self.total_it      = 0

    # ──────────────── action selection ────────────────

    @torch.no_grad()
    def select_action(self, state_seq):
        """
        Deterministic inference (evaluation mode).
        state_seq: numpy (seq_len, state_dim)  or  (state_dim,) [auto-expanded]
        """
        if state_seq.ndim == 1:
            # Single flat obs → expand to (1, 1, S) for minimal seq
            s = torch.FloatTensor(state_seq).unsqueeze(0).unsqueeze(0)
        else:
            # (T, S) → (1, T, S)
            s = torch.FloatTensor(state_seq).unsqueeze(0)
        s = s.to(self.device)
        action = self.actor.forward_single(s)
        return action.cpu().numpy().flatten()

    def select_action_with_noise(self, state_seq, noise_std=0.1):
        action = self.select_action(state_seq)
        noise  = np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action + noise, -1.0, 1.0)

    # ──────────────── training step ────────────────

    def train(self, replay_buffer, batch_size=256):
        """One gradient update step."""
        self.total_it += 1

        # Sample sequences
        state_seq, action, next_seq, reward, done = \
            replay_buffer.sample(batch_size, self.device)
        # state_seq: (B, T, S),  action: (B, A),  etc.

        # ── Critic update (Eq.7-8) ──
        with torch.no_grad():
            # Target policy smoothing (Eq.9)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (
                self.actor_target.forward_single(next_seq) + noise
            ).clamp(-1.0, 1.0)

            # Clipped double Q (Eq.7)
            tq1, tq2 = self.critic_target(next_seq, next_action)
            target_Q = reward + (1 - done) * self.discount * torch.min(tq1, tq2)

        q1, q2 = self.critic(state_seq, action)
        critic_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Delayed actor update (Eq.10) ──
        actor_loss_val = None
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(
                state_seq, self.actor.forward_single(state_seq)
            ).mean()

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
        for p, pt in zip(net.parameters(), target_net.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

    # ──────────────── save / load ────────────────

    def save(self, path):
        torch.save({
            'actor':         self.actor.state_dict(),
            'actor_target':  self.actor_target.state_dict(),
            'critic':        self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'total_it':      self.total_it,
            'seq_len':       self.seq_len,
        }, path)
        print(f"[ATL-TD3] Saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.total_it = ckpt.get('total_it', 0)
        print(f"[ATL-TD3] Loaded from {path}  (iter={self.total_it})")
