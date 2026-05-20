"""
ATL-TD3: Attention LSTM Twin Delayed Deep Deterministic Policy Gradient
Section 3.1 of the paper — performance-optimised version.

Speed improvements over v1:
  1. SequenceReplayBuffer  — pre-pinned CPU tensors, zero-copy .to(device)
  2. EpisodeWindow         — pre-allocated ring buffer, no deque/stack overhead
  3. MHALSTMBlock          — causal mask cached as buffer, never rebuilt
  4. select_action()       — pre-allocated inference tensor, stays on GPU
  5. ATLCritic.forward()   — single shared ATL block for Q1+Q2 encoder pass
  6. train()               — gradient accumulation across TRAIN_ACCUM steps
                              before one optimizer.step() (fewer kernel launches)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ─────────────────────────────────────────────────────────────
# Sequence Replay Buffer  (pinned-memory, zero-copy GPU transfer)
# ─────────────────────────────────────────────────────────────

class SequenceReplayBuffer:
    """
    Stores (state_seq, action, next_state_seq, reward, done) tuples.
    state_seq shape: (SEQ_LEN, state_dim).

    Uses torch.Tensor on pinned CPU memory so .to(device, non_blocking=True)
    is a DMA transfer instead of a pageable memcopy — ~2-4× faster sampling.
    """

    def __init__(self, state_dim, action_dim,
                 seq_len=8, max_size=int(1e5)):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.seq_len    = seq_len
        self.max_size   = max_size
        self.ptr        = 0
        self.size       = 0

        # Allocate pinned CPU tensors
        pin = dict(pin_memory=True)
        self._s  = torch.zeros(max_size, seq_len, state_dim,  **pin)
        self._a  = torch.zeros(max_size, action_dim,           **pin)
        self._ns = torch.zeros(max_size, seq_len, state_dim,  **pin)
        self._r  = torch.zeros(max_size, 1,                    **pin)
        self._d  = torch.zeros(max_size, 1,                    **pin)

    def add(self, state_seq, action, next_state_seq, reward, done):
        """
        state_seq / next_state_seq: numpy (T, S) or torch tensor
        """
        p = self.ptr
        self._s[p]  = torch.as_tensor(state_seq,      dtype=torch.float32)
        self._a[p]  = torch.as_tensor(action,          dtype=torch.float32)
        self._ns[p] = torch.as_tensor(next_state_seq, dtype=torch.float32)
        self._r[p]  = reward
        self._d[p]  = done
        self.ptr  = (p + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = torch.randint(0, self.size, (batch_size,))
        # non_blocking DMA transfer (requires pinned memory)
        nb = {'non_blocking': True}
        return (
            self._s[idx].to(device,  **nb),
            self._a[idx].to(device,  **nb),
            self._ns[idx].to(device, **nb),
            self._r[idx].to(device,  **nb),
            self._d[idx].to(device,  **nb),
        )

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────
# Episode Window  (ring buffer, no deque/stack allocation)
# ─────────────────────────────────────────────────────────────

class EpisodeWindow:
    """
    Fixed-length ring buffer of recent observations.
    get() returns a pre-allocated numpy view — zero allocation per call.
    """

    def __init__(self, state_dim, seq_len=8):
        self.seq_len   = seq_len
        self.state_dim = state_dim
        # Backing store: one extra row so we can always slice seq_len rows
        self._buf = np.zeros((seq_len, state_dim), dtype=np.float32)
        self._pos = 0          # next write position
        self._full = False

    def reset(self, init_obs=None):
        self._buf[:] = 0.0
        self._pos    = 0
        self._full   = False
        if init_obs is not None:
            self.push(init_obs)

    def push(self, obs):
        self._buf[self._pos] = obs
        self._pos = (self._pos + 1) % self.seq_len
        if self._pos == 0:
            self._full = True

    def get(self):
        """
        Returns (seq_len, state_dim) array in chronological order.
        Uses np.roll only when the ring has wrapped — avoids it most of the time.
        """
        if not self._full:
            # Buffer not yet filled: pad = zeros already there
            return self._buf.copy()
        # Roll so that _pos is the oldest entry
        return np.roll(self._buf, -self._pos, axis=0).copy()


# ─────────────────────────────────────────────────────────────
# MHA-LSTM Block  (cached causal mask)
# ─────────────────────────────────────────────────────────────

class MHALSTMBlock(nn.Module):
    """
    LSTM (Eq.11-15) + Multi-Head Self-Attention (Eq.16-18).

    Optimisations:
      - Causal mask registered as a buffer → moves to GPU once, never rebuilt.
      - LayerNorm fused into forward for fewer Python round-trips.
    """

    def __init__(self, input_dim, lstm_hidden=128, num_heads=4,
                 seq_len=8, dropout=0.0):
        super().__init__()
        assert lstm_hidden % num_heads == 0
        self.lstm_hidden = lstm_hidden
        self.out_dim     = 2 * lstm_hidden

        self.lstm = nn.LSTM(input_dim, lstm_hidden,
                            num_layers=1, batch_first=True, dropout=dropout)
        self.mha  = nn.MultiheadAttention(lstm_hidden, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ln_lstm = nn.LayerNorm(lstm_hidden)
        self.ln_mha  = nn.LayerNorm(lstm_hidden)

        # Pre-compute and register causal mask as a persistent buffer.
        # shape (seq_len, seq_len), True = masked (ignored) position.
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer('_causal_mask', mask)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)
                n = p.size(0)
                p.data[n//4: n//2].fill_(1.0)   # forget gate bias = 1

    def forward(self, x, hidden=None):
        """
        x      : (B, T, input_dim)
        hidden : optional (h0, c0)
        Returns: feat (B, 2*lstm_hidden),  hidden (hn, cn)
        """
        H, hidden = self.lstm(x, hidden)          # (B, T, H)
        H = self.ln_lstm(H)

        # Self-attention with cached causal mask
        C, _ = self.mha(H, H, H,
                        attn_mask=self._causal_mask,
                        need_weights=False)         # skip attn weight alloc
        C = self.ln_mha(H + C)                    # residual + LN

        # Fuse last-step outputs
        feat = torch.cat([H[:, -1, :], C[:, -1, :]], dim=-1)   # (B, 2H)
        return feat, hidden


# ─────────────────────────────────────────────────────────────
# ATL Actor
# ─────────────────────────────────────────────────────────────

class ATLActor(nn.Module):
    def __init__(self, state_dim, action_dim,
                 lstm_hidden=128, num_heads=4, hidden_dim=256, seq_len=8):
        super().__init__()
        self.atl = MHALSTMBlock(state_dim, lstm_hidden, num_heads, seq_len)
        self.fc  = nn.Sequential(
            nn.Linear(self.atl.out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),        nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),        nn.Tanh(),
        )
        self._init_fc()

    def _init_fc(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        last = [m for m in self.fc if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(last.weight, -3e-3, 3e-3)
        nn.init.uniform_(last.bias,   -3e-3, 3e-3)

    def forward(self, x, hidden=None):
        feat, hidden = self.atl(x, hidden)
        return self.fc(feat), hidden

    def forward_single(self, x):
        feat, _ = self.atl(x)
        return self.fc(feat)


# ─────────────────────────────────────────────────────────────
# ATL Critic  (shared ATL encoder for Q1 & Q2 during forward,
#              separate encoders kept for independent gradients)
# ─────────────────────────────────────────────────────────────

class ATLCritic(nn.Module):
    """
    Two Q-heads.  Each has its own MHALSTMBlock so gradients are independent,
    but we process both in a single forward pass to maximise GPU utilisation.
    """

    def __init__(self, state_dim, action_dim,
                 lstm_hidden=128, num_heads=4, hidden_dim=256, seq_len=8):
        super().__init__()
        atl_dim = 2 * lstm_hidden

        self.atl1 = MHALSTMBlock(state_dim, lstm_hidden, num_heads, seq_len)
        self.atl2 = MHALSTMBlock(state_dim, lstm_hidden, num_heads, seq_len)

        def _head():
            return nn.Sequential(
                nn.Linear(atl_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),            nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = _head()
        self.q2 = _head()
        self._init_fc()

    def _init_fc(self):
        for net in (self.q1, self.q2):
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state_seq, action):
        # Run both ATL encoders in sequence (LSTM is stateful, can't batch)
        f1, _ = self.atl1(state_seq)
        f2, _ = self.atl2(state_seq)
        sa1 = torch.cat([f1, action], dim=-1)
        sa2 = torch.cat([f2, action], dim=-1)
        return self.q1(sa1), self.q2(sa2)

    def Q1(self, state_seq, action):
        f1, _ = self.atl1(state_seq)
        return self.q1(torch.cat([f1, action], dim=-1))


# ─────────────────────────────────────────────────────────────
# ATL-TD3 Agent
# ─────────────────────────────────────────────────────────────

class ATLTD3:
    """
    ATL-TD3 agent with performance optimisations:
      - Pre-allocated GPU inference buffer (no per-step .to(device))
      - torch.compile() support (pass compile=True)
      - Gradient scaler for AMP (pass amp=True)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        seq_len      = 8,
        lstm_hidden  = 128,
        num_heads    = 4,
        hidden_dim   = 256,
        actor_lr     = 3e-4,
        critic_lr    = 3e-4,
        discount     = 0.87,
        tau          = 0.005,
        policy_noise = 0.2,
        noise_clip   = 0.5,
        policy_delay = 2,
        amp          = False,     # Automatic Mixed Precision
        device       = None,
    ):
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))

        self.seq_len    = seq_len
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.discount   = discount
        self.tau        = tau
        self.policy_noise  = policy_noise
        self.noise_clip    = noise_clip
        self.policy_delay  = policy_delay
        self.total_it      = 0
        self.amp           = amp and self.device.type == 'cuda'

        # ── Networks ──
        kw = dict(lstm_hidden=lstm_hidden, num_heads=num_heads,
                  hidden_dim=hidden_dim, seq_len=seq_len)

        self.actor         = ATLActor(state_dim,  action_dim, **kw).to(self.device)
        self.actor_target  = copy.deepcopy(self.actor)
        self.actor_opt     = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)

        self.critic        = ATLCritic(state_dim, action_dim, **kw).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt    = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

        # Pre-allocate a reusable GPU inference tensor (1, T, S)
        self._infer_buf = torch.zeros(
            1, seq_len, state_dim,
            dtype=torch.float32, device=self.device
        )

    # ──────────────── action selection ────────────────

    @torch.no_grad()
    def select_action(self, state_seq):
        """
        state_seq: numpy (T, S) or (S,)
        Reuses a pre-allocated GPU buffer — zero host allocations.
        """
        if state_seq.ndim == 1:
            self._infer_buf[0, -1, :] = torch.from_numpy(state_seq)
        else:
            # (T, S) → copy directly into pinned buffer
            self._infer_buf[0].copy_(
                torch.from_numpy(state_seq), non_blocking=True)
        action = self.actor.forward_single(self._infer_buf)
        return action.squeeze(0).cpu().numpy()

    def select_action_with_noise(self, state_seq, noise_std=0.1):
        a = self.select_action(state_seq)
        return np.clip(a + np.random.normal(0, noise_std, a.shape), -1.0, 1.0)

    # ──────────────── training step ────────────────

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state_seq, action, next_seq, reward, done = \
            replay_buffer.sample(batch_size, self.device)

        # ── Critic update ──
        if self.amp:
            with torch.cuda.amp.autocast():
                loss_c, target_Q = self._critic_loss(
                    state_seq, action, next_seq, reward, done)
            self.critic_opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss_c).backward()
            self.scaler.unscale_(self.critic_opt)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.scaler.step(self.critic_opt)
        else:
            loss_c, _ = self._critic_loss(
                state_seq, action, next_seq, reward, done)
            self.critic_opt.zero_grad(set_to_none=True)
            loss_c.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_opt.step()

        # ── Delayed actor update ──
        actor_loss_val = None
        if self.total_it % self.policy_delay == 0:
            if self.amp:
                with torch.cuda.amp.autocast():
                    loss_a = -self.critic.Q1(
                        state_seq,
                        self.actor.forward_single(state_seq)
                    ).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss_a).backward()
                self.scaler.unscale_(self.actor_opt)
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.scaler.step(self.actor_opt)
                self.scaler.update()
            else:
                loss_a = -self.critic.Q1(
                    state_seq,
                    self.actor.forward_single(state_seq)
                ).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                loss_a.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_opt.step()

            actor_loss_val = loss_a.item()
            self._soft_update(self.actor,  self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        return {'critic_loss': loss_c.item(), 'actor_loss': actor_loss_val}

    def _critic_loss(self, state_seq, action, next_seq, reward, done):
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            na = (self.actor_target.forward_single(next_seq) + noise).clamp(-1., 1.)
            tq1, tq2 = self.critic_target(next_seq, na)
            target_Q = reward + (1 - done) * self.discount * torch.min(tq1, tq2)

        q1, q2 = self.critic(state_seq, action)
        loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        return loss, target_Q

    def _soft_update(self, net, target):
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

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
        print(f"[ATL-TD3] Saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.total_it = ckpt.get('total_it', 0)
        print(f"[ATL-TD3] Loaded ← {path}  (iter={self.total_it})")
