# USV Autonomous Collision Avoidance — TD3 & ATL-TD3 Reproduction

Reproduction of both the **TD3 baseline** and the **ATL-TD3 proposed method** from:

> Cui et al., "Autonomous collision avoidance decision-making method for USV based on ATL-TD3 algorithm"  
> *Ocean Engineering* 312 (2024) 119297

---

## File Structure

```
usv_td3/
├── ship_model.py       # 3-DOF nonlinear USV dynamics (Fossen 2011, Table 1)
├── colregs_model.py    # QSD ship domain, arena, COLREGs classifier,
│                       # 48-beam vectorised radar sensor
├── environment.py      # RL environment — reward (Eq.19-25), state/action
├── td3_agent.py        # Standard TD3 — FC Actor/Critic + ReplayBuffer
├── atl_td3_agent.py    # ATL-TD3 — MHA-LSTM networks + optimised buffer/window
├── imazu_scenarios.py  # All 20 Imazu case definitions
├── train.py            # Unified training (--algo td3 | atl_td3, --amp)
├── evaluate.py         # Evaluation + trajectory/rudder plots + compare mode
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.8+, PyTorch 1.12+.

---

## Training

```bash
# Standard TD3
python train.py --algo td3 --episodes 5000

# ATL-TD3
python train.py --algo atl_td3 --episodes 5000

# ATL-TD3 with Automatic Mixed Precision (CUDA, ~1.5-2× faster)
python train.py --algo atl_td3 --amp

# Resume from checkpoint
python train.py --algo atl_td3 --resume checkpoints/atl_td3_best.pt

# With wind/wave disturbance
python train.py --algo atl_td3 --amp --wind-wave
```

Models saved to `./checkpoints/`:
`td3_best.pt`, `td3_final.pt`, `atl_td3_best.pt`, `atl_td3_final.pt`

---

## Evaluation

```bash
# All 20 Imazu cases
python evaluate.py --algo atl_td3 --model checkpoints/atl_td3_best.pt

# Single case + detail plot
python evaluate.py --algo atl_td3 --model checkpoints/atl_td3_best.pt --case 11

# Save trajectory + rudder figures (Fig.9 / Fig.10 style)
python evaluate.py --algo atl_td3 --model checkpoints/atl_td3_best.pt --save-fig

# Side-by-side TD3 vs ATL-TD3 comparison
python evaluate.py --compare \
    --td3-model    checkpoints/td3_best.pt \
    --atltd3-model checkpoints/atl_td3_best.pt \
    --case 11 --save-fig
```

---

## Performance Optimisations (vs original slow version)

The following changes bring GPU training from ~10 min/50 ep down to ~1-2 min/50 ep:

| # | Bottleneck | Fix | Speedup |
|---|-----------|-----|---------|
| 1 | `RadarSensor.scan()` — Python loop over 48 beams × N obstacles | Full numpy vectorisation with broadcasting, zero Python loops | **~30-50×** on radar |
| 2 | `SequenceReplayBuffer.sample()` — pageable memcopy every batch | Pinned CPU tensors + `non_blocking=True` DMA transfer | ~2-4× on sampling |
| 3 | `EpisodeWindow.get()` — `np.stack(list(deque))` per step | Pre-allocated ring buffer, `np.roll` only when wrapped | ~5× on windowing |
| 4 | `select_action()` — new tensor + `.to(device)` per step | Pre-allocated GPU inference buffer, `copy_()` in-place | ~3× on inference |
| 5 | `train()` called every env step | `TRAIN_FREQ=4`: collect 4 steps then 1 update | ~4× fewer kernel launches |
| 6 | `_causal_mask` rebuilt every forward | Registered as `nn.Buffer`, moved to GPU once | eliminates per-forward alloc |
| 7 | Env object reconstructed each episode | Persistent env, update config then call `reset()` | ~2× on episode setup |
| 8 | Full-precision LSTM on GPU | `--amp` flag: `torch.cuda.amp.autocast()` + `GradScaler` | ~1.5-2× on GPU |

### Recommended GPU launch command

```bash
python train.py --algo atl_td3 --amp --episodes 5000
```

Expected training time on a modern GPU (RTX 3090 / A100):
- ~1-2 min per 50 episodes (vs ~10 min before)
- ~2-4 hours for 5000 episodes total

---

## ATL-TD3 Architecture (Section 3.1, Fig.5–7)

### MHA-LSTM Block

```
Input sequence  (B, T=8, state_dim=52)
        │
        ▼
    LSTM  hidden=128                ← Eq.11–15
        │  H: (B, T, 128)
        ▼
  LayerNorm(H)
        │
        ▼
  MultiheadAttention(H,H,H)        ← Eq.16–18
  heads=4, causal mask (cached)
        │  C: (B, T, 128)
        ▼
  Residual H + C  →  LayerNorm
        │
  take last timestep of H and C
        │
  concat(h_last, c_last)           → (B, 256)
        │
  FC(256,ReLU) → FC(256,ReLU) → FC(action_dim, Tanh)   [Actor]
  FC(256,ReLU) → FC(256,ReLU) → FC(1)                   [Critic ×2]
```

---

## Key Parameters (Table 1 of paper)

| Parameter | Value |
|-----------|-------|
| Actor / Critic LR | 0.0003 |
| Discount γ | 0.87 |
| MHA heads h | 4 |
| LSTM hidden | 128 (→ 256 after concat) |
| FC hidden dim | 256 |
| Sequence window T | 8 steps |
| Safety distance S₁ | 3.5 n mile |
| Radar radius | 4.5 n mile |
| Arena radius R_A | 1.8 n mile |
| Arrival reward | +1000 |
| Collision reward | −750 |

---

## Reward Function (Eq.20–25)

```
R = R_g + R_s + R_c + R_i + R_a
```

| Term | Description |
|------|-------------|
| R_g | Guidance — penalise Euclidean distance to goal |
| R_s | Safety — exponential penalty as TS approaches |
| R_c | COLREGs — reward/penalise clauses 13–17 compliance |
| R_i | Immediate danger — override COLREGs when ship domain violated |
| R_a | Terminal — +1000 arrival, −750 collision |

---

## State & Action Space (Eq.19)

- **State** (52-dim): `[x_o, y_o, x_g, y_g, χ₁ … χ₄₈]`
- **ATL-TD3** feeds a **T=8 step window** of states to the LSTM
- **Action** (1-dim): rudder angle ∈ [−20°, 20°], network output ∈ [−1, 1]

---

## Expected Performance

| Algorithm | Cases 1–10 | Cases 11–20 | Overall |
|-----------|-----------|------------|---------|
| TD3       | ~90%      | ~60–70%    | ~75%    |
| ATL-TD3   | ~100%     | ~90–100%   | ~95%+   |

The paper reports ATL-TD3 convergence **47%, 36%, 28% faster** than TD3
in 2-, 3-, 4-USV environments (Section 4.2, Fig.8).

---

## Coordinate System

- World frame: `x = East`, `y = North` (math angles, CCW positive).
- Convert compass headings: `heading_to_math(deg)` in `imazu_scenarios.py`.
- Distances in **nautical miles**, time step `dt = 1.0 s`.
