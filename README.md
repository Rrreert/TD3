# USV Autonomous Collision Avoidance — TD3 & ATL-TD3 Reproduction

Reproduction of both the **TD3 baseline** and the **ATL-TD3 proposed method** from:

> Cui et al., "Autonomous collision avoidance decision-making method for USV based on ATL-TD3 algorithm"  
> *Ocean Engineering* 312 (2024) 119297

Tested on all **20 Imazu classic encounter scenarios**.

---

## File Structure

```
usv_td3/
├── ship_model.py       # 3-DOF nonlinear USV dynamics (Fossen 2011, Table 1)
├── colregs_model.py    # QSD ship domain, arena model, COLREGs classifier, 48-beam radar
├── environment.py      # RL environment — reward function (Eq.19-25), state/action spaces
├── td3_agent.py        # Standard TD3 — Actor/Critic FC networks + ReplayBuffer
├── atl_td3_agent.py    # ATL-TD3 — MHA-LSTM networks + SequenceReplayBuffer + EpisodeWindow
├── imazu_scenarios.py  # All 20 Imazu case definitions
├── train.py            # Unified training script (--algo td3 | atl_td3)
├── evaluate.py         # Evaluation + trajectory/rudder plots + comparison mode
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
# Train standard TD3
python train.py --algo td3 --episodes 5000

# Train ATL-TD3 (proposed method)
python train.py --algo atl_td3 --episodes 5000

# With wind/wave disturbance (paper: 45° dir, H_s=0.15m, T_p=5.5s)
python train.py --algo atl_td3 --wind-wave

# Resume from checkpoint
python train.py --algo atl_td3 --resume checkpoints/atl_td3_best.pt
```

Models are saved to `./checkpoints/`:
- `td3_best.pt` / `td3_final.pt`
- `atl_td3_best.pt` / `atl_td3_final.pt`

Evaluation on 20 Imazu cases runs every 200 episodes; best-so-far is saved automatically.

---

## Evaluation

```bash
# Evaluate TD3 on all 20 cases
python evaluate.py --algo td3 --model checkpoints/td3_best.pt

# Evaluate ATL-TD3 on all 20 cases + save figures
python evaluate.py --algo atl_td3 --model checkpoints/atl_td3_best.pt --save-fig

# Single case detail plot
python evaluate.py --algo atl_td3 --model checkpoints/atl_td3_best.pt --case 11

# Side-by-side comparison on one case
python evaluate.py --compare \
    --td3-model    checkpoints/td3_best.pt \
    --atltd3-model checkpoints/atl_td3_best.pt \
    --case 11 --save-fig
```

Output figures reproduce **Fig.9** (trajectories) and **Fig.10** (rudder angles) of the paper.

---

## ATL-TD3 Architecture (Section 3.1, Fig.5–7)

### MHA-LSTM Block

```
Input sequence (B, T, state_dim)
        │
        ▼
  LSTM (hidden=128)           ← Eq.11-15: input/forget/output gates
        │
        ▼ H (B, T, 128)
  Multi-Head Self-Attention   ← Eq.16-18: 4 heads, causal mask
  (h=4, embed=128)
        │
        ▼ C (B, T, 128)
  Residual: H + C
        │
  take last timestep
  concat(h_last, c_last)
        │
        ▼ (B, 256)
```

### Actor

```
MHALSTMBlock(state_dim → 256)
→ FC(256, ReLU) → FC(256, ReLU) → FC(action_dim, Tanh)
```

### Critic (×2)

```
MHALSTMBlock(state_dim → 256)
concat with action (256+1=257)
→ FC(256, ReLU) → FC(256, ReLU) → FC(1)
```

---

## Key Parameters (Table 1)

| Parameter | Value |
|-----------|-------|
| Actor / Critic LR | 0.0003 |
| Discount γ | 0.87 |
| MHA heads h | **4** |
| LSTM hidden | 128 (→ 256 after concat) |
| FC hidden | 256 |
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
| R_s | Safety — exponential penalty when TS enters radar range |
| R_c | COLREGs — reward/penalise compliance with clauses 13–17 |
| R_i | Immediate danger — override COLREGs when ship domain violated |
| R_a | Terminal — +1000 arrival, −750 collision |

---

## State & Action Space (Eq.19)

- **State** (52-dim): `[x_o, y_o, x_g, y_g, χ₁ … χ₄₈]`
  - Normalised OS position, goal position, 48 radar beam readings
  - ATL-TD3 feeds a **window of T=8 consecutive states** to the LSTM

- **Action** (1-dim): rudder angle ∈ [−20°, 20°], output as [−1, 1]×20

---

## Expected Performance

| Algorithm | Cases 1–10 | Cases 11–20 | Overall |
|-----------|-----------|------------|---------|
| TD3       | ~90%      | ~60–70%    | ~75%    |
| ATL-TD3   | ~100%     | ~90–100%   | ~95%+   |

The paper reports ATL-TD3 convergence speed enhanced by **47%, 36%, 28%** over TD3  
in 2-, 3-, and 4-USV training environments respectively (Section 4.2).

---

## Coordinate System

- World frame: `x = East`, `y = North` (math convention).
- Headings: radians, `0 = East`, CCW positive.  
  Convert compass headings via `heading_to_math(deg)` in `imazu_scenarios.py`.
- Distances in **nautical miles (n mile)**, time step `dt = 1.0 s`.
