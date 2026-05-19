# USV Autonomous Collision Avoidance — TD3 Reproduction

Reproduction of the **TD3 baseline** from:

> Cui et al., "Autonomous collision avoidance decision-making method for USV based on ATL-TD3 algorithm",  
> *Ocean Engineering* 312 (2024) 119297.

This code implements the standard TD3 algorithm (without LSTM / multi-head attention) and tests it on all **20 Imazu classic encounter scenarios**.

---

## File Structure

```
usv_td3/
├── ship_model.py       # 3-DOF nonlinear USV dynamics (Fossen 2011, Table 1)
├── colregs_model.py    # Ship domain (QSD), arena model, COLREGs classifier, radar sensor
├── environment.py      # RL environment — state/action/reward (Eq.19-25)
├── td3_agent.py        # TD3 algorithm — Actor, Critic, ReplayBuffer
├── imazu_scenarios.py  # All 20 Imazu case definitions (Fig.9)
├── train.py            # Training loop
├── evaluate.py         # Evaluation + trajectory/rudder plots
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.8+, PyTorch 1.12+.

---

## Quick Start

### 1. Train

```bash
# Standard training (5000 episodes, ~30-60 min on CPU, ~10 min on GPU)
python train.py

# With wind/wave disturbance
python train.py --wind-wave

# Custom episode count and save directory
python train.py --episodes 3000 --save-dir ./my_checkpoints

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pt --episodes 2000
```

Training prints every 50 episodes and evaluates on all 20 Imazu cases every 200 episodes.  
The **best model** (highest Imazu success rate) is saved to `checkpoints/best_model.pt`.

---

### 2. Evaluate

```bash
# Evaluate all 20 Imazu cases and plot trajectories
python evaluate.py --model checkpoints/best_model.pt

# Single case with detailed plot
python evaluate.py --model checkpoints/best_model.pt --case 11

# Save figures to PNG
python evaluate.py --model checkpoints/best_model.pt --save-fig
```

Output plots match the style of **Fig.9** (trajectories) and **Fig.10** (rudder angles) in the paper.

---

## Key Parameters (from Table 1 of the paper)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Actor learning rate | Ar | 0.0003 |
| Critic learning rate | Cr | 0.0003 |
| Discount rate | γ | 0.87 |
| Safety distance | S₁ | 3.5 n mile |
| Radar radius | R_radar | 4.5 n mile |
| Arena radius | R_A | 1.8 n mile |
| COLREGs weight (a) | λ_c1 | 0.58 |
| COLREGs weight (b) | λ_c2 | 0.73 |
| Safety reward weight | λ_s | 0.77 |
| Immediate danger weight | λ_i | 0.94 |
| Arrival reward | r_arrival | 1000 |
| Collision reward | r_collision | −750 |
| MHA heads | h | 4 (ATL version only) |

---

## State and Action Space

- **State** (52-dim, Eq.19):  
  `[x_o, y_o, x_g, y_g, χ₁, χ₂, ..., χ₄₈]`  
  — normalized OS position, goal position, and 48 radar detection beam readings.

- **Action** (1-dim):  
  Continuous rudder angle in `[-20°, 20°]`, output from Actor as `[-1, 1]` scaled by 20.

---

## Reward Function (Eq.20–25)

```
R = R_g + R_s + R_c + R_i + R_a
```

| Component | Role |
|-----------|------|
| R_g | Guidance — penalise distance to goal |
| R_s | Safety — penalise proximity to target ships |
| R_c | COLREGs — reward compliant manoeuvres |
| R_i | Immediate danger — override COLREGs when ship domain violated |
| R_a | Terminal — +1000 arrival, −750 collision |

---

## Imazu Test Cases

| Cases | Type |
|-------|------|
| 1–4   | Two-ship: head-on, overtaking, crossing (GW/SO) |
| 5–10  | Three-ship: mixed encounters |
| 11–20 | Four-ship (three TS): complex multi-ship situations |

---

## Expected Results

The paper reports that the **standard TD3** (without ATL enhancement):
- Passes most of the simpler cases (1–10).
- Shows **high collision risk in cases 11–20** (complex multi-ship).
- Exhibits longer navigation paths and rougher trajectories vs ATL-TD3.

This is consistent with Fig.11 and Fig.13 of the paper, where TD3 achieves ~75–80% success rate  
versus ATL-TD3's near-100% success.

**To improve success rate** on cases 11–20, consider:
1. Increasing training episodes to 8000–10000.
2. Increasing replay buffer size (default 1e5).
3. Tuning exploration noise downward after 3000 episodes.
4. Using prioritised experience replay.

---

## Tuning Guide

If the agent fails too many cases, try adjusting in `train.py`:

```python
EXPLORE_NOISE  = 0.10     # reduce from 0.15 for more stable late training
WARMUP_STEPS   = 2000     # more random exploration at start
BATCH_SIZE     = 512      # larger batches for stability
```

And in `td3_agent.py`:

```python
discount = 0.90    # slightly higher discount for long-horizon tasks
tau      = 0.003   # slower target network update
```

---

## Notes on Coordinate System

- World frame: `x = East`, `y = North` (math convention).
- Headings: radians, `0 = East`, CCW positive.  
  Use `heading_to_math(compass_deg)` from `imazu_scenarios.py` to convert.
- All distances in **nautical miles (n mile)**.
- Time step: `dt = 1.0 s`.

---

## Differences from Paper's ATL-TD3

This code reproduces only the **TD3 baseline**. The ATL-TD3 adds:
1. LSTM layer between input and hidden layers (retains historical state).
2. Multi-head self-attention (MHA, 4 heads) on top of LSTM output.
3. Optimised experience replay (Section 3.1 of paper).

To implement ATL-TD3, replace the `Actor` and `Critic` networks in `td3_agent.py`  
with LSTM+MHA variants, keeping all other training logic identical.
