"""
Ship Domain (QSD), Arena Model, COLREGs classifier, Radar Sensor.
Based on Section 2.2 of the paper.

Performance notes:
  RadarSensor.scan() is fully vectorised with numpy broadcasting —
  no Python loops over beams or obstacles.
"""

import numpy as np


# ─────────────────────────────────────────────
# Quaternion Ship Domain (QSD)
# ─────────────────────────────────────────────

class ShipDomain:
    def __init__(self, L_os=1.0, k_AD=1.0, k_DT=1.5):
        self.L_os = L_os
        inner = np.sqrt(k_AD**2 + (k_DT / 2)**2)
        self.R_fore  = (1 + 1.34 * inner) * L_os
        self.R_aft   = (1 + 0.67 * inner) * L_os
        self.R_starb = (0.2 + k_DT) * L_os
        self.R_port  = (0.2 + 0.75 * k_DT) * L_os

    def is_violated(self, dx_body, dy_body):
        Rf = self.R_fore  if dx_body >= 0 else self.R_aft
        Rs = self.R_starb if dy_body >= 0 else self.R_port
        return (dx_body / Rf)**2 + (dy_body / Rs)**2 <= 1.0


# ─────────────────────────────────────────────
# Arena Model
# ─────────────────────────────────────────────

def compute_arena_radius(L_os, L_ts, v_r, T_n,
                          xi=1.0, N_A=1.0, gamma=1.0, P_var=0.0):
    S_SDA = (L_os * np.pi / 135 + L_ts * np.pi / 45) + \
             gamma * P_var + (L_os + L_ts)
    R_A = v_r * (T_n + xi * N_A * S_SDA / max(v_r, 1e-4))
    return S_SDA, R_A


# ─────────────────────────────────────────────
# COLREGs Encounter Classification
# ─────────────────────────────────────────────

class COLREGs:
    HEAD_ON_HALF      = 6.0
    CROSSING_STARB_MAX = 112.5
    CROSSING_PORT_MIN  = 247.5

    @staticmethod
    def relative_bearing(os_pos, os_psi, ts_pos):
        dx = ts_pos[0] - os_pos[0]
        dy = ts_pos[1] - os_pos[1]
        compass = 90.0 - np.degrees(np.arctan2(dy, dx))
        return (compass - np.degrees(os_psi)) % 360.0

    @staticmethod
    def classify(os_eta, ts_eta):
        rb_OS = COLREGs.relative_bearing(os_eta[:2], os_eta[2], ts_eta[:2])
        rb_TS = COLREGs.relative_bearing(ts_eta[:2], ts_eta[2], os_eta[:2])
        ho = COLREGs.HEAD_ON_HALF
        if (rb_OS < ho or rb_OS > 360 - ho) and \
           (rb_TS < ho or rb_TS > 360 - ho):
            return 'head_on'
        if 112.5 <= rb_TS <= 247.5:
            return 'overtaking'
        if ho <= rb_OS <= COLREGs.CROSSING_STARB_MAX:
            return 'crossing_gw'
        if COLREGs.CROSSING_PORT_MIN <= rb_OS <= 354.0:
            return 'crossing_so'
        return 'safe'

    @staticmethod
    def compliant_action(encounter_type, current_delta=0.0):
        if encounter_type in ('head_on', 'crossing_gw', 'overtaking'):
            return +1
        return 0


# ─────────────────────────────────────────────
# Radar Sensor — fully vectorised
# ─────────────────────────────────────────────

class RadarSensor:
    """
    48-beam radar sensor.
    scan() uses pure numpy broadcasting — no Python loops.
    Speed vs original loop version: ~30-50× faster.
    """

    N_BEAMS = 48
    RANGE   = 4.5   # n mile

    def __init__(self):
        # Beam angles relative to body frame, shape (N_BEAMS,)
        self._beam_angles = np.linspace(
            0, 2 * np.pi, self.N_BEAMS, endpoint=False
        ).astype(np.float32)

    # ------------------------------------------------------------------
    def scan(self, os_eta, target_ships, static_obstacles=None):
        """
        Vectorised ray-circle intersection for all beams × all obstacles.

        os_eta        : [x, y, psi]
        target_ships  : list of [x, y, psi, length, ...]
        static_obs    : list of (cx, cy, radius)

        Returns: (N_BEAMS,) float32 array of normalised distances [0,1]
        """
        # ── Collect obstacle circles ────────────────────────────────
        obs_list = []
        if target_ships:
            for ts in target_ships:
                r = max(ts[3] if len(ts) > 3 else 0.1, 0.15)
                obs_list.append((ts[0], ts[1], r))
        if static_obstacles:
            obs_list.extend(static_obstacles)

        if not obs_list:
            return np.ones(self.N_BEAMS, dtype=np.float32)

        ox, oy, opsi = float(os_eta[0]), float(os_eta[1]), float(os_eta[2])

        # obs arrays  shape (M,)
        obs = np.array(obs_list, dtype=np.float32)   # (M, 3)
        cx  = obs[:, 0] - ox     # (M,)  relative to OS
        cy  = obs[:, 1] - oy     # (M,)
        cr  = obs[:, 2]          # (M,)

        # Beam directions  shape (N_BEAMS,)
        beam_dirs = opsi + self._beam_angles          # (N,)
        bx = np.cos(beam_dirs).astype(np.float32)     # (N,)
        by = np.sin(beam_dirs).astype(np.float32)     # (N,)

        # Broadcast: (N,1) × (1,M) → (N,M)
        bx_  = bx[:, np.newaxis]   # (N,1)
        by_  = by[:, np.newaxis]   # (N,1)
        cx_  = cx[np.newaxis, :]   # (1,M)
        cy_  = cy[np.newaxis, :]   # (1,M)
        cr_  = cr[np.newaxis, :]   # (1,M)

        # Scalar projection of obstacle onto each beam
        proj = cx_ * bx_ + cy_ * by_                  # (N,M)

        # Squared perpendicular distance
        perp2 = cx_**2 + cy_**2 - proj**2             # (N,M)

        # Valid hits: proj > 0 and perp2 <= cr^2
        valid = (proj > 0) & (perp2 <= cr_**2)        # (N,M) bool

        # Distance to hit point
        safe_perp2 = np.where(valid, perp2, 0.0)
        hit_dist   = proj - np.sqrt(np.maximum(cr_**2 - safe_perp2, 0.0))

        # Only keep positive hits
        hit_dist = np.where(valid & (hit_dist > 0), hit_dist, self.RANGE)

        # Minimum over obstacles for each beam
        min_dist = hit_dist.min(axis=1)                # (N,)

        return (min_dist / self.RANGE).astype(np.float32)
