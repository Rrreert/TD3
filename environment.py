"""
USV Collision Avoidance Environment
Implements the full reward function (Eq.20-25) and state/action spaces
from Section 3.2-3.3 of the paper.

State space (Eq.19):  S = [x_o, y_o, x_g, y_g, chi_1..chi_48]  → 52 dims
Action space:  continuous rudder angle in [-20°, 20°], resolution 0.5°

Reward components:
  R_g  – guidance reward (Eq.20)
  R_s  – safety reward  (Eq.21)
  R_c  – COLREGs reward (Eq.22)
  R_i  – immediate danger reward (Eq.23)
  R_a  – arrival / collision reward (Eq.24)
"""

import numpy as np
from ship_model  import ShipModel
from colregs_model import ShipDomain, RadarSensor, COLREGs, compute_arena_radius


# ─────────────────────────────────────────────
# Target Ship (simple kinematic model)
# ─────────────────────────────────────────────

class TargetShip:
    """Constant-heading, constant-speed target ship."""

    def __init__(self, x, y, psi, speed, length=0.15, dt=1.0):
        self.eta   = np.array([x, y, psi], dtype=float)
        self.speed = speed
        self.length = length
        self.dt    = dt
        # Simple kinematic: straight-line motion
        self.vx = speed * np.cos(psi)
        self.vy = speed * np.sin(psi)

    def step(self):
        self.eta[0] += self.vx * self.dt
        self.eta[1] += self.vy * self.dt

    @property
    def pos(self):
        return self.eta[:2].copy()

    @property
    def psi(self):
        return self.eta[2]


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

class USVEnv:
    """
    Single-episode USV collision avoidance environment.

    Parameters (from Table 1 of the paper):
      safety_dist  = 3.5 n mile   (S1)
      radar_range  = 4.5 n mile   (R_radar)
      arena_radius = 1.8 n mile   (R_A)
      arrival_r    = 1000         (r_arrival)
      collision_r  = -750         (r_collision)
      lambda_g     = 1.0          (guidance weight)
      lambda_s     = 0.77
      lambda_c1    = 0.58
      lambda_c2    = 0.73
      lambda_i     = 0.94
    """

    # --- dimensions ---
    STATE_DIM  = 52   # [x_o, y_o, x_g, y_g, 48 radar beams]
    ACTION_DIM = 1    # rudder angle

    # --- from Table 1 ---
    RADAR_RANGE   = 4.5
    SAFETY_DIST   = 3.5
    ARENA_RADIUS  = 1.8
    ARRIVAL_R     =  1000.0
    COLLISION_R   =  -750.0
    LAMBDA_G      =  1.0
    LAMBDA_S      =  0.77
    LAMBDA_C1     =  0.58
    LAMBDA_C2     =  0.73
    LAMBDA_I      =  0.94
    SIGMA_S       =  5.0    # positive safety reward (bonus for safe passing)
    SIGMA_C       =  3.0    # positive COLREGs reward
    RS_THRESHOLD  =  2.0    # safety reward range threshold (r_s)

    ARRIVAL_DIST  = 0.3     # n mile – goal reached threshold
    MAX_STEPS     = 2000

    def __init__(self, target_ships_config=None, goal=None,
                 os_init=None, dt=1.0, wind_wave=False):
        """
        target_ships_config: list of dicts with keys
            x, y, psi (rad), speed, length
        goal: [x_g, y_g]
        os_init: dict with x, y, psi, speed
        """
        self.dt = dt
        self.wind_wave = wind_wave

        self.ship      = ShipModel(dt=dt)
        self.radar     = RadarSensor()
        self.domain    = ShipDomain(L_os=0.15, k_AD=1.0, k_DT=1.5)

        # Scenario setup
        self._ts_config  = target_ships_config or []
        self._goal_cfg   = goal
        self._os_init    = os_init

        self.reset()

    # ─────────────── reset ───────────────

    def reset(self):
        cfg = self._os_init or {}
        x   = cfg.get('x',   0.0)
        y   = cfg.get('y',   0.0)
        psi = cfg.get('psi', 0.0)
        spd = cfg.get('speed', 0.5)

        self.eta, self.nu = self.ship.reset(x, y, psi, u=spd)
        self.delta = 0.0   # current rudder angle (deg)

        # Goal
        if self._goal_cfg is not None:
            self.goal = np.array(self._goal_cfg, dtype=float)
        else:
            # default: straight ahead 8 n mile
            self.goal = np.array([x + 8.0 * np.cos(psi),
                                   y + 8.0 * np.sin(psi)])

        # Spawn target ships
        self.targets = []
        for cfg_ts in self._ts_config:
            ts = TargetShip(
                x=cfg_ts['x'], y=cfg_ts['y'],
                psi=cfg_ts['psi'], speed=cfg_ts['speed'],
                length=cfg_ts.get('length', 0.15),
                dt=self.dt
            )
            self.targets.append(ts)

        self.step_count = 0
        self.done       = False
        return self._get_obs()

    # ─────────────── step ───────────────

    def step(self, action):
        """
        action: float in [-1, 1]  → mapped to [-20°, 20°] rudder
        Returns: obs, reward, done, info
        """
        assert not self.done, "Call reset() before step()."
        self.step_count += 1

        # Map action to rudder angle
        delta_cmd = float(action) * 20.0   # [-20, 20] deg
        # Smooth rudder rate limit (optional realism)
        max_rate = 3.0  # deg/step
        self.delta = np.clip(
            delta_cmd,
            self.delta - max_rate,
            self.delta + max_rate
        )
        self.delta = np.clip(self.delta, -20.0, 20.0)

        # Environmental disturbance
        tau_w = self._wind_wave_disturbance() if self.wind_wave else None

        # Propagate OS
        self.eta, self.nu = self.ship.step(self.eta, self.nu,
                                            self.delta, tau_w)

        # Propagate target ships
        for ts in self.targets:
            ts.step()

        # Compute reward components
        reward, info = self._compute_reward()

        # Termination
        if info.get('collision') or info.get('arrived'):
            self.done = True
        if self.step_count >= self.MAX_STEPS:
            self.done = True
            info['timeout'] = True

        obs = self._get_obs()
        return obs, reward, self.done, info

    # ─────────────── observation ───────────────

    def _get_obs(self):
        """State vector: [x_o, y_o, x_g, y_g, chi_1..chi_48] normalised."""
        ts_list = [[ts.eta[0], ts.eta[1], ts.eta[2], ts.length]
                   for ts in self.targets]

        radar_readings = self.radar.scan(self.eta, ts_list)

        # Normalise position by radar range
        scale = self.RADAR_RANGE
        obs = np.concatenate([
            [self.eta[0] / scale,
             self.eta[1] / scale,
             self.goal[0] / scale,
             self.goal[1] / scale],
            radar_readings
        ])
        return obs.astype(np.float32)

    # ─────────────── reward ───────────────

    def _compute_reward(self):
        info = {'collision': False, 'arrived': False,
                'colregs_violation': False}

        Rg = self._guidance_reward()
        Rs, collision = self._safety_reward()
        Rc, colregs_v = self._colregs_reward()
        Ri = self._immediate_danger_reward(colregs_v)
        Ra, arrived   = self._arrival_collision_reward(collision)

        info['collision']        = collision
        info['arrived']          = arrived
        info['colregs_violation'] = colregs_v

        R = Rg + Rs + Rc + Ri + Ra
        return float(R), info

    def _guidance_reward(self):
        """Eq.20 – penalise distance to goal."""
        dist = np.linalg.norm(self.eta[:2] - self.goal)
        return -self.LAMBDA_G * dist

    def _safety_reward(self):
        """Eq.21 – safety reward based on nearest TS distance."""
        collision = False
        Rs = 0.0
        for ts in self.targets:
            dT = np.linalg.norm(self.eta[:2] - ts.pos)
            if dT > self.RADAR_RANGE:
                continue  # outside radar range, no penalty
            if dT <= self.SAFETY_DIST:
                if dT > 0.05:   # ship domain (immediate danger threshold)
                    # Inside safety range but outside ship domain
                    Rs += -self.LAMBDA_S * np.exp(abs(dT)) / self.RS_THRESHOLD
                else:
                    # Collision
                    collision = True
                    Rs += self.COLLISION_R
            else:
                # Safe – small positive reward for exploring
                Rs += self.SIGMA_S * 0.01

        return Rs, collision

    def _colregs_reward(self):
        """Eq.22 – reward/penalise COLREGs compliance."""
        Rc = 0.0
        any_violation = False

        for ts in self.targets:
            dT = np.linalg.norm(self.eta[:2] - ts.pos)
            if dT > self.RADAR_RANGE:
                continue

            # Only active when TS has entered OS arena
            if dT >= self.ARENA_RADIUS:
                continue

            encounter = COLREGs.classify(self.eta, ts.eta)
            hint      = COLREGs.compliant_action(encounter, self.delta)

            # Check compliance: rudder should match hint direction
            compliant = self._is_colregs_compliant(hint)

            if dT > self.SAFETY_DIST:
                # Between arena and safety range
                inner = -self.LAMBDA_S * np.exp(abs(dT)) / \
                         self.RS_THRESHOLD - self.SIGMA_S
                if compliant:
                    Rc += self.SIGMA_C
                else:
                    Rc += self.LAMBDA_C1 * inner
                    any_violation = True
            else:
                # Inside safety range (close quarters)
                if not compliant:
                    Rc += -self.LAMBDA_C2 * abs(self.ARENA_RADIUS - dT)
                    any_violation = True
                else:
                    Rc += self.SIGMA_C

        return Rc, any_violation

    def _immediate_danger_reward(self, colregs_violated):
        """Eq.23 – override COLREGs when ship domain violated."""
        Ri = 0.0
        for ts in self.targets:
            dT = np.linalg.norm(self.eta[:2] - ts.pos)
            if dT >= self.ARENA_RADIUS:
                continue

            # Check if TS is inside OS ship domain
            dx_body, dy_body = self._world_to_body(ts.pos - self.eta[:2])
            domain_violated = self.domain.is_violated(dx_body, dy_body)

            if domain_violated and colregs_violated:
                # COLREGs departed + domain violated → penalise based on dist
                inner = (self.LAMBDA_C2 * abs(self.ARENA_RADIUS - dT)
                         - self.SIGMA_C)
                Ri += self.LAMBDA_I * inner

        return Ri

    def _arrival_collision_reward(self, collision):
        """Eq.24."""
        if collision:
            return self.COLLISION_R, False
        dist_goal = np.linalg.norm(self.eta[:2] - self.goal)
        if dist_goal <= self.ARRIVAL_DIST:
            return self.ARRIVAL_R, True
        return 0.0, False

    # ─────────────── helpers ───────────────

    def _is_colregs_compliant(self, hint):
        """
        hint: +1 starboard, -1 port, 0 any.
        Check current rudder direction matches.
        """
        if hint == 0:
            return True
        if hint == +1:
            return self.delta >= -2.0   # turning starboard (or neutral)
        if hint == -1:
            return self.delta <= 2.0
        return True

    def _world_to_body(self, vec_world):
        """Rotate 2D vector from world frame to OS body frame."""
        psi = self.eta[2]
        c, s = np.cos(psi), np.sin(psi)
        x_b =  c * vec_world[0] + s * vec_world[1]
        y_b = -s * vec_world[0] + c * vec_world[1]
        return x_b, y_b

    def _wind_wave_disturbance(self):
        """
        Simple wind/wave disturbance.
        Paper: 45° wave/wind direction, H_s=0.15m, T_p=5.5s, v_wind=0.45m/s.
        Returns tau_w (3-vector) in body frame.
        """
        wave_dir = np.deg2rad(45.0)
        # Project onto OS heading
        rel = wave_dir - self.eta[2]
        amplitude = 0.02
        tau_w = np.array([
            amplitude * np.cos(rel),
            amplitude * np.sin(rel),
            amplitude * 0.1 * np.sin(rel)
        ])
        return tau_w

    # ─────────────── rendering ───────────────

    def get_state_for_render(self):
        """Return dict for visualisation."""
        return {
            'os_pos':  self.eta[:2].copy(),
            'os_psi':  self.eta[2],
            'ts_pos':  [ts.pos for ts in self.targets],
            'ts_psi':  [ts.psi for ts in self.targets],
            'goal':    self.goal.copy(),
            'delta':   self.delta,
            'step':    self.step_count
        }
