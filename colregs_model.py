"""
Ship Domain (QSD) and Arena Model
COLREGs encounter classification (clauses 13-17)
Based on Section 2.2 of the paper.
"""

import numpy as np


# ─────────────────────────────────────────────
# Quaternion Ship Domain (QSD)
# ─────────────────────────────────────────────

class ShipDomain:
    """
    Quaternion Ship Domain from Wang (2010).
    Parameters calibrated for the USV scale used in the paper.
    Domain radii in nautical miles (n mile).
    """

    def __init__(self, L_os=1.0, k_AD=1.0, k_DT=1.5):
        """
        L_os : length of own ship (n mile scale, ~1 unit here)
        k_AD : coefficient of advance (default 1.0)
        k_DT : tactical diameter coefficient (default 1.5)
        """
        self.L_os = L_os
        self.k_AD = k_AD
        self.k_DT = k_DT
        self._compute_radii()

    def _compute_radii(self):
        kAD = self.k_AD
        kDT = self.k_DT
        L   = self.L_os
        inner = np.sqrt(kAD**2 + (kDT / 2)**2)
        self.R_fore  = (1 + 1.34 * inner) * L
        self.R_aft   = (1 + 0.67 * inner) * L
        self.R_starb = (0.2 + kDT) * L
        self.R_port  = (0.2 + 0.75 * kDT) * L

    def f_qsd(self, dx, dy):
        """
        QSD membership function f(x,y) <= 1 means inside domain.
        dx, dy: relative position of TS in OS body frame
                (x forward, y starboard)
        """
        def sgn(val):
            return 1.0 if val >= 0 else -1.0

        Rf = self.R_fore  if dx >= 0 else self.R_aft
        Rs = self.R_starb if dy >= 0 else self.R_port

        denom_x = (1 + sgn(dx)) * self.R_fore - (1 - sgn(dx)) * self.R_aft
        denom_y = (1 + sgn(dy)) * self.R_starb - (1 + sgn(dy)) * self.R_port

        # Use direct half-axis selection (cleaner formulation)
        term_x = (dx / Rf) ** 2
        term_y = (dy / Rs) ** 2
        return term_x + term_y

    def is_violated(self, dx_body, dy_body):
        """True if TS is inside OS ship domain."""
        return self.f_qsd(dx_body, dy_body) <= 1.0


# ─────────────────────────────────────────────
# Arena Model
# ─────────────────────────────────────────────

def compute_arena_radius(L_os, L_ts, v_r, T_n, xi=1.0,
                          N_A=1.0, gamma=1.0, P_var=0.0):
    """
    Arena radius from Eq.(5)-(6) of the paper.
    L_os, L_ts : ship lengths
    v_r        : relative speed
    T_n        : manoeuvring time constant
    Returns: S_SDA (safe domain size), R_A (arena radius)
    """
    S_SDA = (L_os * np.pi / 135 + L_ts * np.pi / 45) + \
             gamma * P_var + (L_os + L_ts)
    R_A = v_r * (T_n + xi * N_A * S_SDA / max(v_r, 1e-4))
    return S_SDA, R_A


# ─────────────────────────────────────────────
# COLREGs Encounter Classification
# Clauses 13-17, Chapter 2
# ─────────────────────────────────────────────

class COLREGs:
    """
    Classify encounter situation between OS and TS.
    Based on relative bearing (CBOS: compass bearing of TS from OS)
    and the TS heading relative to OS heading.

    Encounter zones (from Fig.2 of the paper):
      Head-on  (HO) : TS bearing ~0°, TS heading ~180° rel to OS => both
                       see each other ahead.  OS bearing from TS ~0°.
      Overtaking     : OS approaches TS from TS's stern arc [112.5°, 247.5°]
      Crossing give-way (CR1): TS on OS starboard [0°,112.5°]
      Crossing stand-on (CR2): TS on OS port      [247.5°,360°]
      Safe (SF)      : otherwise
    """

    # Angular boundaries (degrees, measured from OS heading, clockwise)
    HEAD_ON_HALF   = 6.0    # ±6° from dead ahead → head-on arc
    OVERTAKING_FWD = 112.5  # TS bearing > 112.5° and < 247.5° → OS overtaking
    CROSSING_PORT_MIN = 247.5
    CROSSING_STARB_MAX = 112.5

    @staticmethod
    def relative_bearing(os_pos, os_psi, ts_pos):
        """
        Bearing of TS as seen from OS, relative to OS heading [0,360).
        """
        dx = ts_pos[0] - os_pos[0]
        dy = ts_pos[1] - os_pos[1]
        abs_bearing = np.degrees(np.arctan2(dy, dx))
        # Convert to compass: north=0, clockwise positive
        compass = 90.0 - abs_bearing          # math->compass
        rel = (compass - np.degrees(os_psi)) % 360.0
        return rel

    @staticmethod
    def classify(os_eta, ts_eta, os_nu=None, ts_nu=None):
        """
        Returns one of: 'head_on', 'overtaking', 'crossing_gw',
                        'crossing_so', 'safe'
        os_eta: [x, y, psi]  OS world-frame state
        ts_eta: [x, y, psi]  TS world-frame state
        """
        # Bearing of TS from OS (relative to OS heading)
        rel_bearing_OS = COLREGs.relative_bearing(
            os_eta[:2], os_eta[2], ts_eta[:2])

        # Bearing of OS from TS (relative to TS heading) - for head-on check
        rel_bearing_TS = COLREGs.relative_bearing(
            ts_eta[:2], ts_eta[2], os_eta[:2])

        # Head-on: TS roughly ahead of OS AND OS roughly ahead of TS
        if (rel_bearing_OS < COLREGs.HEAD_ON_HALF or
                rel_bearing_OS > 360 - COLREGs.HEAD_ON_HALF):
            if (rel_bearing_TS < COLREGs.HEAD_ON_HALF or
                    rel_bearing_TS > 360 - COLREGs.HEAD_ON_HALF):
                return 'head_on'

        # Overtaking: OS approaches TS from TS's stern sector
        # i.e., OS bearing as seen from TS is in [112.5, 247.5]
        if 112.5 <= rel_bearing_TS <= 247.5:
            return 'overtaking'

        # Crossing give-way: TS on OS starboard [6, 112.5]
        if COLREGs.HEAD_ON_HALF <= rel_bearing_OS <= COLREGs.CROSSING_STARB_MAX:
            return 'crossing_gw'   # OS must give way (turn starboard)

        # Crossing stand-on: TS on OS port [247.5, 354]
        if COLREGs.CROSSING_PORT_MIN <= rel_bearing_OS <= 354.0:
            return 'crossing_so'   # OS stands on

        return 'safe'

    @staticmethod
    def compliant_action(encounter_type, current_delta=0.0):
        """
        Returns preferred rudder direction hint for COLREGs compliance.
        +1 → starboard (positive rudder), -1 → port, 0 → maintain
        """
        if encounter_type in ('head_on', 'crossing_gw', 'overtaking'):
            return +1   # alter course to starboard
        elif encounter_type == 'crossing_so':
            return 0    # stand on (maintain course)
        return 0


# ─────────────────────────────────────────────
# Radar sensor: 48 vector lines
# ─────────────────────────────────────────────

class RadarSensor:
    """
    48 radar detection lines as used in the paper (Eq.19).
    Each line returns distance to nearest obstacle (0..1 normalized).
    Radar range: 4.5 n mile (from Table 1).
    """

    N_BEAMS   = 48
    RANGE     = 4.5   # n mile

    def __init__(self):
        # Beam angles in body frame (0 = ahead, clockwise positive)
        self.angles = np.linspace(0, 2 * np.pi, self.N_BEAMS, endpoint=False)

    def scan(self, os_eta, target_ships, static_obstacles=None):
        """
        Returns array of 48 normalised distances [0,1].
        1.0 = no obstacle in that beam.
        os_eta: [x, y, psi]
        target_ships: list of [x, y, psi, length, beam]  (world frame)
        static_obstacles: list of (cx, cy, radius)
        """
        readings = np.ones(self.N_BEAMS)

        all_obs = []
        if target_ships:
            for ts in target_ships:
                # Approximate each ship as a circle of radius ~ half length
                r = max(ts[3] if len(ts) > 3 else 0.1, 0.15)
                all_obs.append((ts[0], ts[1], r))

        if static_obstacles:
            for obs in static_obstacles:
                all_obs.append(obs)

        if not all_obs:
            return readings

        ox, oy, opsi = os_eta

        for i, ang in enumerate(self.angles):
            beam_dir = opsi + ang   # world frame beam direction
            bx = np.cos(beam_dir)
            by = np.sin(beam_dir)

            min_dist = self.RANGE
            for (cx, cy, cr) in all_obs:
                # Ray-circle intersection
                dx = cx - ox
                dy = cy - oy
                proj = dx * bx + dy * by
                if proj < 0:
                    continue
                perp2 = dx**2 + dy**2 - proj**2
                if perp2 > cr**2:
                    continue
                hit = proj - np.sqrt(max(cr**2 - perp2, 0.0))
                if 0 < hit < min_dist:
                    min_dist = hit

            readings[i] = min_dist / self.RANGE

        return readings
