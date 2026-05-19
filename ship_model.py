"""
3-DOF Nonlinear Ship Motion Model
Based on: Fossen (2011), parameters from Table 1 of the paper.
State: eta = [x, y, psi], nu = [u, v, r]
Control input: rudder angle delta (degrees)
"""

import numpy as np


class ShipModel:
    """
    3-DOF (surge, sway, yaw) nonlinear USV motion model.
    Hydrodynamic derivatives from Table 1 of the paper.
    """

    def __init__(self, dt=1.0):
        self.dt = dt  # time step (seconds)

        # Ship length (used for ship domain scaling)
        self.L = 1.0  # normalized, actual scale handled by domain model

        # Hydrodynamic derivatives (from Table 1)
        self.Xu_dot = -0.0075
        self.Yv_dot = -0.1553
        self.Yu_dot = -0.1553   # NOTE: paper lists Yu_dot = Yv_dot
        self.Nv_dot = -0.0007927
        self.Nr_dot = -0.0074

        self.Xu    = -3.5655
        self.X_uu  = -0.0697    # X|u|u
        self.Xuuu  = -0.0697    # Xuuu (same value in table)

        self.Yv    = -0.3032
        self.Y_vv  = -0.7834    # Y|v|v
        self.Y_vr  = -0.2099    # Y|v|r
        self.Y_rr  = -0.0543    # Y|r|r

        self.Nv    = -0.0999
        self.N_vv  = -0.1822    # N|v|v
        self.N_rv  = -0.1561    # N|r|v
        self.N_vr  = -0.0561    # N|v|r
        self.N_rr  = -0.3390    # N|r|r

        self.Iz_dot = -0.0120
        self.Yr     =  0.0832
        self.Nr     = -0.0455

        # Inertia matrix M (3x3), surge-sway-yaw
        # M = M_RB + M_A  (rigid body + added mass)
        # Simplified: M diagonal dominant, off-diag from added mass
        # Using non-dimensional form common in literature
        m  = 1.0   # non-dimensional mass
        Iz = 0.05  # non-dimensional moment of inertia

        # Added mass terms (negative of derivatives)
        Xu_a  = -self.Xu_dot
        Yv_a  = -self.Yv_dot
        Nr_a  = -self.Nr_dot
        Nv_a  = -self.Nv_dot
        Yr_a  = 0.0

        self.M = np.array([
            [m - Xu_a,     0,          0       ],
            [0,         m - Yv_a,   -Nv_a      ],
            [0,        -Nv_a,      Iz - Nr_a   ]
        ])
        self.M_inv = np.linalg.inv(self.M)

        # Steering force coefficients (rudder)
        # From paper: tau = [X_delta*delta, Y_delta*delta, N_delta*delta]
        # Tuned for reasonable response
        self.X_delta = 0.0
        self.Y_delta = 0.35
        self.N_delta = -0.20

        # Design speed (knots -> non-dim)
        self.u0 = 0.5  # nominal surge speed (non-dimensional)

    def rotation_matrix(self, psi):
        """Rotation matrix from body to world frame."""
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    def C_matrix(self, nu):
        """Coriolis-centripetal matrix (simplified, body-frame)."""
        u, v, r = nu
        m  = 1.0
        Iz = 0.05
        Yv_a = -self.Yv_dot
        Xu_a = -self.Xu_dot
        Nr_a = -self.Nr_dot
        Nv_a = -self.Nv_dot

        C = np.array([
            [0,                  0,     -(m - Yv_a)*v - Nv_a*r],
            [0,                  0,      (m - Xu_a)*u          ],
            [(m - Yv_a)*v + Nv_a*r, -(m - Xu_a)*u,   0        ]
        ])
        return C

    def D_matrix(self, nu):
        """Nonlinear damping matrix."""
        u, v, r = nu
        d11 = -self.Xu - self.X_uu*abs(u) - self.Xuuu*u**2
        d22 = -self.Yv - self.Y_vv*abs(v) - self.Y_vr*abs(r)
        d23 = -self.Y_rr*abs(r)
        d32 = -self.N_vv*abs(v) - self.N_rv*abs(r)
        d33 = -self.Nr  - self.N_vr*abs(v) - self.N_rr*abs(r)

        D = np.array([
            [d11,  0,    0  ],
            [0,   d22,  d23 ],
            [0,   d32,  d33 ]
        ])
        return D

    def step(self, eta, nu, delta_deg, tau_w=None):
        """
        Integrate one time step.
        eta: [x, y, psi]  world frame position + heading
        nu:  [u, v, r]    body frame velocities
        delta_deg: rudder angle in degrees  [-20, 20]
        tau_w: environmental disturbance (optional, 3-vector)
        Returns: eta_new, nu_new
        """
        delta = np.deg2rad(np.clip(delta_deg, -20.0, 20.0))

        # Control forces
        tau = np.array([
            self.X_delta * delta,
            self.Y_delta * delta,
            self.N_delta * delta
        ])

        if tau_w is None:
            tau_w = np.zeros(3)

        # Equations of motion: M * nu_dot = tau - C*nu - D*nu + tau_w
        C = self.C_matrix(nu)
        D = self.D_matrix(nu)
        nu_dot = self.M_inv @ (tau - C @ nu - D @ nu + tau_w)

        # Euler integration
        nu_new = nu + self.dt * nu_dot

        # Clamp velocities to physical limits
        nu_new[0] = np.clip(nu_new[0], 0.05, 1.5)   # surge: forward only
        nu_new[1] = np.clip(nu_new[1], -0.5, 0.5)   # sway
        nu_new[2] = np.clip(nu_new[2], -0.3, 0.3)   # yaw rate

        # Kinematics: eta_dot = R(psi) * nu
        R = self.rotation_matrix(eta[2])
        eta_dot = R @ nu_new
        eta_new = eta.copy()
        eta_new[:2] += self.dt * eta_dot[:2]
        eta_new[2]  += self.dt * eta_dot[2]
        eta_new[2]   = self._wrap_angle(eta_new[2])

        return eta_new, nu_new

    @staticmethod
    def _wrap_angle(angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def reset(self, x=0.0, y=0.0, psi=0.0, u=None):
        """Initialize state."""
        if u is None:
            u = self.u0
        eta = np.array([x, y, psi], dtype=float)
        nu  = np.array([u, 0.0, 0.0], dtype=float)
        return eta, nu
