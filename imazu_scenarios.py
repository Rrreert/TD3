"""
20 Imazu Classic Encounter Scenarios
Reference: Woerner et al. (2016) and Fig.2, Fig.9 of the paper.

Coordinate system:
  - World frame: x = East, y = North  (math convention)
  - All positions in nautical miles (n mile)
  - All headings in radians (math convention, 0 = East, CCW positive)
  - Ship speeds in n mile / step  (calibrated to match paper scale)

OS always starts at (0, -4) heading North (psi = pi/2).
Goal is at (0, +4).

Target ships are configured to create each encounter type.

Cases 1-4:    Two-ship encounters (HO, OT, CR)
Cases 5-10:   Three-ship encounters
Cases 11-20:  Four-ship (three TS) complex encounters
"""

import numpy as np

# Convenient angle helpers
def deg2rad(d):
    return np.deg2rad(d)

def heading_to_math(compass_deg):
    """Convert compass heading (N=0, CW) to math angle (E=0, CCW)."""
    return np.deg2rad(90.0 - compass_deg)


# OS initial configuration (same for all cases)
OS_INIT = {
    'x':     0.0,
    'y':    -4.0,
    'psi':   heading_to_math(0.0),   # heading North
    'speed': 0.40
}
OS_GOAL = [0.0, 4.5]

TS_SPEED = 0.30   # target ship speed (n mile / step)
OS_SPEED = 0.40


def make_ts(x, y, compass_heading_deg, speed=None):
    """Helper to create a target ship config dict."""
    spd = speed if speed is not None else TS_SPEED
    return {
        'x':      float(x),
        'y':      float(y),
        'psi':    heading_to_math(compass_heading_deg),
        'speed':  float(spd),
        'length': 0.15
    }


# ─────────────────────────────────────────────
# Case definitions
# ─────────────────────────────────────────────

IMAZU_CASES = {

    # ── Two-ship encounters (cases 1-4) ──────────────────────────

    # Case 1: Head-on (HO) – TS coming straight from ahead
    1: {
        'desc': 'Two-ship Head-on',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts(0.0, 3.5, 180.0)   # TS heading South, meeting OS
        ]
    },

    # Case 2: Overtaking – OS overtakes TS (TS slower, same direction)
    2: {
        'desc': 'Two-ship Overtaking',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts(0.0, 0.5, 0.0, speed=0.15)   # TS heading North slowly
        ]
    },

    # Case 3: Crossing give-way – TS on OS starboard
    3: {
        'desc': 'Two-ship Crossing (OS give-way, TS starboard)',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts(3.5, 0.0, 270.0)   # TS heading West, crosses ahead
        ]
    },

    # Case 4: Crossing stand-on – TS on OS port
    4: {
        'desc': 'Two-ship Crossing (OS stand-on, TS port)',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts(-3.5, 0.0, 90.0)   # TS heading East, crosses ahead
        ]
    },

    # ── Three-ship encounters (cases 5-10) ───────────────────────

    # Case 5: Head-on + overtaking
    5: {
        'desc': 'Three-ship: HO + OT',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 0.0,  3.5, 180.0),            # TS-01 HO
            make_ts(-0.5, -1.5,   0.0, speed=0.15) # TS-02 being overtaken
        ]
    },

    # Case 6: Head-on + crossing (starboard behind)
    6: {
        'desc': 'Three-ship: HO + CR starboard behind',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 0.0,  3.5, 180.0),   # TS-01 HO
            make_ts( 3.0, -2.0, 315.0)    # TS-02 from starboard rear
        ]
    },

    # Case 7: Crossing + overtaking (starboard)
    7: {
        'desc': 'Three-ship: CR + OT starboard',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 3.5,  0.0, 270.0),            # TS-01 crossing starboard
            make_ts( 0.5, -1.5,   0.0, speed=0.15) # TS-02 being overtaken
        ]
    },

    # Case 8: Head-on + crossing both sides
    8: {
        'desc': 'Three-ship: HO + CR both sides',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 0.0,  3.5, 180.0),   # TS-01 HO
            make_ts( 3.0,  0.5, 270.0),   # TS-02 crossing starboard
            # only 2 TS for case 8 in original
        ]
    },

    # Case 9: Three crossing ships from different bearings
    9: {
        'desc': 'Three-ship: multiple crossings',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 3.0,  1.0, 270.0),   # TS-01 crossing starboard
            make_ts(-3.0,  1.0,  90.0),   # TS-02 crossing port
        ]
    },

    # Case 10: Overtaking + crossing (starboard behind)
    10: {
        'desc': 'Three-ship: OT + CR starboard behind',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 0.2, -1.5,   0.0, speed=0.12), # TS-01 being overtaken
            make_ts( 3.0, -3.0, 330.0)               # TS-02 from starboard
        ]
    },

    # ── Complex encounters (cases 11-20, three TS) ───────────────

    # Case 11: Head-on + two crossings
    11: {
        'desc': 'Four-ship: HO + two crossings',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 0.0,  3.5, 180.0),   # TS-01 HO
            make_ts( 3.5,  0.0, 270.0),   # TS-02 crossing starboard
            make_ts(-3.5,  0.5,  90.0),   # TS-03 crossing port
        ]
    },

    # Case 12: HO + CR starboard + OT
    12: {
        'desc': 'Four-ship: HO + CR starboard + OT',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 0.0,  3.5, 180.0),             # TS-01 HO
            make_ts( 3.5,  0.5, 270.0),             # TS-02 crossing starboard
            make_ts( 0.3, -2.0,   0.0, speed=0.12), # TS-03 overtaking
        ]
    },

    # Case 13: CR starboard + OT + CR port
    13: {
        'desc': 'Four-ship: CR starboard + OT + CR port',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 3.5,  0.5, 270.0),             # TS-01 crossing starboard
            make_ts( 0.3, -2.0,   0.0, speed=0.12), # TS-02 being overtaken
            make_ts(-3.5,  0.5,  90.0),             # TS-03 crossing port
        ]
    },

    # Case 14: HO + CR starboard + CR from behind starboard
    14: {
        'desc': 'Four-ship: HO + CR starboard + CR behind starboard',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 0.0,  3.5, 180.0),   # TS-01 HO
            make_ts( 3.0,  1.5, 250.0),   # TS-02 crossing starboard
            make_ts( 3.0, -3.0, 330.0),   # TS-03 from starboard rear
        ]
    },

    # Case 15: HO + OT + CR starboard ahead
    15: {
        'desc': 'Four-ship: HO + OT + CR ahead',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 0.0,  3.5, 180.0),             # TS-01 HO
            make_ts( 0.3, -2.5,   0.0, speed=0.12), # TS-02 being overtaken
            make_ts( 2.5,  2.0, 240.0),             # TS-03 crossing ahead starboard
        ]
    },

    # Case 16: Three ships: two CR + one OT
    16: {
        'desc': 'Four-ship: two CR + OT (complex)',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 3.5,  0.0, 270.0),             # TS-01 crossing starboard
            make_ts(-2.5,  1.5,  60.0),             # TS-02 crossing port
            make_ts( 0.3, -2.5,   0.0, speed=0.12), # TS-03 being overtaken
        ]
    },

    # Case 17: Two HO + one CR
    17: {
        'desc': 'Four-ship: two HO + CR',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts(-0.5,  3.5, 180.0),   # TS-01 HO (slight offset)
            make_ts( 0.5,  3.0, 200.0),   # TS-02 near HO
            make_ts( 3.5,  0.5, 270.0),   # TS-03 crossing starboard
        ]
    },

    # Case 18: CR + HO + CR from different angles
    18: {
        'desc': 'Four-ship: CR + HO + CR (angled)',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 2.5,  2.5, 225.0),   # TS-01 crossing diag starboard
            make_ts( 0.0,  3.5, 180.0),   # TS-02 HO
            make_ts(-2.5,  2.5, 135.0),   # TS-03 crossing diag port
        ]
    },

    # Case 19: Dense multi-ship – three approaching from ahead sector
    19: {
        'desc': 'Four-ship: dense ahead sector',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts(-0.8,  3.0, 150.0),   # TS-01 slightly port ahead
            make_ts( 0.0,  3.5, 180.0),   # TS-02 dead ahead
            make_ts( 0.8,  3.0, 210.0),   # TS-03 slightly starboard ahead
        ]
    },

    # Case 20: Completely surrounded (port, starboard, ahead)
    20: {
        'desc': 'Four-ship: surrounded (all quadrants)',
        'os':   OS_INIT,
        'goal': OS_GOAL,
        'targets': [
            make_ts( 0.0,  3.5, 180.0),   # TS-01 HO ahead
            make_ts( 3.5,  0.0, 270.0),   # TS-02 CR starboard
            make_ts(-3.5,  0.0,  90.0),   # TS-03 CR port
        ]
    },
}


def get_case(case_id):
    """Return scenario config for a given case ID (1-20)."""
    if case_id not in IMAZU_CASES:
        raise ValueError(f"Case {case_id} not defined. Valid: 1-20.")
    return IMAZU_CASES[case_id]


def list_cases():
    """Print all case descriptions."""
    for k, v in sorted(IMAZU_CASES.items()):
        n_ts = len(v['targets'])
        print(f"  Case {k:2d}: {v['desc']}  ({n_ts} target ship(s))")


if __name__ == '__main__':
    list_cases()
