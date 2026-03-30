"""Simulation configuration parameters."""

import numpy as np

# --- World ---
WORLD_SIZE = 10_000.0          # meters, square arena side
NUM_AGENTS = 1000
SEED = 42

# --- Agent kinematics ---
PREFERRED_SPEED = 15.0         # m/s  (nominal cruise)
MAX_SPEED = 20.0               # m/s
MIN_SPEED = 0.5                # m/s  (near-hover floor)
AGENT_RADIUS = 1.5             # m    (collision body)
SAFETY_MARGIN = 3.0            # m    (separation distance = 2*AGENT_RADIUS + SAFETY_MARGIN)

# --- Simulation ---
DT = 0.1                       # seconds per tick
MAX_TIME = 600.0               # seconds, hard cutoff
COLLISION_DIST = 2 * AGENT_RADIUS  # center-to-center collision threshold

# --- Wind / perturbation model ---
WIND_ENABLED = True
WIND_BASE_STRENGTH = 2.0       # m/s average wind magnitude
WIND_GUST_STRENGTH = 4.0       # m/s peak gust magnitude
WIND_CHANGE_INTERVAL = 5.0     # seconds between wind direction shifts
EVASION_SPEED_PENALTY = 0.3    # fraction of speed lost during evasion maneuvers

# --- Metrics ---
COLLISION_RADIUS = COLLISION_DIST  # for collision counting

# --- Reproducibility ---
RNG = np.random.RandomState(SEED)
