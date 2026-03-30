"""
2-D simulation environment for speed-only deconfliction benchmarking.

Each agent travels a fixed straight-line route from origin to destination.
The only control input is scalar speed along that heading.
Wind and evasion perturbations dynamically reduce effective speed.

Fully vectorized with numpy for performance with 1000+ agents.
Uses scipy KDTree for O(N log N) collision detection.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass, field

import config as cfg


@dataclass
class WindField:
    """Simple spatially-uniform wind with periodic gusts."""
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    strength: float = cfg.WIND_BASE_STRENGTH
    _next_change: float = 0.0

    def update(self, t: float, rng: np.random.RandomState):
        if t >= self._next_change:
            angle = rng.uniform(0, 2 * np.pi)
            self.direction = np.array([np.cos(angle), np.sin(angle)])
            gust = rng.uniform(0, cfg.WIND_GUST_STRENGTH)
            self.strength = cfg.WIND_BASE_STRENGTH + gust
            self._next_change = t + cfg.WIND_CHANGE_INTERVAL

    def speed_penalties(self, headings: np.ndarray) -> np.ndarray:
        """Return (N,) speed reductions due to headwind component."""
        headwind = -headings @ self.direction * self.strength
        return np.maximum(headwind, 0.0)


class Environment:
    """Manages agents, wind, collision detection, and stepping — fully vectorized."""

    def __init__(self, seed: int = cfg.SEED):
        self.rng = np.random.RandomState(seed)
        self.wind = WindField()
        self.t = 0.0
        self.collisions: int = 0
        self._collision_pairs: set = set()

        # arrays (populated by generate_agents)
        self.n = 0
        self.origins: np.ndarray = np.empty((0, 2))
        self.destinations: np.ndarray = np.empty((0, 2))
        self.headings: np.ndarray = np.empty((0, 2))
        self.route_lengths: np.ndarray = np.empty(0)
        self.positions: np.ndarray = np.empty((0, 2))
        self.progress: np.ndarray = np.empty(0)
        self.speeds: np.ndarray = np.empty(0)
        self.arrived: np.ndarray = np.empty(0, dtype=bool)
        self.collided: np.ndarray = np.empty(0, dtype=bool)
        self.arrival_times: np.ndarray = np.empty(0)

    def generate_agents(self, n: int = cfg.NUM_AGENTS):
        self.n = n
        self.origins = self.rng.uniform(0, cfg.WORLD_SIZE, size=(n, 2))
        self.destinations = self.rng.uniform(0, cfg.WORLD_SIZE, size=(n, 2))

        # ensure minimum route length
        diff = self.destinations - self.origins
        dists = np.linalg.norm(diff, axis=1)
        short = dists < 200.0
        while np.any(short):
            self.destinations[short] = self.rng.uniform(0, cfg.WORLD_SIZE, size=(int(short.sum()), 2))
            diff = self.destinations - self.origins
            dists = np.linalg.norm(diff, axis=1)
            short = dists < 200.0

        self.route_lengths = dists
        self.headings = diff / dists[:, None]
        self._init_state()

    def _init_state(self):
        n = self.n
        self.positions = self.origins.copy()
        self.progress = np.zeros(n)
        self.speeds = np.full(n, cfg.PREFERRED_SPEED)
        self.arrived = np.zeros(n, dtype=bool)
        self.collided = np.zeros(n, dtype=bool)
        self.arrival_times = np.full(n, np.nan)

    def reset(self):
        self.t = 0.0
        self.collisions = 0
        self._collision_pairs.clear()
        self.wind = WindField()
        self._init_state()

    # --- observation accessors ---

    def get_positions(self) -> np.ndarray:
        return self.positions.copy()

    def get_headings(self) -> np.ndarray:
        return self.headings.copy()

    def get_speeds(self) -> np.ndarray:
        return self.speeds.copy()

    def get_preferred_speeds(self) -> np.ndarray:
        return np.full(self.n, cfg.PREFERRED_SPEED)

    def get_active_mask(self) -> np.ndarray:
        return ~self.arrived

    # --- stepping ---

    def step(self, commanded_speeds: np.ndarray):
        dt = cfg.DT
        active = ~self.arrived

        if cfg.WIND_ENABLED:
            self.wind.update(self.t, self.rng)

        # clamp commanded speeds
        cmd = np.clip(commanded_speeds, cfg.MIN_SPEED, cfg.MAX_SPEED)

        # wind penalty (vectorized)
        if cfg.WIND_ENABLED:
            penalties = self.wind.speed_penalties(self.headings)
        else:
            penalties = 0.0

        effective = np.maximum(cmd - penalties, cfg.MIN_SPEED)
        self.speeds = np.where(active, effective, 0.0)

        # advance progress
        self.progress = np.where(active, self.progress + self.speeds * dt, self.progress)

        # check arrivals
        just_arrived = active & (self.progress >= self.route_lengths)
        self.arrived |= just_arrived
        self.progress = np.where(just_arrived, self.route_lengths, self.progress)
        self.arrival_times = np.where(just_arrived, self.t + dt, self.arrival_times)

        # update positions: origin + heading * progress
        self.positions = self.origins + self.headings * self.progress[:, None]
        # snap arrived agents to destination
        self.positions[self.arrived] = self.destinations[self.arrived]

        # collision detection
        self._detect_collisions()
        self.t += dt

    def _detect_collisions(self):
        active_idx = np.where(~self.arrived)[0]
        if len(active_idx) < 2:
            return

        pos = self.positions[active_idx]
        tree = cKDTree(pos)
        pairs = tree.query_pairs(cfg.COLLISION_DIST, output_type='ndarray')
        if len(pairs) == 0:
            return

        # map local indices back to global
        for local_i, local_j in pairs:
            gi, gj = int(active_idx[local_i]), int(active_idx[local_j])
            pair_key = (min(gi, gj), max(gi, gj))
            if pair_key not in self._collision_pairs:
                self._collision_pairs.add(pair_key)
                self.collisions += 1
                self.collided[gi] = True
                self.collided[gj] = True

    # --- termination ---

    def all_arrived(self) -> bool:
        return bool(np.all(self.arrived))

    def timed_out(self) -> bool:
        return self.t >= cfg.MAX_TIME
