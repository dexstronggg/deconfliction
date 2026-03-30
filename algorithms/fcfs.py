"""
FCFS (First-Come-First-Served) priority-based deceleration.

Agents are prioritised by ID (lower = higher priority, i.e. first-come).
When two agents are on a collision course, the lower-priority agent decelerates.

Uses KDTree for O(N log N) neighbor lookup instead of brute-force O(N^2).
"""

import numpy as np
from scipy.spatial import cKDTree
from .base import BaseAlgorithm
import config as cfg


class FCFS(BaseAlgorithm):
    name = "FCFS"
    complexity = "O(N log N)"

    def __init__(self, lookahead: float = 5.0, neighbor_radius: float = 150.0):
        self.lookahead = lookahead
        self.neighbor_radius = neighbor_radius

    def compute_speeds(self, positions, headings, speeds, preferred, active, dt):
        n = len(positions)
        cmd = preferred.copy()
        sep = 2 * cfg.AGENT_RADIUS + cfg.SAFETY_MARGIN

        active_idx = np.where(active)[0]
        if len(active_idx) < 2:
            return cmd

        pos_a = positions[active_idx]
        tree = cKDTree(pos_a)

        head_a = headings[active_idx]
        spd_a = speeds[active_idx]
        future = pos_a + head_a * (spd_a[:, None] * self.lookahead)

        for ii in range(len(active_idx)):
            i = active_idx[ii]
            neighbors = tree.query_ball_point(pos_a[ii], self.neighbor_radius)

            for jj in neighbors:
                if jj <= ii:
                    continue
                j = active_idx[jj]

                cur_dist = np.linalg.norm(pos_a[ii] - pos_a[jj])
                fut_dist = np.linalg.norm(future[ii] - future[jj])

                if fut_dist < sep and fut_dist < cur_dist:
                    # lower id = higher priority, higher id decelerates
                    loser = j
                    ratio = max(cur_dist / sep, 0.1) if sep > 0 else 1.0
                    safe_speed = preferred[loser] * min(ratio, 1.0)
                    cmd[loser] = min(cmd[loser], max(safe_speed, cfg.MIN_SPEED))

        return cmd
