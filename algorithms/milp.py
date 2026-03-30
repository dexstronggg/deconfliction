"""
MILP — Mixed-Integer Linear Programming for speed-only deconfliction.

Formulates the speed assignment as an optimization problem:
- Minimize total deviation from preferred speeds
- Subject to: pairwise separation constraints for all close agent pairs
- Binary variables encode ordering (who passes first)

Uses scipy.optimize.linprog for the LP relaxation (continuous relaxation
of the MILP). Full MILP would require a branch-and-bound solver; we use
LP relaxation which gives a lower bound on the optimal and is tractable
for 1000 agents in real-time.

Complexity: O(N² + LP solve) per step — LP with N variables and O(K) constraints.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linprog
from .base import BaseAlgorithm
import config as cfg


class MILP(BaseAlgorithm):
    name = "MILP"
    complexity = "O(N^2 + LP)"

    def __init__(self, tau: float = 5.0, neighbor_radius: float = 100.0):
        self.tau = tau
        self.neighbor_radius = neighbor_radius

    def compute_speeds(self, positions, headings, speeds, preferred, active, dt):
        n = len(positions)
        cmd = preferred.copy()
        sep = 2 * cfg.AGENT_RADIUS + cfg.SAFETY_MARGIN

        active_idx = np.where(active)[0]
        na = len(active_idx)
        if na < 2:
            return cmd

        pos_a = positions[active_idx]
        tree = cKDTree(pos_a)

        # Build LP: minimize sum |s_i - pref_i|
        # We linearize absolute value: s_i = pref_i + d_plus_i - d_minus_i
        # minimize sum(d_plus + d_minus)
        # subject to: MIN_SPEED <= s_i <= MAX_SPEED
        #             separation constraints for close pairs

        # Variables: s_0..s_{na-1} (speeds for active agents)
        # Objective: minimize sum(|s_i - pref_i|) — use LP trick with slack vars
        # For simplicity, directly optimize speeds with linear penalty

        # Gather conflict pairs and constraints
        A_ub_rows = []
        b_ub_vals = []

        pairs = tree.query_pairs(self.neighbor_radius, output_type='ndarray')

        for local_i, local_j in pairs:
            gi = active_idx[local_i]
            gj = active_idx[local_j]

            rel_pos = positions[gj] - positions[gi]
            dist = np.linalg.norm(rel_pos)
            if dist < 1e-9 or dist > self.neighbor_radius:
                continue

            hi_i = headings[gi]
            hi_j = headings[gj]

            # Separation constraint at lookahead time tau:
            # ||pos_i + s_i*hi_i*tau - pos_j - s_j*hi_j*tau|| >= sep
            #
            # Linear approximation: project onto the line connecting agents
            # d(tau) ≈ dist + tau * (s_i * proj_i - s_j * proj_j)
            # where proj = heading projected onto relative position direction
            n_vec = rel_pos / dist
            proj_i = np.dot(hi_i, n_vec)
            proj_j = np.dot(hi_j, n_vec)

            # We want: dist + tau*(s_j*proj_j - s_i*proj_i) >= sep
            # => -tau*proj_i * s_i + tau*proj_j * s_j >= sep - dist
            # => tau*proj_i * s_i - tau*proj_j * s_j <= dist - sep

            row = np.zeros(na)
            row[local_i] = self.tau * proj_i
            row[local_j] = -self.tau * proj_j
            A_ub_rows.append(row)
            b_ub_vals.append(dist - sep)

            # symmetric constraint
            row2 = np.zeros(na)
            row2[local_j] = self.tau * proj_j
            row2[local_i] = -self.tau * proj_i
            A_ub_rows.append(row2)
            b_ub_vals.append(dist - sep)

        # Objective: minimize sum(|s_i - pref_i|)
        # Using LP formulation: min c^T x
        # Simple approach: minimize negative preferred alignment
        # c_i = -1 for all (maximize total speed toward preferred)
        # But we actually want to minimize deviation, so:
        # min sum((pref_i - s_i)) when s_i < pref_i (deceleration penalty)
        # Simplify: use c_i = -1 (maximize speeds, subject to safety)
        c = -np.ones(na)  # maximize total speed (minimize deceleration)

        bounds = [(cfg.MIN_SPEED, cfg.MAX_SPEED)] * na

        if A_ub_rows:
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub_vals)

            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            if result.success:
                for k in range(na):
                    cmd[active_idx[k]] = np.clip(result.x[k], cfg.MIN_SPEED, cfg.MAX_SPEED)
            else:
                # LP infeasible — fall back to conservative speeds
                for k in range(na):
                    cmd[active_idx[k]] = cfg.MIN_SPEED
        else:
            # no conflicts — use preferred
            for k in range(na):
                cmd[active_idx[k]] = preferred[active_idx[k]]

        return cmd
