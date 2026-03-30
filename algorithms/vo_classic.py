"""
Classic Velocity Obstacle (VO) — projected to speed-only control.

Each agent computes the full VO cone for every neighbor, then selects
the speed along its fixed heading that avoids all cones. Unlike the
SSD variant, this uses the standard VO formulation (no reciprocal sharing
of avoidance responsibility — the ego agent avoids the full obstacle).

Complexity: O(N·K) per step where K = avg neighbors via KDTree.
"""

import numpy as np
from scipy.spatial import cKDTree
from .base import BaseAlgorithm
import config as cfg


class VOClassic(BaseAlgorithm):
    name = "VO"
    complexity = "O(N^2)"

    def __init__(self, tau: float = 5.0, neighbor_radius: float = 150.0):
        self.tau = tau
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

        for ii in range(len(active_idx)):
            i = active_idx[ii]
            hi = headings[i]
            blocked_intervals = []

            neighbors = tree.query_ball_point(pos_a[ii], self.neighbor_radius)

            for jj in neighbors:
                if jj == ii:
                    continue
                j = active_idx[jj]

                rel_pos = positions[j] - positions[i]
                dist = np.linalg.norm(rel_pos)
                if dist < 1e-9:
                    continue

                # VO cone: velocity of agent i that will cause collision with j
                # assuming j continues at constant velocity
                vj = speeds[j] * headings[j]

                # Truncated VO at time horizon tau:
                # The set of velocities v_i such that there exists t in [0, tau]
                # where ||p_ij + (v_j - v_i)*t|| < sep
                #
                # With v_i = s * hi, we solve for the blocked speed interval.

                # Parameterize: let v_i = s * hi
                # ||rel_pos + (vj - s*hi)*t||^2 < sep^2
                # We want the range of s that makes minimum distance < sep
                # over t in [0, tau].

                d = rel_pos
                d_dot_hi = np.dot(d, hi)
                d_dot_vj = np.dot(d, vj)
                vj_dot_hi = np.dot(vj, hi)
                vj_sq = np.dot(vj, vj)
                d_sq = np.dot(d, d)

                # Closest approach: min_t |d + (vj - s*hi)*t|^2
                # A(s) = |vj - s*hi|^2 = s^2 - 2s*vj_dot_hi + vj_sq
                # B(s) = d.(vj - s*hi) = d_dot_vj - s*d_dot_hi
                # t_min = -B/A, min_dist^2 = d_sq - B^2/A
                #
                # Collision when d_sq - B^2/A < sep^2
                # => (d_sq - sep^2)*A < B^2
                # => (d_sq - sep^2)(s^2 - 2s*c1 + c2) < (c3 - s*c4)^2

                lhs_coeff = d_sq - sep * sep
                c1, c2, c3, c4 = vj_dot_hi, vj_sq, d_dot_vj, d_dot_hi

                a_coef = lhs_coeff - c4 * c4
                b_coef = -2 * lhs_coeff * c1 + 2 * c3 * c4
                c_coef = lhs_coeff * c2 - c3 * c3

                disc = b_coef * b_coef - 4 * a_coef * c_coef
                if disc < 0 or abs(a_coef) < 1e-12:
                    continue

                sqrt_disc = np.sqrt(disc)
                s1 = (-b_coef - sqrt_disc) / (2 * a_coef)
                s2 = (-b_coef + sqrt_disc) / (2 * a_coef)
                s_lo, s_hi = min(s1, s2), max(s1, s2)
                s_lo = max(s_lo, cfg.MIN_SPEED)
                s_hi = min(s_hi, cfg.MAX_SPEED)

                if s_lo < s_hi:
                    # verify time constraint
                    s_mid = (s_lo + s_hi) / 2
                    w_mid = vj - s_mid * hi
                    A_mid = np.dot(w_mid, w_mid)
                    B_mid = np.dot(d, w_mid)
                    if A_mid > 1e-12:
                        t_min = np.clip(-B_mid / A_mid, 0.0, self.tau)
                        pos_at_t = d + w_mid * t_min
                        if np.dot(pos_at_t, pos_at_t) < sep * sep:
                            blocked_intervals.append((s_lo, s_hi))

            if not blocked_intervals:
                cmd[i] = preferred[i]
                continue

            # merge & find best safe speed
            blocked_intervals.sort()
            merged = [blocked_intervals[0]]
            for lo, hi_ in blocked_intervals[1:]:
                if lo <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hi_))
                else:
                    merged.append((lo, hi_))

            pref = preferred[i]
            best_speed = None
            best_dist = float('inf')
            candidates = [cfg.MIN_SPEED, cfg.MAX_SPEED]
            for lo, hi_ in merged:
                candidates.append(max(lo - 0.1, cfg.MIN_SPEED))
                candidates.append(min(hi_ + 0.1, cfg.MAX_SPEED))

            for s_cand in candidates:
                if not any(lo <= s_cand <= hi_ for lo, hi_ in merged):
                    d_ = abs(s_cand - pref)
                    if d_ < best_dist:
                        best_dist = d_
                        best_speed = s_cand

            cmd[i] = best_speed if best_speed is not None else cfg.MIN_SPEED

        return cmd
