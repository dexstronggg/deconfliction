"""
Reciprocal Velocity Obstacle (RVO) — projected to speed-only control.

Unlike classic VO where the ego agent bears full responsibility,
RVO splits the avoidance equally: each agent takes half the adjustment.
This reduces oscillations compared to classic VO.

The blocked interval is computed the same way as VO, but the avoidance
velocity is averaged between the current velocity and the VO-free velocity
(i.e., each agent adjusts by half).

Complexity: O(N²) per step (with KDTree pruning).
"""

import numpy as np
from scipy.spatial import cKDTree
from .base import BaseAlgorithm
import config as cfg


class RVO(BaseAlgorithm):
    name = "RVO"
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
            current_speed = speeds[i]
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

                vj = speeds[j] * headings[j]
                d = rel_pos

                # RVO: the velocity obstacle is centered at (v_i + v_j)/2
                # instead of v_j. For speed-only: we compute the VO blocked
                # interval and then shift it to account for reciprocal sharing.

                d_dot_hi = np.dot(d, hi)
                d_dot_vj = np.dot(d, vj)
                vj_dot_hi = np.dot(vj, hi)
                vj_sq = np.dot(vj, vj)
                d_sq = np.dot(d, d)

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
                s_lo, s_hi_val = min(s1, s2), max(s1, s2)

                # RVO adjustment: shift blocked interval toward current speed
                # (each agent takes half the responsibility)
                mid_blocked = (s_lo + s_hi_val) / 2
                half_width = (s_hi_val - s_lo) / 2
                rvo_center = (mid_blocked + current_speed) / 2
                s_lo = rvo_center - half_width
                s_hi_val = rvo_center + half_width

                s_lo = max(s_lo, cfg.MIN_SPEED)
                s_hi_val = min(s_hi_val, cfg.MAX_SPEED)

                if s_lo < s_hi_val:
                    s_mid = (s_lo + s_hi_val) / 2
                    w_mid = vj - s_mid * hi
                    A_mid = np.dot(w_mid, w_mid)
                    B_mid = np.dot(d, w_mid)
                    if A_mid > 1e-12:
                        t_min = np.clip(-B_mid / A_mid, 0.0, self.tau)
                        pos_at_t = d + w_mid * t_min
                        if np.dot(pos_at_t, pos_at_t) < sep * sep:
                            blocked_intervals.append((s_lo, s_hi_val))

            if not blocked_intervals:
                cmd[i] = preferred[i]
                continue

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
