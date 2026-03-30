"""
CSORCA — Cooperative Speed ORCA.

Extension of ORCA specifically designed for speed-only deconfliction.
Each agent computes ORCA half-planes and projects them onto its heading line,
but additionally applies cooperative speed sharing: when two agents must
adjust, the one with more speed margin (further from min/max) takes a
larger share of the avoidance burden.

Complexity: O(N²) per step (with KDTree pruning).
"""

import numpy as np
from scipy.spatial import cKDTree
from .base import BaseAlgorithm
import config as cfg


class CSORCA(BaseAlgorithm):
    name = "CSORCA"
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
            si = speeds[i]
            pref_vel = preferred[i] * hi

            lower_bounds = [cfg.MIN_SPEED]
            upper_bounds = [cfg.MAX_SPEED]

            neighbors = tree.query_ball_point(pos_a[ii], self.neighbor_radius)

            for jj in neighbors:
                if jj == ii:
                    continue
                j = active_idx[jj]

                rel_pos = positions[j] - positions[i]
                dist = np.linalg.norm(rel_pos)
                if dist < 1e-9:
                    continue

                sj = speeds[j]
                vj = sj * headings[j]
                rel_vel = si * hi - vj

                # Cooperative sharing ratio: agent with more speed margin
                # takes more of the avoidance
                margin_i = min(si - cfg.MIN_SPEED, cfg.MAX_SPEED - si)
                margin_j = min(sj - cfg.MIN_SPEED, cfg.MAX_SPEED - sj)
                total_margin = margin_i + margin_j
                if total_margin > 1e-9:
                    share_i = margin_i / total_margin  # higher margin = more responsibility
                else:
                    share_i = 0.5

                if dist < sep:
                    n_vec = rel_pos / dist
                    u = (sep - dist + 0.1) / (2 * self.tau) * n_vec
                    plane_normal = n_vec
                    plane_point = pref_vel + u * share_i  # cooperative share
                else:
                    cutoff_center = rel_pos / self.tau
                    w = rel_vel - cutoff_center
                    w_len = np.linalg.norm(w)
                    rdist = sep / self.tau

                    if w_len < 1e-9:
                        plane_normal = rel_pos / dist
                        u = rdist * plane_normal
                    elif w_len < rdist:
                        plane_normal = w / w_len
                        u = (rdist - w_len) * plane_normal
                    else:
                        leg = np.sqrt(max(dist * dist - sep * sep, 0.0))
                        rel_dir = rel_pos / dist
                        perp = np.array([-rel_dir[1], rel_dir[0]])
                        left_leg = (rel_dir * leg + perp * sep) / dist
                        right_leg = (rel_dir * leg - perp * sep) / dist
                        dot_left = np.dot(rel_vel, np.array([-left_leg[1], left_leg[0]]))

                        if dot_left > 0:
                            proj = np.dot(rel_vel, left_leg) * left_leg
                            u = proj - rel_vel
                            plane_normal = np.array([-left_leg[1], left_leg[0]])
                        else:
                            proj = np.dot(rel_vel, right_leg) * right_leg
                            u = proj - rel_vel
                            plane_normal = np.array([right_leg[1], -right_leg[0]])

                        pn_len = np.linalg.norm(plane_normal)
                        if pn_len > 1e-9:
                            plane_normal /= pn_len

                    plane_point = pref_vel + u * share_i  # cooperative share

                n_dot_h = np.dot(plane_normal, hi)
                rhs = np.dot(plane_point, plane_normal)

                if abs(n_dot_h) < 1e-9:
                    if np.dot(pref_vel, plane_normal) < rhs:
                        lower_bounds.append(cfg.MAX_SPEED + 1)
                    continue

                s_bound = rhs / n_dot_h
                if n_dot_h > 0:
                    lower_bounds.append(s_bound)
                else:
                    upper_bounds.append(s_bound)

            lo = max(max(lower_bounds), cfg.MIN_SPEED)
            hi_ = min(min(upper_bounds), cfg.MAX_SPEED)

            if lo <= hi_:
                cmd[i] = np.clip(preferred[i], lo, hi_)
            else:
                cmd[i] = cfg.MIN_SPEED

        return cmd
