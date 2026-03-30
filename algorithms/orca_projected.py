"""
ORCA with heading projection for speed-only control.

Full ORCA computes half-planes in 2-D velocity space. Since heading is fixed,
we project each ORCA half-plane constraint onto the 1-D speed line
(v = s * heading_i) and solve the resulting set of linear constraints
for the speed closest to preferred.

Uses KDTree for neighbor pruning.
"""

import numpy as np
from scipy.spatial import cKDTree
from .base import BaseAlgorithm
import config as cfg


class ORCAProjected(BaseAlgorithm):
    name = "ORCA-Projected"
    complexity = "O(N log N)"

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

                vj = speeds[j] * headings[j]
                rel_vel = speeds[i] * hi - vj

                if dist < sep:
                    n_vec = rel_pos / dist
                    u = (sep - dist + 0.1) / (2 * self.tau) * n_vec
                    plane_normal = n_vec
                    plane_point = pref_vel + u * 0.5
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

                    plane_point = pref_vel + u * 0.5

                # project onto heading line: v = s * hi
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

            lo = max(lower_bounds)
            hi_ = min(upper_bounds)
            lo = max(lo, cfg.MIN_SPEED)
            hi_ = min(hi_, cfg.MAX_SPEED)

            if lo <= hi_:
                cmd[i] = np.clip(preferred[i], lo, hi_)
            else:
                cmd[i] = cfg.MIN_SPEED

        return cmd
