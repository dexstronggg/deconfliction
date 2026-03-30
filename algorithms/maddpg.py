"""
MADDPG — Multi-Agent Deep Deterministic Policy Gradient (untrained baseline).

Architecture: decentralized actors with centralized critic.
Each actor maps local observation to a deterministic speed action.
The critic (not used at inference) would see all agents' observations.

Untrained — random initialization. Shows architecture overhead.

Complexity: O(N·K·d) per step — K neighbors, d = hidden dimension.
"""

import numpy as np
from .base import BaseAlgorithm
import config as cfg

K_NEIGHBORS = 10
OBS_DIM = 2 + K_NEIGHBORS * 4
HIDDEN_DIM = 128  # DDPG typically uses larger networks


class DDPGActor:
    """Deterministic actor: obs -> speed (two hidden layers)."""

    def __init__(self, obs_dim: int, hidden_dim: int, seed: int = 123):
        rng = np.random.RandomState(seed)
        self.w1 = rng.randn(obs_dim, hidden_dim) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.w3 = rng.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(1)

    def forward_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0, obs_batch @ self.w1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        out = (h2 @ self.w3 + self.b3).ravel()
        sig = 1.0 / (1.0 + np.exp(-np.clip(out, -10, 10)))
        return cfg.MIN_SPEED + sig * (cfg.MAX_SPEED - cfg.MIN_SPEED)


class MADDPG(BaseAlgorithm):
    name = "MADDPG"
    complexity = "O(N*K*d^2)"

    def __init__(self):
        self.actor = DDPGActor(OBS_DIM, HIDDEN_DIM)

    def _build_observations(self, positions, headings, speeds, preferred, active):
        from scipy.spatial import cKDTree

        n = len(positions)
        obs = np.zeros((n, OBS_DIM))
        active_idx = np.where(active)[0]
        if len(active_idx) < 1:
            return obs

        pos_a = positions[active_idx]
        tree = cKDTree(pos_a) if len(active_idx) > 1 else None

        for ii, i in enumerate(active_idx):
            obs[i, 0] = speeds[i] / cfg.MAX_SPEED
            obs[i, 1] = speeds[i] / preferred[i] if preferred[i] > 0 else 0

            if tree is None or len(active_idx) < 2:
                continue

            k = min(K_NEIGHBORS, len(active_idx) - 1)
            dists, idxs = tree.query(pos_a[ii], k=k + 1)

            slot = 0
            for nn_idx, nn_dist in zip(idxs[1:], dists[1:]):
                if slot >= K_NEIGHBORS:
                    break
                j = active_idx[nn_idx]
                rel = positions[j] - positions[i]
                base = 2 + slot * 4
                obs[i, base] = rel[0] / cfg.WORLD_SIZE
                obs[i, base + 1] = rel[1] / cfg.WORLD_SIZE
                obs[i, base + 2] = speeds[j] / cfg.MAX_SPEED
                obs[i, base + 3] = np.dot(headings[i], headings[j])
                slot += 1

        return obs

    def compute_speeds(self, positions, headings, speeds, preferred, active, dt):
        obs = self._build_observations(positions, headings, speeds, preferred, active)
        cmd = self.actor.forward_batch(obs)
        cmd[~active] = 0.0
        return cmd
