"""Base class for speed-only deconfliction algorithms."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """
    Interface for speed-only deconfliction.

    Every algorithm receives the full state and returns a (N,) array
    of commanded scalar speeds along each agent's fixed heading.
    """

    name: str = "base"
    complexity: str = "N/A"  # asymptotic complexity string

    @abstractmethod
    def compute_speeds(
        self,
        positions: np.ndarray,    # (N, 2)
        headings: np.ndarray,     # (N, 2) unit vectors
        speeds: np.ndarray,       # (N,) current speeds
        preferred: np.ndarray,    # (N,) preferred speeds
        active: np.ndarray,       # (N,) bool mask
        dt: float,
    ) -> np.ndarray:
        """Return (N,) array of commanded speeds."""
        ...
