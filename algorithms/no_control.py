"""No-Control baseline: every agent flies at preferred constant speed."""

import numpy as np
from .base import BaseAlgorithm


class NoControl(BaseAlgorithm):
    name = "No-Control"
    complexity = "O(1)"

    def compute_speeds(self, positions, headings, speeds, preferred, active, dt):
        return preferred.copy()
