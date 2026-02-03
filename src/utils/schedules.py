"""Simple ramp-up schedules."""

from __future__ import annotations

import math


def linear_rampup(epoch: int, rampup_length: int) -> float:
    if rampup_length == 0:
        return 1.0
    return min(1.0, epoch / rampup_length)


def cosine_rampdown(epoch: int, total_epochs: int) -> float:
    if total_epochs == 0:
        return 1.0
    return 0.5 * (math.cos(math.pi * epoch / total_epochs) + 1)
