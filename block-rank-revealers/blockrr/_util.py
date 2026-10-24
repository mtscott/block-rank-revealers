"""Small helpers shared across the package."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_rng(rng: Any) -> np.random.Generator:
    """A NumPy Generator from a Generator, a seed, or None."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
