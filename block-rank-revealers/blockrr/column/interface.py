"""Common interface shared by the column-pivoted QR methods.

Convention used everywhere in this package
------------------------------------------
Given ``A`` of shape ``(m, n)`` an algorithm returns a permutation ``p`` and
factors ``Q``, ``R`` such that::

    A[:, p] == Q @ R

``p`` is a 0-based integer array of length ``n``.  ``R`` is upper triangular of
shape ``(nn, n)`` where ``nn = min(m, n)`` for full factorizations, or ``(r, n)``
for algorithms that truncate at a detected numerical rank ``r`` (CQRRPT does
this).  ``Q`` is the economy factor of shape ``(m, nn)`` or ``(m, r)``, and is
``None`` when the caller passes ``want_q=False``.

Every algorithm accepts ``k`` (target rank).  Algorithms that produce a full
factorization ignore it except for reporting; sRRQR requires it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class QRCPResult:
    """Uniform return type for all pivoted QR algorithms."""

    R: np.ndarray
    p: np.ndarray
    Q: Optional[np.ndarray] = None
    rank: Optional[int] = None
    name: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.p = np.asarray(self.p, dtype=np.int64)
        if self.R.ndim != 2:
            raise ValueError(f"{self.name}: R must be 2-D, got shape {self.R.shape}")
        if self.p.ndim != 1 or self.p.size != self.R.shape[1]:
            raise ValueError(
                f"{self.name}: p has length {self.p.size} but R has "
                f"{self.R.shape[1]} columns"
            )
        if not np.array_equal(np.sort(self.p), np.arange(self.p.size)):
            raise ValueError(f"{self.name}: p is not a permutation of 0..n-1")

    @property
    def k_max(self) -> int:
        """Largest truncation rank this result can be evaluated at."""
        return self.R.shape[0]


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[..., QRCPResult]] = {}
DESCRIPTIONS: Dict[str, str] = {}


def register(name: str, description: str = "") -> Callable:
    """Decorator registering an algorithm under ``name``."""

    def deco(fn: Callable[..., QRCPResult]) -> Callable[..., QRCPResult]:
        if name in REGISTRY:
            raise KeyError(f"algorithm {name!r} already registered")
        REGISTRY[name] = fn
        DESCRIPTIONS[name] = description or (fn.__doc__ or "").strip().split("\n")[0]
        fn.registry_name = name  # type: ignore[attr-defined]
        return fn

    return deco


def get(name: str) -> Callable[..., QRCPResult]:
    if name not in REGISTRY:
        raise KeyError(
            f"unknown algorithm {name!r}; available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


def available() -> Dict[str, str]:
    return dict(DESCRIPTIONS)


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------

from .._util import as_rng  # noqa: E402,F401  (re-exported for the column methods)


def householder(x: np.ndarray) -> tuple[np.ndarray, float]:
    """LAPACK-style reflector: ``(I - beta v v^T) x = -sign(x0) ||x|| e_1``.

    Returns ``(v, beta)`` with ``v[0] == 1``.  ``beta == 0`` means no reflection
    is needed.
    """
    x = np.asarray(x, dtype=np.float64)
    nrm = np.linalg.norm(x)
    if nrm == 0.0:
        v = np.zeros_like(x)
        v[0] = 1.0
        return v, 0.0
    alpha = x[0]
    sigma = nrm if alpha >= 0 else -nrm
    v = x.copy()
    v[0] = alpha + sigma
    if v[0] == 0.0:
        v = np.zeros_like(x)
        v[0] = 1.0
        return v, 0.0
    beta = v[0] / sigma
    v = v / v[0]
    return v, float(beta)


def apply_reflectors_to_identity(
    V: list[np.ndarray], betas: list[float], m: int, ncols: int
) -> np.ndarray:
    """Form the economy orthogonal factor ``Q`` (shape ``(m, ncols)``)."""
    Q = np.zeros((m, ncols), dtype=np.float64)
    Q[:ncols, :ncols] = np.eye(ncols)
    for j in range(len(V) - 1, -1, -1):
        beta = betas[j]
        if beta == 0.0:
            continue
        v = V[j]
        block = Q[j:, :]
        block -= beta * np.outer(v, v @ block)
    return Q
