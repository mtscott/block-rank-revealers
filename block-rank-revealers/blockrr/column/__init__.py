"""Column-level QR with column pivoting, behind one interface.

``get(name)(A, **options)`` returns a :class:`QRCPResult` with ``A[:, p] = Q R``.  The
methods compared in the paper are

    businger_golub   textbook Businger--Golub (trailing norms recomputed every step)
    geqp3            LAPACK dgeqp3 via SciPy
    hqrrp            blocked randomized pivoted QR (HQRRP)
    cqrrpt           CQRRPT: sparse sign sketch -> dgeqp3 -> CholeskyQR
    srrqr            Gu--Eisenstat strong RRQR, Algorithm 4 (fixed rank k)
    rand_srrqr       randomized sRRQR: CQRRPT with sRRQR on the sketch

A column method selects blocks through ``blockrr.core.induced_block_order``.
"""

from . import businger_golub, cqrrpt, geqp3, hqrrp, srrqr  # noqa: F401  (registers the methods)
from .interface import REGISTRY, QRCPResult, available, get, register  # noqa: F401

#: Methods whose output depends on a random seed.
RANDOMIZED = {"cqrrpt", "rand_srrqr", "cqrrpt_hqrrp", "hqrrp"}

#: Methods that need the target rank k.
NEEDS_K = {"srrqr"}

#: Methods designed for tall matrices (m >> n).
TALL_ONLY = {"cqrrpt", "rand_srrqr", "cqrrpt_hqrrp"}

__all__ = ["REGISTRY", "QRCPResult", "get", "register", "available",
           "RANDOMIZED", "NEEDS_K", "TALL_ONLY"]
