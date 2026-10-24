"""blockrr -- block rank revealers under a column-block constraint.

Code for J. Haddock, R. Murray and M. T. Scott, "Rank revealers under a column-block
constraint: sharp limits, an instance-dependent theory, and the star_M viewpoint".

Columns come in atomic blocks A = [A_1, ..., A_n] of width b; choosing a block means
taking all of its columns.  The package selects blocks by greedy block pivoting, the
log-det rule of the paper, in two forms:

    algorithm2(A, b, k, variant="a" | "b" | "c")   tensor block QR with greedy pivoting
    algorithm1(A, b, k)                             the same pivots, in matrix form

The three variants of Algorithm 2 choose the same blocks and differ in the structure of
their factors (see ``blockrr.algorithm2``).  ``check_algorithm2`` evaluates the identities
each variant satisfies.

Submodules: ``core`` (volumes and references), ``tensor`` (the star_M product),
``starm_qr`` (pivoted star_M-QR of the literature), ``column`` (column-pivoted QR methods),
``problems`` (synthetic test problems), ``sketching``, ``perdigao`` (the tower data).
"""

__version__ = "1.0.0"

from . import column, core, problems, sketching, starm_qr, tensor  # noqa: F401  (perdigao: import explicitly)
from .algorithm2 import (VARIANTS, Algorithm2Result, algorithm2,  # noqa: F401
                         check_algorithm2, select_blocks)
from .core import (augment, compress, exhaustive, induced_block_order,  # noqa: F401
                   log_vol, log_vol_S, trajectory)


def algorithm1(A, b, k):
    """Algorithm 1 (block QR with greedy pivoting): dict(pi, Q, R, log_vols).

    Thin wrapper of ``core.block_greedy_qr`` with Householder reductions, run on the
    compressed factor of a tall A."""
    return core.block_greedy_qr(core.compress(A), b, k, mode="householder")


__all__ = ["algorithm1", "algorithm2", "select_blocks", "check_algorithm2", "VARIANTS",
           "Algorithm2Result", "augment", "compress", "exhaustive", "induced_block_order",
           "log_vol", "log_vol_S", "trajectory",
           "column", "core", "problems", "sketching", "starm_qr", "tensor"]
