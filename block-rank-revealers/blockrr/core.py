"""Block volumes, Algorithm 1, and the reference selections used in the experiments.

Conventions (as in the paper):
    A = [A_1 ... A_n] in R^{m x nb}, A_j in R^{m x b}; blocks are atomic.
    A selection S is a set of block indices; A_S = [A_j]_{j in S}.
    Vol(M) = prod sigma_i(M) = sqrt(det M^T M); we work with log Vol (natural log).

Volumes of column subsets of A equal those of the same column subsets of R0, where
A = Q0 R0 is a thin QR, so tall problems are compressed once to the nb x nb factor R0
and every volume is computed there.  This is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Iterable, Optional

import numpy as np

_TINY = 1e-300


# --------------------------------------------------------------------------
# volumes
# --------------------------------------------------------------------------

def block_cols(j: int, b: int) -> np.ndarray:
    """Column indices of block j."""
    return np.arange(j * b, (j + 1) * b)


def cols_of(S: Iterable[int], b: int) -> np.ndarray:
    """Column indices of the blocks in S, in the order given."""
    S = list(S)
    if not S:
        return np.zeros(0, dtype=int)
    return np.concatenate([block_cols(j, b) for j in S])


def log_vol(M: np.ndarray) -> float:
    """log Vol(M) = sum log sigma_i(M), from the singular values (-inf if rank deficient)."""
    if M.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(M, compute_uv=False)
    if s.size < M.shape[1] or s[-1] <= _TINY:
        return -np.inf
    return float(np.sum(np.log(s)))


def compress(A: np.ndarray) -> np.ndarray:
    """R0 with A = Q0 R0 (A itself if it is not tall); column-subset volumes are unchanged."""
    m, N = A.shape
    if m <= N:
        return np.asarray(A, dtype=np.float64)
    return np.linalg.qr(np.asarray(A, dtype=np.float64), mode="r")


def log_vol_S(A: np.ndarray, S: Iterable[int], b: int) -> float:
    """log Vol(A_S)."""
    return log_vol(A[:, cols_of(S, b)])


def augment(X: np.ndarray, R: float) -> np.ndarray:
    """Bayesian design matrix A = [R X; I_{nb}], so log Vol(A_S) = 1/2 log det(I + R^2 X_S^T X_S)."""
    N = X.shape[1]
    return np.vstack([R * X, np.eye(N)])


def trajectory(A: np.ndarray, order: list[int], b: int, kmax: Optional[int] = None) -> list[float]:
    """log Vol of the first t blocks of ``order``, t = 1, ..., kmax."""
    kmax = len(order) if kmax is None else min(kmax, len(order))
    return [log_vol_S(A, order[:t], b) for t in range(1, kmax + 1)]


@dataclass
class BlockSelection:
    name: str
    order: list[int]                                      # selection order
    nested: bool = True
    log_vols: list[float] = field(default_factory=list)   # log Vol after t blocks
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Algorithm 1 -- block QR with greedy (log det) block pivoting
# --------------------------------------------------------------------------

def block_greedy_qr(A: np.ndarray, b: int, k: int, *, mode: str = "svd") -> dict:
    """Algorithm 1 of the paper, literally.

    At step t the block whose residual C_j has the largest log det(C_j^T C_j) is
    pivoted to position t, and its residual is reduced by an orthogonal factor:
    mode="svd" uses the full U of the SVD of C_p, as written in the paper;
    mode="householder" uses a Householder QR of C_p instead (same selection, upper
    triangular R).  Ties go to the lowest index.

    Returns dict(pi, Q, R, log_vols) with A[:, blocks pi] = Q R.
    """
    A = np.array(A, dtype=np.float64, copy=True)
    m, N = A.shape
    n = N // b
    if N != n * b:
        raise ValueError("number of columns must be a multiple of b")
    if k * b > m:
        raise ValueError("need kb <= m")
    pi = list(range(n))
    Q = np.eye(m)
    log_vols, running = [], 0.0
    for t in range(k):
        r = t * b
        scores = [2.0 * log_vol(A[r:, j * b:(j + 1) * b]) for j in range(t, n)]
        p = t + int(np.argmax(scores))
        # the volume increment is the volume of the residual block (a Schur complement)
        running += log_vol(A[r:, p * b:(p + 1) * b])
        log_vols.append(running)
        if p != t:
            ct, cp = block_cols(t, b), block_cols(p, b)
            A[:, np.concatenate([ct, cp])] = A[:, np.concatenate([cp, ct])]
            pi[t], pi[p] = pi[p], pi[t]
        Cp = A[r:, t * b:(t + 1) * b]
        if mode == "svd":
            U, _, _ = np.linalg.svd(Cp, full_matrices=True)
        elif mode == "householder":
            U, _ = np.linalg.qr(Cp, mode="complete")
        else:
            raise ValueError(mode)
        A[r:, :] = U.T @ A[r:, :]
        Q[:, r:] = Q[:, r:] @ U
    return {"pi": pi, "Q": Q, "R": A, "log_vols": log_vols}


def select_block_greedy(A, b, k) -> BlockSelection:
    """The first k pivots of Algorithm 1 (on the compressed factor), with their volumes."""
    R0 = compress(A)
    out = block_greedy_qr(R0, b, k, mode="householder")
    order = out["pi"][:k]
    return BlockSelection("block greedy (Algorithm 1)", order, True, trajectory(R0, order, b))


# --------------------------------------------------------------------------
# references: exhaustive optimum / worst, random selections, a cheap upper bound
# --------------------------------------------------------------------------

def n_subsets(n: int, k: int) -> int:
    return comb(n, k)


def exhaustive(A, b, k, *, limit: int = 400_000) -> Optional[dict]:
    """Largest and smallest k-block volume by enumeration (None if more than ``limit`` sets)."""
    R0 = compress(A)
    n = R0.shape[1] // b
    if n_subsets(n, k) > limit:
        return None
    best, worst = (-np.inf, None), (np.inf, None)
    for S in combinations(range(n), k):
        v = log_vol_S(R0, S, b)
        if v > best[0]:
            best = (v, S)
        if v < worst[0]:
            worst = (v, S)
    return {"opt": best[0], "S_opt": list(best[1]),
            "worst": worst[0], "S_worst": list(worst[1])}


def random_selection(A, b, k, rng) -> float:
    """log Vol of a uniformly random k-block selection."""
    R0 = compress(A)
    n = R0.shape[1] // b
    S = rng.choice(n, size=k, replace=False)
    return log_vol_S(R0, S, b)


def single_block_upper_bound(A, b, k) -> float:
    """Fischer: Vol(A_S) <= prod_{j in S} Vol(A_j) <= product of the k largest single-block volumes."""
    R0 = compress(A)
    n = R0.shape[1] // b
    v = sorted((log_vol_S(R0, [j], b) for j in range(n)), reverse=True)
    return float(sum(v[:k]))


# --------------------------------------------------------------------------
# column methods: the block order they induce
# --------------------------------------------------------------------------

def induced_block_order(p: np.ndarray, b: int) -> list[int]:
    """Blocks in the order in which the first of their columns appears in the pivot order p.

    This is what a column-level method that ignores the blocks selects when blocks are
    atomic: wanting one column of a block means buying all b of them.
    """
    seen, order = set(), []
    for c in np.asarray(p):
        j = int(c) // b
        if j not in seen:
            seen.add(j)
            order.append(j)
    return order
