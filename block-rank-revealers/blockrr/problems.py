"""Synthetic test problems: Example 2.1 of the paper, the ceil(k/b)-trap family that
generalizes it, and a block devil's-stairs matrix."""

from __future__ import annotations

from math import ceil

import numpy as np


def example_2_1(eps: float = 0.5, delta: float = 1e-2) -> np.ndarray:
    """The 6 x 6 matrix of Example 2.1: n = 3 blocks of width b = 2 (use k = 2).

    Block B1 (columns 0-1) has the largest volume, but the best pair is {B2, B3}.
    """
    s = np.sqrt(1 - eps**2)
    return np.array([
        [1, 0, s, 0, 0, 0],
        [0, 0, eps, 1, 0, 0],
        [0, 1, 0, 0, s, 0],
        [0, 0, 0, 0, eps, 1],
        [delta, 0, 0, 0, 0, 0],
        [0, delta, 0, 0, 0, 0],
    ], dtype=float)


def example_2_1_predictions(eps: float, delta: float) -> dict:
    """Closed forms for Example 2.1."""
    s = np.sqrt(1 - eps**2)
    return {
        "Vol(B1)": 1 + delta**2,
        "Vol(B2)=Vol(B3)": s,
        "Vol(greedy)": delta * s * np.sqrt(1 + delta**2),
        "Vol(S*)": 1 - eps**2,
        "ratio": delta * np.sqrt(1 + delta**2) / s,
    }


def trap_family(k: int, b: int, eps: float = 0.5, delta: float = 1e-2) -> tuple[np.ndarray, list[int]]:
    """Example 2.1 generalized to any budget k >= 2 and block width b >= 2.

    k good blocks G_i: first column sqrt(1-eps^2) g_i + eps f_i, second column f_i,
    remaining b-2 columns private unit vectors; Vol(G_i) = sqrt(1-eps^2).
    ceil(k/b) trap blocks; trap t has the columns g_i + delta p for b of the good
    indices i (one column per poisoned good block), padded with private unit columns.
    Greedy takes every trap first (each has volume >= 1 > sqrt(1-eps^2)), after which
    each poisoned good block's residual volume is O(delta).

    The traps are listed first.  Returns (X, indices of the good blocks).
    """
    n_traps = ceil(k / b)
    s = np.sqrt(1 - eps**2)
    dirs = {}

    def d(name):
        if name not in dirs:
            dirs[name] = len(dirs)
        return dirs[name]

    cols_spec = []          # list of blocks, each a list of {direction: coefficient}
    for i in range(k):
        blk = [{d(f"g{i}"): s, d(f"f{i}"): eps}, {d(f"f{i}"): 1.0}]
        blk += [{d(f"h{i}_{l}"): 1.0} for l in range(b - 2)]
        cols_spec.append(blk)
    for t in range(n_traps):
        blk = []
        covered = list(range(t * b, min((t + 1) * b, k)))
        for l in range(b):
            if l < len(covered):
                i = covered[l]
                blk.append({d(f"g{i}"): 1.0, d(f"p{t}_{l}"): delta})
            else:
                blk.append({d(f"q{t}_{l}"): 1.0})
        cols_spec.append(blk)
    m = len(dirs)
    X = np.zeros((m, len(cols_spec) * b))
    for j, blk in enumerate(cols_spec):
        for l, col in enumerate(blk):
            for r, v in col.items():
                X[r, j * b + l] = v
    # put the traps first so that the good set is not the first k blocks
    n = len(cols_spec)
    perm = list(range(k, n)) + list(range(k))
    Xp = np.concatenate([X[:, j * b:(j + 1) * b] for j in perm], axis=1)
    good = [perm.index(i) for i in range(k)]
    return Xp, sorted(good)


def block_devil_stairs(m: int, n: int, b: int, *, n_steps: int = 5,
                       drop: float = 1e-2, rng=None) -> np.ndarray:
    """m x nb matrix U Sigma V^T whose singular values descend in n_steps equal steps of
    ``drop`` (the last step absorbs the remainder); consecutive columns form the blocks."""
    rng = np.random.default_rng(rng)
    N = n * b
    nn = min(m, N)
    per = max(1, nn // n_steps)
    sigma = np.concatenate([np.full(per, drop**i) for i in range(n_steps)])
    sigma = np.concatenate([sigma, np.full(max(0, nn - sigma.size), sigma[-1])])[:nn]
    U, _ = np.linalg.qr(rng.standard_normal((m, nn)))
    V, _ = np.linalg.qr(rng.standard_normal((N, nn)))
    return (U * sigma) @ V.T
