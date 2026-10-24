"""Algorithm 2 of the paper (tensor block QR with greedy pivoting) and its three variants.

    variant  ORTHOFACTORS(C_p)     SCOPE(t, n)     identity it satisfies
    "a"      [Q_hat, I_b]  (QR)    [n]             facewise (M = I): A *_I P = Q *_I R
    "b"      [U, V]        (SVD)   {t, ..., n}     one M = W_j^T per tubal column j
    "c"      [U, V]        (SVD)   [n]             a single M = W^T for all tubal columns

All three pick the same blocks as Algorithm 1 (Proposition 4 of the paper); they differ
only in the structure of the factors: the diagonal blocks of R are upper triangular for
(a), diagonal for (b), and unstructured for (c), whose transform is shared by all blocks.

Example
-------
>>> import numpy as np
>>> from blockrr import algorithm2
>>> A = np.random.default_rng(0).standard_normal((200, 40))      # 10 blocks of width 4
>>> res = algorithm2(A, b=4, k=5, variant="b")
>>> res.selected                                                   # the 5 blocks chosen
>>> res.log_volumes[-1]                                            # log Vol(A_S) of those 5
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core, tensor

VARIANTS = {
    "a": {"ortho": "qr", "scope": "all"},
    "b": {"ortho": "svd", "scope": "trailing"},
    "c": {"ortho": "svd", "scope": "all"},
}

DESCRIPTIONS = {
    "a": "ORTHOFACTORS = [Q_hat, I_b] (QR), SCOPE = [n]: facewise product, triangular diagonal blocks",
    "b": "ORTHOFACTORS = [U, V] (SVD), SCOPE = {t..n}: per-column transforms W_j, diagonal diagonal blocks",
    "c": "ORTHOFACTORS = [U, V] (SVD), SCOPE = [n]: one transform W shared by every block",
}


def _check_variant(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    return VARIANTS[variant]


@dataclass
class Algorithm2Result:
    """Output of :func:`algorithm2`.

    order        pivot order of all n blocks; the first k were chosen greedily
    log_volumes  log Vol of the first t pivoted blocks, t = 1, ..., k (natural log)
    Q            m x m orthogonal factor (of R0 when the input was compressed)
    R            Rtilde as an m x n x b tensor (``R_matrix`` unfolds it)
    W            b x n x b; W(:, j, :) is the tube transform applied to tubal column j
    V            the V of every step (the identity for variant "a")
    compressed   True when a tall A was replaced by R0, A = Q0 R0, before the run
    """

    variant: str
    b: int
    k: int
    order: list[int]
    log_volumes: list[float]
    Q: np.ndarray = field(repr=False)
    R: np.ndarray = field(repr=False)
    W: np.ndarray = field(repr=False)
    V: list = field(repr=False)
    compressed: bool = False

    @property
    def selected(self) -> list[int]:
        """The k selected blocks, in the order chosen."""
        return self.order[: self.k]

    @property
    def R_matrix(self) -> np.ndarray:
        """Rtilde unfolded to an m x nb matrix."""
        return tensor.unfold1(self.R)


def algorithm2(A: np.ndarray, b: int, k: int | None = None, *, variant: str = "a",
               compress: bool = True) -> Algorithm2Result:
    """Run Algorithm 2 with variant "a", "b" or "c" for k steps (default: all n blocks).

    A        m x nb matrix of n blocks of width b
    compress replace a tall A by R0 from A = Q0 R0 first (the selection, the volumes and
             R are unchanged; Q is then the factor of R0).  The algorithm forms an m x m
             Q, so leave this on for tall data.
    """
    opts = _check_variant(variant)
    A = np.asarray(A, dtype=np.float64)
    m, N = A.shape
    if N % b:
        raise ValueError("the number of columns must be a multiple of b")
    n = N // b
    k = n if k is None else int(k)
    X = core.compress(A) if compress else A
    if k * b > X.shape[0]:
        raise ValueError(f"need k*b <= {X.shape[0]} rows")
    pi, Q, Rt, W, Vs = tensor.algorithm2_tensor(tensor.fold1(X, b), k, **opts)
    order = [int(j) for j in pi]
    return Algorithm2Result(variant=variant, b=b, k=k, order=order,
                            log_volumes=core.trajectory(X, order[:k], b, k),
                            Q=Q, R=Rt, W=W, V=Vs, compressed=bool(compress and m > N))


def select_blocks(A: np.ndarray, b: int, k: int, *, variant: str = "a") -> list[int]:
    """The k blocks Algorithm 2 selects (the same for every variant)."""
    return algorithm2(A, b, k, variant=variant).selected


def check_algorithm2(A: np.ndarray, b: int, *, variant: str = "a", mode3_with: str = "VT") -> dict:
    """Residuals of the identities the paper states for one variant, with k = n.

    Runs on A as given (pass R0 for tall data).  ``mode3_with="V"`` replaces the x3 V^T
    of line 9 by x3 V, to show that the transpose matters.  See
    :func:`blockrr.tensor.check_algorithm2_tensor` for the keys.
    """
    return tensor.check_algorithm2_tensor(A, b, **_check_variant(variant), mode3_with=mode3_with)
