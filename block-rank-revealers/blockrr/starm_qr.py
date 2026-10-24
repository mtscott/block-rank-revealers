"""Pivoted *_M-QR factorizations from the literature, used as block-selection rules.

With A = fold1(A) in R^{m x n x b}, tubal column j of A is block j of the matrix.
In the transform domain Ahat = A x3 M every frontal slice Ahat(:, :, l) is an m x n
matrix whose column j belongs to tubal column j.  A *_M-QR factors every slice with
its own QR, using one column permutation P shared by all slices,

    Ahat(:, :, l) P = Qhat_l Rhat_l,        l = 1, ..., b,                    (*)

so that A *_M P = Q *_M R with Q orthogonal and R f-upper triangular (every
transform-domain slice upper triangular).  The shared permutation orders the tubal
columns, i.e. the blocks, and its first k entries are a k-block selection.

Pivot rules (``rule``):
    "none"        unpivoted t-QR / *_M-QR (Kilmer, Martin and Perrone 2008, Thm 8.1):
                  P = I, so the selection is the input order.
    "first_slice" pivoted t-QR (Hao, Kilmer, Braman and Hoover 2013, Thm 6.1): P is the
                  Businger-Golub permutation of the first transform-domain slice
                  Ahat(:, :, 1) alone; the other slices follow it.
    "tube_norm"   the all-slices variant of Section 3.4 of the draft: at each step take
                  the tubal column whose residual, summed over all slices, has the
                  largest Frobenius norm; all slices are updated in parallel.

Transforms (``transform``): "I" (*_I, facewise), "DFT" (unitary DFT, the t-product),
"DCT" (orthonormal DCT-II, the *_c product of Kernfeld, Kilmer and Aeron), "Haar"
(orthonormal Haar matrix, b a power of two).  For DFT, DCT and Haar the first row of M
is constant, so the first slice is the (scaled) sum of each block's columns and the
"first_slice" rule picks the same blocks for all three.

The factorization is invariant under a left orthogonal factor, so it can be run on
R0 from A = Q0 R0 and gives the same permutation and R.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

from .tensor import fold1, mode3

TRANSFORMS = ("I", "DFT", "DCT", "Haar")
RULES = ("none", "first_slice", "tube_norm")


def haar_matrix(b: int) -> np.ndarray:
    """Orthonormal Haar matrix of order b (a power of two), rows = basis functions."""
    if b < 1 or b & (b - 1):
        raise ValueError("Haar transform needs b a power of two")
    H = np.array([[1.0]])
    while H.shape[0] < b:
        h = H.shape[0]
        top = np.kron(H, [1.0, 1.0])
        bot = np.kron(np.eye(h), [1.0, -1.0])
        H = np.vstack([top, bot]) / np.sqrt(2.0)
    return H


def transform(name: str, b: int) -> np.ndarray:
    """The b x b transform M (unitary or orthogonal) named by ``name``."""
    key = name.lower()
    if key in ("i", "identity"):
        return np.eye(b)
    if key in ("dft", "fft", "t"):
        return np.fft.fft(np.eye(b), axis=0) / np.sqrt(b)
    if key == "dct":
        return dct(np.eye(b), type=2, norm="ortho", axis=0)
    if key == "haar":
        return haar_matrix(b)
    raise KeyError(f"unknown transform {name!r}; use one of {TRANSFORMS}")


def _house(x: np.ndarray):
    """Householder vector v and beta with (I - beta v v^H) x = alpha e_1 (real or complex)."""
    nrm = np.linalg.norm(x)
    if nrm == 0.0:
        return np.zeros_like(x), 0.0
    x0 = x[0]
    phase = x0 / abs(x0) if x0 != 0 else 1.0
    v = x.astype(np.result_type(x, phase), copy=True)
    v[0] += phase * nrm
    beta = 2.0 / np.real(np.vdot(v, v))
    return v, beta


def starm_qr(A: np.ndarray, b: int, *, transform_name: str = "DFT", rule: str = "tube_norm",
             k: int | None = None, want_q: bool = False) -> dict:
    """Pivoted (or unpivoted) *_M-QR of fold1(A) with a shared permutation.

    Returns dict(order, Rhat, Qhat (if want_q), M, rule, transform) where ``order`` is
    the full block order (the first k entries are the pivots chosen).
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    M = transform(transform_name, b)
    T = fold1(np.asarray(A, dtype=np.float64), b)
    m, n, _ = T.shape
    kk = min(m, n) if k is None else min(k, m, n)
    dtype = np.complex128 if np.iscomplexobj(M) else np.float64
    R = mode3(T.astype(dtype), M).copy()                # transform domain, m x n x b
    Q = np.repeat(np.eye(m, dtype=dtype)[:, :, None], b, axis=2) if want_q else None
    order = np.arange(n)
    for t in range(kk):
        if rule == "none":
            p = t
        else:
            res = np.abs(R[t:, t:, :]) ** 2
            score = res[:, :, 0].sum(0) if rule == "first_slice" else res.sum(axis=(0, 2))
            p = t + int(np.argmax(score))
        if p != t:
            R[:, [t, p], :] = R[:, [p, t], :]
            order[[t, p]] = order[[p, t]]
        for l in range(b):
            v, beta = _house(R[t:, t, l])
            if beta == 0.0:
                continue
            blk = R[t:, t:, l]
            blk -= beta * np.outer(v, v.conj() @ blk)
            R[t + 1:, t, l] = 0.0
            if want_q:
                Ql = Q[:, t:, l]
                Ql -= beta * np.outer(Ql @ v, v.conj())
    return {"order": [int(j) for j in order], "Rhat": R, "Qhat": Q, "M": M,
            "rule": rule, "transform": transform_name, "steps": kk}


def check_starm_qr(A: np.ndarray, b: int, out: dict) -> dict:
    """Residual of (*) in the transform domain, orthogonality of every Qhat_l, and the
    f-upper-triangular structure of Rhat (relative to ||Rhat||)."""
    T = fold1(np.asarray(A, dtype=np.float64), b)
    That = mode3(T.astype(out["Rhat"].dtype), out["M"])
    R, Q = out["Rhat"], out["Qhat"]
    m, n, _ = R.shape
    lhs = That[:, out["order"], :]
    rhs = np.einsum("ikl,kjl->ijl", Q, R)
    nrm = np.linalg.norm(R)
    below = np.sqrt(sum(np.linalg.norm(np.tril(R[:, :, l], -1)) ** 2 for l in range(b))) / nrm
    orth = max(np.linalg.norm(Q[:, :, l].conj().T @ Q[:, :, l] - np.eye(m)) for l in range(b))
    return {"identity": float(np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs)),
            "orth_Q": float(orth), "f_upper_below": float(below)}
