"""Sketching operators S A, applied without forming S.

Gaussian (scaled by 1/sqrt(d)), sparse sign (``nnz`` nonzeros per column of S, as in
Martinsson & Tropp 2020), and a subsampled randomized trigonometric transform.  The
Gaussian and sparse sign operators work on row blocks, so tall inputs never need a
dense d x m matrix.
"""

from __future__ import annotations

import numpy as np

from ._util import as_rng

DEFAULT_ROW_BLOCK = 1 << 16


def gaussian(A: np.ndarray, d: int, rng, row_block: int = DEFAULT_ROW_BLOCK) -> np.ndarray:
    """Blocked Gaussian sketch, scaled by ``1/sqrt(d)``.

    Accumulates ``sum_b S_b @ A_b`` over row blocks so peak extra memory is
    ``row_block * d`` doubles rather than ``m * d``.
    """
    rng = as_rng(rng)
    m, n = A.shape
    out = np.zeros((d, n), dtype=np.float64)
    for start in range(0, m, row_block):
        stop = min(start + row_block, m)
        S_blk = rng.standard_normal((d, stop - start))
        out += S_blk @ A[start:stop]
    out /= np.sqrt(d)
    return out


def sparse_sign(A: np.ndarray, d: int, rng, nnz: int = 8) -> np.ndarray:
    """Sparse sign embedding (a SASO / CountSketch-style operator).

    Each row of ``A`` is scattered, with random signs, into ``nnz`` of the ``d``
    sketch rows.  Cost is ``O(nnz * m * n)`` with no dense operator formed.
    This is the operator RandBLAS defaults to for CQRRPT.
    """
    rng = as_rng(rng)
    m, n = A.shape
    nnz = int(min(nnz, d))
    out = np.zeros((d, n), dtype=np.float64)
    scale = 1.0 / np.sqrt(nnz)

    block = max(1, DEFAULT_ROW_BLOCK // max(nnz, 1))
    for start in range(0, m, block):
        stop = min(start + block, m)
        rows = A[start:stop]
        b = stop - start
        # nnz distinct target rows per source row
        targets = np.argsort(rng.random((b, d)), axis=1)[:, :nnz]
        signs = rng.integers(0, 2, size=(b, nnz)) * 2.0 - 1.0
        for c in range(nnz):
            np.add.at(out, targets[:, c], rows * signs[:, c, None])
    out *= scale
    return out


def srft(A: np.ndarray, d: int, rng) -> np.ndarray:
    """Subsampled randomised Fourier/cosine transform.

    Random sign flip, DCT-II along rows, then uniform row subsampling.  Uses
    ``scipy.fft`` so it is ``O(m n log m)``.  Unlike the two above it touches
    all of ``A`` at once, so prefer ``sparse_sign`` for very tall inputs.
    """
    from scipy.fft import dct

    rng = as_rng(rng)
    m, n = A.shape
    signs = rng.integers(0, 2, size=m) * 2.0 - 1.0
    B = dct(A * signs[:, None], type=2, axis=0, norm="ortho")
    idx = rng.choice(m, size=d, replace=False)
    return B[idx] * np.sqrt(m / d)


OPERATORS = {
    "gaussian": gaussian,
    "sparse_sign": sparse_sign,
    "srft": srft,
}


def apply_sketch(A: np.ndarray, d: int, kind: str = "sparse_sign", rng=None, **kw):
    """Dispatch to a named operator, returning the ``d x n`` sketch of ``A``."""
    if kind not in OPERATORS:
        raise KeyError(f"unknown sketch {kind!r}; available: {sorted(OPERATORS)}")
    d = int(min(d, A.shape[0]))
    return OPERATORS[kind](A, d, rng, **kw)
