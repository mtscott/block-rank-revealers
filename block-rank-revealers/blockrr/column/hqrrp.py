"""HQRRP-style blocked randomized pivoted QR.

Householder QR with Randomization for Pivoting, after Martinsson (2015), Duersch & Gu
(2017), and Martinsson, Quintana-Orti, Heavner & van de Geijn (SISC 2017).  A sample
``Y = Omega * A_trailing`` with ``block + oversample`` rows is pivoted with GEQP3 to
choose ``block`` columns at a time; the defaults are ``block = 64``, ``oversample = 10``.

This is a reference implementation for comparing pivot quality: it re-sketches the
trailing matrix for every block instead of downdating the sample, so its running time
is not representative of optimized implementations.  Like RQRCP, HQRRP comes with no
strong-RRQR guarantee.
"""

from __future__ import annotations

import numpy as np

from .interface import (
    QRCPResult,
    apply_reflectors_to_identity,
    as_rng,
    householder,
    register,
)


@register("hqrrp", "HQRRP: blocked randomized pivoted QR (no strong-RRQR guarantee)")
def hqrrp(
    A: np.ndarray,
    *,
    block: int = 64,
    oversample: int = 10,
    rng=None,
    want_q: bool = True,
    **_,
) -> QRCPResult:
    rng = as_rng(rng)
    R = np.array(A, dtype=np.float64, order="F", copy=True)
    m, n = R.shape
    nn = min(m, n)
    p = np.arange(n, dtype=np.int64)

    V: list[np.ndarray] = []
    betas: list[float] = []
    b = int(max(1, min(block, nn)))
    s = int(max(0, oversample))
    n_blocks = 0

    j = 0
    while j < nn:
        bs = min(b, nn - j)
        rows = m - j
        cols = n - j
        ell = min(bs + s, rows)

        # --- randomized pivot selection on the trailing submatrix ---------
        Omega = rng.standard_normal((ell, rows))
        Y = Omega @ R[j:, j:]                       # (ell, cols)
        from scipy.linalg import lapack

        _, jpvt, _, _, info = lapack.dgeqp3(np.asfortranarray(Y))
        if info != 0:
            raise RuntimeError(f"dgeqp3 on the sample matrix returned info={info}")
        order = jpvt.astype(np.int64) - 1

        # apply the block of pivots to the full trailing matrix
        R[:, j:] = R[:, j + order]
        p[j:] = p[j + order]
        n_blocks += 1

        # --- unpivoted Householder QR on the panel -------------------------
        for c in range(j, j + bs):
            v, beta = householder(R[c:, c])
            V.append(v)
            betas.append(beta)
            if beta != 0.0:
                blk = R[c:, c:]
                blk -= beta * np.outer(v, v @ blk)
            R[c + 1 :, c] = 0.0
        j += bs

    R_out = np.triu(R[:nn, :])
    Q = apply_reflectors_to_identity(V, betas, m, nn) if want_q else None
    return QRCPResult(
        R=R_out,
        p=p,
        Q=Q,
        rank=None,
        name="hqrrp",
        meta={"block": b, "oversample": s, "n_blocks": n_blocks,
              "sample_update": "re-sketched per block (not Duersch-Gu downdate)"},
    )
