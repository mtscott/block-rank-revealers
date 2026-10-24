"""Businger & Golub (1965) QR with column pivoting, Level-2, in two flavors.

``downdate=False`` (default, ``businger_golub``) is the textbook algorithm: the
trailing column norms are recomputed from scratch at every step.  This is the
Businger--Golub method of the paper.

``downdate=True`` (``businger_golub_downdate``) downdates the norms via
``||c||^2 -= r^2``, with the recomputation safeguard of LAPACK's ``dlaqp2``, and
reports how often the safeguard fired in ``meta['norm_recomputations']``.  The two
can select different pivots on ill-conditioned input.
"""

from __future__ import annotations

import numpy as np

from .interface import QRCPResult, apply_reflectors_to_identity, householder, register

# LAPACK dlaqp2 uses tol3z = sqrt(eps) for the downdating safeguard.
_TOL3Z = np.sqrt(np.finfo(np.float64).eps)


def _businger_golub(
    A: np.ndarray,
    *,
    downdate: bool = False,
    want_q: bool = True,
    **_,
) -> QRCPResult:
    R = np.array(A, dtype=np.float64, order="F", copy=True)
    m, n = R.shape
    nn = min(m, n)
    p = np.arange(n, dtype=np.int64)

    V: list[np.ndarray] = []
    betas: list[float] = []
    recomputes = 0

    if downdate:
        partial = np.linalg.norm(R, axis=0)
        exact = partial.copy()

    for j in range(nn):
        if downdate:
            norms = partial[j:]
        else:
            # textbook: recompute trailing norms every step
            norms = np.linalg.norm(R[j:, j:], axis=0)

        l = j + int(np.argmax(norms))
        if l != j:
            R[:, [j, l]] = R[:, [l, j]]
            p[[j, l]] = p[[l, j]]
            if downdate:
                partial[[j, l]] = partial[[l, j]]
                exact[[j, l]] = exact[[l, j]]

        v, beta = householder(R[j:, j])
        V.append(v)
        betas.append(beta)
        if beta != 0.0:
            block = R[j:, j:]
            block -= beta * np.outer(v, v @ block)
        R[j + 1 :, j] = 0.0

        if downdate and j + 1 < n:
            for c in range(j + 1, n):
                if partial[c] == 0.0:
                    continue
                temp = 1.0 - (abs(R[j, c]) / partial[c]) ** 2
                temp = max(temp, 0.0)
                # LAPACK's cancellation test (dlaqp2)
                temp2 = temp * (partial[c] / exact[c]) ** 2 if exact[c] > 0 else 0.0
                if temp2 <= _TOL3Z:
                    partial[c] = np.linalg.norm(R[j + 1 :, c])
                    exact[c] = partial[c]
                    recomputes += 1
                else:
                    partial[c] *= np.sqrt(temp)

    R_out = np.triu(R[:nn, :])
    Q = apply_reflectors_to_identity(V, betas, m, nn) if want_q else None
    return QRCPResult(
        R=R_out,
        p=p,
        Q=Q,
        rank=None,
        name="businger_golub_downdate" if downdate else "businger_golub",
        meta={"downdate": downdate, "norm_recomputations": recomputes},
    )


@register(
    "businger_golub",
    "Businger-Golub 1965, Level-2, trailing norms recomputed from scratch",
)
def businger_golub(A, *, want_q=True, **kw) -> QRCPResult:
    return _businger_golub(A, downdate=False, want_q=want_q)


@register(
    "businger_golub_downdate",
    "Businger-Golub with LAPACK-style norm downdating + cancellation safeguard",
)
def businger_golub_downdate(A, *, want_q=True, **kw) -> QRCPResult:
    return _businger_golub(A, downdate=True, want_q=want_q)
