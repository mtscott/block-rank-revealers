"""LAPACK ``dgeqp3`` -- the blocked, Level-3 QR with column pivoting.

Reached through ``scipy.linalg.lapack``, so the BLAS used is whatever SciPy was linked
against; ``backend_info()`` returns its name and version.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lapack

from .interface import QRCPResult, register


def backend_info() -> str:
    try:
        import scipy

        cfg = scipy.__config__.show(mode="dicts")  # type: ignore[attr-defined]
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        return f"{blas.get('name', '?')} {blas.get('version', '?')}"
    except Exception:
        return "unknown"


@register("geqp3", "LAPACK dgeqp3 (blocked Level-3 QRCP) via scipy")
def geqp3(A: np.ndarray, *, want_q: bool = True, **kw) -> QRCPResult:
    A = np.asarray(A, dtype=np.float64, order="F")
    m, n = A.shape
    nn = min(m, n)

    qr, jpvt, tau, work, info = lapack.dgeqp3(A, overwrite_a=False)
    if info != 0:
        raise RuntimeError(f"dgeqp3 returned info={info}")

    p = jpvt.astype(np.int64) - 1  # LAPACK is 1-based
    R = np.triu(qr[:nn, :])

    Q = None
    if want_q:
        q, work2, info2 = lapack.dorgqr(qr, tau)
        if info2 != 0:
            raise RuntimeError(f"dorgqr returned info={info2}")
        Q = q[:, :nn]

    return QRCPResult(
        R=R, p=p, Q=Q, rank=None, name="geqp3",
        meta={"lwork": int(work[0]), "blas": backend_info()},
    )
