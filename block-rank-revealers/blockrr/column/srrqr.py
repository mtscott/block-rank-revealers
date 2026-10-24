"""Gu & Eisenstat (1996) strong rank-revealing QR.

Algorithm 4 (fixed rank ``k``) and Algorithm 6 (error threshold) of "Efficient
algorithms for computing a strong rank-revealing QR factorization", SIAM J. Sci.
Comput. 17(4), 848-869.

The algorithm starts from a pivoted QR of ``A`` (``init``, GEQP3 by default) and then
swaps columns until the f-bound holds.  The swaps act only on the ``nn x n`` triangular
factor, never on the ``m x n`` input; ``meta`` records the time of the initial pivoted
QR and of the swaps separately, and the number of swaps.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import solve_triangular

from .interface import QRCPResult, get, register


def _swap_quality(R: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rho, AB, gamma)`` where ``rho[i, j]`` is the determinant gain
    from swapping column ``i`` (<= k) with column ``k + j``.

    Gu-Eisenstat eq. (3.2):  rho_ij = sqrt(AB_ij^2 + (gamma_j * omega_i)^2)
    with ``AB = R11^{-1} R12``, ``gamma_j = ||R22 e_j||``, and
    ``omega_i = ||e_i^T R11^{-1}||`` (the reciprocal of their omega).
    """
    R11 = R[:k, :k]
    R12 = R[:k, k:]
    R22 = R[k:, k:]

    AB = solve_triangular(R11, R12, lower=False)
    R11_inv = solve_triangular(R11, np.eye(k), lower=False)
    omega = np.linalg.norm(R11_inv, axis=1)          # ||e_i^T R11^{-1}||
    gamma = np.linalg.norm(R22, axis=0) if R22.size else np.zeros(R.shape[1] - k)

    rho = np.sqrt(AB**2 + np.outer(omega, gamma) ** 2)
    return rho, AB, gamma


def _srrqr_fixed_rank(
    A: np.ndarray,
    k: int,
    *,
    f: float = 2.0,
    want_q: bool = True,
    max_swaps: int | None = None,
    init: str = "geqp3",
    **_,
) -> QRCPResult:
    if f < 1.0:
        raise ValueError("f must be >= 1")
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    nn = min(m, n)
    k = int(min(k, nn))
    if max_swaps is None:
        max_swaps = 20 * n

    t0 = time.perf_counter()
    base = get(init)(A, want_q=want_q)
    t_init = time.perf_counter() - t0

    R = np.array(base.R, dtype=np.float64, copy=True)   # (nn, n)
    p = base.p.copy()
    Qacc = np.eye(nn) if want_q else None

    t1 = time.perf_counter()
    swaps = 0
    max_rho = np.nan
    if k < n:
        for swaps in range(max_swaps + 1):
            rho, _, _ = _swap_quality(R, k)
            i, j = np.unravel_index(int(np.argmax(rho)), rho.shape)
            max_rho = float(rho[i, j])
            if max_rho <= f or not np.isfinite(max_rho):
                break
            # swap column i (in the leading block) with column k + j
            perm = np.arange(n)
            perm[[i, k + j]] = perm[[k + j, i]]
            R = R[:, perm]
            p = p[perm]
            # retriangularise; R is nn x n so this is cheap relative to m x n
            Qw, R = np.linalg.qr(R, mode="reduced")
            if want_q:
                Qacc = Qacc @ Qw
        else:
            import warnings

            warnings.warn(
                f"sRRQR hit max_swaps={max_swaps} with max rho={max_rho:.3e} > f={f}; "
                "the f-bound is NOT certified for this result",
                RuntimeWarning,
            )
    t_swap = time.perf_counter() - t1

    Q = None
    if want_q:
        Q = base.Q @ Qacc

    return QRCPResult(
        R=np.triu(R),
        p=p,
        Q=Q,
        rank=k,
        name="srrqr",
        meta={
            "f": f,
            "k": k,
            "swaps": swaps,
            "max_rho_final": max_rho,
            "f_bound_certified": bool(np.isfinite(max_rho) and max_rho <= f),
            "t_init_qrcp": t_init,
            "t_swaps": t_swap,
            "init": init,
        },
    )


@register("srrqr", "Gu-Eisenstat strong RRQR, Algorithm 4 (fixed rank k)")
def srrqr(A, *, k=None, f=2.0, want_q=True, **kw) -> QRCPResult:
    if k is None:
        raise ValueError("srrqr requires a target rank k (pass k=... )")
    return _srrqr_fixed_rank(A, k, f=f, want_q=want_q, **kw)


@register("srrqr_tol", "Gu-Eisenstat strong RRQR, Algorithm 6 (error threshold)")
def srrqr_tol(A, *, tol=1e-10, f=2.0, want_q=True, k=None, **kw) -> QRCPResult:
    """Smallest ``k`` with ``sigma_max(R22) <= tol * ||A||_2``, refined by Alg. 4.

    Searches upward from the rank suggested by the initial QRCP diagonal, which
    is nearly always correct, so the loop typically executes once.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[1]
    nn = min(A.shape)

    probe = get("geqp3")(A, want_q=False)
    d = np.abs(np.diag(probe.R))
    thresh = tol * (d[0] if d.size else 1.0)
    k0 = int(np.count_nonzero(d > thresh)) or 1

    for kk in range(k0, nn + 1):
        res = _srrqr_fixed_rank(A, kk, f=f, want_q=want_q, **kw)
        R22 = res.R[kk:, kk:]
        err = np.linalg.norm(R22, 2) if R22.size else 0.0
        if err <= tol * np.linalg.norm(res.R, 2) or kk == nn:
            res.name = "srrqr_tol"
            res.meta.update({"tol": tol, "k_selected": kk, "sigma_max_R22": float(err)})
            return res
    raise AssertionError("unreachable")
