"""CQRRPT and randomized strong RRQR.

CQRRPT (Melnichenko, Balabanov, Murray, Demmel, Mahoney, Luszczek, arXiv:2311.08316)
is CholeskyQR with Randomization and Pivoting for Tall matrices:

  1. sketch  ``A_sk = S A``  with ``d = ceil(gamma * n)`` rows (sparse sign by default);
  2. pivoted QR of the sketch -> permutation ``p``, triangular ``R_sk``, rank estimate r;
  3. precondition ``A_pre = A[:, p[:r]] R_sk[:r, :r]^{-1}``;
  4. CholeskyQR of ``A_pre`` -> ``Q``, ``R_pre``;
  5. ``R = R_pre @ R_sk[:r, :]``.

The pivots come entirely from step 2, whose kernel is ``pivot_kernel``: ``cqrrpt`` uses
GEQP3; ``rand_srrqr`` ("randomized sRRQR" in the paper) uses the Gu--Eisenstat strong
RRQR, so the f-bound certified on ``S A`` transfers to ``A`` up to the distortion of the
sketch; ``cqrrpt_hqrrp`` uses HQRRP.

CQRRPT assumes ``m >> n``; it warns when the aspect ratio is below ``MIN_ASPECT``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from .interface import QRCPResult, as_rng, get, register
from ..sketching import apply_sketch

MIN_ASPECT = 4.0


def _cholesky_qr(A_pre: np.ndarray, iters: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """CholeskyQR, optionally repeated.  Returns ``(Q, R, meta)``."""
    Q = A_pre
    R_total = None
    meta: dict = {"chol_fallback": False, "chol_iters_done": 0}
    for it in range(iters):
        G = Q.T @ Q
        try:
            R = cholesky(G, lower=False)
        except np.linalg.LinAlgError:
            # Sketch failed to precondition adequately; fall back so the
            # benchmark still produces a row instead of dying.
            warnings.warn(
                "CQRRPT: Cholesky of the Gram matrix failed -- the sketch did "
                "not precondition A well enough. Falling back to Householder QR "
                "for this run; raise gamma or use a denser sketch.",
                RuntimeWarning,
            )
            Q, R = np.linalg.qr(Q, mode="reduced")
            meta["chol_fallback"] = True
            R_total = R if R_total is None else R @ R_total
            meta["chol_iters_done"] = it + 1
            break
        Q = solve_triangular(R, Q.T, lower=False, trans="T").T
        R_total = R if R_total is None else R @ R_total
        meta["chol_iters_done"] = it + 1
    assert R_total is not None
    return Q, R_total, meta


def _cqrrpt(
    A: np.ndarray,
    *,
    gamma: float = 1.25,
    sketch: str = "sparse_sign",
    pivot_kernel: str = "geqp3",
    kernel_opts: dict | None = None,
    rank_tol: float = 1e-12,
    chol_iters: int = 1,
    rng=None,
    want_q: bool = True,
    k: int | None = None,
    name: str = "cqrrpt",
    **_,
) -> QRCPResult:
    rng = as_rng(rng)
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    aspect = m / n
    if aspect < MIN_ASPECT:
        warnings.warn(
            f"{name}: aspect ratio m/n = {aspect:.1f} < {MIN_ASPECT}; CQRRPT is "
            "designed for tall matrices and this comparison point is not "
            "representative.",
            RuntimeWarning,
        )

    d = int(min(m, np.ceil(gamma * n)))
    A_sk = apply_sketch(A, d, kind=sketch, rng=rng)

    kernel_opts = dict(kernel_opts or {})
    if pivot_kernel in ("srrqr", "srrqr_tol"):
        kernel_opts.setdefault("k", k if k is not None else min(d, n))
    sk_res = get(pivot_kernel)(A_sk, want_q=False, **kernel_opts)

    p = sk_res.p
    R_sk = sk_res.R
    diag = np.abs(np.diag(R_sk))
    if diag.size == 0 or diag[0] == 0.0:
        raise np.linalg.LinAlgError(f"{name}: sketch is numerically zero")
    r = int(np.count_nonzero(diag > rank_tol * diag[0]))
    r = max(r, 1)

    R11 = R_sk[:r, :r]
    A_p1 = np.ascontiguousarray(A[:, p[:r]])
    # A_pre = A_p1 @ inv(R11)   <=>   R11^T A_pre^T = A_p1^T
    A_pre = solve_triangular(R11, A_p1.T, lower=False, trans="T").T

    Q, R_pre, chol_meta = _cholesky_qr(A_pre, chol_iters)
    R = R_pre @ R_sk[:r, :]

    return QRCPResult(
        R=R,
        p=p,
        Q=Q if want_q else None,
        rank=r,
        name=name,
        meta={
            "gamma": gamma,
            "d": d,
            "sketch": sketch,
            "pivot_kernel": pivot_kernel,
            "rank_estimate": r,
            "aspect_ratio": aspect,
            "kappa_A_pre": float(np.linalg.cond(A_pre)) if n <= 2000 else None,
            "sketch_kernel_meta": sk_res.meta,
            **chol_meta,
        },
    )


@register("cqrrpt", "CQRRPT (RandLAPACK): sketch -> GEQP3 -> CholeskyQR, tall matrices")
def cqrrpt(A, **kw) -> QRCPResult:
    kw.setdefault("pivot_kernel", "geqp3")
    return _cqrrpt(A, name="cqrrpt", **kw)


@register(
    "rand_srrqr",
    "Randomized strong RRQR: sketch -> Gu-Eisenstat sRRQR on the sketch -> CholeskyQR",
)
def rand_srrqr(A, *, f=2.0, **kw) -> QRCPResult:
    """CQRRPT with the sketch-side pivoting kernel replaced by sRRQR.

    This is the construction to prefer over calling RQRCP or HQRRP a
    "randomized strong RRQR" -- those come with no theoretical bound on the
    error, whereas the f-bound certified on the sketch here transfers to A
    through the embedding.
    """
    kw["pivot_kernel"] = "srrqr"
    kernel_opts = dict(kw.pop("kernel_opts", None) or {})
    kernel_opts.setdefault("f", f)
    return _cqrrpt(A, name="rand_srrqr", kernel_opts=kernel_opts, **kw)


@register("cqrrpt_hqrrp", "CQRRPT with HQRRP as the sketch-side pivoting kernel")
def cqrrpt_hqrrp(A, **kw) -> QRCPResult:
    kw["pivot_kernel"] = "hqrrp"
    return _cqrrpt(A, name="cqrrpt_hqrrp", **kw)
