"""The star_M product and Algorithm 2 (tensor block QR with greedy pivoting), literally.

Conventions, as in the paper:
    fold1(A)(i, j, l) = A_j(i, l)            (tube A(i, j, :) = row i of block j)
    (A x3 M)(i, j, :) = M @ A(i, j, :)
    a *M b = M^{-1} (M a  .*  M b)
    A *M B = (Ahat_l @ Bhat_l facewise, with Ahat = A x3 M) x3 M^{-1}

Algorithm 2 has two subroutines,
    ORTHOFACTORS(C_p) = [Q_hat, I_b]  (QR of C_p)   or  [U, V]  (SVD of C_p)
    SCOPE(t, n)       = [n]                          or  {t, ..., n},
and its line 9 rotates the tubal columns in J = SCOPE(t, n) by  x3 V^T, i.e. every
block A_j with j in J becomes A_j V.  The three instantiations studied in the paper
are (a) QR with SCOPE = [n], (b) SVD with SCOPE = {t, ..., n}, (c) SVD with SCOPE = [n];
``blockrr.algorithm2`` exposes them by name.

``check_algorithm2_tensor`` evaluates every identity the paper states for a choice and
reports the residual, so a convention mismatch shows up as an O(1) residual.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- basics

def fold1(A: np.ndarray, b: int) -> np.ndarray:
    """m x nb matrix -> m x n x b tensor; tubal column j is block j."""
    m, N = A.shape
    n = N // b
    return A.reshape(m, n, b)          # A[i, j*b + l] -> T[i, j, l]


def unfold1(T: np.ndarray) -> np.ndarray:
    """Inverse of fold1."""
    m, n, b = T.shape
    return T.reshape(m, n * b)


def mode3(T: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(T x3 M)(i, j, :) = M @ T(i, j, :)."""
    return np.einsum("pl,ijl->ijp", M, T)


def facewise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Frontal-slice-wise product: C(:, :, l) = A(:, :, l) @ B(:, :, l)."""
    return np.einsum("ikl,kjl->ijl", A, B)


def mprod(A: np.ndarray, B: np.ndarray, M: np.ndarray) -> np.ndarray:
    """A *M B."""
    return mode3(facewise(mode3(A, M), mode3(B, M)), np.linalg.inv(M))


def stack3(Mat: np.ndarray, b: int) -> np.ndarray:
    """The tensor with every frontal slice equal to Mat."""
    return np.repeat(Mat[:, :, None], b, axis=2)


def _perm(n: int, pi) -> np.ndarray:
    return np.eye(n)[:, list(pi)]


def _rel(X, Y):
    return float(np.linalg.norm(X - Y) / max(np.linalg.norm(Y), 1e-300))


# ---------------------------------------------------------------- Algorithm 2

def algorithm2_tensor(T: np.ndarray, k: int, *, ortho: str = "svd", scope: str = "all",
                      mode3_with: str = "VT"):
    """Algorithm 2 on the folded tensor T (m x n x b), k steps.

    ortho      : "qr" -> ORTHOFACTORS = [full Q of C_p, I_b];  "svd" -> [U, V].
    scope      : "all" -> SCOPE = [n];  "trailing" -> SCOPE = {t, ..., n}.
    mode3_with : "VT" (line 9 of the paper, x3 V^T) or "V" (x3 V, used only to show
                 that the transpose matters).

    Returns pi, Q, Rtilde, W, Vs (the V of every step).
    """
    T = np.array(T, dtype=np.float64, copy=True)
    m, n, b = T.shape
    pi = list(range(n))
    W = np.repeat(np.eye(b)[:, None, :], n, axis=1)            # W(:, j, :) = I_b
    Q = np.eye(m)
    Vs = []
    for t in range(k):
        r = t * b
        scores = []
        for j in range(t, n):
            s = np.linalg.svd(T[r:, j, :], compute_uv=False)
            scores.append(2.0 * float(np.sum(np.log(np.maximum(s, 1e-300)))))
        p = t + int(np.argmax(scores))
        T[:, [t, p], :] = T[:, [p, t], :]
        W[:, [t, p], :] = W[:, [p, t], :]
        pi[t], pi[p] = pi[p], pi[t]
        C = T[r:, t, :]
        if ortho == "qr":
            O, _ = np.linalg.qr(C, mode="complete")
            V = np.eye(b)
        elif ortho == "svd":
            O, _, Vt = np.linalg.svd(C, full_matrices=True)
            V = Vt.T
        else:
            raise ValueError(ortho)
        J = list(range(n)) if scope == "all" else list(range(t, n))
        Mv = V.T if mode3_with == "VT" else V
        T[:, J, :] = mode3(T[:, J, :], Mv)
        W[:, J, :] = mode3(W[:, J, :], Mv)
        G = np.eye(m)
        G[r:, r:] = O
        Q = Q @ G
        T = facewise(stack3(G.T, b), T)
        Vs.append(V)
    return pi, Q, T, W, Vs


def _wmat(W: np.ndarray, j: int) -> np.ndarray:
    """W(:, j, :) as a b x b matrix, rows = mode-1 index, columns = tube index."""
    return W[:, j, :]


def check_algorithm2_tensor(A: np.ndarray, b: int, *, ortho: str, scope: str,
                            mode3_with: str = "VT") -> dict:
    """Residuals of every identity the paper states for one choice (k = n).

    identity            : the star_M factorization identity for this choice
                          (QR: facewise with M = I; SVD, SCOPE = {t..n}: one identity per
                          tubal column with M = W_j^T; SVD, SCOPE = [n]: one M = W^T)
    orth_Q              : ||Q^T Q - I||
    below_block_diag    : unfold1(Rtilde) below its diagonal blocks, relative
    diag_blocks_offdiag : off-diagonal part of the diagonal blocks, relative
    diag_blocks_strict_lower : strictly lower part of the diagonal blocks, relative
    W_chain             : (SVD, {t..n}) max_j ||W(:, j+1, :) - W(:, j, :) V_{j+1}||
    W_all_equal         : (SVD, [n]) max_j ||W(:, j, :) - W(:, 1, :)||
    """
    Tn = fold1(A, b)
    m, n, _ = Tn.shape
    pi, Q, Rt, W, Vs = algorithm2_tensor(Tn, n, ortho=ortho, scope=scope, mode3_with=mode3_with)
    Ru = unfold1(Rt)
    out = {"pi": pi,
           "orth_Q": float(np.linalg.norm(Q.T @ Q - np.eye(m))),
           "below_block_diag": float(np.sqrt(sum(
               np.linalg.norm(Ru[(t + 1) * b:, t * b:(t + 1) * b]) ** 2 for t in range(n))) / np.linalg.norm(Ru)),
           "diag_blocks_offdiag": float(max(
               np.linalg.norm(Ru[t*b:(t+1)*b, t*b:(t+1)*b] - np.diag(np.diag(Ru[t*b:(t+1)*b, t*b:(t+1)*b])))
               for t in range(n)) / np.linalg.norm(Ru)),
           "diag_blocks_strict_lower": float(max(
               np.linalg.norm(np.tril(Ru[t*b:(t+1)*b, t*b:(t+1)*b], -1)) for t in range(n)) / np.linalg.norm(Ru))}
    Pt = stack3(_perm(n, pi), b)
    if ortho == "qr":
        # A *_{I_b} stack(I_n(:, pi)) = stack(Q) *_{I} Rtilde
        out["identity"] = _rel(facewise(Tn, Pt), facewise(stack3(Q, b), Rt))
    elif scope == "trailing":
        res = []
        for j in range(n):
            Wj = _wmat(W, j)
            Pj = mode3(Pt[:, j:j + 1, :], Wj)
            Rj = mode3(Rt[:, j:j + 1, :], Wj)
            Qj = mode3(stack3(Q, b), Wj)
            res.append(_rel(mprod(Tn, Pj, Wj.T), mprod(Qj, Rj, Wj.T)))
        out["identity"] = float(max(res))
        # W(:, j+1, :) = W(:, j, :) V, V from step j+1
        out["W_chain"] = float(max(
            np.linalg.norm(_wmat(W, j + 1) - _wmat(W, j) @ Vs[j + 1]) for j in range(n - 1)))
    else:
        W0 = _wmat(W, 0)
        out["W_all_equal"] = float(max(np.linalg.norm(_wmat(W, j) - W0) for j in range(n)))
        P = mode3(Pt, W0)
        Qs = mode3(stack3(Q, b), W0)
        R = mode3(Rt, W0)
        out["identity"] = _rel(mprod(Tn, P, W0.T), mprod(Qs, R, W0.T))
    return out
