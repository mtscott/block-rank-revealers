"""Volumes, Algorithm 1, enumeration, and the bounds of the paper on small instances."""

from itertools import combinations

import numpy as np

import blockrr
from blockrr import core


def test_algorithm1_factorization():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((20, 15))
    b = 3
    for mode in ("svd", "householder"):
        out = core.block_greedy_qr(A, b, 5, mode=mode)
        AP = A[:, core.cols_of(out["pi"], b)]
        assert np.linalg.norm(AP - out["Q"] @ out["R"]) < 1e-12 * np.linalg.norm(A)
        assert np.allclose(out["log_vols"], core.trajectory(A, out["pi"][:5], b))
    assert blockrr.algorithm1(A, b, 5)["pi"] == core.block_greedy_qr(A, b, 5)["pi"]


def test_compression_keeps_volumes():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((300, 12))
    R0 = core.compress(A)
    for S in ([0], [1, 3], [0, 2, 3]):
        assert np.isclose(core.log_vol_S(A, S, 3), core.log_vol_S(R0, S, 3))


def test_exhaustive_matches_brute_force():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((12, 12))
    for k in (1, 2, 3):
        vals = {S: core.log_vol_S(A, S, 2) for S in combinations(range(6), k)}
        e = core.exhaustive(A, 2, k)
        assert np.isclose(e["opt"], max(vals.values())) and np.isclose(e["worst"], min(vals.values()))
    assert core.exhaustive(A, 2, 3, limit=5) is None


def test_augmented_volume_is_the_information_gain():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((5, 8))
    alpha = 2.0
    A = core.augment(X, alpha)
    S = [1, 3]
    XS = X[:, core.cols_of(S, 2)]
    expected = 0.5 * np.linalg.slogdet(np.eye(4) + alpha**2 * XS.T @ XS)[1]
    assert np.isclose(core.log_vol_S(A, S, 2), expected)


def test_induced_block_order():
    assert core.induced_block_order(np.array([5, 4, 0, 1, 2, 3]), 2) == [2, 0, 1]


def test_worst_case_bound_via_vq():
    """Vlow_k >= Vbar_k / Vbar_q with q = min(k, n - k), on random augmented instances."""
    rng = np.random.default_rng(3)
    for _ in range(60):
        m0, n, b = rng.integers(2, 7), rng.integers(3, 7), rng.integers(1, 3)
        k = int(rng.integers(1, n))
        X = rng.standard_normal((m0, n * b))
        A = core.augment(X, float(rng.choice([0.5, 2.0])))
        vols = [core.log_vol_S(A, S, b) for S in combinations(range(n), k)]
        q = min(k, n - k)
        Vq = max(core.log_vol_S(A, S, b) for S in combinations(range(n), q))
        assert min(vols) >= max(vols) - Vq - 1e-9


def test_greedy_guarantee():
    """log Vol(A_G) >= (1 - (1 - 1/k)^k) log Vbar_k for augmented A (submodularity)."""
    rng = np.random.default_rng(4)
    for _ in range(40):
        n, b = int(rng.integers(3, 7)), int(rng.integers(1, 3))
        A = core.augment(rng.standard_normal((int(rng.integers(1, 6)), n * b)), float(10 ** rng.uniform(-1, 1)))
        for k in range(1, n + 1):
            opt = core.exhaustive(A, b, k)["opt"]
            g = blockrr.algorithm2(A, b, k).log_volumes[-1]
            assert g >= (1 - (1 - 1 / k) ** k) * opt - 1e-10
