"""Column-pivoted QR methods (blockrr.column)."""

import numpy as np
import pytest
from scipy.linalg import solve_triangular

from blockrr import column
from blockrr.column import get

FULL = ["businger_golub", "businger_golub_downdate", "geqp3", "hqrrp"]
TALL = ["cqrrpt", "rand_srrqr"]


def exp_decay(m, n, rate, rng):
    nn = min(m, n)
    U, _ = np.linalg.qr(rng.standard_normal((m, nn)))
    V, _ = np.linalg.qr(rng.standard_normal((n, nn)))
    return (U * np.exp(-rate * np.arange(nn))) @ V.T


def kahan(n, c=0.5):
    s = np.sqrt(1.0 - c * c)
    return np.diag(s ** np.arange(n, dtype=float)) @ (np.eye(n) - c * np.triu(np.ones((n, n)), 1))


def f_bound(R, k):
    return float(np.max(np.abs(solve_triangular(R[:k, :k], R[:k, k:], lower=False))))


@pytest.fixture(scope="module")
def tall():
    return exp_decay(2000, 40, 0.15, np.random.default_rng(0))


@pytest.mark.parametrize("name", FULL)
def test_full_factorization(tall, name):
    res = get(name)(tall, k=10, rng=np.random.default_rng(1))
    assert np.linalg.norm(tall[:, res.p] - res.Q @ res.R) < 1e-12 * np.linalg.norm(tall)
    assert np.linalg.norm(res.Q.T @ res.Q - np.eye(res.Q.shape[1])) < 1e-11
    assert np.allclose(res.R, np.triu(res.R), atol=0)


def test_businger_golub_and_geqp3_agree_on_benign_input(tall):
    assert np.array_equal(get("businger_golub")(tall, want_q=False).p, get("geqp3")(tall, want_q=False).p)


def test_srrqr_certifies_the_f_bound():
    K = kahan(96)
    bg = get("businger_golub")(K, want_q=False)
    sr = get("srrqr")(K, k=48, f=2.0)
    assert sr.meta["f_bound_certified"] and f_bound(sr.R, 48) <= 2.0 * (1 + 1e-8)
    assert f_bound(bg.R, 48) > 1e3 * f_bound(sr.R, 48)


@pytest.mark.parametrize("name", TALL)
def test_sketched_methods(tall, name):
    res = get(name)(tall, k=10, rng=np.random.default_rng(3))
    r = res.R.shape[0]
    assert res.rank is not None and res.rank > 0
    assert np.linalg.norm(tall[:, res.p[:r]] - res.Q @ res.R[:, :r]) < 1e-8 * np.linalg.norm(tall)
    assert np.array_equal(np.sort(res.p), np.arange(tall.shape[1]))


def test_cqrrpt_detects_exact_rank():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((4000, 12)) @ rng.standard_normal((12, 50))
    assert get("cqrrpt")(A, rng=rng, rank_tol=1e-10).rank == 12


def test_cqrrpt_warns_on_square_input():
    with pytest.warns(RuntimeWarning, match="aspect ratio"):
        get("cqrrpt")(np.random.default_rng(5).standard_normal((80, 80)), rng=0)


def test_registry():
    assert {"businger_golub", "geqp3", "hqrrp", "cqrrpt", "srrqr", "rand_srrqr"} <= set(column.available())
    assert column.RANDOMIZED >= {"cqrrpt", "rand_srrqr", "hqrrp"}
