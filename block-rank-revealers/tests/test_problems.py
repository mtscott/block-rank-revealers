"""The block Kahan problem: Kahan's ties, the closed form for the shifted selection, and
block greedy against the column methods."""

import numpy as np
import pytest

from blockrr import core
from blockrr import problems as P
from blockrr.column import get


def test_kahan_unit_columns_and_ties():
    c = 0.3
    s = np.sqrt(1 - c * c)
    K = P.kahan(40, c, pert=0.0)
    assert np.allclose(np.linalg.norm(K, axis=0), 1.0)
    # in the natural order every remaining residual has norm s^t after t steps: R_tt = s^t
    R = np.linalg.qr(K, mode="r")
    assert np.allclose(np.abs(np.diag(R)), s ** np.arange(40))
    # MATLAB's perturbation of the diagonal
    Kp = P.kahan(40, c)
    assert np.allclose(np.diag(Kp) - np.diag(K), 25 * np.finfo(float).eps * np.arange(40, 0, -1))


@pytest.mark.parametrize("n,b,c", [(6, 3, 1 / 3), (8, 2, 0.5), (5, 4, 0.25), (6, 4, 0.5)])
def test_kahan_shift_closed_form(n, b, c):
    A = P.block_kahan(n, b, c=c, pert=0.0)
    for k in range(1, n):
        direct = (core.log_vol_S(A, list(range(1, k + 1)), b) - core.log_vol_S(A, list(range(k)), b)) / np.log(10)
        assert np.isclose(direct, P.kahan_shift_log10_ratio(b, c, k), rtol=0, atol=1e-10)
    # the per-block increase tends to the rate
    inc = P.kahan_shift_log10_ratio(b, c, 40) - P.kahan_shift_log10_ratio(b, c, 39)
    assert np.isclose(inc, P.kahan_shift_rate(b, c), atol=1e-8)


@pytest.mark.parametrize("b", [2, 3, 4, 8])
def test_kahan_rate_largest_at_one_over_b(b):
    cs = np.linspace(0.01, 0.99, 9801)
    rates = [P.kahan_shift_rate(b, c) for c in cs]
    assert abs(cs[int(np.argmax(rates))] - 1 / b) < 1e-3
    assert 0.217 < P.kahan_shift_rate(b, 1 / b) < 0.228


def test_block_kahan_greedy_skips_first_block():
    n, b = 8, 4
    A = P.block_kahan(n, b, rows=4 * n * b)
    assert A.shape == (128, 32)
    R0 = core.compress(A)
    g = core.select_block_greedy(R0, b, n - 1)
    assert g.order == list(range(1, n))
    for k in range(1, n):
        assert sorted(core.exhaustive(R0, b, k)["S_opt"]) == list(range(1, k + 1))
    for name in ("geqp3", "businger_golub"):
        order = core.induced_block_order(get(name)(A, want_q=False).p, b)
        assert order[: n - 1] == list(range(n - 1))
