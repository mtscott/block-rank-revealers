"""Algorithm 2 and its variants (a), (b), (c)."""

import numpy as np
import pytest

import blockrr
from blockrr import core, problems as P

VARIANTS = sorted(blockrr.VARIANTS)


def _random(m, n, b, seed, spread=3.0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n * b)) * 10 ** rng.uniform(-spread, spread, size=n * b)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_select_the_blocks_of_algorithm1(variant):
    for seed in range(30):
        rng = np.random.default_rng(seed)
        b, n = int(rng.integers(1, 5)), int(rng.integers(2, 9))
        A = _random(int(rng.integers(n * b, n * b + 12)), n, b, seed)
        assert blockrr.algorithm2(A, b, variant=variant).order == core.block_greedy_qr(A, b, n)["pi"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_identities_hold(variant):
    A = _random(30, 6, 4, 1)
    r = blockrr.check_algorithm2(A, 4, variant=variant)
    assert r["identity"] < 1e-12 and r["orth_Q"] < 1e-12 and r["below_block_diag"] < 1e-13
    if variant == "a":
        assert r["diag_blocks_strict_lower"] < 1e-13          # triangular diagonal blocks
    if variant == "b":
        assert r["diag_blocks_offdiag"] < 1e-12 and r["W_chain"] < 1e-12
    if variant == "c":
        assert r["W_all_equal"] < 1e-12


def test_line_9_needs_the_transpose():
    A = _random(30, 6, 4, 2)
    right = blockrr.check_algorithm2(A, 4, variant="b", mode3_with="VT")
    wrong = blockrr.check_algorithm2(A, 4, variant="b", mode3_with="V")
    assert right["diag_blocks_offdiag"] < 1e-12 and wrong["diag_blocks_offdiag"] > 1e-3
    assert wrong["identity"] < 1e-12                          # the identities still hold


@pytest.mark.parametrize("variant", VARIANTS)
def test_result_fields_and_compression(variant):
    A = _random(400, 8, 3, 3, spread=1.0)
    res = blockrr.algorithm2(A, 3, 5, variant=variant)
    assert res.compressed and res.Q.shape == (24, 24) and res.R.shape == (24, 8, 3)
    assert len(res.selected) == 5 and sorted(res.order) == list(range(8))
    assert np.allclose(res.log_volumes, core.trajectory(A, res.selected, 3))
    small = blockrr.algorithm2(A[:30], 3, 5, variant=variant, compress=False)
    assert small.Q.shape == (30, 30) and not small.compressed
    assert blockrr.select_blocks(A, 3, 5, variant=variant) == res.selected


def test_bad_variant():
    with pytest.raises(ValueError):
        blockrr.algorithm2(np.eye(4), 2, variant="d")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("eps,delta", [(0.5, 1e-2), (0.9, 1e-3)])
def test_example_2_1(variant, eps, delta):
    X = P.example_2_1(eps, delta)
    pred = P.example_2_1_predictions(eps, delta)
    g = blockrr.algorithm2(X, 2, 2, variant=variant)
    e = core.exhaustive(X, 2, 2)
    assert g.selected[0] == 0                                  # greedy takes B1 ...
    assert np.isclose(np.exp(g.log_volumes[-1]), pred["Vol(greedy)"], rtol=1e-10)
    assert sorted(e["S_opt"]) == [1, 2]                        # ... but the optimum is {B2, B3}
    assert np.isclose(np.exp(e["opt"]), pred["Vol(S*)"], rtol=1e-10)


@pytest.mark.parametrize("k,b", [(4, 2), (6, 3), (8, 4)])
def test_trap_family_rate(k, b):
    for delta in (1e-2, 1e-4):
        X, _ = P.trap_family(k, b, 0.5, delta)
        g = blockrr.algorithm2(X, b, k)
        e = core.exhaustive(X, b, k)
        slope = k - int(np.ceil(k / b))
        assert abs((g.log_volumes[-1] - e["opt"]) / np.log(10) - slope * np.log10(delta)) < 0.2


def test_prior_floor_scales_like_one_over_alpha():
    X = P.example_2_1(0.5, 1e-8)
    r = []
    for alpha in (100, 1000, 10000):
        A = core.augment(X, alpha)
        r.append(np.exp(blockrr.algorithm2(A, 2, 2).log_volumes[-1] - core.exhaustive(A, 2, 2)["opt"]) * alpha)
    assert np.ptp(r) < 0.05 and 1.8 < r[0] < 2.0
