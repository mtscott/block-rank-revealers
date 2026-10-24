"""Pivoted star_M-QR factorizations of the literature (blockrr.starm_qr)."""

import numpy as np
import pytest

from blockrr import core, starm_qr as SM
from blockrr.column import get


@pytest.mark.parametrize("tname", ["I", "DFT", "DCT", "Haar"])
@pytest.mark.parametrize("rule", ["none", "first_slice", "tube_norm"])
def test_identity_and_structure(tname, rule):
    rng = np.random.default_rng(5)
    A = rng.standard_normal((30, 7 * 4)) * 10 ** rng.uniform(-2, 2, size=28)
    out = SM.starm_qr(A, 4, transform_name=tname, rule=rule, want_q=True)
    chk = SM.check_starm_qr(A, 4, out)
    assert chk["identity"] < 1e-13 and chk["orth_Q"] < 1e-12 and chk["f_upper_below"] < 1e-14
    assert sorted(out["order"]) == list(range(7))
    if rule == "none":
        assert out["order"] == list(range(7))


def test_first_slice_rule_is_transform_independent_when_first_row_is_constant():
    rng = np.random.default_rng(6)
    A = rng.standard_normal((40, 9 * 4))
    orders = {t: SM.starm_qr(A, 4, transform_name=t, rule="first_slice")["order"] for t in ("DFT", "DCT", "Haar")}
    assert orders["DFT"] == orders["DCT"] == orders["Haar"]
    # ... and it is Businger-Golub on the sum of each block's columns
    S = A.reshape(40, 9, 4).sum(axis=2)
    assert list(get("businger_golub")(S, want_q=False).p) == orders["DFT"]


def test_same_order_on_the_compressed_factor():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((500, 6 * 3))
    R0 = core.compress(A)
    for t in ("I", "DFT", "DCT"):
        assert SM.starm_qr(A, 3, transform_name=t)["order"] == SM.starm_qr(R0, 3, transform_name=t)["order"]


def test_tube_norm_with_b1_is_businger_golub():
    A = np.random.default_rng(8).standard_normal((25, 12))
    assert SM.starm_qr(A, 1, transform_name="I")["order"] == list(get("businger_golub")(A, want_q=False).p)


def test_haar_is_orthogonal():
    for b in (1, 2, 4, 8):
        H = SM.haar_matrix(b)
        assert np.allclose(H @ H.T, np.eye(b)) and np.allclose(H[0], 1 / np.sqrt(b))
