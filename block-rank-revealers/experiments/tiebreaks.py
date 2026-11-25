#!/usr/bin/env python3
"""Tie-breaks on the Perdigão problems, treated like seeds.

    python experiments/tiebreaks.py --npz data/perdigao_iop.npz      # ~2 min

The Perdigão columns are standardized, so every column, and every tower's block, has the
same norm.  The first step of every method that ranks columns or blocks by a norm is then
an exact tie, which rounding decides:

    dgeqp3, Businger-Golub     the first column, a tie among all N columns
    sRRQR                      starts from that column-pivoted QR
    first-slice *_M-QR, M = I  the first column of each block, a tie among the n towers
    tube-norm *_M-QR, any M    the Frobenius norm of each block, a tie among the n towers

Block greedy (the log-determinant of each block) and first-slice *_M-QR with the DFT or DCT
(the column sum of each block) are not tied.  For each tied method this script runs it once
from every possible first pick, then lets it continue by its own rule, and writes
results/tiebreaks.json: log Vol (natural log) of the first k blocks, k = 1..K, per first
pick.  analysis.py reports the median over first picks, and counts the best one in the best
known volume, as it does for the seeds of the randomized methods.

Each replay, started from the library's own first pick, reproduces the run saved in
results/benchmarks.json; the script checks this and stops if not.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from blockrr import core, starm_qr as SM  # noqa: E402
from blockrr.column.srrqr import _swap_quality  # noqa: E402
from blockrr.perdigao import MEANS, TURB, load  # noqa: E402
import run_benchmarks as RB  # noqa: E402

PROBLEMS = (("pmeans", "Perdigão: mean wind + temperature"),
            ("pturb", "Perdigão: wind, temperature + turbulence"),
            ("pbayes", "Perdigão: Bayesian D-optimal (alpha=3)"))
STARM_TIED = (("first_slice", "I"), ("tube_norm", "I"), ("tube_norm", "DFT"), ("tube_norm", "DCT"))


def prefix_logvols(R0, order, b, K):
    """log Vol of the first t blocks of order, t = 1..K, from one QR of their columns."""
    cols = core.cols_of(order[:K], b)
    d = np.log(np.abs(np.diag(np.linalg.qr(R0[:, cols], mode="r"))))
    return np.add.reduceat(d, np.arange(0, K * b, b)).cumsum()


def column_pivoting(R0, first):
    """Businger-Golub column pivoting with the first pivot forced: the full column order."""
    R = R0.copy()
    order, rem = [], list(range(R.shape[1]))
    for t in range(R.shape[1]):
        piv = first if t == 0 else rem[int(np.argmax((R[:, rem] ** 2).sum(0)))]
        order.append(piv)
        rem.remove(piv)
        q = R[:, piv].copy()
        nq = np.linalg.norm(q)
        if nq > 0:
            q /= nq
            R -= np.outer(q, q @ R)
    return np.array(order)


def srrqr_from(R0, p0, kcol, f=2.0):
    """The Gu-Eisenstat exchanges of blockrr.column.srrqr, started from the column order p0."""
    R = np.linalg.qr(R0[:, p0], mode="r")
    n = R.shape[1]
    p = np.arange(n)
    swaps = 0
    if kcol < n:
        for swaps in range(20 * n + 1):
            rho, _, _ = _swap_quality(R, kcol)
            i, j = np.unravel_index(int(np.argmax(rho)), rho.shape)
            if rho[i, j] <= f or not np.isfinite(rho[i, j]):
                break
            perm = np.arange(n)
            perm[[i, kcol + j]] = perm[[kcol + j, i]]
            R = R[:, perm]
            p = p[perm]
            _, R = np.linalg.qr(R, mode="reduced")
    return p0[p], swaps


def starm_order(R0, b, K, tn, rule, first):
    """Pivoted *_M-QR (first-slice or tube-norm pivoting) with the first block forced."""
    n = R0.shape[1] // b
    M = SM.transform(tn, b)
    T = SM.mode3(SM.fold1(R0, b).astype(np.complex128 if np.iscomplexobj(M) else float), M)
    faces = [T[:, :, l].copy() for l in range(b)]
    order, rem = [], list(range(n))
    for t in range(K):
        if t == 0:
            piv = first
        else:
            score = ((np.abs(faces[0][:, rem]) ** 2).sum(0) if rule == "first_slice"
                     else sum((np.abs(F[:, rem]) ** 2).sum(0) for F in faces))
            piv = rem[int(np.argmax(score))]
        order.append(piv)
        rem.remove(piv)
        for F in faces:
            q = F[:, piv].copy()
            nq = np.linalg.norm(q)
            if nq > 0:
                q /= nq
                F -= np.outer(q, q.conj() @ F)
    return order


def run(key, A, b, saved):
    t0 = time.time()
    R0 = core.compress(A)
    N = A.shape[1]
    n = N // b
    K = n - 1
    ks = list(range(1, K + 1))
    stored = {nm: np.array(m["logvol"][:K], float) for nm, m in saved["methods"].items()}
    cn = np.linalg.norm(A, axis=0)
    bn = np.array([np.linalg.norm(A[:, j * b:(j + 1) * b]) for j in range(n)])
    info = {"column_norm_spread": float(np.ptp(cn) / cn.max()), "block_norm_spread": float(np.ptp(bn) / bn.max())}
    runs = {}
    # column pivoting and sRRQR, from every first column
    cp, sr, swaps = [], [], 0
    for c in range(N):
        p0 = column_pivoting(R0, c)
        cp.append(prefix_logvols(R0, core.induced_block_order(p0, b), b, K))
        row = []
        for k in ks:
            pk, s = srrqr_from(R0, p0, min(k * b, N - 1))
            swaps += s
            row.append(core.log_vol_S(R0, core.induced_block_order(pk, b)[:k], b))
        sr.append(row)
    cp, sr = np.array(cp), np.array(sr)
    lib = int(RB._col_order(A, "geqp3", 0)[0])
    for nm, r in (("dgeqp3", cp), ("Businger-Golub", cp), ("sRRQR (Gu-Eisenstat)", sr)):
        if not np.allclose(r[lib], stored[nm], atol=1e-6):
            raise RuntimeError(f"[{key}] {nm}: the replay from the library's first column differs from the saved run")
        runs[nm] = r
    info["srrqr_exchanges"] = swaps
    # pivoted *_M-QR, from every first block
    for rule, tn in STARM_TIED:
        nm = f"{rule}|{tn}"
        lib_order = SM.starm_qr(R0, b, transform_name=tn, rule=rule, k=K)["order"]
        r = np.array([prefix_logvols(R0, starm_order(R0, b, K, tn, rule, j0), b, K) for j0 in range(n)])
        if not np.allclose(r[int(lib_order[0])], stored[nm], atol=1e-6):
            raise RuntimeError(f"[{key}] {nm}: the replay from the library's first block differs from the saved run")
        runs[nm] = r
    print(f"[{key}] N={N} n={n} b={b}: {time.time() - t0:.0f}s, {json.dumps(info)}", flush=True)
    return {"ks": ks, "first_picks": {nm: int(r.shape[0]) for nm, r in runs.items()}, "info": info,
            "runs": {nm: r.tolist() for nm, r in runs.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Perdigão .npz from blockrr.perdigao")
    ap.add_argument("--results", default=str(ROOT / "results/benchmarks.json"))
    ap.add_argument("--out", default=str(ROOT / "results/tiebreaks.json"))
    args = ap.parse_args()
    saved = {p["name"]: p for p in json.load(open(args.results))["problems"]}
    P1, P2 = load(args.npz, MEANS), load(args.npz, TURB)
    mats = {"pmeans": (P1["A"], P1["b"]), "pturb": (P2["A"], P2["b"]),
            "pbayes": (core.augment(P1["A"] / math.sqrt(P1["m"]), 3.0), P1["b"])}
    out = {key: run(key, *mats[key], saved[name]) for key, name in PROBLEMS}
    out["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
