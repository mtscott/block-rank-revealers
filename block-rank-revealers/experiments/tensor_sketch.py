#!/usr/bin/env python3
"""The star_M experiments of the paper (Section 4.7) and the sketched Example 2.1 (Figure 1).

1. Tensor methods on the trap family and the block devil's stairs: the three
   instantiations of Algorithm 2 (QR/[n], SVD/{t..n}, SVD/[n]) and the pivoted
   *_M-QR factorizations of the literature (unpivoted, Kilmer-Martin-Perrone 2008;
   first-slice pivoting, Hao-Kilmer-Braman-Hoover 2013; and the all-slices
   tube-norm variant of Section 3.4 of the paper) for M in {I, DFT, DCT, Haar}, run on A and on
   Gaussian sketches Omega A with d = ceil(1.25 N) and d = 2N rows (500 each).
2. Example 2.1 run on a Gaussian sketch Omega A with d = ceil(1.25 N) and d = 2N
   rows (N = 6 columns), for X alone and for A = [alpha X; I].

    python experiments/tensor_sketch.py      # ~2 min -> results/tensor_sketch.json  (needs results/analysis.json)
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import blockrr  # noqa: E402
from blockrr import core, problems as P, starm_qr as SM  # noqa: E402
from blockrr.column import get  # noqa: E402

LN10 = math.log(10.0)
OUT = ROOT / "results/tensor_sketch.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------
# 1. tensor methods
# ----------------------------------------------------------------------------

def tensor_methods(results):
    probs = {"trap": (P.trap_family(8, 4, 0.5, 1e-2)[0], 4, 9),
             "devil": (P.block_devil_stairs(3000, 16, 4, rng=7), 4, 15)}
    out = {}
    for key, (A, b, K) in probs.items():
        R0 = core.compress(A)
        best = np.array(results[key]["best"])
        opt = np.array([np.nan if v is None else v for v in results[key]["opt"]])
        traj = lambda order: np.array(core.trajectory(R0, list(order)[:K], b, K))
        g = core.select_block_greedy(R0, b, K).order
        tg = traj(g)
        rec = {"K": K, "n": R0.shape[1] // b, "b": b, "methods": {}}

        def add(name, order, *, family, M, rule, check=None, seconds=None):
            tr = traj(order)
            gap = (best - tr) / LN10
            rec["methods"][name] = {
                "family": family, "M": M, "rule": rule, "order": [int(j) for j in order],
                "gap": gap, "max_gap": float(gap.max()), "k_at_max": int(np.argmax(gap)) + 1,
                "n_exact_opt": int(np.sum(np.abs(tr - opt) < 1e-8)),
                "max_dlog10_vs_greedy": float(np.max(np.abs(tr - tg)) / LN10),
                "check": check, "seconds": seconds}

        for v in blockrr.VARIANTS:                              # Algorithm 2 (a), (b), (c)
            lab = f"Alg. 2 ({v})"
            t0 = time.perf_counter()
            pi = blockrr.algorithm2(R0, b, K, variant=v).order
            dt = time.perf_counter() - t0
            add(lab, pi, family="alg2", M={"Alg. 2 (a)": "I", "Alg. 2 (b)": "W_j (learned)",
                                           "Alg. 2 (c)": "W (learned)"}[lab],
                rule="log det, whole tubal column", seconds=dt)
        for rule in SM.RULES:
            for tn in SM.TRANSFORMS:
                t0 = time.perf_counter()
                res = SM.starm_qr(R0, b, transform_name=tn, rule=rule, want_q=True)
                dt = time.perf_counter() - t0
                chk = SM.check_starm_qr(R0, b, res)
                add(f"{rule}|{tn}", res["order"], family=rule, M=tn, rule=rule, check=chk, seconds=dt)
        rec["greedy_order"] = [int(j) for j in g]
        out[key] = rec
        log("tensor", key, "done")
    return out


def tensor_sketched(results, tensor_unsketched, n_seeds=500):
    """The same methods run on a Gaussian sketch Omega A (d = ceil(1.25 N) and 2N rows),
    with every selection evaluated on A against the exact optimum.

    For each seed one sketch is drawn and every method is run on it, so the methods
    are compared on the same sketches.  Algorithm 2 (a)-(c) are run on every sketch
    and checked against Algorithm 1 (Proposition 4); the unpivoted *_M-QR keeps the
    input order, so the sketch does not change it.
    """
    probs = {"trap": (P.trap_family(8, 4, 0.5, 1e-2), 4, 9),
             "devil": ((P.block_devil_stairs(3000, 16, 4, rng=7), None), 4, 15)}
    rules = [(r, t) for r in ("first_slice", "tube_norm") for t in SM.TRANSFORMS]
    out = {}
    for pi_, (key, ((A, good), b, K)) in enumerate(probs.items()):
        R0 = core.compress(A)
        m, N = A.shape
        opt = np.array(results[key]["opt"], float)
        assert np.all(np.isfinite(opt)) and len(opt) == K
        sizes = {"1.25": int(math.ceil(1.25 * N)), "2.0": 2 * N}
        unsk = {nm: r["order"][:K] for nm, r in tensor_unsketched[key]["methods"].items()}
        unsk["Alg. 1"] = tensor_unsketched[key]["greedy_order"][:K]
        unsk["dgeqp3"] = core.induced_block_order(get("geqp3")(A, want_q=False).p, b)[:K]
        rec = {"K": K, "N": N, "m": m, "b": b, "sizes": sizes, "seeds": n_seeds, "by_size": {}}
        for si, (gname, d) in enumerate(sizes.items()):
            t0 = time.perf_counter()
            loss, same, alg2_same = {}, {}, {v: 0 for v in blockrr.VARIANTS}
            traps_taken = []
            for s in range(n_seeds):
                rng = np.random.default_rng([20261023, pi_, si, s])
                SA = (rng.standard_normal((d, m)) / math.sqrt(d)) @ A
                Rs = core.compress(SA)
                orders = {"Alg. 1": core.select_block_greedy(Rs, b, K).order[:K]}
                for v in blockrr.VARIANTS:
                    pi2 = blockrr.algorithm2(Rs, b, K, variant=v).order
                    alg2_same[v] += int([int(j) for j in pi2[:K]] == [int(j) for j in orders["Alg. 1"]])
                for rule, tn in rules:
                    orders[f"{rule}|{tn}"] = SM.starm_qr(Rs, b, transform_name=tn, rule=rule, k=K)["order"][:K]
                orders["dgeqp3"] = core.induced_block_order(get("geqp3")(SA, want_q=False).p, b)[:K]
                for nm, o in orders.items():
                    tr = np.array(core.trajectory(R0, list(o), b, K))
                    loss.setdefault(nm, []).append((opt - tr) / LN10)
                    same.setdefault(nm, []).append([int(j) for j in o] == [int(j) for j in unsk[nm]])
                if good is not None:
                    traps = [j for j in range(A.shape[1] // b) if j not in good]
                    traps_taken.append([sum(1 for j in orders[nm][:k] if j in traps)
                                        for nm in ("Alg. 1", "first_slice|DFT") for k in (2, 8)])
            stats = {}
            for nm, L in loss.items():
                L = np.array(L)                                 # (seeds, K)
                mx = L.max(1)
                stats[nm] = {"median_by_k": np.median(L, 0), "mean_by_k": L.mean(0),
                             "p10_by_k": np.quantile(L, 0.1, 0), "p90_by_k": np.quantile(L, 0.9, 0),
                             "max_by_k": L.max(0),
                             "median_max": float(np.median(mx)), "mean_max": float(mx.mean()),
                             "worst_max": float(mx.max()), "best_max": float(mx.min()),
                             "frac_same_as_unsketched": float(np.mean(same[nm])),
                             "frac_attains_opt": float(np.mean(L < 1e-8))}
            rec["by_size"][gname] = {"d": d, "stats": stats, "alg2_same_as_alg1": alg2_same,
                                     "seconds": time.perf_counter() - t0}
            if good is not None:
                tt = np.array(traps_taken)                      # (seeds, 4)
                rec["by_size"][gname]["traps"] = {
                    "alg1_k2_mean": float(tt[:, 0].mean()), "alg1_k8_both": float(np.mean(tt[:, 1] == 2)),
                    "alg1_k8_none": float(np.mean(tt[:, 1] == 0)),
                    "fs_k2_mean": float(tt[:, 2].mean()), "fs_k8_both": float(np.mean(tt[:, 3] == 2)),
                    "fs_k8_none": float(np.mean(tt[:, 3] == 0))}
            log("tensor sketch", key, gname, d, f"{time.perf_counter() - t0:.1f}s",
                "alg2 same", alg2_same)
        out[key] = rec
    return out


# ----------------------------------------------------------------------------
# 2. Example 2.1 on a Gaussian sketch
# ----------------------------------------------------------------------------

PAIRS = [(0, 1), (0, 2), (1, 2)]


def _logvol_batch(M):
    """log volume of each matrix in a stack (S, d, c), via singular values."""
    s = np.linalg.svd(M, compute_uv=False)
    return np.sum(np.log(np.maximum(s, 1e-300)), axis=-1)


def sketch_example(n_seeds=2000, eps=0.5):
    deltas = np.logspace(-8, -0.5, 31)
    cases = {"X": None, "10": 10.0, "100": 100.0, "1000": 1000.0, "10000": 10000.0}
    N = 6
    sizes = {"1.25": int(math.ceil(1.25 * N)), "2.0": 2 * N}
    rng = np.random.default_rng(20261023)
    out = {"N": N, "sizes": sizes, "deltas": deltas.tolist(), "seeds": n_seeds, "eps": eps, "curves": {}}
    for gname, d in sizes.items():
        for case, alpha in cases.items():
            rows = []
            for delta in deltas:
                X = P.example_2_1(eps, delta)
                A = X if alpha is None else core.augment(X, alpha)
                m = A.shape[0]
                cols = lambda S: np.concatenate([[2 * j, 2 * j + 1] for j in S])
                true_pair = np.array([core.log_vol(A[:, cols(S)]) for S in PAIRS])
                vbar = true_pair.max()
                G = rng.standard_normal((n_seeds, d, m)) / math.sqrt(d)
                SA = G @ A                                              # (S, d, 6)
                single = np.stack([_logvol_batch(SA[:, :, cols([j])]) for j in range(3)], 1)
                pair = np.stack([_logvol_batch(SA[:, :, cols(S)]) for S in PAIRS], 1)
                first = np.argmax(single, axis=1)
                # second step: the pair containing `first` with the larger volume
                cand = {0: [0, 1], 1: [0, 2], 2: [1, 2]}       # pairs (by index in PAIRS) containing block j
                chosen = np.array([cand[f][int(np.argmax(pair[s, cand[f]]))] for s, f in enumerate(first)])
                true_loss = (true_pair[chosen] - vbar) / LN10                 # on A
                in_sketch = (pair[np.arange(n_seeds), chosen] - pair.max(1)) / LN10   # on Omega A
                rows.append({"delta": float(delta),
                             "p_first_B1": float(np.mean(first == 0)),
                             "p_trap": float(np.mean(chosen != 2)),        # selected pair contains B1
                             "mean": float(true_loss.mean()), "median": float(np.median(true_loss)),
                             "p10": float(np.quantile(true_loss, 0.1)), "p90": float(np.quantile(true_loss, 0.9)),
                             "sketch_mean": float(in_sketch.mean()), "sketch_median": float(np.median(in_sketch)),
                             "unsketched": float((true_pair[0] - vbar) / LN10),   # greedy on A takes {B1,B2}
                             "trap_loss": float((true_pair[0] - vbar) / LN10)})
            out["curves"][f"{gname}|{case}"] = rows
            log("sketch", gname, case, "p_trap at smallest delta", rows[0]["p_trap"])
    return out


def main():
    res = json.load(open(ROOT / "results/analysis.json"))["results"]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    out["tensor"] = tensor_methods(res)
    out["tensor_sketch"] = tensor_sketched(res, out["tensor"])
    out["sketch_example"] = sketch_example()

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    OUT.write_text(json.dumps(conv(out)))
    log("wrote", OUT)


if __name__ == "__main__":
    main()
