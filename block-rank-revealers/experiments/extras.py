#!/usr/bin/env python3
"""Small additions used in the text: optimal sets at small k on the Perdigão problems
(with tower names), and wall-clock times.

    python experiments/extras.py --npz data/perdigao_iop.npz      # -> results/extras.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import blockrr  # noqa: E402
from blockrr import core  # noqa: E402
from blockrr.column import get  # noqa: E402
from blockrr.perdigao import load, MEANS, TURB  # noqa: E402


def timeit(fn, reps=3):
    best = np.inf
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/extras.json"))
    args = ap.parse_args()
    P1 = load(args.npz, MEANS)
    P2 = load(args.npz, TURB)
    Xw = P1["A"] / np.sqrt(P1["m"])
    probs = {"pmeans": (P1["A"], 4, P1["names"]), "pturb": (P2["A"], 8, P2["names"]),
             "pbayes": (core.augment(Xw, 3.0), 4, P1["names"])}
    out = {}
    for key, (A, b, names) in probs.items():
        R0 = core.compress(A)
        n = A.shape[1] // b
        g = blockrr.algorithm2(R0, b, 8)                     # block greedy, variant (a)
        rec = {"greedy_first8": [names[j] for j in g.selected]}
        for k in (1, 2, 3, 4):
            e = core.exhaustive(R0, b, k, limit=10**6)
            rec[f"opt{k}"] = [names[j] for j in sorted(e["S_opt"])]
            rec[f"worst{k}"] = [names[j] for j in sorted(e["S_worst"])]
            rec[f"opt{k}_log10"] = e["opt"] / np.log(10)
            rec[f"greedy{k}_log10"] = g.log_volumes[k - 1] / np.log(10)
            rec[f"greedy{k}_set"] = sorted(names[j] for j in g.selected[:k])
            rec[f"greedy{k}_is_opt"] = sorted(g.selected[:k]) == sorted(e["S_opt"])
        # single-tower volumes: is the greedy first tower the best single tower (it must be)
        sv = [core.log_vol_S(R0, [j], b) for j in range(n)]
        rec["best_single"] = names[int(np.argmax(sv))]
        # timings (seconds), K = 36 blocks, on the compressed factor where the method allows
        K = 36
        tc, _ = timeit(lambda: core.compress(A), reps=3)
        rec["t_compress"] = tc
        rec["t_greedy_K36_on_R0"], _ = timeit(lambda: core.select_block_greedy(R0, b, K), reps=3)   # Algorithm 1
        for v in blockrr.VARIANTS:                                                                  # Algorithm 2
            rec[f"t_algorithm2{v}_K36_on_R0"], _ = timeit(lambda: blockrr.algorithm2(R0, b, K, variant=v), reps=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec["t_geqp3_on_A"], _ = timeit(lambda: get("geqp3")(A, want_q=False), reps=3)
            rec["t_bg_on_A"], _ = timeit(lambda: get("businger_golub")(A, want_q=False), reps=1)
            rec["t_srrqr_k72_on_A"], r = timeit(lambda: get("srrqr")(A, k=min(18 * b, A.shape[1] - 1), want_q=False), reps=1)
            rec["srrqr_k72_swaps"] = int(r.meta["swaps"])
        rec["m"], rec["N"] = A.shape
        out[key] = rec
        print(key, json.dumps({k: v for k, v in rec.items() if not k.startswith("t_")})[:400], flush=True)
        print(key, {k: round(v, 3) for k, v in rec.items() if k.startswith("t_")}, flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
