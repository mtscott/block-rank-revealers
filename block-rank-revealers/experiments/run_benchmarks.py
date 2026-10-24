#!/usr/bin/env python3
"""Volume of the selected blocks against the number of blocks, for every method and problem.

    python experiments/run_benchmarks.py --npz data/perdigao_iop.npz   # ~20 min
    python experiments/run_benchmarks.py                                # synthetic problems only

Writes results/benchmarks.json: for each problem and budget k = 1..K, log Vol of the k
blocks each method selects, the exact optimum and worst selection where enumeration is
affordable, random selections, and a Fischer upper bound; plus the Example 2.1 and trap
family sweeps.

Block greedy is Algorithm 2, run with each of its variants (a), (b), (c); the three must
select blocks of the same volume at every k (Proposition 4 of the paper), and are checked
against Algorithm 1.  A column method selects a block the first time one of its columns
is pivoted (``core.induced_block_order``).
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
from blockrr import core, problems as P  # noqa: E402
from blockrr.column import RANDOMIZED, get  # noqa: E402
from blockrr.perdigao import MEANS, TURB, load  # noqa: E402

GREEDY = "Block greedy (Alg. 2)"
COLUMN_METHODS = ["geqp3", "businger_golub", "hqrrp", "cqrrpt"]   # nested
PER_K_COLUMN = ["srrqr", "rand_srrqr"]                             # need k
N_SEEDS = 10


def _col_order(A, name, seed, kcol=None):
    opts = {"want_q": False}
    if name in RANDOMIZED:
        opts["rng"] = np.random.default_rng(seed)
    if kcol is not None:
        opts["k"] = kcol
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get(name)(A, **opts).p


def block_greedy(R0, b, K):
    """Algorithm 2 with variants (a), (b), (c), checked against each other and Algorithm 1."""
    runs = {v: blockrr.algorithm2(R0, b, K, variant=v) for v in blockrr.VARIANTS}
    alg1 = core.select_block_greedy(R0, b, K)
    ref = np.array(alg1.log_vols)
    dev = {v: float(np.max(np.abs(np.array(r.log_volumes) - ref))) for v, r in runs.items()}
    if max(dev.values()) > 1e-9 * max(1.0, float(np.max(np.abs(ref)))):
        raise RuntimeError(f"Algorithm 2 variants disagree with Algorithm 1: {dev}")
    meta = {"orders": {v: r.selected for v, r in runs.items()}, "algorithm1_order": alg1.order,
            "max_abs_dev_from_algorithm1": dev}
    return runs["a"], meta


def run_problem(name, A, b, K=None, *, desc="", exhaustive_limit=200_000, tall=None):
    t0 = time.time()
    A = np.asarray(A, dtype=float)
    R0 = core.compress(A)
    N = A.shape[1]
    n = N // b
    K = min(K or n - 1, n - 1, A.shape[0] // b)
    ks = list(range(1, K + 1))
    tall = (A.shape[0] / N >= 4.0) if tall is None else tall
    lv = lambda S: core.log_vol_S(R0, S, b)
    out = {"name": name, "desc": desc, "m": A.shape[0], "n": n, "b": b,
           "ks": ks, "methods": {}, "ref": {}}

    def add(label, vals, *, nested, family, lo=None, hi=None, note=""):
        out["methods"][label] = {"logvol": [None if not np.isfinite(v) else float(v) for v in vals],
                                 "nested": nested, "family": family, "note": note}
        if lo is not None:
            out["methods"][label]["lo"] = [float(v) for v in lo]
            out["methods"][label]["hi"] = [float(v) for v in hi]

    # block greedy = Algorithm 2 (a), (b), (c)
    g, meta = block_greedy(R0, b, K)
    add(GREEDY, g.log_volumes, nested=True, family="paper",
        note="Algorithm 2, variants (a), (b), (c): the same blocks at every k; checked against Algorithm 1")
    out["block_greedy"] = meta

    # column methods, block order induced by first appearance
    for nm in COLUMN_METHODS:
        if nm == "cqrrpt" and not tall:
            continue
        seeds = range(N_SEEDS) if nm in RANDOMIZED else [0]
        runs = []
        for s in seeds:
            order = core.induced_block_order(_col_order(A, nm, s), b)
            runs.append([lv(order[:k]) for k in ks])
        runs = np.array(runs)
        label = {"geqp3": "dgeqp3", "businger_golub": "Businger-Golub",
                 "hqrrp": "HQRRP", "cqrrpt": "CQRRPT"}[nm]
        if len(runs) > 1:
            add(label, np.median(runs, 0), nested=True, family="column",
                lo=runs.min(0), hi=runs.max(0), note=f"median of {len(runs)} seeds")
        else:
            add(label, runs[0], nested=True, family="column")
    for nm in PER_K_COLUMN:
        if nm == "rand_srrqr" and not tall:
            continue
        seeds = range(5) if nm in RANDOMIZED else [0]
        runs = []
        for s in seeds:
            row = []
            for k in ks:
                kcol = min(k * b, N - 1)
                order = core.induced_block_order(_col_order(A, nm, s, kcol=kcol), b)
                row.append(lv(order[:k]))
            runs.append(row)
        runs = np.array(runs)
        label = {"srrqr": "sRRQR (Gu-Eisenstat)", "rand_srrqr": "Randomized sRRQR"}[nm]
        if len(runs) > 1:
            add(label, np.median(runs, 0), nested=False, family="column",
                lo=runs.min(0), hi=runs.max(0), note=f"median of {len(runs)} seeds")
        else:
            add(label, runs[0], nested=False, family="column")

    # references
    rng = np.random.default_rng(123)
    rnd = np.array([[core.random_selection(R0, b, k, rng) for k in ks] for _ in range(200)])
    out["ref"]["random_median"] = [float(v) for v in np.median(rnd, 0)]
    out["ref"]["random_p10"] = [float(v) for v in np.quantile(rnd, 0.1, 0)]
    out["ref"]["random_p90"] = [float(v) for v in np.quantile(rnd, 0.9, 0)]
    out["ref"]["upper_fischer"] = [core.single_block_upper_bound(R0, b, k) for k in ks]
    opt, worst = [], []
    for k in ks:
        e = core.exhaustive(R0, b, k, limit=exhaustive_limit)
        opt.append(None if e is None else float(e["opt"]))
        worst.append(None if e is None else float(e["worst"]))
    out["ref"]["optimum"] = opt
    out["ref"]["worst"] = worst
    out["seconds"] = round(time.time() - t0, 1)
    print(f"[{name}] m={A.shape[0]} n={n} b={b} K={K}  {out['seconds']}s", flush=True)
    return out


def example_sweeps():
    """Example 2.1 (with and without the prior) and the ceil(k/b)-trap family over delta."""
    rows = []
    for delta in np.logspace(-8, -0.5, 16):
        X = P.example_2_1(0.5, delta)
        g = core.select_block_greedy(X, 2, 2)
        e = core.exhaustive(X, 2, 2)
        row = {"delta": float(delta),
               "plain": float((g.log_vols[-1] - e["opt"]) / np.log(10)),
               "predicted": float(np.log10(P.example_2_1_predictions(0.5, delta)["ratio"]))}
        for alpha in (10, 100, 1000, 10000):
            Aa = core.augment(X, alpha)
            ga = core.select_block_greedy(Aa, 2, 2)
            ea = core.exhaustive(Aa, 2, 2)
            row[f"alpha={alpha}"] = float((ga.log_vols[-1] - ea["opt"]) / np.log(10))
        rows.append(row)
    traps = []
    for (k, b) in [(2, 2), (4, 2), (6, 3), (8, 4)]:
        for delta in np.logspace(-6, -1, 11):
            X, _ = P.trap_family(k, b, 0.5, delta)
            g = core.select_block_greedy(X, b, k)
            e = core.exhaustive(X, b, k)
            traps.append({"k": k, "b": b, "delta": float(delta),
                          "log10_ratio": float((g.log_vols[-1] - e["opt"]) / np.log(10))})
    return {"example_2_1": rows, "trap_family": traps}


def perdigao_problems(npz):
    P1 = load(npz, MEANS)
    P2 = load(npz, TURB)
    span = (f"{np.datetime64(P1['t0'], 's')} to {np.datetime64(P1['t1'], 's')}").replace("T", " ")

    def basetxt(Pz):
        return (f"{Pz['m']:,} five-minute records from {span} UTC x {len(Pz['names'])} towers; towers under "
                f"90% complete left out, the {Pz['dropped_rows']:,} records in which a kept tower has a gap "
                f"dropped, columns standardized.")

    def first8(pr, names):
        pr["greedy_first"] = [names[j] for j in pr["block_greedy"]["orders"]["a"][:8]]
        pr["tower_names"] = names
        return pr

    n = len(P1["names"])
    out = []
    pr = run_problem("Perdigão: mean wind + temperature", P1["A"], P1["b"], n - 1,
                     desc=f"Block = one tower's 10 m sonic u, v, w, tc. {basetxt(P1)}")
    pr["latlon"] = P1["latlon"]
    out.append(first8(pr, P1["names"]))
    pr = run_problem("Perdigão: wind, temperature + turbulence", P2["A"], P2["b"], len(P2["names"]) - 1,
                     desc="Block = one tower's 10 m sonic u, v, w, tc and the second moments w'w', tc'tc', "
                          "u'w', w'tc'. " + basetxt(P2))
    out.append(first8(pr, P2["names"]))
    Xw = P1["A"] / np.sqrt(P1["m"])
    pr = run_problem("Perdigão: Bayesian D-optimal (R=3)", core.augment(Xw, 3.0), P1["b"], n - 1,
                     desc="log Vol [R X; I], X = data / sqrt(m) so that X^T X is the correlation matrix "
                          "of the channels of problem (i).", tall=True)
    out.append(first8(pr, P1["names"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=None, help="Perdigão .npz from blockrr.perdigao (omit to skip)")
    ap.add_argument("--out", default=str(ROOT / "results/benchmarks.json"))
    args = ap.parse_args(argv)

    probs = []
    X, good = P.trap_family(8, 4, 0.5, 1e-2)
    probs.append(run_problem("Trap family (k=8, b=4)", X, 4, 9,
                             desc="ceil(k/b)-trap family: 2 trap blocks poison all 8 good blocks; "
                                  "eps = 0.5, delta = 1e-2"))
    probs.append(run_problem("Block devil's staircase", P.block_devil_stairs(3000, 16, 4, rng=7), 4, 15,
                             desc="3000 x 64, 16 blocks of 4, five-step spectrum dropping 1e-2 per step"))
    if args.npz:
        probs += perdigao_problems(args.npz)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "problems": probs, "sweeps": example_sweeps()}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
