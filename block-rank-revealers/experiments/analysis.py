#!/usr/bin/env python3
"""Every number in the experiments section, recomputed from the benchmark runs and the data.

    python experiments/analysis.py --npz data/perdigao_iop.npz      # ~5 min

Reads   results/benchmarks.json   (experiments/run_benchmarks.py)
Writes  results/analysis.json

Sections (keys of the output):
  results      gap-to-best tables from the saved benchmark runs
  perdigao     data facts (towers, completeness, rows, dates)
  reproduce    the saved trajectories recomputed from scratch, plus selections,
               swap counts and tower orders
  bayes_bounds Theorems 2.2/2.3 and Corollary 2.1 against exhaustive worst/optimum
               and block greedy on the Perdigao Bayesian problem
  verify       Theorem 2.3, Corollary 2.1 and the greedy guarantee on random
               augmented instances, plus the tightness construction
  alg2         identities of Algorithm 2 (three ORTHOFACTORS/SCOPE choices)
  sweeps       Example 2.1 (plain and augmented) and the ceil(k/b)-trap family
  rr           order-revelation and interpolation factors (Definitions 2.1-2.2)
  sketch       block greedy on a sketch Omega A

Block greedy is Algorithm 2, variant (a) (blockrr.algorithm2).  ``reproduce`` recomputes
the saved trajectories from scratch, block greedy also with Algorithm 1 as a cross-check,
and ``alg2`` checks the identities of all three variants of Algorithm 2.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import warnings
from itertools import combinations
from math import ceil

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import blockrr  # noqa: E402
from blockrr import core, problems as P  # noqa: E402
from blockrr.column import get  # noqa: E402
from blockrr.perdigao import load, MEANS, TURB  # noqa: E402
from blockrr.sketching import apply_sketch  # noqa: E402

LN10 = math.log(10.0)
TOL = 1e-8                      # natural-log tolerance for "same volume"
GREEDY = "Block greedy (Alg. 2)"
SHORT = {
    "Trap family (k=8, b=4)": "trap",
    "Block Kahan (c=1/4)": "kahan",
    "Block devil's staircase": "devil",
    "Perdigão: mean wind + temperature": "pmeans",
    "Perdigão: wind, temperature + turbulence": "pturb",
    "Perdigão: Bayesian D-optimal (alpha=3)": "pbayes",
}


PAPER_PROBLEMS = ("trap", "kahan", "devil", "pmeans", "pturb", "pbayes")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def l10(x):
    return np.asarray(x, dtype=float) / LN10


def arr(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


# ----------------------------------------------------------------------------
# 1. the saved benchmark runs
# ----------------------------------------------------------------------------

def summarize_results(res, tiebreaks=None):
    out = {}
    for p in res["problems"]:
        key = SHORT[p["name"]]
        ks = np.array(p["ks"])
        meth = {nm: arr(m["logvol"]) for nm, m in p["methods"].items()}
        opt, worst = arr(p["ref"]["optimum"]), arr(p["ref"]["worst"])
        # best known = every method compared, including the best seed of each randomized
        # method, the best tie-break of each tied method, and the exact optimum where enumerated
        his = [arr(m["hi"]) for m in p["methods"].values() if "hi" in m]
        # methods whose first step is an exact tie (tiebreaks.py): the median over first picks
        # stands for the method, as the median over seeds does for a randomized one
        tbr = {nm: np.array(r, float) for nm, r in (tiebreaks or {}).get(key, {}).get("runs", {}).items()}
        for nm, r in tbr.items():
            meth[nm] = np.median(r, 0)
            his.append(r.max(0))
        best = np.fmax(np.vstack(list(meth.values()) + his).max(0), opt)
        exact = ~np.isnan(opt)
        gaps = {nm: l10(best - v) for nm, v in meth.items()}
        summ = {}
        for nm, g in gaps.items():
            i = int(np.argmax(g))
            d = {"max_gap": g[i], "k_at_max": int(ks[i]),
                 "n_best": int(np.sum(g < TOL / LN10)),
                 "n_exact_opt": int(np.sum(np.abs(meth[nm][exact] - opt[exact]) < TOL)),
                 "nested": p["methods"][nm]["nested"], "family": p["methods"][nm]["family"],
                 "gap": g}
            m = p["methods"][nm]
            if "lo" in m:      # randomized: range over seeds
                d["gap_worst_seed"] = l10(best - arr(m["lo"]))
                d["gap_best_seed"] = l10(best - arr(m["hi"]))
                d["max_gap_worst_seed"] = float(np.max(d["gap_worst_seed"]))
            if nm in tbr:      # tied: range over first picks
                d["gap_worst_tiebreak"] = l10(best - tbr[nm].min(0))
                d["gap_best_tiebreak"] = l10(best - tbr[nm].max(0))
                d["max_gap_worst_tiebreak"] = float(np.max(d["gap_worst_tiebreak"]))
                d["n_tiebreaks"] = int(tbr[nm].shape[0])
            summ[nm] = d
        rnd = {q: l10(best - arr(p["ref"][f"random_{q}"])) for q in ("median", "p10", "p90")}
        same = {}
        names = list(meth)
        for a, b_ in combinations(names, 2):
            if np.allclose(meth[a], meth[b_], atol=1e-9, rtol=0):
                same.setdefault(a, []).append(b_)
        out[key] = {
            "name": p["name"], "m": p["m"], "n": p["n"], "b": p["b"], "K": int(ks[-1]),
            "ks": ks, "seconds": p["seconds"], "exact_ks": ks[exact],
            "best": best, "opt": opt, "worst": worst,
            "worst_gap": l10(best - worst),
            "fischer_above_best": l10(arr(p["ref"]["upper_fischer"]) - best),
            "methods": summ, "random": rnd,
            "random_max_median_gap": float(np.max(rnd["median"])),
            "identical_lines": same,
            "tower_names": p.get("tower_names"), "latlon": p.get("latlon"),
            "greedy_first": p.get("greedy_first"),
        }
    return out


# ----------------------------------------------------------------------------
# 2. Perdigao data facts
# ----------------------------------------------------------------------------

def perdigao_facts(npz):
    z = np.load(npz)
    keys = [str(k) for k in z["keys"]]
    X = z["data"].astype(np.float64)
    X[np.abs(X) >= 1e36] = np.nan
    idx = {k: i for i, k in enumerate(keys)}
    sites = sorted({k.split("|")[2] for k in keys})
    t = z["time"]
    out = {"n_sites": len(sites), "sites": sites, "n_times": int(len(t)),
           "t_first": int(t[0]), "t_last": int(t[-1]),
           "n_files": int(len(z["files"])), "file_first": str(z["files"][0]),
           "file_last": str(z["files"][-1])}
    avail = {}
    for v in ["u", "v", "w", "tc", "T", "RH", "P", "w'w'", "u'w'", "w'tc'", "tc'tc'"]:
        for h in ["2m", "10m"]:
            c = [s for s in sites if f"{v}|{h}|{s}" in idx]
            if c:
                avail[f"{v}@{h}"] = len(c)
    out["channel_tower_counts"] = avail
    for chans, label in ((MEANS, "means"), (TURB, "turb")):
        frac = {}
        for s in sites:
            ii = [idx.get(f"{c}|10m|{s}") for c in chans]
            if None in ii:
                continue
            frac[s] = float(np.isfinite(X[:, ii]).all(1).mean())
        kept = sorted(s for s, f in frac.items() if f >= 0.90)
        out[label] = {"towers_with_all_channels": len(frac), "complete_frac": frac,
                      "kept": kept, "dropped": sorted(s for s in frac if s not in kept),
                      "no_channels": sorted(s for s in sites if s not in frac)}
    # all-channel raw completeness of the 13,248 timestamps, before dropping towers
    P1 = load(npz, MEANS)
    P2 = load(npz, TURB)
    out["rows_kept"] = P1["m"]
    out["rows_dropped"] = P1["dropped_rows"]
    out["same_rows_turb"] = bool(P1["m"] == P2["m"] and P1["t0"] == P2["t0"] and P1["t1"] == P2["t1"])
    out["same_towers_turb"] = P1["names"] == P2["names"]
    out["t0"], out["t1"] = P1["t0"], P1["t1"]
    out["towers"] = P1["names"]
    out["latlon"] = P1["latlon"]
    out["n_towers"] = len(P1["names"])
    return out, P1, P2


# ----------------------------------------------------------------------------
# 3. problems, selections, reproduction of the saved trajectories
# ----------------------------------------------------------------------------

def build_problems(P1, P2):
    probs = {}
    X, good = P.trap_family(8, 4, 0.5, 1e-2)
    probs["trap"] = {"A": X, "b": 4, "K": 9, "good": good}
    probs["kahan"] = {"A": P.block_kahan(16, 4, rows=256), "b": 4, "K": 15}
    probs["devil"] = {"A": P.block_devil_stairs(3000, 16, 4, rng=7), "b": 4, "K": 15}
    probs["pmeans"] = {"A": P1["A"], "b": 4, "K": 36}
    probs["pturb"] = {"A": P2["A"], "b": 8, "K": 36}
    Xw = P1["A"] / np.sqrt(P1["m"])
    probs["pbayes"] = {"A": core.augment(Xw, 3.0), "b": 4, "K": 36, "X": Xw, "alpha": 3.0}
    for d in probs.values():
        d["R0"] = core.compress(d["A"])
        d["n"] = d["A"].shape[1] // d["b"]
    return probs


def col_perm(A, name, kcol=None):
    opts = {"want_q": False}
    if kcol is not None:
        opts["k"] = kcol
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get(name)(A, **opts)


def reproduce(probs, res_by_key):
    out = {}
    for key, d in probs.items():
        A, R0, b, K, n = d["A"], d["R0"], d["b"], d["K"], d["n"]
        saved = res_by_key[key]
        g = blockrr.algorithm2(R0, b, K)                  # block greedy: Algorithm 2, variant (a)
        g1 = core.select_block_greedy(R0, b, K)           # Algorithm 1, as an independent check
        geqp3 = core.induced_block_order(col_perm(A, "geqp3").p, b)
        bg = core.induced_block_order(col_perm(A, "businger_golub").p, b)
        sr_orders, sr_swaps, sr_vals = [], [], []
        N = A.shape[1]
        for k in range(1, K + 1):
            r = col_perm(A, "srrqr", kcol=min(k * b, N - 1))
            o = core.induced_block_order(r.p, b)
            sr_orders.append(o[:k]); sr_swaps.append(int(r.meta["swaps"]))
            sr_vals.append(core.log_vol_S(R0, o[:k], b))
        traj = lambda o: np.array(core.trajectory(R0, o, b, K))
        mine = {GREEDY: np.array(g.log_volumes), "dgeqp3": traj(geqp3),
                "Businger-Golub": traj(bg), "sRRQR (Gu-Eisenstat)": np.array(sr_vals)}
        diffs = {nm: float(np.max(np.abs(v - arr(saved["methods_raw"][nm])))) for nm, v in mine.items()}
        diffs["Algorithm 1"] = float(np.max(np.abs(np.array(g1.log_vols) - np.array(g.log_volumes))))
        out[key] = {"greedy_order": g.selected, "algorithm1_order": g1.order,
                    "geqp3_order": geqp3[:K], "bg_order": bg[:K],
                    "srrqr_swaps": sr_swaps, "srrqr_orders": sr_orders,
                    "max_abs_diff_vs_saved": diffs}
        d["sel"] = {"greedy": g.selected, "geqp3": geqp3, "bg": bg}
        log("reproduce", key, "max diff", max(diffs.values()), "srrqr swaps total", sum(sr_swaps))
    return out


# ----------------------------------------------------------------------------
# 4. bounds versus worst case and greedy, Perdigao Bayesian problem
# ----------------------------------------------------------------------------

def anti_greedy(R0, b, K, n):
    """Upper estimate of the worst k-block volume for every k <= K: Algorithm 1 with the
    smallest instead of the largest score, i.e. add the block of least volume gain."""
    S, vals = [], []
    for _ in range(K):
        v, j = min((core.log_vol_S(R0, S + [j], b), j) for j in range(n) if j not in S)
        S.append(j)
        vals.append(v)
    return np.array(vals), S


def bounds_curves(X, alpha, b, ks, logVbar, logVbar_exact, logVq_exact_fn, v_sorted):
    """All lower bounds on log V_k (natural log), given (an underestimate of) log Vbar_k."""
    s = np.linalg.svd(alpha * X, compute_uv=False)
    tolr = max(X.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > tolr))
    lam = 0.5 * np.log1p(s[0] ** 2)                         # log Lambda
    logVolA = float(np.sum(0.5 * np.log1p(s ** 2)))
    trc = float((alpha ** 2) * np.sum(X * X))
    n = X.shape[1] // b
    rows = []
    for i, k in enumerate(ks):
        q = min(k, n - k)
        r = min(b * q, rank)
        logPhi = float(np.sum(0.5 * np.log1p(s[:r] ** 2)))
        Vb = logVbar[i]
        row = {"k": k, "q": q, "r": r, "logVbar": Vb, "exact": bool(logVbar_exact[i]),
               "b11": Vb - logPhi, "b10": Vb - float(np.sum(v_sorted[:q])),
               "ci": Vb - r * lam, "civ": Vb - logVolA,
               "ciii": Vb - 0.5 * r * np.log1p(trc / r) if r >= 1 else np.nan,
               # s = sigma(alpha X), so s[0] = alpha ||X||_2 already (Corollary 2.1(ii))
               "cii": (Vb - 0.5 * r * np.log(2) - r * np.log(s[0])) if s[0] >= 1 else np.nan,
               "b9": (Vb - logVq_exact_fn(q)) if logVq_exact_fn(q) is not None else np.nan,
               "greedy_guarantee": (1 - (1 - 1 / k) ** k) * Vb,
               "logPhi": logPhi}
        rows.append(row)
    meta = {"rank_X": rank, "sigma1_alphaX": float(s[0]), "log_Lambda": lam, "logVolA": logVolA,
            "alpha2_trace": trc, "alpha_normX": float(s[0])}
    return rows, meta


def bayes_bounds(probs, results):
    d = probs["pbayes"]
    R0, b, n, K = d["R0"], d["b"], d["n"], d["K"]
    S = results["pbayes"]
    ks = list(S["ks"])
    v = np.array(sorted((core.log_vol_S(R0, [j], b) for j in range(n)), reverse=True))
    opt = S["opt"]
    worst = S["worst"]
    exact = ~np.isnan(opt)
    logVbar = np.where(exact, opt, S["best"])
    def Vq(q):
        if q == 0:
            return 0.0
        i = ks.index(q) if q in ks else None
        return None if (i is None or not exact[i]) else float(opt[i])
    rows, meta = bounds_curves(d["X"], d["alpha"], b, ks, logVbar, exact, Vq, v)
    # worst: exact where enumerated, otherwise the anti-greedy volume (an upper estimate)
    wh, _ = anti_greedy(R0, b, max(ks), n)
    wh = wh[np.array(ks) - 1]
    greedy = S["best"] - S["methods"][GREEDY]["gap"] * LN10
    for i, row in enumerate(rows):
        row["worst_exact"] = None if np.isnan(worst[i]) else float(worst[i])
        row["worst_found"] = float(wh[i])
        row["greedy"] = float(greedy[i])
        row["random_median"] = float(S["best"][i] - S["random"]["median"][i] * LN10)
    heur_ok = [abs(wh[i] - worst[i]) < 1e-8 for i in range(len(ks)) if not np.isnan(worst[i])]
    heur_gap = [float((wh[i] - worst[i]) / LN10) for i in range(len(ks)) if not np.isnan(worst[i])]
    viol = []
    for row in rows:
        lowest = row["worst_exact"] if row["worst_exact"] is not None else row["worst_found"]
        for key in ("b9", "b10", "b11", "ci", "cii", "ciii", "civ"):
            if row[key] is not None and np.isfinite(row[key]) and row["exact"]:
                viol.append(row[key] - lowest)
        if row["exact"]:
            viol.append(row["greedy_guarantee"] - row["greedy"])
    meta.update({"single_block_logvols_sorted": v, "heuristic_matches_exact_worst": heur_ok,
                 "heuristic_minus_exact_worst_log10": heur_gap,
                 "max_violation_exact_k": float(max(viol)) if viol else None})
    log("bayes bounds done; anti-greedy matches exact worst at", sum(heur_ok), "of", len(heur_ok),
        "largest excess (log10)", max(heur_gap))
    return {"rows": rows, "meta": meta}


# ----------------------------------------------------------------------------
# 5. random verification of Theorem 2.3 / Corollary 2.1 / the greedy guarantee
# ----------------------------------------------------------------------------

def verify_random(n_inst=300, seed=2026):
    rng = np.random.default_rng(seed)
    keys = ["eq8", "b9", "b10", "b11", "ci", "cii", "ciii", "civ", "chain_Vq_le_prod", "chain_Vq_le_Phi", "greedy"]
    worst_viol = {k: -np.inf for k in keys}
    min_slack = {k: np.inf for k in keys}
    n_checks = {k: 0 for k in keys}
    shapes = []
    for _ in range(n_inst):
        n = int(rng.integers(3, 8)); b = int(rng.integers(1, 4)); m0 = int(rng.integers(1, 9))
        X = rng.standard_normal((m0, n * b))
        scale = 10 ** rng.uniform(-1.5, 1.0, size=n)
        scale[rng.random(n) < 0.15] = 0.0                       # some uninformative blocks
        X = X * np.repeat(scale, b)[None, :]
        alpha = float(10 ** rng.uniform(-1, 1.5))
        A = core.augment(X, alpha)
        R0 = core.compress(A)
        shapes.append((m0, n, b))
        lv = {(): 0.0}
        for k in range(1, n + 1):
            for Sset in combinations(range(n), k):
                lv[Sset] = core.log_vol_S(R0, Sset, b)
        vs = np.array(sorted((lv[(j,)] for j in range(n)), reverse=True))
        s = np.linalg.svd(alpha * X, compute_uv=False)
        rank = int(np.sum(s > max(X.shape) * np.finfo(float).eps * max(s[0], 1e-300)))
        lam = 0.5 * np.log1p(s[0] ** 2)
        logVolA = float(np.sum(0.5 * np.log1p(s ** 2)))
        trc = float(alpha ** 2 * np.sum(X * X))
        Vbar = {k: max(v for Sset, v in lv.items() if len(Sset) == k) for k in range(0, n + 1)}
        for k in range(1, n + 1):
            sets = [Sset for Sset in lv if len(Sset) == k]
            vals = np.array([lv[Sset] for Sset in sets])
            Vb, Vu = vals.max(), vals.min()
            q = min(k, n - k); r = min(b * q, rank)
            logPhi = float(np.sum(0.5 * np.log1p(s[:r] ** 2)))
            def rec(key, lhs, rhs):        # check lhs <= rhs
                worst_viol[key] = max(worst_viol[key], lhs - rhs)
                min_slack[key] = min(min_slack[key], rhs - lhs)
                n_checks[key] += 1
            for Sset in sets:
                for Tset in sets:
                    diff = tuple(sorted(set(Sset) - set(Tset)))
                    rec("eq8", lv[Sset], lv[Tset] + lv[diff])
            rec("b9", Vb - Vbar[q], Vu)
            rec("b10", Vb - float(np.sum(vs[:q])), Vu)
            rec("b11", Vb - logPhi, Vu)
            rec("ci", Vb - r * lam, Vu)
            if s[0] >= 1:                  # s[0] = alpha ||X||_2
                rec("cii", Vb - 0.5 * r * np.log(2) - r * np.log(s[0]), Vu)
            if r >= 1:
                rec("ciii", Vb - 0.5 * r * np.log1p(trc / r), Vu)
            rec("civ", Vb - logVolA, Vu)
            rec("chain_Vq_le_prod", Vbar[q], float(np.sum(vs[:q])))
            rec("chain_Vq_le_Phi", Vbar[q], logPhi)
            g = blockrr.algorithm2(R0, b, k, variant="a")
            rec("greedy", (1 - (1 - 1 / k) ** k) * Vb, g.log_volumes[-1])
    # tightness: k orthogonal informative blocks + n-k >= k zero blocks
    tight = []
    for (k, b, rho) in [(2, 2, 3.0), (3, 1, 10.0), (2, 3, 0.5), (4, 2, 2.0)]:
        n = 2 * k
        m0 = k * b
        alpha = 1.0
        X = np.zeros((m0, n * b))
        X[:, : k * b] = rho * np.eye(m0)
        R0 = core.compress(core.augment(X, alpha))
        vals = [core.log_vol_S(R0, Sset, b) for Sset in combinations(range(n), k)]
        s = np.linalg.svd(alpha * X, compute_uv=False)
        rr = min(b * min(k, n - k), int(np.sum(s > 1e-12)))
        logPhi = float(np.sum(0.5 * np.log1p(s[:rr] ** 2)))
        tight.append({"k": k, "b": b, "rho": rho, "gap": max(vals) - min(vals), "logPhi": logPhi,
                      "r_log_Lambda": rr * 0.5 * np.log1p(s[0] ** 2)})
    log("random verification done")
    return {"instances": n_inst, "seed": seed, "worst_violation": worst_viol, "min_slack": min_slack,
            "n_checks": n_checks, "tightness": tight,
            "m0_range": [min(s[0] for s in shapes), max(s[0] for s in shapes)],
            "n_range": [min(s[1] for s in shapes), max(s[1] for s in shapes)],
            "b_range": [min(s[2] for s in shapes), max(s[2] for s in shapes)]}


# ----------------------------------------------------------------------------
# 6. Algorithm 2 identities
# ----------------------------------------------------------------------------

def alg2_checks(probs):
    mats = {"Example 2.1": (P.example_2_1(0.5, 1e-2), 2)}
    for key in PAPER_PROBLEMS:
        d = probs[key]
        A = d["A"] if d["A"].shape[0] <= 100 else d["R0"]      # tall data: run on R0
        mats[key] = (A, d["b"])
    rng = np.random.default_rng(11)
    for i in range(200):
        b = int(rng.integers(1, 5)); n = int(rng.integers(2, 9)); m = int(rng.integers(n * b, n * b + 12))
        mats[f"random{i}"] = (rng.standard_normal((m, n * b)) * 10 ** rng.uniform(-3, 3, size=n * b), b)
    # the three variants of Algorithm 2, and (b) with x3 V in place of x3 V^T in line 9
    variants = {"a": ("a", "VT"), "b": ("b", "VT"), "c": ("c", "VT"), "b, x3 V": ("b", "V")}
    out = {v: {"same_pivots_as_alg1": 0, "count": 0} for v in variants}
    per_problem = {}
    for name, (A, b) in mats.items():
        n = A.shape[1] // b
        pi1 = core.block_greedy_qr(A, b, n)["pi"]
        for v, (var, mw) in variants.items():
            r = blockrr.check_algorithm2(A, b, variant=var, mode3_with=mw)
            st = out[v]
            st["count"] += 1
            st["same_pivots_as_alg1"] += int(r["pi"] == pi1)
            for kk, val in r.items():
                if kk == "pi":
                    continue
                st[f"max_{kk}"] = max(st.get(f"max_{kk}", 0.0), val)
                st.setdefault(f"min_{kk}", np.inf)
                st[f"min_{kk}"] = min(st[f"min_{kk}"], val)
            if not name.startswith("random"):
                per_problem.setdefault(name, {})[v] = {kk: val for kk, val in r.items() if kk != "pi"}
                per_problem[name][v]["same_pivots"] = r["pi"] == pi1
    log("alg2 checks done")
    return {"summary": out, "per_problem": per_problem, "n_random": 200}


# ----------------------------------------------------------------------------
# 7. sweeps: Example 2.1 and the trap family
# ----------------------------------------------------------------------------

def aug_limit(alpha, eps):
    """delta -> 0 limit of Vol(greedy)/Vol(opt) for A = [alpha X; I], X from Example 2.1."""
    a2 = alpha ** 2
    return (1 + a2) * np.sqrt(1 + 2 * a2) / (1 + 2 * a2 + (1 - eps ** 2) * a2 ** 2)


def sweeps(saved):
    ex = saved["example_2_1"]
    plain_err = max(abs(r["plain"] - r["predicted"]) for r in ex)
    alphas = [10, 100, 1000, 10000]
    floors = {a: float(np.log10(aug_limit(a, 0.5))) for a in alphas}
    at_small = {a: ex[0][f"alpha={a}"] for a in alphas}       # delta = 1e-8
    # recompute the augmented curves on a finer delta grid, and check greedy picks B1 first
    fine = []
    for delta in np.logspace(-8, -0.5, 31):
        X = P.example_2_1(0.5, delta)
        row = {"delta": float(delta),
               "plain": float((blockrr.algorithm2(X, 2, 2).log_volumes[-1] - core.exhaustive(X, 2, 2)["opt"]) / LN10),
               "predicted": float(np.log10(P.example_2_1_predictions(0.5, delta)["ratio"]))}
        for a in alphas:
            Aa = core.augment(X, a)
            g = blockrr.algorithm2(Aa, 2, 2)
            row[f"alpha={a}"] = float((g.log_volumes[-1] - core.exhaustive(Aa, 2, 2)["opt"]) / LN10)
            row[f"alpha={a}_first"] = int(g.selected[0])
        fine.append(row)
    combos = [(2, 2), (3, 3), (4, 2), (5, 2), (6, 3), (8, 4)]
    deltas = np.logspace(-6, -1, 11)
    traps = []
    for (k, b) in combos:
        ys = []
        for delta in deltas:
            X, good = P.trap_family(k, b, 0.5, delta)
            g = blockrr.algorithm2(X, b, k)
            e = core.exhaustive(X, b, k)
            ys.append((g.log_volumes[-1] - e["opt"]) / LN10)
            traps.append({"k": k, "b": b, "delta": float(delta), "log10_ratio": float(ys[-1]),
                          "greedy_takes_all_traps": all(j not in good for j in g.selected[:ceil(k / b)]),
                          "opt_is_good_set": sorted(e["S_opt"]) == sorted(good)})
        ys = np.array(ys)
        mask = deltas <= 1e-3
        slope = float(np.polyfit(np.log10(deltas[mask]), ys[mask], 1)[0])
        traps.append({"k": k, "b": b, "summary": True, "fitted_slope": slope,
                      "predicted_slope": k - ceil(k / b), "n_traps": ceil(k / b)})
    # agreement with the saved sweep
    saved_tr = {(r["k"], r["b"], round(math.log10(r["delta"]), 6)): r["log10_ratio"] for r in saved["trap_family"]}
    diffs = [abs(saved_tr[(r["k"], r["b"], round(math.log10(r["delta"]), 6))] - r["log10_ratio"])
             for r in traps if not r.get("summary") and (r["k"], r["b"], round(math.log10(r["delta"]), 6)) in saved_tr]
    log("sweeps done")
    return {"example_2_1_plain_max_abs_err": plain_err, "aug_floor_pred": floors, "aug_at_delta_1e-8": at_small,
            "aug_floor_err": max(abs(floors[a] - at_small[a]) for a in alphas),
            "example_2_1_fine": fine, "trap": traps, "trap_saved_max_diff": max(diffs) if diffs else None,
            "saved_example_2_1": ex}


# ----------------------------------------------------------------------------
# 8. order revelation and interpolation (Definitions 2.1, 2.2)
# ----------------------------------------------------------------------------

def rr_metrics(R0, S, b, sA):
    N = R0.shape[1]
    cols = core.cols_of(S, b)
    kb = len(cols)
    Qs, Rs = np.linalg.qr(R0[:, cols])
    B = Qs.T @ R0
    sp = np.linalg.svd(B, compute_uv=False)
    E = R0 - Qs @ B
    se = np.linalg.svd(E, compute_uv=False)
    mu_ord = float(np.max(sA[:kb] / np.maximum(sp[:kb], 1e-300)))
    tail = sA[kb:]
    ok = tail > 1e-14 * sA[0]
    mu_res = float(np.max(se[: N - kb][ok] / tail[ok])) if np.any(ok) else float("nan")
    try:
        Z = np.linalg.solve(Rs, B)
        nu = float(np.linalg.norm(Z, 2))
    except np.linalg.LinAlgError:
        nu = float("inf")
    return {"mu_ord": mu_ord, "mu_res": mu_res, "nu": nu}


def rr_study(probs, results, reproduced, k_by_problem):
    out = {}
    rng = np.random.default_rng(99)
    for key, ks in k_by_problem.items():
        d = probs[key]
        R0, b, n = d["R0"], d["b"], d["n"]
        sA = np.linalg.svd(R0, compute_uv=False)
        rows = {}
        for k in ks:
            sel = {"Block greedy": d["sel"]["greedy"][:k],
                   "dgeqp3 (induced)": d["sel"]["geqp3"][:k]}
            ex = core.exhaustive(R0, b, k, limit=200_000)
            if ex is not None:
                sel["Exact optimum"] = ex["S_opt"]
                sel["Exact worst"] = ex["S_worst"]
            r = {nm: dict(rr_metrics(R0, S, b, sA), logvol10=core.log_vol_S(R0, S, b) / LN10)
                 for nm, S in sel.items()}
            rand = [rr_metrics(R0, list(rng.choice(n, size=k, replace=False)), b, sA) for _ in range(200)]
            r["Random (median of 200)"] = {kk: float(np.median([x[kk] for x in rand])) for kk in ("mu_ord", "mu_res", "nu")}
            rows[k] = r
        out[key] = rows
    log("rr study done")
    return out


# ----------------------------------------------------------------------------
# 9. block greedy on a sketch
# ----------------------------------------------------------------------------

def sketch_study(probs, results, n_seeds=20):
    out = {}
    for key in ("pmeans", "pturb", "pbayes"):
        d = probs[key]
        A, R0, b, K, n = d["A"], d["R0"], d["b"], d["K"], d["n"]
        N = A.shape[1]
        g = blockrr.algorithm2(R0, b, K)
        gtraj = np.array(g.log_volumes)
        g_order = g.selected
        best = results[key]["best"]
        res = {}
        for kind in ("gaussian", "sparse_sign"):
            for gamma in (1.25, 2.0, 4.0):
                dsk = int(np.ceil(gamma * N))
                gaps, gaps_best, same_set, same_order = [], [], [], []
                for sd in range(n_seeds):
                    rng = np.random.default_rng(10_000 + 97 * sd)
                    SA = apply_sketch(A, dsk, kind=kind, rng=rng)
                    o = blockrr.select_blocks(SA, b, K)
                    tr = np.array(core.trajectory(R0, o, b, K))
                    gaps.append(l10(gtraj - tr)); gaps_best.append(l10(best - tr))
                    same_set.append([set(o[:k]) == set(g_order[:k]) for k in range(1, K + 1)])
                    same_order.append(o == g_order)
                gaps, gaps_best, same_set = np.array(gaps), np.array(gaps_best), np.array(same_set)
                res[f"{kind}|{gamma}"] = {
                    "d": dsk, "gamma": gamma, "kind": kind, "seeds": n_seeds,
                    "max_gap_vs_greedy": float(gaps.max()), "min_gap_vs_greedy": float(gaps.min()),
                    "median_max_gap_vs_greedy": float(np.median(gaps.max(1))),
                    "max_gap_vs_best": float(gaps_best.max()),
                    "frac_same_order": float(np.mean(same_order)),
                    "frac_same_set_all_k": float(np.mean(same_set.all(1))),
                    "frac_same_set_mean_over_k": float(same_set.mean()),
                    "gap_vs_greedy_median_by_k": np.median(gaps, 0),
                    "gap_vs_greedy_max_by_k": gaps.max(0)}
        out[key] = res
        log("sketch", key, "done")
    # Example 2.1, augmented, embedded in R^M and sketched
    eps, delta, alpha, M = 0.5, 1e-4, 1000.0, 4000
    Aaug = core.augment(P.example_2_1(eps, delta), alpha)              # 12 x 6
    rng = np.random.default_rng(5)
    U, _ = np.linalg.qr(rng.standard_normal((M, Aaug.shape[0])))
    Atall = U @ Aaug
    ex = core.exhaustive(Aaug, 2, 2)
    trap_rows = []
    for dsk in (12, 24, 48, 96, 192, 384, 768):
        took_b1, ratios = [], []
        for sd in range(500):
            SA = apply_sketch(Atall, dsk, kind="gaussian", rng=np.random.default_rng(777 + sd))
            o = blockrr.select_blocks(SA, 2, 2)
            took_b1.append(0 in o)
            ratios.append((core.log_vol_S(Aaug, o, 2) - ex["opt"]) / LN10)
        trap_rows.append({"d": dsk, "gamma": dsk / 6, "p_trap": float(np.mean(took_b1)),
                          "median_log10_ratio": float(np.median(ratios)), "mean_log10_ratio": float(np.mean(ratios))})
    g0 = blockrr.algorithm2(Aaug, 2, 2)
    out["example_2_1_sketch"] = {"eps": eps, "delta": delta, "alpha": alpha, "M": M, "rows": trap_rows,
                                 "unsketched_log10_ratio": float((g0.log_volumes[-1] - ex["opt"]) / LN10),
                                 "unsketched_order": g0.selected}
    log("sketch trap done")
    return out


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--results", default=str(ROOT / "results/benchmarks.json"))
    ap.add_argument("--out", default=str(ROOT / "results/analysis.json"))
    ap.add_argument("--tiebreaks", default=str(ROOT / "results/tiebreaks.json"),
                    help="tiebreaks.py output; the tied methods on Perdigão are reported over first picks")
    ap.add_argument("--skip", default="", help="comma list of sections to skip")
    ap.add_argument("--only", default="",
                    help="recompute only these sections (bounds,verify,alg2,sweeps,rr,sketch) "
                         "and merge them into the existing --out file")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    if args.only:
        only = set(args.only.split(","))
        skip = {"bounds", "verify", "alg2", "rr", "sketch"} - only
        prev = json.load(open(args.out))
    res = json.load(open(args.results))
    raw = {SHORT[p["name"]]: {nm: m["logvol"] for nm, m in p["methods"].items()} for p in res["problems"]}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "numpy": np.__version__}
    import scipy
    out["scipy"] = scipy.__version__
    tb = json.load(open(args.tiebreaks)) if pathlib.Path(args.tiebreaks).exists() else None
    out["tiebreaks"] = None if tb is None else tb.get("generated")
    results = summarize_results(res, tb)
    for k in results:
        results[k]["methods_raw"] = raw[k]
    out["results"] = results
    log("results summarized")
    facts, P1, P2 = perdigao_facts(args.npz)
    out["perdigao"] = facts
    probs = build_problems(P1, P2)
    out["reproduce"] = reproduce(probs, results)
    if "bounds" not in skip:
        out["bayes_bounds"] = bayes_bounds(probs, results)
    if "verify" not in skip:
        out["verify"] = verify_random()
    if "alg2" not in skip:
        out["alg2"] = alg2_checks(probs)
    out["sweeps"] = sweeps(res["sweeps"])
    if "rr" not in skip:
        out["rr"] = rr_study(probs, results, out["reproduce"],
                             {"trap": [2, 8], "devil": [8, 12],
                              "pmeans": [2, 4, 8], "pturb": [4, 8], "pbayes": [4, 8]})
    if "sketch" not in skip:
        out["sketch"] = sketch_study(probs, results)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = jsonable(out)
    if args.only:
        keymap = {"bounds": "bayes_bounds", "verify": "verify", "alg2": "alg2", "sweeps": "sweeps",
                  "rr": "rr", "sketch": "sketch"}
        for sec in only:
            prev[keymap[sec]] = out[keymap[sec]]
        prev["generated_" + "_".join(sorted(only))] = out["generated"]
        out = prev
    with open(args.out, "w") as f:
        json.dump(out, f)
    log("wrote", args.out)


if __name__ == "__main__":
    main()
