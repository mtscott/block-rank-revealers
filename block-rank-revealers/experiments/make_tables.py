#!/usr/bin/env python3
"""LaTeX tables of the experiments section, generated from results/*.json.

    python experiments/make_tables.py  # writes tables/*.tex

Every number in a table comes straight from analysis.json / extras.json, so the
tables never need retyping.  experiments.tex \\input's nothing: build_tex.py
pastes these files into it between %%BEGIN table:<name> / %%END markers.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "results"
TAB = ROOT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
LN10 = math.log(10.0)

PROBS = ["kahan", "devil", "pmeans", "pturb", "pbayes"]          # the losses table (Figure 2 + Perdigao)
PROB_HEAD = [r"Kahan", r"stairs", r"(i)", r"(ii)", r"(iii)"]
N_SYNTH = 2                       # the first N_SYNTH problems are synthetic
ALL_PROBS = ["trap", "kahan", "devil", "pmeans", "pturb", "pbayes"]  # the problems table
METHODS = [
    ("Block greedy (Alg. 2)", r"block greedy (Alg.~\exAlgGreedy)"),
    ("Businger-Golub", r"Businger--Golub"),
    ("dgeqp3", r"\texttt{dgeqp3}"),
    ("sRRQR (Gu-Eisenstat)", r"sRRQR"),
    ("HQRRP", r"HQRRP, median"),
    ("CQRRPT", r"CQRRPT, median"),
    ("Randomized sRRQR", r"randomized sRRQR, median"),
]
STARM_METHODS = [   # pivoted *_M-QR; first-slice with M = DFT and DCT select the same blocks
    ("first_slice|I", r"\quad first-slice, $\mathbf M=\mathbf I$"),
    ("first_slice|DFT", r"\quad first-slice, $\mathbf M=$ DFT, DCT"),
    ("tube_norm|I", r"\quad tube-norm, $\mathbf M=\mathbf I$"),
    ("tube_norm|DFT", r"\quad tube-norm, $\mathbf M=$ DFT"),
    ("tube_norm|DCT", r"\quad tube-norm, $\mathbf M=$ DCT"),
]


def load(name):
    with open(BUILD / name) as f:
        return json.load(f)


def f2(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    v = round(float(x), nd)
    if v == 0:
        v = 0.0                      # no "-0.00"
    if v < 0:
        return f"${v:.{nd}f}$"       # math minus, not a hyphen
    return f"{v:.{nd}f}"


def write(name, body):
    (TAB / f"{name}.tex").write_text(body)


# --------------------------------------------------------------------------
def table_losses(D):
    R = D["results"]
    lines = []
    ns, nr = N_SYNTH, len(PROBS) - N_SYNTH
    lines.append(r"\begin{tabular}{@{}l" + "r" * ns + r"@{\hspace{1.6em}}" + "r" * nr + "@{}}")
    lines.append(r"\toprule")
    lines.append(rf" & \multicolumn{{{ns}}}{{c}}{{synthetic (\S\ref{{sec:exp-synthetic}})}} & "
                 rf"\multicolumn{{{nr}}}{{c@{{}}}}{{Perdig\~ao (\S\ref{{sec:exp-perdigao}})}}\\")
    lines.append(rf"\cmidrule(r{{1.6em}}){{2-{1 + ns}}}\cmidrule(l){{{2 + ns}-{1 + ns + nr}}}")
    lines.append(r"method & " + " & ".join(PROB_HEAD) + r"\\")
    lines.append(r"\midrule")
    for p in PROBS:
        m = R[p]["methods"]
        assert np.allclose(m["first_slice|DFT"]["gap"], m["first_slice|DCT"]["gap"], atol=1e-12), p
    # bold: the lowest loss in each column, over the method rows
    rows = list(METHODS) + list(STARM_METHODS)
    low = {p: min(round(R[p]["methods"][k]["max_gap"], 2) for k, _ in rows if k in R[p]["methods"]) for p in PROBS}

    def cell(p, key):
        m = R[p]["methods"].get(key)
        if m is None:
            return "--"
        s = f2(m["max_gap"])
        return rf"\textbf{{{s}}}" if round(m["max_gap"], 2) == low[p] else s

    def worst(key, field, label):
        """randomized: the worst seed; tied on Perdigao (tiebreaks.py): the worst first pick"""
        cells = [("(" + f2(R[p]["methods"][key][field], 1) + ")") if field in R[p]["methods"].get(key, {})
                 else "--" for p in PROBS]
        return [label + " & " + " & ".join(cells) + r"\\"] if any(c != "--" for c in cells) else []

    for key, lab in METHODS:
        lines.append(lab + " & " + " & ".join(cell(p, key) for p in PROBS) + r"\\")
        lines += worst(key, "max_gap_worst_seed", r"\quad worst seed")
        lines += worst(key, "max_gap_worst_tiebreak", r"\quad worst tie-break")
    lines.append(r"\addlinespace[2pt]")
    lines.append(rf"\multicolumn{{{1 + len(PROBS)}}}{{@{{}}l}}{{pivoted $\star_{{\mathbf M}}$-QR factorizations:}}\\")
    for key, lab in STARM_METHODS:
        lines.append(lab + " & " + " & ".join(cell(p, key) for p in PROBS) + r"\\")
        lines += worst(key, "max_gap_worst_tiebreak", r"\quad\quad worst tie-break")
    lines.append(r"\midrule")
    cells = [f2(R[p]["random_max_median_gap"], 1) for p in PROBS]
    lines.append(r"random $k$-set (median) & " + " & ".join(cells) + r"\\")
    cells = []
    for p in PROBS:
        wg = np.array([np.nan if v is None else v for v in R[p]["worst_gap"]], float)
        s = f2(np.nanmax(wg), 1)
        if len(R[p]["exact_ks"]) < len(R[p]["ks"]):
            s += r"$^{\dagger}$"
        cells.append(s)
    lines.append(r"worst $k$-set, $\underline V_k$ & " + " & ".join(cells) + r"\\")
    lines.append(r"\midrule")
    cells = []
    for p in PROBS:
        g = R[p]["methods"]["Block greedy (Alg. 2)"]
        cells.append(f"{g['n_exact_opt']}/{len(R[p]['exact_ks'])}")
    lines.append(r"greedy attains $\overline V_k$ & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    write("losses", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
def table_bounds(D):
    B = D["bayes_bounds"]
    rows = {r["k"]: r for r in B["rows"]}
    ks = [1, 2, 4, 8, 12, 18, 24, 30, 33, 34, 35, 36]
    L = lambda r, key: None if r[key] is None else r[key] / LN10
    lines = [r"\begin{tabular}{@{}r r r r r @{\hspace{1em}} r r r r r @{\hspace{1em}} r@{}}",
             r"\toprule",
             r" & \multicolumn{4}{c}{observed} & \multicolumn{5}{c}{lower bounds on $\underline V_k$} & \\",
             r"\cmidrule(r{1em}){2-5}\cmidrule(r{1em}){6-10}",
             r"$k$ & $\widehat V_k$ & greedy & random & worst & \exEqViaVq & \exEqViaSingles & \exEqViaPhi & C(iii) & C(i) & guarantee\\",
             r"\midrule"]
    for k in ks:
        r = rows[k]
        vb = f2(L(r, "logVbar")) + (r"$^{\star}$" if r["exact"] else "")
        worst = L(r, "worst_exact")
        ws = f2(worst) if worst is not None else r"$\le$\," + f2(L(r, "worst_found"))
        cells = [str(k), vb, f2(L(r, "greedy")), f2(L(r, "random_median")), ws,
                 f2(L(r, "b9")) if r["b9"] is not None else "--",
                 f2(L(r, "b10")), f2(L(r, "b11")), f2(L(r, "ciii")), f2(L(r, "ci")),
                 f2(L(r, "greedy_guarantee"))]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("bounds", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
def table_rr(D):
    RR = D["rr"]
    R = D["results"]
    spec = [("trap", "8", r"trap, $k=8$"), ("devil", "8", r"stairs, $k=8$"), ("devil", "12", r"stairs, $k=12$"),
            ("pmeans", "2", r"Perdig\~ao (i), $k=2$"), ("pmeans", "8", r"Perdig\~ao (i), $k=8$"),
            ("pbayes", "8", r"Perdig\~ao (iii), $k=8$")]
    sels = [("Block greedy", "greedy"), ("Exact optimum", r"$\overline V_k$ set"),
            ("dgeqp3 (induced)", r"\texttt{dgeqp3}"),
            ("Exact worst", r"$\underline V_k$ set"), ("Random (median of 200)", "random (median)")]
    lines = [r"\begin{tabular}{@{}l l r r r r@{}}", r"\toprule",
             r"problem & selection & loss & $\mu_{\mathrm{ord}}$ & $\mu_{\mathrm{res}}$ & $\nu$\\", r"\midrule"]
    for key, k, lab in spec:
        rows = RR[key][k]
        best = R[key]["best"][int(k) - 1] / LN10
        first = True
        for skey, slab in sels:
            if skey not in rows:
                continue
            v = rows[skey]
            if "logvol10" in v:
                loss = f2(best - v["logvol10"])
            else:                     # random: median volume loss over the 200 draws of the benchmark run
                loss = f2(R[key]["random"]["median"][int(k) - 1])
            nu = v["nu"]
            nus = f"{nu:.2g}" if nu < 1e3 else (r"$" + f"{nu:.1e}".replace("e+0", r"\times10^{").replace("e+", r"\times10^{") + "}$")
            lines.append((lab if first else "") + " & " + slab + " & " + loss + " & " +
                         f"{v['mu_ord']:.3g}" + " & " + f"{v['mu_res']:.3g}" + " & " + nus + r"\\")
            first = False
        lines.append(r"\addlinespace[2pt]")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("rr", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
def table_sketch(D):
    SK = D["sketch"]
    lines = [r"\begin{tabular}{@{}l r r r r r r r@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{(i) $u,v,w,T_s$} & \multicolumn{2}{c}{(ii) $+$ moments} & \multicolumn{2}{c@{}}{(iii) Bayesian}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
             r"sketch & $\gamma$ & same set & loss & same set & loss & same set & loss\\", r"\midrule"]
    for kind, klab in (("gaussian", "Gaussian"), ("sparse_sign", "sparse sign")):
        for g in ("1.25", "2.0", "4.0"):
            cells = [klab if g == "1.25" else "", g.rstrip("0").rstrip(".") if g != "1.25" else "1.25"]
            for key in ("pmeans", "pturb", "pbayes"):
                r = SK[key][f"{kind}|{g}"]
                cells.append(f"{100 * r['frac_same_set_mean_over_k']:.0f}\\%")
                cells.append(f2(r["median_max_gap_vs_greedy"]) + r"\,{\scriptsize(" + f2(r["max_gap_vs_greedy"]) + ")}")
            lines.append(" & ".join(cells) + r"\\")
        if kind == "gaussian":
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("sketch", "\n".join(lines) + "\n")
    E = SK["example_2_1_sketch"]
    lines = [r"\begin{tabular}{@{}l" + "r" * len(E["rows"]) + "@{}}", r"\toprule",
             r"sketch rows $d$ & " + " & ".join(str(r["d"]) for r in E["rows"]) + r"\\", r"\midrule",
             r"$\Pr[\text{greedy on }\boldsymbol\Omega\mathbf A\text{ takes }B_1]$ & " +
             " & ".join(f"{r['p_trap']:.2f}" for r in E["rows"]) + r"\\",
             r"mean $\log_{10}\mathbb{V}\mathrm{ol}(\mathbf A_{G})/\overline V_2$ & " +
             " & ".join(f2(r["mean_log10_ratio"]) for r in E["rows"]) + r"\\",
             r"\bottomrule", r"\end{tabular}"]
    write("sketch_trap", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
def table_alg2(D):
    S = D["alg2"]["summary"]
    var = [("a", r"$[\hat{\mathbf Q},\mathbf I_b]$ (QR), $[n]$"),
           ("b", r"$[\mathbf U,\mathbf V]$ (SVD), $\{t,\dots,n\}$"),
           ("c", r"$[\mathbf U,\mathbf V]$ (SVD), $[n]$"),
           ("b, x3 V", r"row 2 with $\times_3\mathbf V$ in line 9")]
    def e(x):
        if x is None:
            return "--"
        if x == 0:
            return "0"
        if x >= 0.1:
            return f"{x:.1f}"
        m, ex = f"{x:.1e}".split("e")
        return rf"${m}\cdot10^{{{int(ex)}}}$"
    lines = [r"\begin{tabular}{@{}l c c c c c@{}}", r"\toprule",
             r"\textsc{OrthoFactors}, \textsc{Scope} & same $\pi$ & identity & below & diag.\ blocks & $\boldsymbol{\mathcal W}$\\",
             r"\midrule"]
    for key, lab in var:
        s = S[key]
        diag = s.get("max_diag_blocks_offdiag")
        if key == "a":
            diag_s = e(s["max_diag_blocks_strict_lower"]) + r"$^{\triangle}$"
        else:
            diag_s = e(diag)
        rec = s.get("max_W_chain", s.get("max_W_all_equal"))
        lines.append(" & ".join([lab, f"{s['same_pivots_as_alg1']}/{s['count']}", e(s["max_identity"]),
                                 e(s["max_below_block_diag"]), diag_s, e(rec) if rec is not None else "--"]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("alg2", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
def table_verify(D):
    V = D["verify"]
    names = [("eq8", r"\exEqSwap: $\mathbb V\mathrm{ol}(\mathbf A_S)\le\mathbb V\mathrm{ol}(\mathbf A_T)\,\mathbb V\mathrm{ol}(\mathbf A_{S\setminus T})$, all $S,T$"),
             ("b9", r"\exEqViaVq"), ("b10", r"\exEqViaSingles"), ("b11", r"\exEqViaPhi{} (Theorem~\exThmSummary)"),
             ("ci", r"Corollary~\exCorExplicit(i)"), ("cii", r"Corollary~\exCorExplicit(ii), when $\alpha\|\mathbf X\|_2\ge1$"),
             ("ciii", r"Corollary~\exCorExplicit(iii)"), ("civ", r"Corollary~\exCorExplicit(iv)"),
             ("chain_Vq_le_prod", r"$\overline V_q\le\prod_{j\le q}v_{[j]}$"),
             ("chain_Vq_le_Phi", r"$\overline V_q\le\Phi_r(\alpha\mathbf X)$"),
             ("greedy", r"greedy guarantee $\mathbb V\mathrm{ol}(\mathbf A_G)\ge\overline V_k^{\,1-(1-1/k)^k}$")]
    lines = [r"\begin{tabular}{@{}l r r@{}}", r"\toprule", r"inequality & checks & largest violation\\", r"\midrule"]
    for key, lab in names:
        w = V["worst_violation"][key]
        ws = "0" if w <= 0 else rf"${w:.1e}".replace("e-", r"\cdot10^{-") + "}$"
        lines.append(f"{lab} & {V['n_checks'][key]:,} & {ws}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("verify", "\n".join(lines) + "\n")


def table_problems(D):
    R = D["results"]
    desc = {
        "trap": r"trap family, $k=8$, $\varepsilon=\tfrac12$, $\delta=10^{-2}$",
        "kahan": r"block Kahan, $c=\tfrac14$",
        "devil": r"block devil's stairs",
        "pmeans": r"Perdig\~ao (i): $u,v,w,T_s$",
        "pturb": r"Perdig\~ao (ii): (i) and moments",
        "pbayes": r"Perdig\~ao (iii): $[\alpha\mathbf X;\mathbf I]$, $\alpha=3$",
    }
    lines = [r"\begin{tabular}{@{}l r r r l@{}}", r"\toprule",
             r"problem & $m$ & $n$ & $b$ & $k$ enumerated\\", r"\midrule"]
    for p in ALL_PROBS:
        S = R[p]
        ex = S["exact_ks"]
        if len(ex) == len(S["ks"]):
            exs = rf"all ($1\le k\le {S['ks'][-1]}$)"
        else:
            lo = [k for k in ex if k <= S["n"] // 2]
            hi = [k for k in ex if k > S["n"] // 2]
            exs = rf"$k\le {max(lo)}$, $k\ge {min(hi)}$"
        lines.append(f"{desc[p]} & {S['m']:,} & {S['n']} & {S['b']} & {exs}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("problems", "\n".join(lines) + "\n")



# --------------------------------------------------------------------------
def table_tensor(D):
    """Algorithm 2 and the *_M-QR methods on A and on Gaussian sketches Omega A, trap
    family and devil's stairs: largest loss over k (on A); on sketches the mean over
    sketches and, in parentheses, the worst sketch."""
    TSJ = load("tensor_sketch.json")
    TS, TK = TSJ["tensor"], TSJ["tensor_sketch"]
    R = D["results"]
    sizes = ("1.25", "2.0")

    def on_a(key, nm):
        rec = TS[key]["methods"].get(nm) or R[key]["methods"][nm]
        return f2(rec["max_gap"])

    # on the trap family the worst sketch costs every method the same (both traps), so
    # only the stairs columns carry the worst sketch; the caption says so
    trap_worst = {round(st["worst_max"], 6) for g in sizes for st in TK["trap"]["by_size"][g]["stats"].values()}
    assert len(trap_worst) == 1, trap_worst

    def on_sketch(key, g, nm):
        st = TK[key]["by_size"][g]["stats"][nm]
        if key == "trap":
            return f2(st["mean_max"])
        return f2(st["mean_max"]) + r"\,{\scriptsize(" + f2(st["worst_max"], 1) + ")}"

    def row(lab, mlab, nm_a, nm_s):
        cells = [lab, mlab]
        for key in ("trap", "devil"):
            cells.append(on_a(key, nm_a))
            cells += [on_sketch(key, g, nm_s) for g in sizes]
        return " & ".join(cells) + r"\\"

    # consistency: Algorithm 2 (a)-(c) = Algorithm 1 on every sketch; merged transforms agree
    for key in ("trap", "devil"):
        for g in sizes:
            B = TK[key]["by_size"][g]
            assert all(v == TK[key]["seeds"] for v in B["alg2_same_as_alg1"].values()), (key, g)
            st = B["stats"]
            for t in ("DCT", "Haar"):
                assert st[f"first_slice|{t}"]["median_by_k"] == st["first_slice|DFT"]["median_by_k"]
                assert st[f"first_slice|{t}"]["max_by_k"] == st["first_slice|DFT"]["max_by_k"]
        ords = {repr(TS[key]["methods"][f"first_slice|{t}"]["order"][:TS[key]["K"]]) for t in ("DFT", "DCT", "Haar")}
        assert len(ords) == 1
        ords = {repr(TS[key]["methods"][f"Alg. 2 ({v})"]["order"][:TS[key]["K"]]) for v in "abc"}
        assert max(TS[key]["methods"][f"Alg. 2 ({v})"]["max_dlog10_vs_greedy"] for v in "abc") < 1e-12
    d = {key: [TK[key]["by_size"][g]["d"] for g in sizes] for key in ("trap", "devil")}
    lines = [r"\begin{tabular}{@{}l l rrr rrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{trap family} & \multicolumn{3}{c@{}}{devil's stairs}\\",
             r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
             r"method & $\mathbf M$ & $\mathbf A$ & " + " & ".join(rf"$d={x}$" for x in d["trap"]) +
             r" & $\mathbf A$ & " + " & ".join(rf"$d={x}$" for x in d["devil"]) + r"\\", r"\midrule"]
    lines.append(row(r"Alg.~\exAlgGreedy${}={}$Alg.~\exAlgTensor\,(a)--(c)", "--", "Alg. 2 (a)", "Alg. 1"))
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"\multicolumn{8}{@{}l}{$\star_{\mathbf M}$-QR factorizations:}\\")
    # the unpivoted *_M-QR keeps the input order, so a sketch does not change its selection
    lines.append(" & ".join([r"\quad unpivoted~\cite{KilmerMartinPerrone2008}$^{\dagger}$", "any"]
                            + [on_a("trap", "none|I")] * 3 + [on_a("devil", "none|I")] * 3) + r"\\")
    groups = [("first_slice", r"\quad first-slice~\cite{Hao2013}", [("I", r"$\mathbf I$"), ("DFT", "DFT, DCT, Haar")]),
              ("tube_norm", r"\quad tube-norm", [("I", r"$\mathbf I$"), ("DFT", "DFT"), ("DCT", "DCT"), ("Haar", "Haar")])]
    for rule, lab, rows_ in groups:
        for i, (tn, mlab) in enumerate(rows_):
            lines.append(row(lab if i == 0 else "", mlab, f"{rule}|{tn}", f"{rule}|{tn}"))
    lines.append(r"\addlinespace[2pt]")
    lines.append(row(r"\texttt{dgeqp3}, induced", "--", "dgeqp3", "dgeqp3"))
    rnd = [f2(R[key]["random_max_median_gap"]) for key in ("trap", "devil")]
    lines.append(" & ".join([r"random (median)", "--"] + [rnd[0]] * 3 + [rnd[1]] * 3) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tensor", "\n".join(lines) + "\n")


def main():
    D = load("analysis.json")
    table_losses(D)
    table_bounds(D)
    table_rr(D)
    table_sketch(D)
    table_alg2(D)
    table_verify(D)
    table_problems(D)
    table_tensor(D)
    print("tables written to", TAB, sorted(p.name for p in TAB.glob("*.tex")))


if __name__ == "__main__":
    main()
