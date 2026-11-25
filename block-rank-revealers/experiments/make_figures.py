#!/usr/bin/env python3
"""Figures of the experiments section, drawn from results/*.json.

    python experiments/make_figures.py      # writes figures/*.png (+ vector .pdf copies)

PNGs at 600 dpi, which the paper includes, sized for elsarticle preprint 12pt (text
width 390pt = 5.4in) and set in Latin Modern so they match the paper's Computer Modern
text.  Each figure is also written as a vector PDF.  Latin Modern ships only with CFF
outlines, so the PDFs embed it as Type 3 (pdf.fonttype = 3); Type 42 would label CFF
data as TrueType.

Colour follows the entity and never changes between figures:
  block greedy (= Algorithm 2) = blue, Businger-Golub / dgeqp3 = orange (solid /
  dashed), HQRRP / CQRRPT = magenta (solid / dashed), sRRQR / randomized sRRQR =
  green (solid / dashed), first-slice / tube-norm *_M-QR = violet / red,
  random selection = gray, exact references = black.
Curves that are all "greedy for different instances" use the blue ramp; the
lower bounds of Section 2 use a violet ramp ordered by strength.  The
categorical slots were checked with the dataviz validator (adjacent CVD and
normal-vision separation pass; magenta is below 3:1 on white, so every series
also carries a marker and a line style, and the numbers behind every figure are
tabulated in the text).
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from PIL import Image  # noqa: E402  (a dependency of matplotlib)

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from blockrr.problems import kahan_shift_log10_ratio  # noqa: E402

# Use Latin Modern Roman (the paper's font) when a TeX installation provides it, and
# register only its 10pt optical size; otherwise matplotlib may pick a display size
# (lmroman17) for the whole family.  Without it the figures fall back to a serif font.
_LM_GLOBS = ("/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf",
             "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
             "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
             "/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf")
_LM = sorted({f for g in _LM_GLOBS for f in glob.glob(g)})
for f in _LM:
    fm.fontManager.addfont(f)
fm.fontManager.ttflist = [e for e in fm.fontManager.ttflist
                          if not (e.name == "Latin Modern Roman" and "lmroman10-" not in e.fname)]
fm.fontManager._findfont_cached.cache_clear()
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#7a7974"
plt.rcParams.update({
    "font.family": "Latin Modern Roman" if _LM else "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.5,
    "legend.frameon": False, "pdf.fonttype": 3, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "lines.solid_capstyle": "round",
    "axes.titlelocation": "left", "axes.titlepad": 4,
})
TEXTWIDTH = 390 / 72.27          # inches, elsarticle preprint 12pt

# ---------------------------------------------------------------- entities
BLUE, ORANGE, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#e87ba4", "#008300"
VIOLET, RED = "#6250d6", "#e34948"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]           # dataviz blue steps 250/350/450/600
VIOLET_RAMP = ["#22185e", "#3d2f9c", "#5a4bc8", "#7c6fdc", "#9a90e6"]  # validated ordinal ramp
GRAY, BAND = "#7a7974", "#e9e8e4"
STYLE = {   # label: (colour, marker, linestyle, filled, zorder)
    "Block greedy (Alg. 2)": (BLUE, "o", "-", True, 9),
    "Businger-Golub": (ORANGE, "s", "-", True, 5),
    "dgeqp3": (ORANGE, "s", (0, (4, 2)), False, 5),
    "HQRRP": (MAGENTA, "v", "-", True, 4),
    "CQRRPT": (MAGENTA, "v", (0, (4, 2)), False, 4),
    "sRRQR (Gu-Eisenstat)": (GREEN, "P", "-", True, 4),
    "Randomized sRRQR": (GREEN, "X", (0, (4, 2)), False, 4),
    # pivoted *_M-QR; with M = DFT and DCT first-slice pivoting selects the same blocks,
    # so "first_slice|DFT" stands for both (checked in _starm_coincide)
    "first_slice|I": (VIOLET, "<", (0, (4, 2)), False, 6),
    "first_slice|DFT": (VIOLET, "<", "-", True, 6),
    "tube_norm|I": (RED, ">", (0, (4, 2)), False, 6),
    "tube_norm|DFT": (RED, ">", "-", True, 6),
    "tube_norm|DCT": (RED, "h", (0, (5, 1.5, 1, 1.5)), False, 6),
}
PLAIN = {
    "Block greedy (Alg. 2)": "block greedy (Alg. 1)",
    "Businger-Golub": "Businger–Golub",
    "dgeqp3": "dgeqp3",
    "HQRRP": "HQRRP (median)",
    "CQRRPT": "CQRRPT (median)",
    "sRRQR (Gu-Eisenstat)": "sRRQR",
    "Randomized sRRQR": "randomized sRRQR (median)",
    "first_slice|I": r"first-slice $\star_M$-QR, $M=I$",
    "first_slice|DFT": r"first-slice $\star_M$-QR, $M$ = DFT, DCT",
    "tube_norm|I": r"tube-norm $\star_M$-QR, $M=I$",
    "tube_norm|DFT": r"tube-norm $\star_M$-QR, $M$ = DFT",
    "tube_norm|DCT": r"tube-norm $\star_M$-QR, $M$ = DCT",
}
ORDER = list(STYLE)
LN10 = math.log(10.0)


def A(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def line(ax, x, y, nm, every=1, ms=3.4, lw=1.25, offset=0, label=True):
    c, mk, ls, filled, z = STYLE[nm]
    ax.plot(x, y, color=c, ls=ls, lw=lw, marker=mk, ms=ms, mew=0.85,
            mfc=c if filled else "white", mec=c, zorder=z,
            markevery=(offset % max(every, 1), every), label=PLAIN[nm] if label else None)


def handles(names, random=True, seeds=False, exact_shade=False):
    hs = []
    for nm in names:
        c, mk, ls, filled, _ = STYLE[nm]
        hs.append(Line2D([], [], color=c, ls=ls, lw=1.25, marker=mk, ms=4, mew=0.85,
                         mfc=c if filled else "white", mec=c, label=PLAIN[nm]))
    if seeds:
        hs.append(Patch(facecolor="#d9d8d4", edgecolor="none", label="seed range, in method colour"))
    if random:
        hs.append(Line2D([], [], color=GRAY, lw=1.0, ls=(0, (1, 1.4)), label="random $k$-set (median)"))
        hs.append(Patch(facecolor=BAND, edgecolor="none", label="random $k$-set (10–90%)"))
    if exact_shade:
        hs.append(Patch(facecolor="#f3f2ee", edgecolor="#cfcec9", lw=0.5, label=r"$\overline{V}_k$ enumerated"))
    return hs


def load():
    with open(BUILD / "analysis.json") as f:
        return json.load(f)


PNG_DPI = 600


def save(fig, name):
    # the PDF first: constrained layout is computed with the renderer of the first save
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None})   # reproducible bytes
    png = FIG / f"{name}.png"
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    # Agg writes RGBA; the background is opaque, so drop the alpha channel (pdfTeX would
    # otherwise embed every figure with a soft mask)
    with Image.open(png) as im:
        assert im.mode == "RGBA" and im.getextrema()[3][0] == 255, f"{png} has transparent pixels"
        rgb = im.convert("RGB")
    rgb.save(png, dpi=(PNG_DPI, PNG_DPI), optimize=True)


# ---------------------------------------------------------------- gap panels

def gap_panel(ax, S, title, *, every=1, ylim=None, shade_exact=False, methods=None,
              random=True, seeds=True, ms=3.4):
    ks = np.array(S["ks"])
    names = [n for n in ORDER if n in S["methods"] and (methods is None or n in methods)]
    if shade_exact:
        ex = set(S["exact_ks"])
        for k in ks:
            if k in ex:
                ax.axvspan(k - 0.5, k + 0.5, color="#f3f2ee", lw=0, zorder=-2)
    if random:
        ax.fill_between(ks, -A(S["random"]["p90"]), -A(S["random"]["p10"]), color=BAND, lw=0, zorder=0)
        ax.plot(ks, -A(S["random"]["median"]), color=GRAY, lw=1.0, ls=(0, (1, 1.4)), zorder=1)
    if seeds:
        for nm in names:
            m = S["methods"][nm]
            if "gap_worst_seed" in m:
                ax.fill_between(ks, -A(m["gap_worst_seed"]), -A(m["gap_best_seed"]),
                                color=STYLE[nm][0], alpha=0.13, lw=0, zorder=1)
    for i, nm in enumerate(names):
        line(ax, ks, -A(S["methods"][nm]["gap"]), nm, every=every, offset=i, ms=ms)
    ax.axhline(0, color=INK, lw=0.6, zorder=3)
    ax.set_title(title)
    ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
        lo = float(np.nanmin(-A(S["random"]["median"]))) if random else 0.0
        if random and lo < ylim[0]:
            ax.annotate(f"random median\nreaches {lo:.1f}", xy=(0.98, 0.03), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=6.5, color=MUTED)
    return names


def _starm_coincide(S):
    """First-slice pivoting with M = DFT and with M = DCT selects the same blocks."""
    m = S["methods"]
    if "first_slice|DFT" in m:
        assert np.allclose(A(m["first_slice|DFT"]["gap"]), A(m["first_slice|DCT"]["gap"]), atol=1e-12)


def _kahan_natural_order(ax, S):
    """Dashed: -log10 Vol(K_{2..k+1}) / Vol(K_{1..k}), the loss of blocks 1..k, which every
    method that keeps the natural order incurs (closed form, c = 1/b as in run_benchmarks)."""
    b = S["b"]
    ks = list(S["ks"])
    y = [-kahan_shift_log10_ratio(b, 1.0 / b, k) for k in ks]
    ax.plot(ks, y, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=30)
    k0 = ks[min(10, len(ks) - 1)]
    ax.annotate(r"$-\log_{10}\,\mathbb{V}\mathrm{ol}(\mathbf{K}_{\{2,\dots,k+1\}})\,/\,"
                r"\mathbb{V}\mathrm{ol}(\mathbf{K}_{\{1,\dots,k\}})$" "\n" r"(blocks $1,\dots,k$)",
                xy=(k0, y[ks.index(k0)]), xytext=(1.25, -2.78), fontsize=6.5, color=INK,
                ha="left", va="top", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=1, shrinkB=1.5,
                                connectionstyle="arc3,rad=0.15"))


def fig_toy(D):
    R = D["results"]
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.35), constrained_layout=True)
    panels = [("kahan", r"(a) block Kahan, $c=1/4$, $b=4$", (-3.45, 0.15)),
              ("devil", r"(b) block devil's stairs, $b=4$", (-3.75, 0.15))]
    used = []
    for ax, (key, title, yl) in zip(axs.flat, panels):
        _starm_coincide(R[key])
        used += gap_panel(ax, R[key], title, ylim=yl, random=False)
        if key == "kahan":
            _kahan_natural_order(ax, R[key])
        ks = R[key]["ks"]
        ax.set_xticks(ks if len(ks) <= 9 else ks[::2])
        ax.set_xlabel(r"number of selected blocks $k$")
    axs[0].set_ylabel(r"$\log_{10}\,\mathbb{V}\mathrm{ol}(\mathbf{A}_S)\,/\,\overline{V}_k$")
    names = [n for n in ORDER if n in set(used)]
    fig.legend(handles=handles(names, random=False, seeds=True), loc="outside lower center", ncol=4,
               handlelength=2.4, columnspacing=1.0, fontsize=7)
    save(fig, "fig_toy_gaps")


def fig_perdigao(D):
    """Methods whose first step ties (tiebreaks.py) are drawn as the median over first picks,
    without a range; the randomized methods keep their seed ranges."""
    R = D["results"]
    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 3.6), constrained_layout=True)
    panels = [("pmeans", r"(a) $u,v,w,T_s$; $b=4$", (-4.0, 0.2)),
              ("pturb", r"(b) $+$ second moments; $b=8$", (-9.8, 0.3)),
              ("pbayes", r"(c) Bayesian, $\alpha=3$; $b=4$", (-1.8, 0.08))]
    used = []
    for j, (key, title, yl) in enumerate(panels):
        ax = axs[j]
        _starm_coincide(R[key])
        used += gap_panel(ax, R[key], title, every=5, ylim=yl, shade_exact=True, random=False)
        ax.set_xticks([1, 10, 20, 30, 36])
        ax.set_xlabel(r"towers selected $k$")
    axs[0].set_ylabel(r"$\log_{10}\,\mathbb{V}\mathrm{ol}(\mathbf{A}_S)\,/\,\widehat V_k$")
    names = [n for n in ORDER if n in set(used)]
    tied = {nm for key, _, _ in panels for nm, m in R[key]["methods"].items() if "n_tiebreaks" in m}
    hs = handles(names, random=False, seeds=True, exact_shade=True)
    for h, nm in zip(hs, names):
        if nm in tied:
            h.set_label(PLAIN[nm] + " (median)")
    fig.legend(handles=hs, loc="outside lower center", ncol=3, handlelength=2.4, columnspacing=1.0, fontsize=7)
    save(fig, "fig_perdigao_gaps")


# ---------------------------------------------------------------- sharp limits

SHARP_LEGEND = dict(fontsize=6.2, handlelength=1.7, handletextpad=0.5, borderaxespad=0.3,
                    labelspacing=0.3)


def _example_panel(ax, sw, title):
    fine = sw["example_2_1_fine"]
    d = np.array([r["delta"] for r in fine])
    for (a, c) in zip(("10", "100", "1000", "10000"), BLUE_RAMP):
        y = np.array([r[f"alpha={a}"] for r in fine])
        e = int(round(math.log10(int(a))))
        ax.plot(d, y, color=c, lw=1.2, marker="o", ms=2.2, markevery=3, zorder=4)
        ax.axhline(sw["aug_floor_pred"][a], color=c, lw=0.7, ls=(0, (1, 1.4)), zorder=2)
        ax.annotate(rf"$\alpha=10^{{{e}}}$", xy=(1.3e-8, sw["aug_floor_pred"][a]), xytext=(0, 2.0),
                    textcoords="offset points", fontsize=6.5, color=INK2, va="bottom")
    y = np.array([r["plain"] for r in fine])
    ax.plot(d, y, color=INK, lw=1.2, marker="s", ms=2.2, markevery=3, zorder=5)
    pred = np.array([r["predicted"] for r in fine])
    ax.plot(d, pred, color="white", lw=0.7, ls=(0, (2, 2)), zorder=6)
    ax.annotate(r"$\mathbf{X}$ (no prior)", xy=(1.0e-8, -4.6), fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(7e-9, 0.45)
    ax.set_ylim(-8.3, 0.25)
    ax.set_title(title)
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.2, marker="o", ms=2.4, label="computed"),
                       Line2D([], [], color=INK2, lw=0.7, ls=(0, (1, 1.4)), label=r"limit $\rho(\alpha)$"),
                       Line2D([], [], color=INK, lw=0.8, ls=(0, (2, 2)), label="closed form")],
              loc="lower right", **SHARP_LEGEND)


def fig_sharp(D):
    """One row: (a) the trap family, (b) Example 2.1 with and without the prior."""
    sw = D["sweeps"]
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.3), constrained_layout=True)

    ax = axs[0]
    rows = [r for r in sw["trap"] if r.get("summary")]
    expo_colour = {1: BLUE_RAMP[0], 2: BLUE_RAMP[1], 4: BLUE_RAMP[2], 6: BLUE_RAMP[3]}
    markers = {(2, 2): "o", (3, 3): "o", (4, 2): "s", (5, 2): "D", (6, 3): "o", (8, 4): "o"}
    for s in rows:
        k, b = s["k"], s["b"]
        pts = [r for r in sw["trap"] if not r.get("summary") and r["k"] == k and r["b"] == b]
        x = np.array([r["delta"] for r in pts]); y = np.array([r["log10_ratio"] for r in pts])
        c = expo_colour[s["predicted_slope"]]
        ax.plot(x, y, color=c, lw=1.1, marker=markers[(k, b)], ms=2.4, mfc=c, mec=c,
                label=rf"$({k},{b})$: {s['fitted_slope']:.2f}")
    ax.set_xscale("log")
    ax.set_xticks([1e-6, 1e-4, 1e-2])
    ax.set_ylim(-41, 1.5)
    ax.set_title(r"(a) $\lceil k/b\rceil$-trap family")
    ax.legend(loc="lower right", title=r"$(k,b)$: slope", title_fontsize=6.2, **SHARP_LEGEND)
    ax.set_ylabel(r"$\log_{10}\,\mathbb{V}\mathrm{ol}(\mathbf{A}_{G})\,/\,\overline{V}_k$")

    _example_panel(axs[1], sw, r"(b) $\mathbf{X}$ and $[\alpha\mathbf{X};\mathbf{I}]$")
    axs[1].set_ylabel(r"$\log_{10}\,\mathbb{V}\mathrm{ol}(\mathbf{A}_{G})\,/\,\overline{V}_2$")
    for ax in axs:
        ax.set_xlabel(r"$\delta$")
    axs[1].set_xticks([1e-8, 1e-6, 1e-4, 1e-2])
    save(fig, "fig_sharp_limits")


# ---------------------------------------------------------------- tensor methods

# ---------------------------------------------------------------- bounds

BOUNDS = [  # key, label, colour, linestyle; numbering of Theorem 2.5 in the manuscript
    ("b9", r"(11)", VIOLET_RAMP[0], "-"),                                  # Vbar_k / Vbar_q
    ("b10", r"(12)", VIOLET_RAMP[1], (0, (6, 1.5))),                        # Vbar_k / prod v_[j]
    ("b11", r"(13), $\overline{V}_k/\Phi_r(\alpha\mathbf{X})$", VIOLET_RAMP[2], (0, (3, 1.5))),
    ("ci", r"(13), $\Lambda^{-r}\,\overline{V}_k$", VIOLET_RAMP[4], (0, (5, 1.5, 1, 1.5))),
]


def fig_bounds(D):
    B = D["bayes_bounds"]
    rows = B["rows"]
    k = np.array([r["k"] for r in rows])
    L = lambda key: np.array([np.nan if r[key] is None else r[key] for r in rows], float) / LN10
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.75), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, xl, yl in ((axs[0], (0.5, 36.5), (-1.2, 29.5)), (axs[1], (22.5, 36.5), (-0.6, 29.0))):
        ax.axhspan(yl[0] - 5, 0, color="#f3f2ee", lw=0, zorder=-2)
        ax.axhline(0, color=INK2, lw=0.6, zorder=0)
        ax.fill_between(k, L("worst_found"), L("logVbar"), color="#e8f0fb", lw=0, zorder=0)
        ax.plot(k, L("logVbar"), color=INK, lw=1.2, zorder=6)
        ax.plot(k, L("worst_found"), color=INK, lw=0.9, ls=(0, (3, 1.5)), zorder=6)
        ax.plot(k, L("worst_exact"), ls="none", marker="x", ms=4, mew=0.9, color=INK, zorder=7)
        ax.plot(k, L("random_median"), color=GRAY, lw=1.0, ls=(0, (1, 1.4)), zorder=5)
        line(ax, k, L("greedy"), "Block greedy (Alg. 2)", every=3, ms=3.0)
        ax.plot(k, L("greedy_guarantee"), color=BLUE, lw=0.9, ls=(0, (5, 2)), zorder=5)
        for key, lab, c, ls in BOUNDS:
            y = L(key)
            if key == "b9":
                ax.plot(k, y, ls="none", marker="o", ms=3.2, mfc="white", mec=c, mew=1.0, zorder=8)
            else:
                ax.plot(k, y, color=c, lw=1.15, ls=ls, zorder=4)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_xlabel(r"towers selected $k$")
    axs[1].set_xticks([24, 27, 30, 33, 36])
    axs[0].set_xticks([1, 10, 20, 30, 36])
    axs[0].set_ylabel(r"$\log_{10}$ volume")
    axs[0].set_title(r"(a) Perdigão, Bayesian problem ($\alpha=3$, $n=37$)")
    axs[1].set_title(r"(b) detail, $k>n/2$")
    hs = [Line2D([], [], color=INK, lw=1.2, label=r"$\widehat V_k$ (best known)"),
          Line2D([], [], color=STYLE["Block greedy (Alg. 2)"][0], lw=1.25, marker="o", ms=3.4,
                 label="block greedy"),
          Line2D([], [], color=BLUE, lw=0.9, ls=(0, (5, 2)),
                 label=r"greedy guarantee $\overline{V}_k^{\,1-(1-1/k)^k}$, Thm 2.7"),
          Line2D([], [], color=GRAY, lw=1.0, ls=(0, (1, 1.4)), label="random $k$-set (median)"),
          Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 1.5)), label=r"anti-greedy ($\geq\underline{V}_k$)"),
          Line2D([], [], color=INK, ls="none", marker="x", ms=4, label=r"$\underline{V}_k$ enumerated")]
    for key, lab, c, ls in BOUNDS:
        if key == "b9":
            hs.append(Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=c, mew=1.0,
                             label=r"bound (11) on $\underline{V}_k$"))
        else:
            hs.append(Line2D([], [], color=c, lw=1.15, ls=ls, label=f"bound {lab}"))
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=6.8, handlelength=2.4,
               columnspacing=1.0)
    save(fig, "fig_bayes_bounds")


# ---------------------------------------------------------------- map

def fig_map(D):
    P = D["perdigao"]
    rep = D["reproduce"]["pmeans"]
    names = P["towers"]
    ll = np.array(P["latlon"], float)
    order = rep["greedy_order"]
    n = len(names)
    rank = np.full(n, n, dtype=int)
    for i, j in enumerate(order):
        rank[j] = i + 1
    lat0 = ll[:, 0].mean()
    x = (ll[:, 1] - ll[:, 1].mean()) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (ll[:, 0] - lat0) * 110.57
    g8 = order[:8]
    q8 = rep["geqp3_order"][:8]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.66, 2.75), constrained_layout=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "bluesteps", ["#0d366b", "#256abf", "#5598e7", "#9ec5f4", "#e2edfb"])
    ax.scatter(x[q8], y[q8], s=95, facecolor="none", edgecolor=ORANGE, linewidth=1.1, zorder=2)
    sc = ax.scatter(x, y, c=rank, cmap=cmap, s=24, edgecolor=INK2, linewidth=0.4, zorder=3, vmin=1, vmax=n)
    for j in g8:
        t = ax.annotate(f"{rank[j]}", (x[j], y[j]), xytext=(3.5, 3.0), textcoords="offset points",
                        fontsize=7, color=INK, zorder=4)
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    cb = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02, ticks=[1, 10, 20, 30, 37])
    cb.set_label("order of selection by block greedy", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=5.5, mfc="#5598e7", mec=INK2, mew=0.4,
                              label="tower (numbers: first 8 of block greedy)"),
                       Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec=ORANGE, mew=1.1,
                              label="first 8 blocks induced by dgeqp3")],
              loc="upper left", fontsize=6.6, handletextpad=0.3)
    ax.set_xlim(x.min() - 0.18, x.max() + 0.25)
    ax.set_ylim(y.min() - 0.15, y.max() + 0.55)
    save(fig, "fig_perdigao_map")
    return {"overlap_g8_q8": len(set(g8) & set(q8)), "g8": [names[j] for j in g8], "q8": [names[j] for j in q8]}


def main():
    D = load()
    fig_sharp(D)
    fig_toy(D)
    fig_perdigao(D)
    fig_bounds(D)
    info = fig_map(D)
    print("figures written to", FIG, info)


if __name__ == "__main__":
    main()
