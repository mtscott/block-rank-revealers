#!/usr/bin/env python3
"""Numerical check of the three instantiations of Algorithm 2 (Table 2 of the paper).

    python experiments/fig_starm_checks.py      # writes figures/fig_starm_checks.{png,pdf}

Uses the example of experiments/tensorqr_notebook.ipynb: a random 18 x 6 x 3 tensor
(seed 1), b = 3, and k = 5 of the n = 6 tubal columns processed.

The paper's variants and blockrr's variant codes (which keep the earlier lettering):

    paper  Table 2 row                         Rotations / Scope       blockrr
    (a)    one adaptive M      (Prop. 3.4)     SVD, Scope = [n]        variant="c"
    (b)    facewise, M = I     (Prop. 3.5)     QR,  Scope = [n]        variant="a"
    (c)    adaptive M_j per    (Prop. 3.6)     SVD, Scope = {t..n}     variant="b"
           tubal column

For every variant and every tubal column j, with M_j = W^T in (a), I in (b) and
W_j^T in (c), the figure shows
  (i)   the relative residual of identity (16) of the paper,
            (A x3 M_j) facewise stack(I_n(:, pi_j)) = stack(Q) facewise Rtilde(:, j, :),
        whose tubal columns make up (17) in (a) and (18) in (b);
  (ii)  the loss of orthogonality ||Q^T *_{M_j} Q - I||_F of Q = stack(Q) x3 M_j^{-1},
        where Q^T transposes every frontal slice and I is the identity tensor of *_{M_j};
  (iii) and (iv) the nonzero pattern of unfold_1(Rtilde) for (a) and (c): entries with
        |r| > 1e-12 max|r| count as nonzero.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import make_figures as mf                       # noqa: E402  (paper style, colours, save())
from blockrr.algorithm2 import algorithm2      # noqa: E402
from blockrr.tensor import facewise, mode3, mprod, stack3, unfold1   # noqa: E402

plt = mf.plt
from matplotlib.colors import ListedColormap    # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402
from matplotlib.patches import Rectangle        # noqa: E402

M_, N_, B_, K_, SEED = 18, 6, 3, 5, 1           # as in tensorqr_notebook.ipynb
NZ_TOL = 1e-12                                  # |r| > NZ_TOL * max|r| counts as nonzero
DARK_VIOLET = "#3d2f9c"                         # with BLUE and ORANGE: all pairs pass the dataviz validator

# paper letter: blockrr variant, colour, marker, line style, x offset, legend label.
# Colour, marker and line style follow the method: facewise is Algorithm 1, blue as in every figure.
VARIANTS = {
    "a": ("c", DARK_VIOLET, "^", (0, (1, 1.3)), -0.14, r"(a) one adaptive $\mathbf{M}$"),
    "b": ("a", mf.BLUE, "o", "-", 0.0, r"(b) facewise, $\mathbf{M}=\mathbf{I}$"),
    "c": ("b", mf.ORANGE, "s", (0, (4, 1.6)), 0.14, r"(c) adaptive $\mathbf{M}_j$ per tubal column"),
}


def checks(seed=SEED):
    """Residual of (16) and orthogonality per tubal column, and Rtilde, for each variant."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M_, N_, B_))       # the notebook's A0
    Au = unfold1(A)
    T12 = lambda X: np.transpose(X, (1, 0, 2))  # noqa: E731  frontal-slice transpose
    out = {}
    for v, (code, *_rest) in VARIANTS.items():
        r = algorithm2(Au, B_, K_, variant=code, compress=False)
        Qs, Rt = stack3(r.Q, B_), r.R
        resid, orth = [], []
        for j in range(N_):
            W = {"a": r.W[:, 0, :], "b": np.eye(B_), "c": r.W[:, j, :]}[v]
            Mj = W.T
            e = np.eye(N_)[:, [r.order[j]]]                          # I_n(:, pi_j)
            lhs = facewise(mode3(A, Mj), stack3(e, B_))              # (A x3 M_j) facewise stack(...)
            rhs = facewise(Qs, Rt[:, j:j + 1, :])                    # stack(Q) facewise Rtilde(:, j, :)
            resid.append(np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs))
            Qj = mode3(Qs, W)                                        # stack(Q) x3 M_j^{-1}, M_j^{-1} = W
            Id = mode3(stack3(np.eye(M_), B_), W)                    # identity tensor of *_{M_j}
            orth.append(np.linalg.norm(mprod(T12(Qj), Qj, Mj) - Id))
        out[v] = {"order": r.order, "resid": np.array(resid), "orth": np.array(orth),
                  "R": unfold1(Rt)}
    return out


def spy(ax, R, colour, title, ylabel=True):
    nz = np.abs(R) > NZ_TOL * np.abs(R).max()
    ax.imshow(nz, cmap=ListedColormap(["white", mf.INK]), vmin=0, vmax=1,
              interpolation="none", aspect="equal")   # "none": the PDF embeds the 18 x 18 mask as is
    m, N = R.shape
    for x in range(B_, N, B_):
        ax.axvline(x - 0.5, color="#cfcec9", lw=0.5, zorder=2)
    for y in range(B_, m, B_):
        ax.axhline(y - 0.5, color="#cfcec9", lw=0.5, zorder=2)
    for t in range(K_):                          # diagonal blocks of the processed columns
        ax.add_patch(Rectangle((t * B_ - 0.5, t * B_ - 0.5), B_, B_, fill=False,
                               edgecolor=colour, lw=1.1, zorder=3))
    ax.set_xticks([t * B_ + (B_ - 1) / 2 for t in range(N // B_)], [str(t + 1) for t in range(N // B_)])
    ax.set_yticks([t * B_ + (B_ - 1) / 2 for t in range(m // B_)], [str(t + 1) for t in range(m // B_)])
    ax.tick_params(length=0, pad=2)
    if not ylabel:
        ax.tick_params(labelleft=False)
    ax.grid(False)
    for side in ax.spines.values():                # frame the whole matrix
        side.set_visible(True)
        side.set_color("#a8a7a2")
        side.set_linewidth(0.6)
    ax.set_xlabel("tubal column $j$")
    if ylabel:
        ax.set_ylabel("row block")
    ax.text(0.04, 0.04, f"nz = {int(nz.sum())}", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color=mf.INK2)
    ax.set_title(title)


def main():
    D = checks()
    for v, d in D.items():
        print(f"variant ({v}) pivots {[p + 1 for p in d['order']]}  residual {d['resid'].min():.1e}"
              f"-{d['resid'].max():.1e}  orthogonality {d['orth'].min():.1e}-{d['orth'].max():.1e}")
    eps = np.finfo(float).eps
    fig, axs = plt.subplots(1, 4, figsize=(mf.TEXTWIDTH, 2.05), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.04)
    axs[1].sharey(axs[0])
    j = np.arange(1, N_ + 1)
    for ax, key, title in ((axs[0], "resid", "(i) residual"), (axs[1], "orth", "(ii) orthogonality")):
        for v, (_code, c, mk, ls, dx, _lab) in VARIANTS.items():
            ax.plot(j + dx, D[v][key], color=c, ls=ls, lw=0.9, marker=mk, ms=3.4, mew=0.8,
                    mfc=c, mec=c, zorder=4)
        ax.axhline(eps, color=mf.GRAY, lw=0.7, ls=(0, (3, 2)), zorder=2)
        ax.annotate(r"machine $\varepsilon$", xy=(N_ + 0.45, eps), xytext=(0, -2),
                    textcoords="offset points", ha="right", va="top", fontsize=6.5, color=mf.MUTED)
        ax.set_yscale("log")
        ax.set_ylim(1e-17, 1e-13)
        ax.set_xlim(0.5, N_ + 0.5)
        ax.set_xticks(j)
        ax.set_xlabel("tubal column $j$")
        ax.set_title(title)
    axs[1].tick_params(labelleft=False)
    axs[3].sharey(axs[2])
    spy(axs[2], D["a"]["R"], VARIANTS["a"][1], r"(iii) $\widetilde{\mathcal{R}}$ in (a)")
    spy(axs[3], D["c"]["R"], VARIANTS["c"][1], r"(iv) $\widetilde{\mathcal{R}}$ in (c)", ylabel=False)
    hs = [Line2D([], [], color=c, ls=ls, lw=0.9, marker=mk, ms=3.4, mfc=c, mec=c, label=lab)
          for _code, c, mk, ls, _dx, lab in VARIANTS.values()]
    fig.legend(handles=hs, loc="outside upper center", ncol=3, frameon=False,
               handlelength=2.2, columnspacing=1.0, handletextpad=0.5)
    mf.save(fig, "fig_starm_checks")
    print("written", mf.FIG / "fig_starm_checks.png")


if __name__ == "__main__":
    main()
