# Block rank revealers

Code for

> J. Haddock, R. Murray and M. T. Scott, *Rank revealers under a column-block constraint:
> sharp limits, an instance-dependent theory, and the ⋆<sub>M</sub> viewpoint* (preprint, 2026).

The columns of **A** = [**A**<sub>1</sub>, …, **A**<sub>n</sub>] come in atomic blocks of
width *b*: choosing a block means taking all *b* of its columns, as when an experiment, a
sensor or a weather tower is bought as a whole. `blockrr` selects blocks by greedy block
pivoting, the log-det rule of the paper, in matrix form (Algorithm 1) and in tensor form
(Algorithm 2, with its three variants (a), (b), (c)). It also contains the methods the paper
compares against, and the scripts that regenerate every number, figure and table of its
experiments section.

## Install

```bash
git clone https://github.com/<user>/block-rank-revealers.git
cd block-rank-revealers
pip install -e ".[experiments]"      # numpy, scipy; matplotlib for the figures
python -m pytest -q                  # 65 tests, a few seconds
```

Python 3.10 or later. The paper's runs used Python 3.13, NumPy 2.5.3 and SciPy 1.18.1.

## Algorithm 2 and its variants

Algorithm 2 folds **A** into an *m* × *n* × *b* tensor whose tubal columns are the blocks,
and at step *t* pivots the block whose residual has the largest volume. Its two subroutines
give three instantiations:

| variant | `ORTHOFACTORS(C_p)` | `SCOPE(t, n)` | diagonal blocks of **R** | identity |
|:-:|---|---|---|---|
| `"a"` | [**Q̂**, **I**<sub>b</sub>] (QR) | [n] | upper triangular | facewise, **M** = **I** |
| `"b"` | [**U**, **V**] (SVD) | {t, …, n} | diagonal | one **M** = **W**<sub>j</sub><sup>⊤</sup> per tubal column |
| `"c"` | [**U**, **V**] (SVD) | [n] | no structure | a single **M** = **W**<sup>⊤</sup> |

All three pick the same blocks as Algorithm 1. The paper's Table 2 letters them differently:
`"c"` is its variant (a), one adaptive **M**; `"a"` is (b), facewise; and `"b"` is (c),
adaptive **M**<sub>j</sub> per tubal column.

```python
import numpy as np
import blockrr

A = np.random.default_rng(0).standard_normal((500, 40))   # n = 10 blocks of width b = 4

for v in ("a", "b", "c"):
    res = blockrr.algorithm2(A, b=4, k=5, variant=v)
    print(v, res.selected, res.log_volumes[-1])            # the same 5 blocks every time

res = blockrr.algorithm2(A, b=4, variant="b")                 # all n steps
res.order, res.Q, res.R, res.W, res.V                         # pivots and factors (R is m x n x b)
blockrr.check_algorithm2(blockrr.compress(A), 4, variant="b") # residuals of the paper's identities
```

A tall **A** is first replaced by **R**<sub>0</sub> from **A** = **Q**<sub>0</sub>**R**<sub>0</sub>:
the volumes, the pivots and **R** are unchanged, and `res.Q` is the factor of
**R**<sub>0</sub> (`compress=False` keeps the full *m* × *m* **Q**). The algorithm is
written as stated in the paper, one tensor operation per line, rather than for speed;
`blockrr.algorithm1` gives the same pivots in matrix form.

### Other entry points

| | |
|---|---|
| `blockrr.algorithm1(A, b, k)` | Algorithm 1, block QR with greedy pivoting |
| `blockrr.core` | volumes (`log_vol_S`), `exhaustive` optimum and worst selection, `induced_block_order` |
| `blockrr.starm_qr.starm_qr` | pivoted ⋆<sub>M</sub>-QR of the literature: unpivoted, first-slice, tube-norm; **M** = I, DFT, DCT, Haar |
| `blockrr.column.get(name)` | column-pivoted QR: `businger_golub`, `geqp3`, `hqrrp`, `cqrrpt`, `srrqr`, `rand_srrqr` |
| `blockrr.problems` | Example 2.1, the ⌈k/b⌉-trap family, block Kahan, the block devil's stairs |
| `blockrr.perdigao` | the Perdigão tower data: extraction and the three tower problems |
| `blockrr.sketching` | Gaussian, sparse sign and SRTT sketches |

## Reproducing the experiments

The synthetic problems need nothing else. The Perdigão problems need the NCAR/EOL ISFS
data; [`data/README.md`](data/README.md) says where to get them and how to extract the
`.npz` used below.

```bash
pip install -e ".[experiments,data]"
python -m blockrr.perdigao data/nc data/perdigao_iop.npz --heights 2m 10m
python experiments/run_benchmarks.py --npz data/perdigao_iop.npz   # results/benchmarks.json
python experiments/tiebreaks.py      --npz data/perdigao_iop.npz   # results/tiebreaks.json
python experiments/analysis.py       --npz data/perdigao_iop.npz   # results/analysis.json
python experiments/extras.py         --npz data/perdigao_iop.npz   # results/extras.json
python experiments/tensor_sketch.py                                # results/tensor_sketch.json
python experiments/make_figures.py                                 # figures/*.png and *.pdf
python experiments/make_tables.py                                  # tables/*.tex
python experiments/fig_starm_checks.py                             # figures/fig_starm_checks.png and .pdf
```

On a virtual machine with two 2.1 GHz Xeon cores the sequence takes about 45 minutes:
30 for `run_benchmarks.py` (mostly the column methods on the 8-channel problem), 2 for
`tiebreaks.py`, 10 for `analysis.py` and 2 for `tensor_sketch.py`.

Random seeds are fixed in the scripts. `results/` holds the output of our runs, so the
last three commands work without the data. Block greedy is Algorithm 2: the benchmark runs
all three variants on every problem, checks that they select blocks of the same volume at
every budget, and checks them against Algorithm 1. The Perdigão columns are standardized,
so the first step of every method that ranks columns or blocks by a norm (`dgeqp3`,
Businger–Golub, sRRQR, first-slice ⋆_M-QR with M = I, tube-norm ⋆_M-QR) is an exact tie
that rounding decides; `tiebreaks.py` runs these methods from every possible first pick,
and the paper reports the median, with the best run counted in the best known volume.

| output | in the paper |
|---|---|
| `figures/fig_sharp_limits.png` | loss of block greedy on the trap family and on Example 2.1, with and without the prior |
| `figures/fig_toy_gaps.png`, `tables/losses.tex` | block greedy, the column methods and the pivoted ⋆<sub>M</sub>-QR factorizations (**M** = I, DFT, DCT) on block Kahan, with the loss of the natural order (blocks 1, …, k) dashed, and the devil's stairs (and the Perdigão columns of the table) |
| `figures/fig_perdigao_gaps.png`, `figures/fig_perdigao_map.png` | the same methods on the Perdigão tower problems |
| `figures/fig_bayes_bounds.png`, `tables/bounds.tex` | the lower bounds against the worst case, Bayesian problem |
| `figures/fig_starm_checks.png` | the three instantiations of Algorithm 2 (lettered as in the paper's Table 2) on the 18 × 6 × 3 example of `experiments/tensorqr_notebook.ipynb`: residual of the factorization identity and loss of orthogonality for every tubal column, and the structure of R̃ in (a) and (c) |
| `tables/sketch.tex` | block greedy on a sketch of the Perdigão problems |
| `tables/sketch_trap.tex` | Example 2.1 on Gaussian sketches with 12 to 768 rows (no longer in the paper) |
| `tables/tensor.tex` | Algorithm 2 (a)–(c) and the ⋆<sub>M</sub>-QR methods on the trap family and the devil's stairs, with and without sketching |
| `tables/problems.tex`, `tables/verify.tex`, `tables/rr.tex`, `tables/alg2.tex` | appendix: problems, random verification, rank-revealing factors, Algorithm 2 in floating point |

The tables are LaTeX fragments that use the paper's cross-reference macros
(`\exAlgGreedy`, `\exEqViaVq`, …). The paper includes the figures as 600 dpi PNGs;
each is also written as a vector PDF. They use Latin Modern when a TeX installation
provides it.

## Layout

```
blockrr/
  algorithm2.py      Algorithm 2: variants (a), (b), (c), identity checks
  tensor.py          the ⋆_M product; Algorithm 2 on the folded tensor, as written in the paper
  core.py            volumes, Algorithm 1, enumeration, references
  starm_qr.py        pivoted ⋆_M-QR factorizations of the literature
  column/            Businger–Golub, dgeqp3, HQRRP, CQRRPT, sRRQR, randomized sRRQR
  problems.py        synthetic test problems
  perdigao.py        Perdigão data: extraction and loading
  sketching.py       sketching operators
experiments/         scripts that produce results/, figures/ and tables/
results/ figures/ tables/   outputs of our runs
tests/               pytest suite
data/                where the Perdigão data go (not included)
```

## Citation

```bibtex
@misc{HaddockMurrayScott2026,
  author = {Haddock, Jamie and Murray, Riley and Scott, Mitchell T.},
  title  = {Rank revealers under a column-block constraint: sharp limits,
            an instance-dependent theory, and the $\star_M$ viewpoint},
  year   = {2026},
  note   = {Preprint}
}
```
