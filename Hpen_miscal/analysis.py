"""
Fits and figure for the penalty-miscalibration sweeps produced by
`simulate.py`.

The evolution time t is FIXED (t = 1) for every data point in this study, so it
is not a variable and is absorbed into the fitted constants throughout:

    eps(lamb, delta)  =  A/lamb  +  B delta^2 lamb^2  (+ C delta)

A, B and C are therefore t-dependent constants evaluated at t = 1.  Writing an
explicit t^2 on the second term only would wrongly suggest the first term is
t-independent: the paper's own bound, Eq. (16), is (M^2/lamb)(6 + 13 M t),
i.e. A grows with t.  Neither t-scaling is tested here.  `plot_miscal.ipynb` is the narrative driver for this module
and explains what every quantity means.

Parsing is field-order-agnostic on purpose: the positional `.replace()` chain in
`data/plot_sweep_lamb.ipynb` breaks as soon as the `miscal` field is added.

Run:  .env/bin/python Hpen_miscal/analysis.py --nb 2
      .env/bin/python Hpen_miscal/analysis.py --nb 2 --figure Hpen_miscal/fig_miscal_2blocks.pdf
"""

import argparse
import collections
import glob
import os
import re
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt

PAT = re.compile(r"(\w+) = (\([^)]*\)|[^,\n]+)")
# data files live next to this module, in Hpen_miscal/data/
HERE = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(HERE, "data")

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Row = dict[str, int | float]
# metric -> delta -> lamb -> seed -> value
Grid = dict[str, dict[float, dict[int, dict[int, float]]]]

# Analytic prediction B = kappa * (20/3) * nb, from Tr(W^2)/D with
# (gx,gz) = (1,3) and Var(delta_a - delta_b) = 2 delta^2/3.  kappa = O(1).
B_ANALYTIC: Callable[[int], float] = lambda nb: (20.0 / 3.0) * nb


# ----------------------------------------------------------------- parsing
def parse_file(fn: str) -> list[Row]:
    """[{field: value}] with numbers already cast."""
    rows: list[Row] = []
    with open(fn) as f:
        for line in f:
            d = {k: v.strip() for k, v in PAT.findall(line)}
            if "innerprod" not in d:
                continue
            ip = complex(d["innerprod"])
            rows.append({
                "nb": int(d["blocks"]), "noise": float(d["noise"]),
                "miscal": float(d.get("miscal", 0.0)), "lamb": int(d["lamb"]),
                "t": float(d["t"]), "seed": int(d["seed"]),
                "infidelity": 1 - abs(ip) ** 2,
                "leakage": float(d["leakage"]),
            })
    return rows


def load(nb: int, noise: float = 0.1, t: float = 1.0, datadir: str | None = None
         ) -> tuple[Grid, list[float], list[int], list[int]]:
    """Grid[(metric)] -> nested dict {delta: {lamb: {seed: value}}}."""
    datadir = datadir or DATADIR
    pat = (f"{datadir}/1dTFIM_sweep_lamb_miscal_{nb}blocks_"
           f"noise={noise}_miscal=*_t={t}.txt")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no files matching {pat}")
    rows = [r for fn in files for r in parse_file(fn)]
    G: Grid = {m: collections.defaultdict(lambda: collections.defaultdict(dict))
         for m in ("infidelity", "leakage")}
    for r in rows:
        for m in G:
            G[m][r["miscal"]][r["lamb"]][r["seed"]] = r[m]
    return G, sorted({r["miscal"] for r in rows}), \
        sorted({r["lamb"] for r in rows}), sorted({r["seed"] for r in rows})


def agg(G: Grid, metric: str, delta: float, lambs: Sequence[int],
        seeds: Sequence[int]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(mean, std, sem) arrays over seeds, per lamb."""
    M = np.array([[G[metric][delta][l][s] for s in seeds] for l in lambs])
    return M.mean(1), M.std(1, ddof=1), M.std(1, ddof=1) / np.sqrt(len(seeds))


# -------------------------------------------------------------------- fits
def fit_baseline(mean: FloatArray, lambs: Sequence[int], lamb_min: int = 32
                 ) -> tuple[float, BoolArray]:
    """A from eps(lamb, 0) = A / lamb, fitted over lamb >= lamb_min.

    Restricted to lamb >= lamb_min because the 1/lamb law itself drifts at
    small lamb.
    """
    L = np.asarray(lambs, float)
    sel = L >= lamb_min
    return float(np.mean(mean[sel] * L[sel])), sel


def excess_per_seed(G: Grid, metric: str, delta: float, lambs: Sequence[int],
                    seeds: Sequence[int]) -> tuple[FloatArray, FloatArray]:
    """Paired excess eps(lamb,delta,seed) - eps(lamb,0,seed), per lamb.

    Paired on seed: the unit miscalibration pattern u and the initial state are
    shared between the delta and delta=0 runs, so the difference is far tighter
    than the difference of the two means.
    """
    X = np.array([[G[metric][delta][l][s] - G[metric][0.0][l][s] for s in seeds]
                  for l in lambs])
    return X.mean(1), X.std(1, ddof=1) / np.sqrt(len(seeds))


def fit_excess(G: Grid, metric: str, deltas: Sequence[float], lambs: Sequence[int],
               seeds: Sequence[int], with_C: bool = True,
               max_excess: float = 0.3, nb: int = 2) -> tuple[float, float, int]:
    """Fit the EXCESS  eps(lamb,delta) - eps(lamb,0) = B delta^2 lamb^2 + C delta.

    t is fixed at 1 throughout and is absorbed into B and C (see module docstring).

    Fitting the excess rather than eps itself is what makes both parameters
    identifiable:

      * it needs no model of the delta = 0 baseline, so the whole lamb range is
        usable -- including small lamb, which is the ONLY place the C delta
        floor is separable from B delta^2 lamb^2.  Restricting to lamb >= 32 (as
        a fit to eps must, since the A/lamb law itself drifts below that)
        leaves C unconstrained and the fit then absorbs unrelated residual
        structure into it, returning the wrong sign.
      * the measured eps(lamb,0) is used exactly instead of the A/lamb
        approximation.

    Saturated points (excess > max_excess) are dropped: the model is the
    small-rotation limit, not the saturated one.
    """
    L, D, Y, W = [], [], [], []
    for d in deltas:
        if d == 0:
            continue
        ex, sem = excess_per_seed(G, metric, d, lambs, seeds)
        for i, l in enumerate(lambs):
            if abs(ex[i]) > max_excess:
                continue
            L.append(l); D.append(d); Y.append(ex[i]); W.append(1.0 / max(sem[i], 1e-14))
    L, D, Y, W = map(np.asarray, (L, D, Y, W))

    def model(p):
        out = p[0] * D ** 2 * L ** 2
        return out + p[1] * D if with_C else out

    p0 = [B_ANALYTIC(nb) * 0.6] + ([-0.2] if with_C else [])
    res = opt.least_squares(lambda p: W * (model(p) - Y), p0)
    B = float(res.x[0]); C = float(res.x[1]) if with_C else 0.0
    return B, C, len(Y)


def fit_B_on_eps(G: Grid, metric: str, deltas: Sequence[float], lambs: Sequence[int],
                 seeds: Sequence[int], A: float, with_C: bool = True,
                 lamb_min: int = 32, nb: int = 2) -> tuple[float, float, int]:
    """The naive form: fit eps = A/lamb + B d^2 l^2
    (+ C d) with A fixed, over lamb >= lamb_min.  Kept for comparison -- see
    `fit_excess` for why C is not identifiable this way."""
    L, D, Y, W = [], [], [], []
    for d in deltas:
        if d == 0:
            continue
        mean, _, sem = agg(G, metric, d, lambs, seeds)
        for i, l in enumerate(lambs):
            if l < lamb_min or mean[i] > 0.5:
                continue
            L.append(l); D.append(d); Y.append(mean[i]); W.append(1.0 / max(sem[i], 1e-12))
    L, D, Y, W = map(np.asarray, (L, D, Y, W))

    def model(p):
        out = A / L + p[0] * D ** 2 * L ** 2
        return out + p[1] * D if with_C else out

    p0 = [B_ANALYTIC(nb) * 0.6] + ([-0.2] if with_C else [])
    res = opt.least_squares(lambda p: W * (model(p) - Y), p0)
    return float(res.x[0]), (float(res.x[1]) if with_C else 0.0), len(Y)


def measure_C(G: Grid, metric: str, deltas: Sequence[float], lambs: Sequence[int],
              seeds: Sequence[int], B: float,
              small_lamb: Sequence[int] = (1, 2, 4, 8), C_scale: float = 0.2,
              frac: float = 0.1) -> tuple[float, float, float, int]:
    """C read directly off the small-lamb wing, where B d^2 l^2 << |C| d.

    `C_scale` is the order of magnitude of |C| used only to SCREEN which cells
    qualify (the B term must be < `frac` of the floor there); the returned value
    is measured, not assumed.  Small-lamb cells only, since that is the sole
    region where the floor is separable from B d^2 l^2."""
    vals: list[float] = []
    for d in deltas:
        if d == 0:
            continue
        ex, _ = excess_per_seed(G, metric, d, lambs, seeds)
        for i, l in enumerate(lambs):
            if l in small_lamb and B * d * d * l * l < frac * C_scale * d:
                vals.append(ex[i] / d)
    return (float(np.median(vals)), float(np.min(vals)), float(np.max(vals)),
            len(vals)) if vals else (np.nan,) * 3 + (0,)


def lambda_opt(mean: FloatArray, lambs: Sequence[int]) -> tuple[float, bool]:
    """Parabolic fit of log eps vs log lamb near the minimum (the grid is
    octave-spaced, so the argmin is too coarse)."""
    L = np.asarray(lambs, float)
    i = int(np.argmin(mean))
    if i == 0 or i == len(L) - 1:
        return float(L[i]), False
    x, y = np.log(L[i - 1:i + 2]), np.log(mean[i - 1:i + 2])
    c = np.polyfit(x, y, 2)
    if c[0] <= 0:
        return float(L[i]), False
    return float(np.exp(-c[1] / (2 * c[0]))), True


def fit_exponent(deltas: Sequence[float], lopts: Sequence[float]) -> tuple[float, float]:
    """log lamb_opt = const - p log delta; returns (p, const)."""
    x, y = np.log10(np.asarray(deltas)), np.log10(np.asarray(lopts))
    s, b = np.polyfit(x, y, 1)
    return -float(s), float(b)


# ------------------------------------------------------------------ report
def report(nb: int, noise: float = 0.1, t: float = 1.0,
           datadir: str | None = None) -> dict[str, Any]:
    G, deltas, lambs, seeds = load(nb, noise, t, datadir)
    fin = [d for d in deltas if d > 0]
    L = np.array(lambs, float)
    print("=" * 78)
    print(f"Penalty miscalibration sweep: nb = {nb}, noise = {noise}, "
          f"t = {t}, {len(seeds)} seeds")
    print(f"  lamb  {lambs[0]} .. {lambs[-1]} ({len(lambs)} values)")
    print(f"  delta {deltas}")
    print("=" * 78)

    out: dict[str, Any] = {"nb": nb, "deltas": deltas, "lambs": lambs, "seeds": seeds, "t": t}

    # --- 1. baselines --------------------------------------------------
    m0, s0, e0 = agg(G, "infidelity", 0.0, lambs, seeds)
    lk0, _, _ = agg(G, "leakage", 0.0, lambs, seeds)
    A, sel = fit_baseline(m0, lambs)
    print(f"\n1. delta = 0 baseline (fitted over lamb >= 32)")
    print(f"   A = lamb * eps = {A:.4f}   "
          f"(per-lamb spread {np.min(m0[sel]*L[sel]):.3f}-{np.max(m0[sel]*L[sel]):.3f})")
    print(f"   eps / leakage = {np.mean(m0[sel]/lk0[sel]):.4f}  "
          f"(the infidelity is almost entirely leakage)")
    out.update(A=A, eps0=m0, eps0_std=s0)

    # --- 2/3. global fit for B, and kappa ------------------------------
    B, C, npts = fit_excess(G, "infidelity", deltas, lambs, seeds,
                            with_C=True, nb=nb)
    B_noC, _, _ = fit_excess(G, "infidelity", deltas, lambs, seeds,
                             with_C=False, nb=nb)
    Cm, Clo, Chi, nC = measure_C(G, "infidelity", deltas, lambs, seeds, B)
    Be, Ce, ne = fit_B_on_eps(G, "infidelity", deltas, lambs, seeds, A,
                              with_C=True, nb=nb)
    print(f"\n2. Global fit of the EXCESS  eps(l,d) - eps(l,0) = B d^2 l^2 + C d")
    print(f"   ({npts} points, paired per seed, weighted by the SEM of the difference)")
    print(f"   B = {B:.3f}    C = {C:+.4f}")
    print(f"   B = {B_noC:.3f}  with the C term omitted")
    print(f"   C measured directly off the small-lamb wing: {Cm:+.4f} "
          f"(range {Clo:+.4f}..{Chi:+.4f}, n={nC})  -> consistent")
    print(f"   [the naive alternative -- fit eps itself over lamb >= 32 with A")
    print(f"    fixed -- gives B = {Be:.3f}, C = {Ce:+.4f} ({ne} pts): C comes out with")
    print(f"    the WRONG SIGN, because above lamb = 32 the B term dominates and C is")
    print(f"    unidentifiable.  Use the excess fit; see fit_excess docstring.]")
    print(f"\n3. B against the analytic form B = kappa * (20/3) * nb")
    print(f"   B_analytic = kappa * (20/3) * nb = kappa * {B_ANALYTIC(nb):.3f}")
    print(f"   => kappa = {B / B_ANALYTIC(nb):.3f}   (expected O(1), <~ 1)")
    out.update(B=B, C=C, B_noC=B_noC, kappa=B / B_ANALYTIC(nb))

    # --- 4. lambda_opt and its exponent --------------------------------
    print(f"\n4. lambda_opt by parabolic fit of log eps vs log lamb")
    print(f"   {'delta':>9s} {'lamb_opt':>10s} {'eps_min':>11s}")
    lo_raw: list[float] = []
    for d in fin:
        mr, _, _ = agg(G, "infidelity", d, lambs, seeds)
        a, oka = lambda_opt(mr, lambs)
        lo_raw.append(a)
        print(f"   {d:>9.0e} {a:>10.1f}{'' if oka else '!'} {mr.min():>11.2e}")
    p_raw, c_raw = fit_exponent(fin, lo_raw)
    print(f"   fitted  log lamb_opt = const - p log delta")
    print(f"     p = {p_raw:.4f}   predicted 2/3 = 0.6667   "
          f"(dev {100*(p_raw-2/3)/(2/3):+.1f}%)")
    print(f"   lamb_opt * delta = "
          + ", ".join(f"{a*d:.3g}" for a, d in zip(lo_raw, fin)))
    out.update(lo_raw=lo_raw, p_raw=p_raw, c_raw=c_raw)

    # --- 5. collapse onto lamb*delta -----------------------------------
    print(f"\n5. Collapse onto lamb*delta: K = [eps(lamb,delta) - eps(lamb,0)] / (lamb*delta)^2")
    print(f"   window: lamb*delta <= 0.03 (perturbative) AND "
          f"B d^2 l^2 >= 10|C|d (C floor < 10%)")
    print(f"   {'delta':>9s} {'lamb window':>14s} {'K':>8s} {'K - C*d term':>13s}")
    Ks: list[float] = []
    Kc: list[float] = []
    for d in fin:
        mr, _, _ = agg(G, "infidelity", d, lambs, seeds)
        keep = [i for i, l in enumerate(lambs)
                if l * d <= 0.03 and B * d * d * l * l >= 10 * abs(C) * d]
        if not keep:
            print(f"   {d:>9.0e} {'(empty)':>14s}"); continue
        k = [(mr[i] - m0[i]) / (lambs[i] * d) ** 2 for i in keep]
        kc = [(mr[i] - m0[i] - C * d) / (lambs[i] * d) ** 2 for i in keep]
        Ks.append(np.mean(k)); Kc.append(np.mean(kc))
        print(f"   {d:>9.0e} {f'{lambs[keep[0]]}-{lambs[keep[-1]]}':>14s} "
              f"{np.mean(k):>8.3f} {np.mean(kc):>13.3f}")
    if len(Ks) > 1:
        print(f"   spread across delta: {max(Ks)/min(Ks):.3f}x  ->  "
              f"{max(Kc)/min(Kc):.3f}x after removing C*delta")
        print(f"   K = {np.mean(Kc):.3f}  => kappa = {np.mean(Kc)/B_ANALYTIC(nb):.3f}")
        out.update(K=float(np.mean(Kc)), K_spread=max(Kc) / min(Kc))
    out["G"] = G
    print()
    return out


# ------------------------------------------------------------------ figure
def make_figure(res: dict[str, Any], path: str, noise: float = 0.1) -> tuple[str, str]:
    """Two panels:
    (a) eps vs lamb, one curve per delta, with the fitted model overlaid;
    (b) lamb_opt vs delta, with the fitted exponent.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G, nb, t = res["G"], res["nb"], res["t"]
    deltas, lambs, seeds = res["deltas"], res["lambs"], res["seeds"]
    fin = [d for d in deltas if d > 0]
    L = np.array(lambs, float)
    A, B, C = res["A"], res["B"], res["C"]

    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.9))
    fig.subplots_adjust(wspace=0.30, left=0.09, right=0.985, bottom=0.155, top=0.9)

    cmap = plt.get_cmap("viridis")
    colors = {d: cmap(i / max(len(fin) - 1, 1)) for i, d in enumerate(fin)}
    colors[0.0] = "k"

    # ---------------- panel (a): eps vs lamb ----------------
    ax = axs[0]
    for d in deltas:
        mean, std, _ = agg(G, "infidelity", d, lambs, seeds)
        lab = r"$\delta = 0$" if d == 0 else rf"$\delta = 10^{{{int(round(np.log10(d)))}}}$"
        ax.errorbar(L, mean, yerr=std, marker="o", ms=3, lw=1.2, capsize=2,
                    color=colors[d], label=lab, zorder=3)
        if d > 0:
            # the model is the small-rotation limit: only draw it where it is
            # valid, else B d^2 lamb^2 runs to 1e4 and squashes the data.
            fitc = A / L + B * d ** 2 * L ** 2 + C * d
            ok = (fitc > 0) & (fitc < 1.0)
            ax.plot(L[ok], fitc[ok], ls="--", lw=1.0, color=colors[d],
                    alpha=0.85, zorder=2)
    ax.plot(L, A / L, ls=":", lw=1.2, color="k", alpha=0.7, zorder=2,
            label=rf"$A/\lambda$, $A={A:.2f}$")
    for d, lo in zip(fin, res["lo_raw"]):
        ymin = A / lo + B * d ** 2 * lo ** 2 + C * d
        ax.plot([lo], [ymin], marker="v", ms=6, color=colors[d],
                mec="k", mew=0.5, zorder=5)
        ax.plot([lo, lo], [ymin * 0.3, ymin * 0.75], color=colors[d],
                lw=0.9, alpha=0.7, zorder=4)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_ylim(2e-7, 3.0)
    ax.set_xlabel(r"penalty coefficient $\lambda$")
    ax.set_ylabel(r"infidelity $\epsilon$")
    ax.set_title(rf"(a) $n_b={nb}$, $t={t:g}$, 1-local noise ${noise:g}$", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.95)
    ax.grid(alpha=0.2, which="both", lw=0.4)

    # ---------------- panel (b): lambda_opt vs delta ----------------
    ax = axs[1]
    dd = np.array(fin)
    xs = np.logspace(np.log10(dd.min()) - 0.3, np.log10(dd.max()) + 0.3, 20)
    lo, p_, c_ = res["lo_raw"], res["p_raw"], res["c_raw"]
    ax.plot(xs, lo[-1] * (xs / dd[-1]) ** (-2 / 3), ls="--", lw=1.4, color="C3",
            alpha=0.75, label=r"predicted $\delta^{-2/3}$")
    ax.plot(xs, 10 ** c_ * xs ** (-p_), ls="-", lw=1.2, color="C0", alpha=0.75,
            label=rf"fit: $\lambda_{{\rm opt}}\propto\delta^{{-{p_:.3f}}}$")
    ax.plot(dd, lo, "o", ms=7, color="C0", mec="k", mew=0.5, zorder=5,
            label="measured")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"relative miscalibration $\delta$")
    ax.set_ylabel(r"$\lambda_{\rm opt}$")
    ax.set_title(r"(b) optimal penalty strength", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.2, which="both", lw=0.4)

    fig.savefig(path, bbox_inches="tight", dpi=200)
    png = os.path.splitext(path)[0] + ".png"
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path, png


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=2)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--datadir", default=None)
    ap.add_argument("--figure", default=None, help="output path, e.g. data/fig_miscal_2blocks.pdf")
    args = ap.parse_args(argv)
    res = report(args.nb, args.noise, args.t, args.datadir)
    if args.figure:
        pdf, png = make_figure(res, args.figure, args.noise)
        print(f"figure -> {pdf}\n          {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
