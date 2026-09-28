"""
1D TFI encoded simulation with a *miscalibrated* penalty Hamiltonian.

Two things that are easy to get wrong and are deliberate here
-------------------------------------------------------------
* The unit pattern `u` is drawn once per seed from U[-1, 1] and *then* scaled
  by `delta`.  Drawing directly from U[-delta, delta] gives the same
  distribution but a different realisation for each delta; sharing one
  realisation is what makes the lambda*delta collapse exact per
  seed and what makes lambda_opt(delta) extractable with little noise.

* No `qutip.sesolve`.  ||lambda H_pen|| reaches ~1e6, so an adaptive ODE
  solver is both slow and quietly inaccurate.  nb <= 2 uses dense
  `scipy.linalg.eigh` (exact, and the cost does not depend on lambda);
  nb >= 3 uses `scipy.sparse.linalg.expm_multiply`.

RNG draw order per seed (fixed and documented; changing it changes every
number):
    psi0_logical : complex standard normal on 4**nb, normalised  (Haar)
    epsx, epsy, epsz : rng.uniform(-1, 1, n)      x3
    u_x, u_z         : rng.uniform(-1, 1, 2*nb)   x2

Run e.g.:
  .env/bin/python Hpen_miscal/simulate.py \
      --nb 2 --miscal 0 --lamb 32,64,...,16384 --seeds 0-19 --stdout
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from typing import Literal, TextIO, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encoding import (
    X,
    Y,
    Z,
    as_csr,
    build_Henc1,
    build_Henc2,
    build_Htar,
    get_encoder_isometry,
    get_encoder_projector,
    penalty_terms,
)

CArray = npt.NDArray[np.complex128]
RArray = npt.NDArray[np.float64]
# dense ndarray when Model.dense, sparse CSR otherwise
Operator = Union[CArray, sps.csr_matrix]
T = TypeVar("T")


# --------------------------------------------------------------------- setup
class Model:
    """Precomputed ideal operators for a given nb, plus the propagators."""

    def __init__(self, nb: int, dense: bool | None = None) -> None:
        self.nb = nb
        self.n = 4 * nb
        self.nl = 2 * nb
        self.dim = 2 ** self.n
        self.dim_l = 4 ** nb
        # dense is exact and lambda-independent in cost, but 4**(2*nb) memory
        self.dense: bool = (nb <= 2) if dense is None else dense

        self.Uenc = get_encoder_isometry(nb)
        self.Penc = get_encoder_projector(self.Uenc)

        pen = penalty_terms(nb)
        self.pen_kind: list[tuple[Literal["x", "z"], int, float]] = [
            (k, p, g) for k, p, g, _ in pen]
        self.Henc1 = as_csr(build_Henc1(nb))
        self.Henc2 = as_csr(build_Henc2(nb))

        conv: Callable[[sps.csr_matrix], Operator] = (
            (lambda M: np.asarray(M.todense())) if self.dense else (lambda M: M))
        self.pen_ops: list[Operator] = [conv(as_csr(P)) for _, _, _, P in pen]
        self.Henc1_m = conv(self.Henc1)
        self.Henc2_m = conv(self.Henc2)
        self.noise_ops: list[Operator] = [conv(as_csr(O(i, self.n)))
                          for i in range(self.n) for O in (X, Y, Z)]

        # target Hamiltonian: tiny, always dense, diagonalised once
        Htar: CArray = np.asarray(build_Htar(self.nl).full())
        self.tar_w, self.tar_V = sla.eigh(Htar)

    # -- random draws, in the documented order --------------------------
    def draw(self, seed: int, noise: float
             ) -> tuple[CArray, list[RArray], RArray, RArray]:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(self.dim_l) + 1j * rng.standard_normal(self.dim_l)
        psi0 = z / np.linalg.norm(z)
        eps = [noise * rng.uniform(-1.0, 1.0, self.n) for _ in range(3)]
        u_x = rng.uniform(-1.0, 1.0, 2 * self.nb)
        u_z = rng.uniform(-1.0, 1.0, 2 * self.nb)
        return psi0, eps, u_x, u_z

    def noise_operator(self, eps: Sequence[RArray]) -> Operator:
        epsx, epsy, epsz = eps
        coeffs = np.empty(3 * self.n)
        coeffs[0::3], coeffs[1::3], coeffs[2::3] = epsx, epsy, epsz
        return cast(Operator,
                    sum(c * O for c, O in zip(coeffs, self.noise_ops)))

    def Hpen_miscal(self, delta: float, u_x: RArray, u_z: RArray) -> Operator:
        out: Operator | int = 0
        for (kind, p, g), P in zip(self.pen_kind, self.pen_ops):
            u = u_x[p] if kind == "x" else u_z[p]
            out = out + (g * (1.0 + delta * u)) * P
        return cast(Operator, out)

    # -- propagation -----------------------------------------------------
    def evolve_target(self, psi0: CArray, t: float) -> CArray:
        return self.tar_V @ (np.exp(-1j * self.tar_w * t)
                             * (self.tar_V.conj().T @ psi0))

    def evolve_sim(self, Htot: Operator, PSI0: CArray, t: float) -> CArray:
        if self.dense:
            w, V = sla.eigh(Htot)
            return V @ (np.exp(-1j * w * t) * (V.conj().T @ PSI0))
        A = (-1j * t) * sps.csc_matrix(Htot)
        return spla.expm_multiply(A, PSI0)

    # -- one point -------------------------------------------------------
    def run_point(self, lamb: float, t: float, psi0: CArray, Hnoise: Operator,
                  Hpen_mis: Operator) -> tuple[complex, float]:
        Htot: Operator = (lamb * Hpen_mis + self.Henc1_m
                + np.sqrt(lamb) * self.Henc2_m + Hnoise)
        PSI0 = self.Uenc @ psi0
        PSI = self.evolve_sim(Htot, PSI0, t)

        psi_tar = self.evolve_target(psi0, t)
        Uenc_psi = self.Uenc @ psi_tar

        innerprod = complex(np.dot(PSI.conj(), Uenc_psi))
        leakage = 1.0 - float(np.linalg.norm(self.Penc @ PSI)) ** 2
        return innerprod, leakage


# ------------------------------------------------------------------- driver
def parse_list(s: str, conv: Callable[[str], T]) -> list[T]:
    """Comma-separated values; int lists also accept inclusive 'a-b' ranges."""
    out: list[T] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if conv is int and "-" in tok.lstrip("-"):
            a, b = tok.rsplit("-", 1)
            out.extend(conv(str(k)) for k in range(int(a), int(b) + 1))
        else:
            out.append(conv(tok))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nb", type=int, required=True, help="number of blocks")
    ap.add_argument("--noise", type=float, default=0.1, help="1-local noise strength")
    ap.add_argument("--t", type=float, default=1.0, help="evolution time")
    ap.add_argument("--lamb", default=None,
                    help="comma list of penalty coefficients "
                         "(default: 2**0..2**16 for nb<=2, 2**3..2**12 otherwise)")
    ap.add_argument("--miscal", default="0",
                    help="comma list of relative calibration errors delta")
    ap.add_argument("--seeds", default="0-19", help="e.g. '0-49' or '0,1,7'")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--tag", default="",
                    help="extra filename tag, inserted into the output filename")
    ap.add_argument("--dense", dest="dense", action="store_true", default=None)
    ap.add_argument("--sparse", dest="dense", action="store_false")
    ap.add_argument("--stdout", action="store_true",
                    help="print rows instead of writing data files")
    ap.add_argument("--append", action="store_true",
                    help="append to existing output files instead of truncating")
    args = ap.parse_args(argv)

    nb = args.nb
    if args.lamb is None:
        lamb_list: list[int] = ([int(2 ** k) for k in range(17)] if nb <= 2
                     else [int(2 ** k) for k in range(3, 13)])
    else:
        lamb_list = parse_list(args.lamb, int)
    delta_list = parse_list(args.miscal, float)
    seed_list = parse_list(args.seeds, int)

    model = Model(nb, dense=args.dense)
    solver = "dense eigh" if model.dense else "sparse expm_multiply"
    print(f"# nb={nb} n={model.n} dim={model.dim} solver={solver} "
          f"noise={args.noise} t={args.t}", file=sys.stderr)
    print(f"# {len(lamb_list)} lamb x {len(delta_list)} delta x "
          f"{len(seed_list)} seeds = "
          f"{len(lamb_list) * len(delta_list) * len(seed_list)} evolutions",
          file=sys.stderr)

    handles: dict[float, TextIO] = {}
    if not args.stdout:
        os.makedirs(args.outdir, exist_ok=True)
        for delta in delta_list:
            tag = f"_{args.tag}" if args.tag else ""
            fn = (f"{args.outdir}/1dTFIM_sweep_lamb_miscal{tag}_{nb}blocks_"
                  f"noise={args.noise}_miscal={delta}_t={args.t}.txt")
            handles[delta] = open(fn, "a" if args.append else "w")
            print(f"# -> {fn}", file=sys.stderr)

    t0 = time.time()
    ndone = 0
    try:
        for seed in seed_list:
            psi0, eps, u_x, u_z = model.draw(seed, args.noise)
            Hnoise = model.noise_operator(eps)
            for delta in delta_list:
                Hpen_mis = model.Hpen_miscal(delta, u_x, u_z)
                for lamb in lamb_list:
                    ip, leak = model.run_point(lamb, args.t, psi0,
                                               Hnoise, Hpen_mis)
                    row = (f"#blocks = {nb}, noise = {args.noise}, "
                           f"miscal = {delta}, lamb = {lamb}, t = {args.t}, "
                           f"seed = {seed}, innerprod = {ip}, "
                           f"leakage = {leak}")
                    if args.stdout:
                        print(row)
                    else:
                        handles[delta].write(row + "\n")
                        handles[delta].flush()   # so long runs show progress
                    ndone += 1
    finally:
        for h in handles.values():
            h.close()

    dt = time.time() - t0
    print(f"# done: {ndone} evolutions in {dt:.1f}s "
          f"({1e3 * dt / max(ndone, 1):.1f} ms each)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
