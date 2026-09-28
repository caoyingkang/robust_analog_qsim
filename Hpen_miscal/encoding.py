"""
Operators and encoder isometry for the [[4n, 2n, 2]] Hamiltonian code,
reimplemented in QuTiP/SciPy (port of `../src/simulation_encoded_1dTFIM.py`,
which used dynamite).
"""

from __future__ import annotations

import functools
from typing import Literal

import numpy as np
import numpy.typing as npt
import qutip as qt
import scipy.sparse as sps

__all__ = [
    "GX", "GZ", "X", "Y", "Z",
    "as_csr",
    "build_Henc1", "build_Henc2", "build_Hpen", "build_Hpen_miscal",
    "build_Htar",
    "get_encoder_isometry", "get_encoder_projector",
    "ket",
    "logical_X", "logical_Z",
    "penalty_terms",
]

# penalty-Hamiltonian couplings, (g_x, g_z)
GX, GZ = 1.0, 3.0

_I2 = qt.qeye(2)
_PAULI = {"x": qt.sigmax(), "y": qt.sigmay(), "z": qt.sigmaz()}


def _single(which: Literal["x", "y", "z"], k: int, n: int) -> qt.Qobj:
    """Pauli `which` on qubit `k` of an `n`-qubit register (MSB-first)."""
    if not 0 <= k < n:
        raise IndexError(f"qubit {k} out of range for n = {n}")
    return qt.tensor([_PAULI[which] if j == k else _I2 for j in range(n)])


def X(k: int, n: int) -> qt.Qobj:
    return _single("x", k, n)


def Y(k: int, n: int) -> qt.Qobj:
    return _single("y", k, n)


def Z(k: int, n: int) -> qt.Qobj:
    return _single("z", k, n)


def ket(bits: str) -> qt.Qobj:
    """Computational basis ket from a bit string read left-to-right as qubits
    0, 1, 2, ..."""
    return qt.tensor([qt.basis(2, int(b)) for b in bits])


def as_csr(obj: qt.Qobj | sps.spmatrix) -> sps.csr_matrix:
    """scipy CSR view of a Qobj (or passthrough for a scipy sparse matrix)."""
    if isinstance(obj, qt.Qobj):
        return obj.to("CSR").data_as("csr_matrix")
    return sps.csr_matrix(obj)


# ----------------------------------------------------------------- logical ops
def logical_X(k: int, nl: int) -> qt.Qobj:
    return X(k, nl)


def logical_Z(k: int, nl: int) -> qt.Qobj:
    return Z(k, nl)


# -------------------------------------------------------------------- encoder
def _U0() -> sps.csc_matrix:
    """Single-block encoder isometry: 16 x 4, columns ordered
    (v00, v01, v10, v11) for MSB-first logical indexing.
    """
    r2 = np.sqrt(0.5)
    v00 = r2 * (ket("0001") - ket("1110"))
    v01 = r2 * (ket("1101") - ket("0010"))
    v10 = r2 * (ket("0100") - ket("1011"))
    v11 = r2 * (ket("1000") - ket("0111"))
    cols = np.column_stack([v.full().ravel() for v in (v00, v01, v10, v11)])
    return sps.csc_matrix(cols)


def get_encoder_isometry(nb: int) -> sps.csc_matrix:
    """Encoder isometry Uenc: (2**(4*nb)) x (4**nb), scipy CSC."""
    return sps.csc_matrix(functools.reduce(sps.kron, [_U0()] * nb))


def get_encoder_projector(Uenc: sps.spmatrix) -> sps.csr_matrix:
    """Penc = Uenc Uenc^dagger."""
    return sps.csr_matrix(Uenc @ Uenc.conj().T)


# ---------------------------------------------------------------- Hamiltonians
def penalty_terms(
    nb: int,
) -> list[tuple[Literal["x", "z"], int, float, qt.Qobj]]:
    """The 4*nb individual penalty terms.

    Returns a list of (kind, pair_index, coupling, Qobj) with kind in
    {'x', 'z'}; pair `p` covers physical qubits (2p, 2p+1) and block `i` owns
    pairs p = 2i and p = 2i+1.
    """
    n = 4 * nb
    terms: list[tuple[Literal["x", "z"], int, float, qt.Qobj]] = []
    for p in range(2 * nb):
        terms.append(("x", p, GX, X(2 * p, n) * X(2 * p + 1, n)))
        terms.append(("z", p, GZ, Z(2 * p, n) * Z(2 * p + 1, n)))
    return terms


def build_Hpen(nb: int) -> qt.Qobj:
    """Ideal penalty Hamiltonian: sum_p (gx X_2p X_2p+1 + gz Z_2p Z_2p+1)."""
    return sum(g * P for _, _, g, P in penalty_terms(nb))


def build_Hpen_miscal(nb: int, delta: float, u_x: npt.ArrayLike,
                     u_z: npt.ArrayLike) -> qt.Qobj:
    """Miscalibrated penalty Hamiltonian g_a -> g_a (1 + delta * u_a).

    `u_x`, `u_z` are unit miscalibration patterns of length 2*nb drawn from
    U[-1, 1]; scaling by `delta` HERE, rather than drawing directly from
    U[-delta, delta], is deliberate.  Both give the same distribution, but this
    way one realisation is shared across every delta and lambda, which makes
    the delta-dependence of each seed smooth and the lambda*delta collapse
    exact per seed instead of only in the mean.
    """
    u_x = np.asarray(u_x, dtype=float)
    u_z = np.asarray(u_z, dtype=float)
    if u_x.shape != (2 * nb,) or u_z.shape != (2 * nb,):
        raise ValueError(f"u_x, u_z must have shape ({2 * nb},)")
    out = 0
    for kind, p, g, P in penalty_terms(nb):
        u = u_x[p] if kind == "x" else u_z[p]
        out = out + g * (1.0 + delta * u) * P
    return out


def build_Henc1(nb: int) -> qt.Qobj:
    """First-order encoding Hamiltonian (ideal); verbatim port."""
    n = 4 * nb
    H = sum(X(4 * i, n) * X(4 * i + 1, n) - X(4 * i, n) * X(4 * i + 2, n)
            for i in range(nb))                                   # logical X
    H += sum(Z(4 * i, n) * Z(4 * i + 1, n) + Z(4 * i, n) * Z(4 * i + 2, n)
             for i in range(nb))                                  # logical Z
    H += sum(Z(4 * i + 1, n) * Z(4 * i + 2, n)
             for i in range(nb))                                  # inner-block ZZ
    H += sum(Z(4 * i + 4, n) * Z(4 * i + 5, n)
             for i in range(nb - 1))   # cancels cross-block gadget residuals
    return H


def build_Henc2(nb: int) -> qt.Qobj:
    """Perturbative-gadget encoding Hamiltonian (ideal); verbatim port."""
    n = 4 * nb
    if nb < 2:
        return qt.qzero([2] * n, [2] * n)
    return sum(np.sqrt(8.0 / 3.0) * (Z(4 * i + 1, n) * X(4 * i + 6, n)
                                     + Z(4 * i + 3, n) * X(4 * i + 6, n))
               for i in range(nb - 1))


def build_Htar(nl: int) -> qt.Qobj:
    """1D TFI target Hamiltonian on `nl` logical sites, open boundary:
    sum_k X_k + sum_k Z_k + sum_k Z_k Z_{k+1}."""
    H = sum(X(k, nl) for k in range(nl))
    H += sum(Z(k, nl) for k in range(nl))
    H += sum(Z(k, nl) * Z(k + 1, nl) for k in range(nl - 1))
    return H
