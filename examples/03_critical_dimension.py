"""Why 26, and how hot a string can get.

Run:  python examples/03_critical_dimension.py

Extracts ``zeta(-1) = -1/12`` numerically from a cut-off sum, solves for the
critical dimension by both the normal-ordering and the central-charge route,
checks Jacobi's abstruse identity in exact integers, verifies the modular
transformation of the Dedekind eta function, and measures the Hagedorn
temperature from the degeneracies.  Writes ``figures/hagedorn.png``.
"""

from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.quantum.partition import (  # noqa: E402
    dedekind_eta,
    fit_hagedorn,
    jacobi_identity_residual,
    oscillator_degeneracies,
    superstring_degeneracies,
)
from stringsim.quantum.zeta import (  # noqa: E402
    central_charge,
    critical_dimension,
    normal_ordering_constant,
    regularised_sum,
)
from stringsim.viz.plots import plot_degeneracy_growth  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"


def zero_point_energy() -> None:
    print("=" * 72)
    print("The zero-point energy: sum of all the integers")
    print("=" * 72)
    est = regularised_sum()
    print(f"  sum_n n e^{{-eps n}} - 1/eps^2  ->  {est.value:.12f}")
    print(f"  zeta(-1)                        =  {est.exact:.12f}")
    print(f"  error                           =  {est.error:.2e}")
    print("  The 1/eps^2 is the cutoff-dependent divergence a counterterm removes;")
    print("  what is left is universal, and it is what makes a = (D-2)/24.")
    print()


def dimension() -> None:
    print("=" * 72)
    print("The critical dimension, two independent ways")
    print("=" * 72)
    for theory, target in (("bosonic", 1.0), ("superstring", 0.5)):
        d = critical_dimension(theory)
        print(
            f"  {theory:<12s}: a(D) = {normal_ordering_constant(d, theory):.4f} "
            f"= {target}  at  D = {d};  central charge there = {central_charge(d, theory):+.1f}"
        )
    print("  a = 1 is forced by Lorentz invariance: the level-1 states are a vector")
    print("  of SO(D-2), and only a massless vector has that few polarisations.")
    print()


def modularity() -> None:
    print("=" * 72)
    print("Modular checks")
    print("=" * 72)
    residual = jacobi_identity_residual(60)
    worst = max(abs(x) for x in residual)
    print(f"  max |coefficient of theta_3^4 - theta_2^4 - theta_4^4| over 60 orders: {worst}")
    print("  Exactly zero, in integer arithmetic.  Level by level this says the")
    print("  GSO-projected NS sector and the R sector have equally many states:")
    boson = superstring_degeneracies(6)
    print(f"    superstring degeneracies at alpha'M^2 = 0..6:  {boson}")
    print("    the same numbers count bosons and fermions, so the one-loop vacuum")
    print("    amplitude vanishes -- the first fingerprint of spacetime supersymmetry.")
    print()
    tau = 0.31 + 0.87j
    lhs = dedekind_eta(-1.0 / tau)
    rhs = cmath.sqrt(-1j * tau) * dedekind_eta(tau)
    print(f"  eta(-1/tau) - sqrt(-i tau) eta(tau) = {abs(lhs - rhs):.2e}   at tau = {tau}")
    print("  Modular invariance of the torus is what makes the one-loop amplitude finite.")
    print()


def hagedorn() -> None:
    print("=" * 72)
    print("Hagedorn temperature, measured from the degeneracies")
    print("=" * 72)
    n_max = 400
    degen = oscillator_degeneracies(n_max, 24)
    fit = fit_hagedorn(n_max=n_max, n_species=24, n_fit=150)
    print(f"  d_N for N = 0..8:  {degen[:9]}")
    print(f"  d_{n_max} has {len(str(degen[n_max]))} digits")
    print(f"  {fit}")
    print(f"  predicted beta_H = 4 pi = {4 * math.pi:.6f}")
    print(f"  measured / predicted = {fit.beta_hagedorn / fit.predicted_beta:.5f}")
    print("  Above T_H the canonical partition function diverges: the string cannot")
    print("  be heated further, because the number of states outruns the Boltzmann factor.")
    plot_degeneracy_growth(fit, degen, FIG / "hagedorn.png")
    print(f"  wrote {FIG / 'hagedorn.png'}\n")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    zero_point_energy()
    dimension()
    modularity()
    hagedorn()
    print("done.")
