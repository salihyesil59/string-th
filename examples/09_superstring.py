"""The RNS superstring: two sectors, the GSO projection, and type IIA/IIB.

Run:  python examples/09_superstring.py

Derives both sector intercepts from cut-off mode sums, counts the GSO-projected
states by explicit enumeration and checks them against the theta-function
product already in the package, verifies spacetime supersymmetry level by level,
and lays out the massless spectra of type IIA and IIB together with the branes
each theory admits.  Writes ``figures/supersymmetry.png``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.branes.dbrane import dp_brane_tension  # noqa: E402
from stringsim.quantum.partition import (  # noqa: E402
    hagedorn_temperature,
    jacobi_identity_residual,
    oscillator_degeneracies,
    superstring_degeneracies,
)
from stringsim.quantum.zeta import central_charge, critical_dimension  # noqa: E402
from stringsim.superstring.rns import (  # noqa: E402
    Sector,
    critical_dimension_from_intercept,
    fermion_mode_numbers,
    hagedorn_beta,
    intercept,
    ns_series_by_parity,
    open_superstring_levels,
    sector_degeneracies,
    supersymmetry_deficit,
    unprojected_ns_ground_state,
)
from stringsim.superstring.typeii import (  # noqa: E402
    massless_counts,
    open_superstring_massless,
    rr_form_ranks,
    stable_brane_ranks,
    type_ii_massless,
)
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.plots import plot_supersymmetry  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
TEN = Conventions(alpha_prime=1.0, dim=10)


def two_sectors() -> None:
    print("=" * 72)
    print("Two sectors, because a worldsheet fermion may be periodic or not")
    print("=" * 72)
    for sector in (Sector.NS, Sector.R):
        modes = fermion_mode_numbers(sector, 4)
        print(f"  {sector.value:<3s} modes {[str(m) for m in modes]}   "
              f"periodic: {sector.is_periodic}")
    print("  The zero in the Ramond list is the whole story: psi_0^mu obey a Clifford")
    print("  algebra, so the R ground state is a spinor.  That is where spacetime")
    print("  fermions come from.")
    print()
    print("  Ground-state energies, from cut-off mode sums (fermions with the opposite sign):")
    for sector in (Sector.NS, Sector.R):
        print(
            f"    a_{sector.value:<3s} measured {intercept(sector, TEN, exact=False):+.8f}   "
            f"closed form {intercept(sector, TEN):+.6f}"
        )
    print("  So alpha'M^2 = N - 1/2 in NS and N in R: the Ramond ground state is massless.")
    print(f"  The NS ground state is a tachyon at alpha'M^2 = "
          f"{unprojected_ns_ground_state(TEN)}, and GSO is what removes it.")
    print()


def dimension() -> None:
    print("=" * 72)
    print("D = 10, three ways")
    print("=" * 72)
    print(f"  solving a_NS = 1/2:                {critical_dimension_from_intercept()}")
    print(f"  the zeta module's normal ordering: {critical_dimension('superstring')}")
    anomaly = central_charge(10, "superstring")
    print(f"  conformal anomaly c = 3D/2 - 15:   {anomaly:+.1f} at D = 10")
    print("  a_NS = (D-2)/16, and Lorentz invariance forces it to 1/2 for the same")
    print("  reason as in the bosonic string: a massive vector cannot be built from")
    print("  only D-2 oscillators.")
    print()


def gso() -> None:
    print("=" * 72)
    print("The GSO projection, by explicit enumeration")
    print("=" * 72)
    even, odd = ns_series_by_parity(8)
    print("  NS oscillator states split by worldsheet fermion parity (u = q^{1/2}):")
    print(f"    {'u^k':>4s} {'even F':>8s} {'odd F':>8s}")
    for k in range(6):
        print(f"    {k:>4d} {even[k]:>8d} {odd[k]:>8d}")
    print("    u^0 even: the tachyonic ground state, F = 0 -- projected out.")
    print("    u^1 odd:  b_{-1/2}^i |0>, eight states, F = 1 -- the massless vector.")
    print()
    mine = sector_degeneracies(Sector.NS, 6)
    theirs = superstring_degeneracies(6)
    print(f"  NS by enumeration:        {mine}")
    print(f"  by the theta-function product: {theirs}")
    print(f"  identical: {mine == theirs}")
    print("  Two unrelated routes: multiplying out oscillator products and tracking")
    print("  parity, versus a ratio of theta functions.")
    print()


def supersymmetry() -> None:
    print("=" * 72)
    print("Spacetime supersymmetry, counted")
    print("=" * 72)
    for level in open_superstring_levels(5, TEN):
        print("  ", level)
    print(f"  NS - R at each level: {supersymmetry_deficit(9)}")
    print(f"  Jacobi's theta_3^4 - theta_2^4 - theta_4^4 over 40 orders: "
          f"{max(abs(c) for c in jacobi_identity_residual(40))}")
    print("  The same statement twice: once by counting states, once as an identity")
    print("  between theta functions.  The massless level is 8 + 8, which is")
    print("  ten-dimensional N = 1 super Yang-Mills:")
    for field in open_superstring_massless():
        print(f"    {field}")
    print()


def type_ii() -> None:
    print("=" * 72)
    print("Type IIA and IIB: 256 massless states each")
    print("=" * 72)
    for kind in ("IIB", "IIA"):
        print(f"  --- type {kind} ---")
        for sector in type_ii_massless(kind):
            print(sector)
        bosons, fermions = massless_counts(kind)
        print(f"    totals: {bosons} bosons + {fermions} fermions = {bosons + fermions}, "
              f"supersymmetric: {bosons == fermions}")
        print()
    print("  Only the R-R sector differs.  IIB takes the same chirality on both sides")
    print("  and gets even-rank forms; IIA takes opposite chiralities and gets odd.")
    print()


def branes() -> None:
    print("=" * 72)
    print("Which D-branes each theory has, and what they weigh")
    print("=" * 72)
    for kind in ("IIA", "IIB"):
        ranks = stable_brane_ranks(kind)
        print(f"  {kind}: RR potentials C_{rr_form_ranks(kind)}  ->  Dp for p = {ranks}")
        tensions = {
            p: f"{dp_brane_tension(p, 0.1, TEN):.6g}" for p in ranks if p >= 0
        }
        print(f"        tensions at g_s = 0.1: {tensions}")
    print("  A Dp-brane couples to C_{p+1}, whose dual is C_{7-p}, the potential of a")
    print("  D(6-p)-brane.  Closing under that gives even p for IIA and odd for IIB.")
    print("  The p = -1 D-instanton has an action rather than a tension, so the branes")
    print("  module refuses it; the IIA D8 needs massive (Romans) IIA and is not here.")
    print()


def thermodynamics() -> None:
    print("=" * 72)
    print("The superstring runs hotter")
    print("=" * 72)
    print(f"  bosonic     beta_H = 4 pi          = {hagedorn_temperature(24, 1.0):.6f}")
    print(f"  superstring beta_H = 2 pi sqrt(2)  = {hagedorn_beta():.6f}")
    print(f"  ratio {hagedorn_beta() / hagedorn_temperature(24, 1.0):.4f}, "
          f"so T_H is higher by {hagedorn_temperature(24, 1.0) / hagedorn_beta():.4f}x")
    print("  Eight bosons and eight fermions grow like twelve bosons would, not")
    print("  twenty-four: a fermionic mode contributes half as much to the entropy.")
    print(f"  sqrt(2) check: {math.sqrt(2):.6f}")
    print()


def figures() -> None:
    levels = open_superstring_levels(7, TEN)
    # The slope reaches its asymptote only like 1/sqrt(N), so go well past the bars.
    plot_supersymmetry(
        levels,
        oscillator_degeneracies(80, 24),
        sector_degeneracies(Sector.NS, 80),
        FIG / "supersymmetry.png",
    )
    print(f"  wrote {FIG / 'supersymmetry.png'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    two_sectors()
    dimension()
    gso()
    supersymmetry()
    type_ii()
    branes()
    thermodynamics()
    figures()
    print("done.")
