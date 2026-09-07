"""The Myers effect: N D0-branes that turn into a sphere.

Run:  python examples/16_myers_effect.py

``examples/05_dbranes.py`` puts N branes on top of each other and gets ``U(N)``.
``examples/14`` gives one brane an action.  This one does both at once: with N
branes the transverse positions are matrices, and in a background flux the
matrices refuse to commute.

Writes ``figures/fuzzy_sphere.png``, ``figures/myers_landscape.png`` and
``figures/fuzzy_sphere.gif``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.branes.dbrane import dp_brane_tension  # noqa: E402
from stringsim.branes.myers import (  # noqa: E402
    algebra_residual,
    block_configuration,
    configuration_energies,
    fuzzy_radius,
    fuzzy_sphere,
    large_n_ratio,
    latitudes,
    myers_gradient,
    myers_potential,
    noncommutativity,
    partitions,
    shrunk_d2_energy,
    spherical_d2_energy,
    su2_generators,
    trace_j_squared,
)
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.animate import animate_fuzzy_sphere  # noqa: E402
from stringsim.viz.plots import plot_fuzzy_sphere, plot_myers_landscape  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions()
FLUX = 1.7
G_S = 0.3


def the_matrices_refuse_to_commute() -> None:
    print("=" * 72)
    print("Positions become matrices, and the flux will not let them commute")
    print("=" * 72)
    print("  V = Tr( -1/4 [Phi_i,Phi_j][Phi_i,Phi_j] + (i f/3) eps_ijk Phi_i Phi_j Phi_k )")
    print()
    print("  The quartic term wants [Phi_i, Phi_j] = 0 -- separated branes, at V = 0.")
    print("  The cubic term does not, and an SU(2) representation is what wins:")
    print()
    print(f"  {'N':>3s} {'algebra residual':>17s} {'EOM residual':>13s} {'V':>12s}"
          f" {'-f^4 Tr(J^2)/96':>16s}")
    for dim in (2, 3, 5, 9, 14):
        matrices = fuzzy_sphere(dim, FLUX)
        print(f"  {dim:>3d} {algebra_residual(su2_generators(dim)):>17.1e}"
              f" {np.max(np.abs(myers_gradient(matrices, FLUX))):>13.1e}"
              f" {myers_potential(matrices, FLUX):>12.6f}"
              f" {-FLUX**4 * trace_j_squared(dim) / 96.0:>16.6f}")
    print()
    print("  The EOM residual is on the full matrix equations, not on the equations")
    print("  restricted to the ansatz: Phi_i = (f/2) J_i really is a solution.")
    print("  Move away from it and it stops being one:")
    off = fuzzy_sphere(5, FLUX).copy()
    off[0] = off[0] * 1.3
    residual = np.max(np.abs(myers_gradient(off, FLUX)))
    print(f"    scaling one matrix by 1.3: EOM residual {residual:.3f}")
    print()


def one_block_beats_every_split() -> None:
    print("=" * 72)
    print("Which representation, and why")
    print("=" * 72)
    print("  V is proportional to -Tr(J^2), and a split into blocks of sizes N_a")
    print("  gives sum N_a (N_a^2 - 1)/4.  So the question is arithmetic:")
    print()
    for total in (4, 6, 8):
        found = configuration_energies(total, FLUX)
        print(f"  N = {total}   ({len(partitions(total))} partitions)")
        for entry in found[:3]:
            print(f"    {entry}")
        print(f"    {'...':>22s}")
        print(f"    {found[-1]}")
        assert found[0].is_irreducible
        print()
    print("  The single block is always at the bottom, and the configuration one")
    print("  would have called the vacuum -- N commuting matrices, N separated")
    print("  branes -- is at the top, at exactly zero.")
    print()


def it_is_a_sphere() -> None:
    print("=" * 72)
    print("And what the single block is")
    print("=" * 72)
    print(f"  {'N':>4s} {'R':>10s} {'(f/4) sqrt(N^2-1)':>19s} {'R/N':>8s}"
          f" {'|[Phi,Phi]|/R^2':>16s} {'latitudes':>10s}")
    for dim in (2, 5, 12, 30, 80):
        matrices = fuzzy_sphere(dim, FLUX)
        heights, _ = latitudes(matrices)
        print(f"  {dim:>4d} {fuzzy_radius(matrices):>10.5f}"
              f" {FLUX / 4.0 * math.sqrt(dim**2 - 1):>19.5f}"
              f" {fuzzy_radius(matrices) / dim:>8.5f}"
              f" {noncommutativity(matrices):>16.5f} {len(heights):>10d}")
    print()
    print(f"  R/N tends to f/4 = {FLUX / 4.0:.5f}, and the relative size of the")
    print("  commutators falls like 1/N.  A noncommutative sphere becomes an")
    print("  ordinary one, and N point-like branes have become one surface.")
    print()


def the_other_description() -> None:
    print("=" * 72)
    print("The same object, from the other side")
    print("=" * 72)
    print("  A D2-brane wrapping a sphere of radius R with N units of flux has")
    print("    E(R) = 4 pi T_2 sqrt(R^4 + pi^2 alpha'^2 N^2).")
    print("  Shrink it to nothing and see what it weighs:")
    print()
    print(f"  {'N':>4s} {'E(R=0)':>16s} {'N T_0':>16s} {'relative':>10s}")
    for dim in (1, 5, 20, 100):
        shrunk = shrunk_d2_energy(dim, G_S, CONV)
        expected = dim * dp_brane_tension(0, G_S, CONV)
        print(f"  {dim:>4d} {shrunk:>16.10f} {expected:>16.10f}"
              f" {abs(shrunk / expected - 1.0):>10.1e}")
    print()
    print("  Nothing was fitted: 4 pi^2 alpha' T_2 = T_0 comes straight out of the")
    print("  tension formula.  A shrunk D2-brane with N flux quanta IS N D0-branes,")
    print("  which is why the matrix picture and the sphere picture are of one")
    print("  object seen from opposite ends.")
    print()
    print("  Where they differ is the interesting part.  The continuum knows only")
    print("  N^3; the matrices give N(N^2 - 1):")
    print()
    print(f"  {'N':>5s} {'V_matrix / V_continuum':>24s} {'1 - 1/N^2':>12s}")
    for dim in (2, 5, 20, 100):
        print(f"  {dim:>5d} {large_n_ratio(dim):>24.10f} {1.0 - 1.0 / dim**2:>12.10f}")
    print()
    print("  Exact, not fitted.  It is the price of building a sphere out of N")
    print("  points, and it is the leading correction the D2-brane cannot see.")
    print()
    print("  The Born-Infeld energy at finite radius, for reference:")
    for radius in (0.0, 0.5, 1.0):
        print(f"    N = 20, R = {radius}: {spherical_d2_energy(radius, 20, G_S, CONV):.8f}")
    print()


def figures() -> None:
    panels = [
        (rf"$N = {dim}$,  $R = {fuzzy_radius(fuzzy_sphere(dim, FLUX)):.2f}$",
         *latitudes(fuzzy_sphere(dim, FLUX)))
        for dim in (3, 8, 24)
    ]
    plot_fuzzy_sphere(panels, FIG / "fuzzy_sphere.png")
    print(f"  wrote {FIG / 'fuzzy_sphere.png'}")

    plot_myers_landscape(configuration_energies(10, FLUX), FIG / "myers_landscape.png")
    print(f"  wrote {FIG / 'myers_landscape.png'}")

    frames = []
    for dim in (2, 3, 4, 6, 9, 14, 22, 34):
        matrices = fuzzy_sphere(dim, FLUX)
        heights, radii = latitudes(matrices)
        frames.append(
            (
                rf"$N = {dim}$,  $R = {fuzzy_radius(matrices):.2f}$,  "
                rf"$|[\Phi,\Phi]|/R^2 = {noncommutativity(matrices):.3f}$",
                heights,
                radii,
            )
        )
    animate_fuzzy_sphere(frames, FIG / "fuzzy_sphere.gif")
    print(f"  wrote {FIG / 'fuzzy_sphere.gif'}")
    print("    The circles multiply and the surface fills in.  Both are the same")
    print("    statement: the commutators shrink relative to the radius.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_matrices_refuse_to_commute()
    one_block_beats_every_split()
    it_is_a_sphere()
    the_other_description()
    figures()
    print("  (a check with nothing in it to adjust: block_configuration([N]) and")
    print("   fuzzy_sphere(N) are the same matrices, to "
          f"{np.max(np.abs(block_configuration([7], FLUX) - fuzzy_sphere(7, FLUX))):.0e})")
    print("done.")
