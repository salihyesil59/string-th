"""Heterotic strings: two theories, and why there are exactly two.

Run:  python examples/10_heterotic.py

Constructs both sixteen-dimensional even self-dual lattices, verifies evenness
and unimodularity from an explicit basis, shows that counting alone cannot tell
``e8 + e8`` from ``so(32)`` while the root geometry can, and works out the
massless spectrum -- including the fact that level matching alone removes the
tachyon, before GSO is applied.  Writes ``figures/heterotic_roots.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.compactification.torus import identify_algebra  # noqa: E402
from stringsim.heterotic.lattice import (  # noqa: E402
    GAUGE_DIMENSION,
    RootLattice,
    _d_roots,
    d16_plus,
    e8,
    even_self_dual_dimension_rule,
    heterotic_lattices,
)
from stringsim.heterotic.spectrum import (  # noqa: E402
    anomaly_free_dimension,
    central_charges,
    has_tachyon,
    internal_dimension,
    left_mass,
    level_matched_masses,
    massless_content,
    right_mass,
)
from stringsim.quantum.zeta import central_charge, critical_dimension  # noqa: E402
from stringsim.superstring.rns import Sector  # noqa: E402
from stringsim.viz.plots import plot_root_connectivity  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"


def heterosis() -> None:
    print("=" * 72)
    print("Heterosis: two different theories, one string")
    print("=" * 72)
    left, right = central_charges()
    print(
        f"  left-movers  bosonic string,     D = {critical_dimension('bosonic')}, "
        f"c_L = {left:.0f}"
    )
    print(f"  right-movers superstring,        D = {critical_dimension('superstring')}, "
          f"c_R = {right:.0f}")
    print(f"  each side's anomaly cancels against its own ghosts: "
          f"{central_charge(26, 'bosonic'):+.0f} and {central_charge(10, 'superstring'):+.0f}")
    print(f"  so {critical_dimension('bosonic')} - {critical_dimension('superstring')} = "
          f"{internal_dimension()} left-moving directions have nowhere to go.")
    print()
    allowed = [d for d in range(33) if even_self_dual_dimension_rule(d)]
    print(f"  Even self-dual lattices exist only in dimensions {allowed}.")
    print("  16 is on that list, which is the whole reason a heterotic string exists:")
    print("  had the two critical dimensions differed by anything else, there would")
    print("  be no lattice to put the leftover directions on.")
    print()


def lattices() -> None:
    print("=" * 72)
    print("The two lattices, checked rather than quoted")
    print("=" * 72)
    print(f"  E8 alone: {e8().n_roots} roots, covolume {e8().covolume():.10f}, "
          f"algebra {e8().algebra}, dim {e8().algebra_dimension}")
    print()
    print(f"  {'lattice':<14s} {'dim':>4s} {'roots':>6s} {'covol':>8s} {'even':>6s} "
          f"{'self-dual':>10s} {'dim G':>6s}")
    for name, lattice in heterotic_lattices().items():
        print(
            f"  {name:<14s} {lattice.dim:>4d} {lattice.n_roots:>6d} "
            f"{lattice.covolume():>8.5f} {str(lattice.is_even()):>6s} "
            f"{str(lattice.is_unimodular()):>10s} {lattice.algebra_dimension:>6d}"
        )
    print()
    print("  D16 without the spinor class is even but NOT self-dual:")
    bare = RootLattice("D16", _d_roots(16))
    print(f"    covolume {bare.covolume():.5f}, even {bare.is_even()}, "
          f"unimodular {bare.is_unimodular()}")
    spinor = d16_plus().extra[0]
    print(f"    the extra generator (1/2,...,1/2) has norm {float(spinor @ spinor):.0f}, so it is")
    print("    not a root and adds no gauge boson -- it only halves the covolume.")
    print("    That coset is why the group is Spin(32)/Z_2 and not SO(32).")
    print()


def telling_them_apart() -> None:
    print("=" * 72)
    print("Counting cannot separate them; geometry can")
    print("=" * 72)
    print(f"  identify_algebra(16, 480) = {identify_algebra(16, 480)}")
    print("  Rank 16 with 480 roots is genuinely ambiguous, and the code says so")
    print("  rather than picking one.  The root systems differ, though:")
    for name, lattice in heterotic_lattices().items():
        print(f"    {name:<14s} components {lattice.components}  ->  {lattice.algebra}")
    print("  Two orthogonal families of 240 versus one connected system of 480.")
    print()


def level_matching() -> None:
    print("=" * 72)
    print("Level matching removes the tachyon, before GSO is applied")
    print("=" * 72)
    print(f"  left  side alpha'M^2/4 = N_L + p^2/2 - 1, always an integer "
          f"(lowest {left_mass(0, 0)})")
    print(f"  right side alpha'M^2/4 = N_R - a_R: NS starts at "
          f"{right_mass(0, Sector.NS)}, R at {right_mass(0, Sector.R)}")
    print("  An integer cannot equal -1/2, so the NS ground state pairs with nothing;")
    print("  and nothing on the right reaches -1, so the left tachyon pairs with nothing.")
    print()
    for gso in (True, False):
        states = level_matched_masses(3, 6, gso=gso)
        print(f"  GSO={str(gso):<5s}: {len(states):>3d} matched combinations, "
              f"lightest alpha'M^2 = {float(states[0].alpha_m2):.1f}, "
              f"tachyon present: {has_tachyon(gso=gso)}")
    print("  The switch changes nothing, which is the point.")
    print()
    print("  The massless level is reached two ways:")
    for state in level_matched_masses(2, 4):
        if state.is_massless:
            print(f"    {state}")
    print()


def spectrum() -> None:
    print("=" * 72)
    print("The massless spectrum")
    print("=" * 72)
    for name, lattice in heterotic_lattices().items():
        content = massless_content(lattice)
        print(f"  {name}:")
        print(f"    left  {content.left_oscillator_states} oscillators "
              f"+ {content.left_lattice_states} roots = {content.left_states}")
        print(f"    right {content.right_states} (8v from NS, 8s from R)")
        print(f"    total {content.left_states} x {content.right_states} = {content.total}")
        print(f"    = {content.supergravity_states} supergravity "
              f"+ {content.gauge_states} gauge")
        for field in content.supergravity:
            print(f"        {field}")
        print(f"    gauge algebra {content.gauge_algebra}, dimension "
              f"{content.gauge_dimension}")
        print()
    print(f"  Both give dim G = {GAUGE_DIMENSION}, which is also exactly what "
          f"Green-Schwarz")
    print(f"  anomaly cancellation requires ({anomaly_free_dimension()}).  The lattice knows")
    print("  nothing about anomalies; two independent consistency conditions agreeing")
    print("  on 496 is why the heterotic string was taken seriously.")
    print()


def figures() -> None:
    """Draw the one thing that actually separates the two lattices."""
    lattices = heterotic_lattices()
    plot_root_connectivity(
        [
            (f"$E_8 \oplus E_8$: components {lattices['E8 + E8'].components}",
             lattices["E8 + E8"].roots),
            (f"$D_{{16}}^{{+}}$: components {lattices['Spin(32)/Z2'].components}",
             lattices["Spin(32)/Z2"].roots),
        ],
        FIG / "heterotic_roots.png",
        title="480 roots each, rank 16 each -- only the connectivity tells them apart",
    )
    print(f"  wrote {FIG / 'heterotic_roots.png'}")
    print("  A projection to two dimensions would show only a shapeless cloud; what")
    print("  distinguishes the theories is which roots are non-orthogonal to which.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    heterosis()
    lattices()
    telling_them_apart()
    level_matching()
    spectrum()
    figures()
    print("done.")
