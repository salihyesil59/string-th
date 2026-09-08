"""Type I: the fifth superstring, and SO(32) for the third time.

Run:  python examples/23_type_i_orientifold.py

The package had IIA, IIB and both heterotic strings.  This is the missing one,
and it is not a new worldsheet: it is type IIB with worldsheet parity gauged,
plus the D9-branes that gauging it forces you to add.

Two numbers come out that were computed elsewhere by unrelated routes.  The
gauge group is SO(32) -- from Chan-Paton factors and an anomaly, with no lattice
anywhere.  And the massless spectrum is 8064 states split 128 + 7936, which is
what the heterotic root lattice gives.

Writes ``figures/orientifold_spectrum.png``, ``figures/worldsheet_surfaces.png``
and ``figures/worldsheet_parity.gif``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.classical.modes import ClosedString  # noqa: E402
from stringsim.heterotic.anomaly import required_dimension  # noqa: E402
from stringsim.heterotic.lattice import heterotic_lattices  # noqa: E402
from stringsim.heterotic.spectrum import massless_content  # noqa: E402
from stringsim.superstring.orientifold import (  # noqa: E402
    brane_count,
    chan_paton_dimension,
    closed_counts,
    gauge_group,
    massless_total,
    one_loop_surfaces,
    open_massless,
    parity_even,
    parity_image,
    parity_residual,
    project_closed,
    split_by_exchange,
    string_coupling_order,
    supersymmetric_sign,
    surfaces_for,
    tadpole_structure,
)
from stringsim.superstring.typeii import massless_counts, type_ii_massless  # noqa: E402
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.animate import animate_worldsheet_parity  # noqa: E402
from stringsim.viz.plots import (  # noqa: E402
    plot_orientifold_spectrum,
    plot_worldsheet_surfaces,
)

FIG = Path(__file__).resolve().parents[1] / "figures"

SIDES = {
    "torus": {"left": "up", "right": "up", "top": "right", "bottom": "right"},
    "Klein bottle": {"left": "up", "right": "up", "top": "right", "bottom": "left"},
    "cylinder": {"left": "up", "right": "up", "top": "boundary", "bottom": "boundary"},
    "Mobius strip": {"left": "up", "right": "down", "top": "boundary", "bottom": "boundary"},
}


def the_projection() -> None:
    """Half of type IIB, and which half."""
    print("Gauging worldsheet parity")
    print("-" * 68)
    print(f"  type IIB massless level: {sum(massless_counts('IIB'))} states")
    print()
    print("  Omega swaps the two chiralities, so each product splits into a")
    print("  symmetric and an antisymmetric half.  Which irreducible pieces go")
    print("  where is found from the dimensions, not assigned:")
    for sector in ("NS-NS", "R-R"):
        fields = {s.name: s for s in type_ii_massless("IIB")}[sector].fields
        sym, anti = split_by_exchange(fields)
        print(f"    {sector:<6s} symmetric  {sum(f.dimension for f in sym):3d} = "
              f"{' + '.join(str(f.dimension) for f in sym)}")
        print(f"    {sector:<6s} antisym.   {sum(f.dimension for f in anti):3d} = "
              f"{' + '.join(str(f.dimension) for f in anti)}")
    print()
    print("  The Ramond-Ramond sign is usually quoted as part of the definition.")
    print("  It cannot be anything else:")
    for sign in (+1, -1):
        bosons, fermions = closed_counts(sign)
        verdict = "a supermultiplet" if bosons == fermions else "not a supermultiplet"
        print(f"    rr_sign = {sign:+d}:  {bosons} bosons, {fermions} fermions -- {verdict}")
    print(f"  supersymmetric_sign() = {supersymmetric_sign():+d}, chosen by counting.")
    print()
    print("  What survives:")
    for sector in project_closed():
        print(sector)
    print("  That is exactly the N = 1 supergravity multiplet of ten dimensions.")
    print()


def the_branes() -> None:
    """How many D9-branes, and what they carry."""
    print("The open sector")
    print("-" * 68)
    print("  An unoriented closed string alone has a Ramond-Ramond tadpole.")
    print("  Cancelling it needs D9-branes, and Omega acts on their Chan-Paton")
    print("  factors as lambda -> +- gamma lambda^T gamma^-1.")
    print()
    print("      n     SO(n) = n(n-1)/2     Sp(n/2) = n(n+1)/2")
    for n in (16, 30, 32, 34):
        print(f"    {n:3d}   {chan_paton_dimension(n, 'SO'):14d}   "
              f"{chan_paton_dimension(n, 'SP'):19d}")
    print()
    print(f"  The anomaly polynomial demands dim G = {required_dimension()}")
    print("  (examples/22).  Solving n(n-1)/2 for it:")
    print(f"    brane_count() = {brane_count()},  gauge group {gauge_group(brane_count())}")
    print("  n(n+1)/2 = 496 has no integer root, so the symplectic projection")
    print("  is not an option.")
    print()
    fields = open_massless(brane_count())
    for f in fields:
        print(f"    {f}")
    print()
    report = tadpole_structure(brane_count())
    print(f"  Tadpole structure: coefficients {report['coefficients']}, "
          f"discriminant {report['discriminant']}, double root {report['root']}.")
    print("  The square is structural -- it is one state's norm.  The crosscap")
    print("  charge -32 is an input here, not a result; the three one-loop")
    print("  amplitudes that would give it are not built.  That it agrees with")
    print("  the anomaly count is why it appears at all.")
    print()


def the_comparison() -> None:
    """Type I against the heterotic SO(32) string."""
    print("Two constructions, one spectrum")
    print("-" * 68)
    report = massless_total()
    heterotic = massless_content(heterotic_lattices()["Spin(32)/Z2"])
    print("                        type I        heterotic SO(32)")
    print(f"    gauge group      {report['gauge_group']:>10s}        "
          f"{heterotic.gauge_algebra:>10s}")
    print(f"    dim G            {report['gauge_dimension']:>10d}        "
          f"{heterotic.gauge_dimension:>10d}")
    print(f"    supergravity     {report['closed']:>10d}        "
          f"{heterotic.supergravity_states:>10d}")
    print(f"    gauge            {report['open']:>10d}        {heterotic.gauge_states:>10d}")
    print(f"    total            {report['total']:>10d}        {heterotic.total:>10d}")
    print()
    print("  One side gauges a worldsheet symmetry and counts Chan-Paton")
    print("  factors; the other enumerates the roots of an even self-dual")
    print("  lattice.  They share no step.  The agreement is the massless")
    print("  shadow of the strong-weak duality between the two theories.")
    print()


def the_surfaces() -> None:
    """Why exactly four diagrams at one loop."""
    print("Worldsheets")
    print("-" * 68)
    print("  chi = 2 - 2g - b - c, and one loop is chi = 0.  Enumerating:")
    for surface in one_loop_surfaces():
        print(f"    {surface}")
    print()
    for theory in ("closed oriented", "closed unoriented", "open unoriented"):
        names = ", ".join(s.name for s in surfaces_for(theory))
        print(f"    {theory:<20s} {names}")
    print()
    print("  Each carries g_s^-chi, so at g_s = 0.1 the sphere is worth")
    print(f"  {string_coupling_order(one_loop_surfaces(euler=2)[0], 0.1):.0f} and every one-loop "
          "surface exactly 1.")
    print()


def parity_on_a_moving_string() -> tuple[list, int]:
    """The same projection, watched instead of counted."""
    print("The same projection, on a string that is moving")
    print("-" * 68)
    conv = Conventions(dim=4)
    travelling = ClosedString(
        conventions=conv,
        # A right-moving circular wave in mode 1 against a left-moving one in
        # mode 2, with equal levels so the state is level-matched.  Different
        # modes on the two sides keep the curve two-dimensional; equal ones
        # would collapse it to a segment at every instant.
        alphas={1: np.array([0.0, 1.0, 1.0j, 0.0])},
        alphas_tilde={2: np.array([0.0, 1.0, -1.0j, 0.0])},
    )
    image = parity_image(travelling)
    even = parity_even(travelling)

    tau = np.linspace(0.0, 4.0, 5)[:, None]
    sigma = np.linspace(0.0, travelling.sigma_max, 33)[None, :]
    swapped = image.position(tau, sigma)
    reflected = travelling.position(tau, -sigma)
    print(f"  swapping the chiralities equals sigma -> -sigma to "
          f"{np.max(np.abs(swapped - reflected)):.1e}")
    print()
    print(f"  levels (N, N~): travelling {travelling.levels()}, "
          f"even {even.levels()} -- matched either way")
    print()
    print("      solution                 |X(sigma) - X(-sigma)| / size")
    for label, string in (("travelling", travelling), ("its Omega image", image),
                          ("the Omega-even part", even)):
        print(f"    {label:<24s} {parity_residual(string):.3e}")
    print("  The invariant one is a standing wave: an unoriented string cannot")
    print("  carry a wave that goes round it.  Counting states and watching one")
    print("  move are the same statement.")
    print()

    n_frames = 90
    taus = np.linspace(0.0, 2.0 * math.pi, n_frames)[:, None]
    sigmas = np.linspace(0.0, travelling.sigma_max, 120)[None, :]
    panels = [
        (label, string.position(taus, sigmas)[:, :, 1:3])
        for label, string in (
            ("travelling", travelling),
            ("its $\\Omega$ image", image),
            ("$\\Omega$-even: folded", even),
        )
    ]
    return panels, n_frames


def figures(panels: list) -> None:
    print("Figures")
    print("-" * 68)

    sectors = {s.name: s for s in type_ii_massless("IIB")}
    kept = {s.name.split(" (")[0]: s.dimension for s in project_closed()}
    before = [
        ("NS-NS", kept["NS-NS"], sectors["NS-NS"].dimension - kept["NS-NS"]),
        ("R-R", kept["R-R"], sectors["R-R"].dimension - kept["R-R"]),
        ("NS-R + R-NS", kept["NS-R + R-NS"],
         sectors["NS-R"].dimension + sectors["R-NS"].dimension - kept["NS-R + R-NS"]),
    ]
    report = massless_total()
    heterotic = massless_content(heterotic_lattices()["Spin(32)/Z2"])
    comparison = [
        ("type I", report["closed"], report["open"]),
        ("heterotic SO(32)", heterotic.supergravity_states, heterotic.gauge_states),
    ]
    plot_orientifold_spectrum(before, comparison, FIG / "orientifold_spectrum.png")
    print(f"  wrote {FIG / 'orientifold_spectrum.png'}")

    order = ["torus", "Klein bottle", "cylinder", "Mobius strip"]
    lookup = {s.name: s for s in one_loop_surfaces()}
    plot_worldsheet_surfaces(
        [(name, lookup[name].euler, SIDES[name]) for name in order],
        FIG / "worldsheet_surfaces.png",
    )
    print(f"  wrote {FIG / 'worldsheet_surfaces.png'}")

    animate_worldsheet_parity(panels, FIG / "worldsheet_parity.gif")
    print(f"  wrote {FIG / 'worldsheet_parity.gif'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_projection()
    the_branes()
    the_comparison()
    the_surfaces()
    panels, _ = parity_on_a_moving_string()
    figures(panels)
    print("done.")
