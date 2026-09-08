"""A one-screen summary of what the package computes.

``python -m stringsim`` prints the open and closed bosonic spectra with their
particle content, the critical dimension, the Hagedorn temperature and the
Regge trajectory -- every number produced by the modules rather than quoted.

``python -m stringsim --figures DIR`` additionally writes the figures.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .amplitudes.veneziano import veneziano_pole_positions
from .branes.dbrane import dp_brane_tension, gauge_group
from .classical.rotating import regge_trajectory
from .compactification.circle import extra_massless_states, self_dual_radius
from .quantum.partition import fit_hagedorn
from .quantum.spectrum import closed_bosonic_spectrum, open_bosonic_spectrum
from .quantum.states import closed_massless_content, open_level_content
from .quantum.virasoro import central_charge_from_algebra, critical_dimension_from_norms
from .quantum.zeta import central_charge, critical_dimension, regularised_sum
from .units import Conventions

__all__ = ["main", "summary"]


def summary(conventions: Conventions | None = None) -> str:
    """Return the summary as a string, so it can be tested without printing."""
    conv = conventions or Conventions()
    out: list[str] = []
    add = out.append

    add(f"stringsim -- bosonic string, alpha' = {conv.alpha_prime}, D = {conv.dim}")
    add(f"  string length sqrt(alpha') = {conv.string_length:.4f}, "
        f"tension 1/(2 pi alpha') = {conv.tension:.6f}")
    add("")

    add("Critical dimension")
    for theory in ("bosonic", "superstring"):
        d = critical_dimension(theory)
        add(f"  {theory:<12s} D = {d:>3d}   central charge {central_charge(d, theory):+.1f}")
    est = regularised_sum()
    add(f"  zeta(-1) measured {est.value:.10f} against {est.exact:.10f}")
    measured_c = central_charge_from_algebra(conv.dim, m=2, level=0)
    add(f"  c read off [L_2, L_-2] in D = {conv.dim}: {measured_c:.6f}")
    add(f"  largest ghost-free D from the physical norms at level 2: "
        f"{critical_dimension_from_norms(level=2)}")
    add("")

    add("Open bosonic spectrum, alpha' M^2 = N - 1")
    for lv in open_bosonic_spectrum(3, conv):
        try:
            content = ", ".join(
                f"{p.name} ({p.dimension})" for p in open_level_content(lv.n, conv.dim)
            )
        except NotImplementedError:
            content = ""
        add(f"  {lv}   {content}")
    add("")

    add("Closed bosonic spectrum, alpha' M^2 = 4(N - 1)")
    for lv in closed_bosonic_spectrum(3, conv):
        add(f"  {lv}")
    add("  massless level: " + ", ".join(
        f"{p.name} ({p.dimension})" for p in closed_massless_content(conv.dim)
    ))
    add("")

    add("Regge trajectory of the classical rotating string")
    for pt in regge_trajectory(conventions=conv)[:3]:
        add(f"  A = {pt.amplitude:.1f}:  alpha' M^2 = {pt.alpha_m2:.6f},  J = {pt.spin:.6f}")
    add("  quantised, the same trajectory reads J = alpha' M^2 + 1")
    add("")

    fit = fit_hagedorn(n_max=200, n_species=conv.transverse_dim, n_fit=80)
    add("Hagedorn temperature")
    add(f"  {fit}")
    add(f"  4 pi sqrt(alpha') = {4 * math.pi * conv.string_length:.6f}")
    add("")

    r0 = self_dual_radius(conv)
    extra = extra_massless_states(r0, conv, n_max=3, w_max=3, level_max=2)
    add("Compactification on a circle")
    add(f"  self-dual radius sqrt(alpha') = {r0:.4f}, "
        f"{len(extra)} extra massless states there")
    add("")

    add("D-branes")
    add(f"  T_3 at g_s = 0.1: {dp_brane_tension(3, 0.1, conv):.6g}")
    add(f"  three coincident branes: {gauge_group([0, 0, 0])};  "
        f"one pulled away: {gauge_group([0, 0, 2.5])}")
    add("")

    add("Veneziano amplitude")
    add(f"  poles at alpha' s = {veneziano_pole_positions(4, conv.alpha_prime)}")
    add("  identical to the open-string mass levels above")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stringsim", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--alpha-prime", type=float, default=1.0, help="Regge slope (default 1)"
    )
    parser.add_argument("--dim", type=int, default=26, help="spacetime dimension (default 26)")
    parser.add_argument(
        "--figures",
        type=Path,
        default=None,
        metavar="DIR",
        help="also write the figures from the examples into DIR",
    )
    args = parser.parse_args(argv)

    conv = Conventions(alpha_prime=args.alpha_prime, dim=args.dim)
    print(summary(conv))

    if args.figures is not None:
        import matplotlib

        matplotlib.use("Agg")
        from .quantum.partition import oscillator_degeneracies
        from .viz.plots import plot_degeneracy_growth, plot_mass_spectrum, plot_regge_trajectory

        args.figures.mkdir(parents=True, exist_ok=True)
        written = [
            plot_mass_spectrum(open_bosonic_spectrum(6, conv), args.figures / "open_spectrum.png"),
            plot_regge_trajectory(
                regge_trajectory(conventions=conv), args.figures / "regge_trajectory.png"
            ),
            plot_degeneracy_growth(
                fit_hagedorn(n_max=200, n_species=conv.transverse_dim, n_fit=80),
                oscillator_degeneracies(200, conv.transverse_dim),
                args.figures / "hagedorn.png",
            ),
        ]
        print("\nwrote:")
        for path in written:
            print(f"  {path}")
    return 0
