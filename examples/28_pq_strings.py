"""(p, q) strings: type IIB's SL(2,Z), and a junction that balances itself.

Run:  python examples/28_pq_strings.py

Type IIB has a fundamental string and a D1-brane, and they are not different
kinds of object.  A duality group rotates one into the other, so what exists is
a lattice of strings labelled by coprime integers.

Three things come out of machinery that was already here.  The (0,1) tension is
what dp_brane_tension gives for a D1, with no mention of duality in it.  The
whole formula comes again from an M2-brane wrapping a cycle, using only
m2_tension and m_theory_radius.  And the SL(2,Z) that folds the worldsheet
modulus in examples/15 folds the coupling here -- the same function, unchanged.

Writes ``figures/pq_strings.png`` and ``figures/pq_junction.gif``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.amplitudes.oneloop import in_fundamental_domain  # noqa: E402
from stringsim.branes.dbrane import dp_brane_tension  # noqa: E402
from stringsim.branes.mtheory import m2_tension, m_theory_radius  # noqa: E402
from stringsim.branes.pq import (  # noqa: E402
    Junction,
    axio_dilaton,
    binding_energy,
    coupling_from_tau,
    duality_residual,
    einstein_tension,
    is_bound_state,
    junction_angles,
    junction_residual,
    membrane_residual,
    membrane_tension,
    reduce_coupling,
    tension,
    transform_charges,
)
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.animate import animate_pq_junction  # noqa: E402
from stringsim.viz.plots import plot_pq_strings  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
S = ((0, -1), (1, 0))
T = ((1, 1), (0, 1))
CONV = Conventions()


def the_multiplet() -> None:
    """One lattice of strings, and its two familiar corners."""
    print("The SL(2,Z) multiplet")
    print("-" * 70)
    print("  tau = C_0 + i/g_s, and T_{p,q} = |p + q tau| / 2 pi alpha'.")
    print()
    print("      g_s     T(1,0)        1/2 pi a'      T(0,1)        D1 tension")
    for g_s in (0.15, 0.35, 1.0, 2.4):
        tau = axio_dilaton(g_s)
        print(f"    {g_s:5.2f}   {tension(1, 0, tau, CONV):.8f}   "
              f"{1.0 / (2 * math.pi * CONV.alpha_prime):.8f}   "
              f"{tension(0, 1, tau, CONV):.8f}   {dp_brane_tension(1, g_s, CONV):.8f}")
    print("  The second column is the fundamental string and the fourth is what")
    print("  dp_brane_tension returns for a D1.  Neither knows about duality.")
    print()
    print("  S exchanges them; T shifts the axion and gives a D1 one unit of")
    print("  fundamental charge -- the Witten effect:")
    for label, matrix in (("S", S), ("T", T)):
        for p, q in ((1, 0), (0, 1)):
            print(f"    {label} on ({p},{q}) -> {transform_charges(matrix, p, q)}")
    print()


def the_invariant() -> None:
    """What the duality actually preserves."""
    print("What is invariant")
    print("-" * 70)
    tau = axio_dilaton(0.35, 0.4)
    print(f"  tau = {tau:.4f}")
    print("  The string-frame tension is not invariant, and should not be: a")
    print("  duality changes which string is called fundamental.  The Einstein-")
    print("  frame one, |p + q tau| / sqrt(Im tau), is.")
    print()
    print("      element        (p,q)     residual")
    for name, matrix in (("S", S), ("T", T), ("[[2,1],[1,1]]", ((2, 1), (1, 1)))):
        for p, q in ((1, 0), (0, 1), (2, -3)):
            print(f"    {name:<14s} ({p:2d},{q:2d})    {duality_residual(matrix, p, q, tau):.1e}")
    print()


def from_eleven_dimensions() -> None:
    """The formula, derived rather than written down."""
    print("The same tension from eleven dimensions")
    print("-" * 70)
    print("  Type IIB on a circle is M-theory on a torus, and a (p,q) string is")
    print("  an M2-brane wrapping the (p,q) cycle.  A cycle of a torus with")
    print("  modulus tau and side L has length L |p + q tau|, so the wrapped")
    print("  membrane has tension T_M2 L |p + q tau|.")
    print()
    print("  Matching the (1,0) string to the fundamental one forces the side:")
    for g_s in (0.2, 0.7, 1.5):
        forced = 1.0 / (2 * math.pi * CONV.alpha_prime * m2_tension(g_s, CONV))
        circle = 2 * math.pi * m_theory_radius(g_s, CONV)
        print(f"    g_s = {g_s:4.1f}:  L = {forced:.10f}   2 pi R_11 = {circle:.10f}")
    print("  The side is the circumference of the M-theory circle.  Nothing was")
    print("  fitted; l_p^3 = g_s alpha'^{3/2} does it.")
    print()
    print("      g_s   (p,q)     M2 route        IIB formula      residual")
    for g_s in (0.2, 0.7, 1.5):
        tau = axio_dilaton(g_s)
        for p, q in ((1, 0), (0, 1), (3, 2)):
            print(f"    {g_s:5.2f}  ({p},{q})   {membrane_tension(p, q, tau, None, CONV):.10f}   "
                  f"{tension(p, q, tau, CONV):.10f}   {membrane_residual(p, q, tau, CONV):.1e}")
    print()


def bound_states() -> None:
    """Which charges are one string."""
    print("Which charges are a single string")
    print("-" * 70)
    tau = axio_dilaton(0.35, 0.4)
    print("      (p,q)   gcd = 1   T_{p,q}      binding energy")
    for p, q in ((1, 0), (0, 1), (1, 1), (2, 3), (2, 2), (3, 6)):
        print(f"    ({p},{q})   {is_bound_state(p, q)!s:<7s}   "
              f"{tension(p, q, tau, CONV):.6f}   {binding_energy(p, q, tau, CONV):+.6f}")
    print("  A coprime (p,q) is lighter than the p fundamental strings and q")
    print("  D1-branes it is made of -- the triangle inequality is the binding.")
    print("  A repeated charge is that many copies of a lighter string sitting")
    print("  exactly at threshold, and binds by nothing.")
    print()


def the_junction() -> list:
    """Equilibrium is charge conservation."""
    print("A junction")
    print("-" * 70)
    print("  A BPS (p,q) string leaves along the phase of p + q tau, with force")
    print("  |p + q tau|.  So the total force is sum(p_i) + tau sum(q_i), and it")
    print("  vanishes exactly when the charges do.  No angle is ever chosen.")
    print()
    tau = axio_dilaton(0.4, 0.25)
    print("      charges                        total    net force   angles (deg)")
    for charges in (
        ((1, 0), (0, 1), (-1, -1)),
        ((2, 1), (-1, 1), (-1, -2)),
        ((1, 0), (1, 1), (0, 1), (-2, -2)),
        ((1, 0), (0, 1), (-1, 0)),
    ):
        junction = Junction(charges=charges, tau=tau)
        angles = [round(float(a), 1) for a in np.degrees(junction_angles(junction))]
        listed = " ".join(f"({p},{q})" for p, q in charges)
        print(f"    {listed:<30s} {str(junction.total_charge):<8s} "
              f"{junction_residual(junction, CONV):.1e}   {angles}")
    print("  The last one does not conserve charge and does not balance.")
    print()

    charges = ((2, 1), (-1, 1), (-1, -2))
    frames = []
    for axion in np.linspace(-0.9, 0.9, 40):
        for coupling in (0.55,):
            moving = axio_dilaton(coupling, float(axion))
            junction = Junction(charges=charges, tau=moving)
            frames.append((moving, charges, junction_residual(junction, CONV)))
    for coupling in np.linspace(0.55, 2.4, 30):
        moving = axio_dilaton(float(coupling), 0.9)
        junction = Junction(charges=charges, tau=moving)
        frames.append((moving, charges, junction_residual(junction, CONV)))
    return frames


def folding_the_coupling() -> None:
    """The same group, the same function, a different tau."""
    print("Folding the coupling")
    print("-" * 70)
    print("  examples/15 walks the worldsheet modulus into the fundamental")
    print("  domain to show the string has no ultraviolet region.  The same")
    print("  function, unchanged, walks the coupling into it -- and there it")
    print("  says a strongly-coupled vacuum is a weakly-coupled one with the")
    print("  strings relabelled.")
    print()
    print("      g_s    C_0        reduced tau          g_s'    matrix")
    for g_s, axion in ((8.0, 0.0), (0.05, 2.3), (1.7, -3.4), (4.0, 0.5)):
        tau = axio_dilaton(g_s, axion)
        domain = reduce_coupling(tau)
        print(f"    {g_s:5.2f} {axion:6.2f}   {domain.tau.real:+.4f}{domain.tau.imag:+.4f}i   "
              f"{coupling_from_tau(domain.tau):6.3f}   {domain.matrix.tolist()}   "
              f"{'in domain' if in_fundamental_domain(domain.tau) else 'NOT'}")
    print()
    tau = axio_dilaton(8.0)
    domain = reduce_coupling(tau)
    print(f"  At g_s = 8 the strings are relabelled by {domain.matrix.tolist()}:")
    print("      (p,q) -> (p',q')     tension there    tension here")
    for p, q in ((1, 0), (0, 1), (2, -3)):
        moved = transform_charges(domain.matrix, p, q)
        print(f"    ({p:2d},{q:2d}) -> {str(moved):<10s}   "
              f"{einstein_tension(p, q, domain.tau):.8f}   "
              f"{einstein_tension(*moved, tau):.8f}")
    print()


def figures(frames: list) -> None:
    print("Figures")
    print("-" * 70)
    couplings = np.logspace(-1.0, 1.0, 60)
    curves = []
    for p, q in ((1, 0), (0, 1), (1, 1), (2, 1)):
        values = [tension(p, q, axio_dilaton(float(g)), CONV) for g in couplings]
        curves.append((f"$({p},{q})$", couplings, values))
    junctions = [
        ("$g_s = 0.4$, $C_0 = 0.25$", ((2, 1), (-1, 1), (-1, -2)), axio_dilaton(0.4, 0.25)),
        ("$g_s = 1.6$, $C_0 = -0.5$", ((2, 1), (-1, 1), (-1, -2)), axio_dilaton(1.6, -0.5)),
    ]
    plot_pq_strings(curves, junctions, FIG / "pq_strings.png")
    print(f"  wrote {FIG / 'pq_strings.png'}")
    animate_pq_junction(frames, FIG / "pq_junction.gif")
    print(f"  wrote {FIG / 'pq_junction.gif'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_multiplet()
    the_invariant()
    from_eleven_dimensions()
    bound_states()
    frames = the_junction()
    folding_the_coupling()
    figures(frames)
    print("done.")
