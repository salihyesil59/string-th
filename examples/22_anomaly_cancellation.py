"""Green-Schwarz: where 496 comes from, and why only two groups.

Run:  python examples/22_anomaly_cancellation.py

``examples/10`` counts 496 gauge bosons out of a lattice and says, as everyone
does, that anomaly cancellation separately demands that number.  This computes
the second half.  The anomaly polynomial is assembled from index densities, the
conventions are fixed by making type IIB cancel, and then 496 and the two
groups come out.

Writes ``figures/anomaly_conditions.png``, ``figures/anomaly_scan.png`` and
``figures/anomaly_sweep.gif``.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.heterotic.anomaly import (  # noqa: E402
    anomaly_polynomial,
    cancels,
    candidates,
    combine,
    e8,
    factorise,
    gravitational_coefficient,
    required_dimension,
    scan,
    sixth_order_terms,
    so,
    type_iib_residual,
)
from stringsim.heterotic.lattice import GAUGE_DIMENSION, heterotic_lattices  # noqa: E402
from stringsim.viz.animate import animate_anomaly_sweep  # noqa: E402
from stringsim.viz.plots import plot_anomaly_conditions, plot_anomaly_scan  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"

def label(monomial) -> str:
    """``(('R2', 2), ('R4', 1))`` -> ``(trR^2)^2 trR^4``.

    Plain ASCII on purpose: this goes to the console as well as to an axis, and
    the console here is not UTF-8.
    """
    pieces = []
    for name, power in monomial:
        head, _, index = name.partition("_")
        tag = "R" if head[0] == "R" else f"F{int(index) + 1}"
        base = f"tr{tag}^{head[-1]}"
        pieces.append(base if power == 1 else f"({base})^{power}")
    return " ".join(pieces) or "1"


def calibration() -> None:
    """Fix the conventions on a theory whose answer is already known."""
    print("Calibration: type IIB")
    print("-" * 66)
    print("  Two gravitini, two dilatini of the opposite chirality, and one")
    print("  self-dual four-form.  The total anomaly must vanish -- three")
    print("  twelve-form coefficients at once, with nothing to tune.")
    wrong = type_iib_residual(half_argument=True)
    print(f"    L = prod x/tanh x                : residual {type_iib_residual()}")
    print(f"    L = prod (x/2)/tanh(x/2)         : residual {wrong}")
    print("  The second is the version that is easy to write from memory.")
    print("  It does not cancel, so the choice is settled by computation.")
    print()


def four_hundred_and_ninety_six() -> None:
    """The pure-gravity condition."""
    print("The gravitational condition")
    print("-" * 66)
    print("  Coefficient of tr R^6, against the gauge group's dimension:")
    for n in (0, 248, 480, 496, 512, 744):
        c = gravitational_coefficient(n)
        print(f"    dim G = {n:4d}:  {str(c):>16s}   = ({n} - 496)/725760")
    print()
    print("  No gauge field appears in a pure-gravity term, so nothing can")
    print("  cancel this.  Solving it:")
    print(f"    required_dimension()          = {required_dimension()}")
    print(f"    lattice GAUGE_DIMENSION       = {GAUGE_DIMENSION}")
    for name, lattice in heterotic_lattices().items():
        print(f"    {name:<14s} roots + Cartan  = {lattice.algebra_dimension}")
    print("  The lattice number is 16 Cartan directions plus 480 roots and")
    print("  knows nothing about anomalies.  Nothing connects them but 496.")
    print()


def the_group() -> None:
    """The pure-gauge condition, which the dimension does not fix."""
    print("The gauge condition")
    print("-" * 66)
    print("  Tr F^6 = (N - 32) tr F^6 + 15 tr F^2 tr F^4 for SO(N), and no")
    print("  product of a four-form and an eight-form can produce tr F^6.")
    print("      N    dim SO(N)   coefficient of tr F^6")
    for n in (16, 28, 30, 31, 32, 33, 36):
        leftover = sixth_order_terms(anomaly_polynomial([so(n)])).coefficient(("F6_0", 1))
        mark = "   <-- vanishes" if leftover == 0 else ""
        print(f"    {n:3d}   {n * (n - 1) // 2:8d}   {str(leftover):>10s}{mark}")
    print()
    print("  SO(32) has dimension 496.  The two conditions come from different")
    print("  halves of the anomaly and land on the same group.")
    print()


def factorisation() -> None:
    """What is left has to be X_4 X_8."""
    print("Factorisation")
    print("-" * 66)
    for name, factors in (("SO(32)", [so(32)]), ("E8 x E8", [e8(0), e8(1)])):
        poly = anomaly_polynomial(factors)
        result = factorise(poly, factors)
        show = lambda form: " + ".join(  # noqa: E731
            f"{c} {label(m)}" for m, c in sorted(form.items())
        )
        print(f"  {name}:")
        print(f"    X_4 = {show(result.x_four)}")
        print(f"    X_8 = {show(result.x_eight)}")
        print(f"    residual of the division: {result.residual}")
        print(f"    b per factor: {', '.join(f'{k} -> {v}' for k, v in result.coefficients)}")
    print("  Both give 1/30 -- the textbook tr R^2 - (1/30) Tr F^2.  It is read")
    print("  off one coefficient; every other one is then a prediction, and the")
    print("  exact division is what checks them.")
    print()


def the_negative_test() -> None:
    """A group with the right dimension that is still anomalous."""
    print("Right dimension, wrong group")
    print("-" * 66)
    for name, factors in (
        ("SO(26) x SO(19)", [so(26, 0), so(19, 1)]),
        ("SO(16) x SO(16)", [so(16, 0), so(16, 1)]),
        ("E8 x SO(16)", [e8(0), so(16, 1)]),
    ):
        verdict, report = cancels(factors)
        leftovers = ", ".join(
            f"{label(m)} -> {c}" for m, c in sorted(report["sixth_order"].items())
        )
        print(f"  {name:<16s} dim {report['dimension']:4d}  "
              f"tr R^6 {str(report['gravitational']):>12s}  "
              f"cancels: {verdict}")
        if leftovers:
            print(f"                   left over: {leftovers}")
    print("  SO(26) x SO(19) passes the gravitational condition exactly and")
    print("  fails the gauge one.  That is why the two are stated separately.")
    print()


def the_scan() -> list:
    """Search the family."""
    print("The scan")
    print("-" * 66)
    family = candidates(max_so=40, max_factors=2)
    survivors = scan(max_so=40, max_factors=2)
    print(f"  {len(family)} candidates: products of up to two SO(N), N <= 40, and E8.")
    print(f"  Anomaly-free: {len(survivors)}")
    for group in survivors:
        print(f"    {' x '.join(f.name for f in group)}   dim {combine(group)[0]}")
    print()
    print("  A search over a family, not a uniqueness proof: SU(N), Sp(N), the")
    print("  smaller exceptional algebras and the abelian solutions are all")
    print("  outside it, and E8 x U(1)^248 and U(1)^496 are known to work too.")
    print()
    return family


def figures(family: list) -> None:
    print("Figures")
    print("-" * 66)

    sampled = sorted(set(range(300, 701, 20)) | {496})
    gravity = [(n, gravitational_coefficient(n)) for n in sampled]
    gauge = [
        (n, sixth_order_terms(anomaly_polynomial([so(n)])).coefficient(("F6_0", 1)),
         n * (n - 1) // 2)
        for n in range(20, 45)
    ]
    plot_anomaly_conditions(gravity, gauge, FIG / "anomaly_conditions.png")
    print(f"  wrote {FIG / 'anomaly_conditions.png'}")

    points, survivors = [], []
    for group in family:
        dimension, _ = combine(group)
        poly = anomaly_polynomial(group)
        residue = sixth_order_terms(poly).largest()
        name = " x ".join(f.name for f in group)
        points.append((dimension, residue, name))
        if cancels(group)[0]:
            survivors.append((dimension, name))
    plot_anomaly_scan(points, survivors, FIG / "anomaly_scan.png")
    print(f"  wrote {FIG / 'anomaly_scan.png'}")

    sweep = range(24, 41)
    # A monomial that is fatal for *some* N stays marked for all of them: at
    # N = 32 the bar is zero, and the point is watching it get there.
    FATAL = {m for n in sweep for m in sixth_order_terms(anomaly_polynomial([so(n)]))}
    monomials = sorted(
        {m for n in sweep for m in anomaly_polynomial([so(n)])},
        key=lambda m: (len(m), m),
    )
    frames = []
    for n in sweep:
        poly = anomaly_polynomial([so(n)])
        values = [poly.get(m, Fraction(0)) for m in monomials]
        # Which monomials no X_4 X_8 product can reach -- asked of the module
        # rather than re-derived here, so the picture cannot disagree with it.
        doomed = set(sixth_order_terms(poly))
        fatal = [m in doomed or m in FATAL for m in monomials]
        verdict = "anomaly cancels" if cancels([so(n)])[0] else "anomalous"
        frames.append((n, [label(m) for m in monomials], values, fatal, verdict))
    animate_anomaly_sweep(frames, FIG / "anomaly_sweep.gif")
    print(f"  wrote {FIG / 'anomaly_sweep.gif'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    calibration()
    four_hundred_and_ninety_six()
    the_group()
    factorisation()
    the_negative_test()
    family = the_scan()
    figures(family)
    print("done.")
