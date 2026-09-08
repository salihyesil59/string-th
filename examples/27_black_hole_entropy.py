"""Strominger-Vafa: the same entropy from a generating function and a horizon.

Run:  python examples/27_black_hole_entropy.py

The D1-D5-P system is a black hole in five dimensions.  Its entropy can be got
at two ways with no step in common: count the states of the effective string,
or measure the area of the horizon and divide by 4G.

The counting side is a product expanded in exact integers, and the Cardy
exponent is *measured* out of it rather than quoted -- slowly, because the
subleading log is not a correction.  The horizon side is three harmonic radii
and Newton's constant, and what it has to do is forget every modulus.

Writes ``figures/black_hole_entropy.png`` and ``figures/cardy_fit.gif``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.branes.entropy import (  # noqa: E402
    VOLUME_CONVENTIONS,
    Moduli,
    bekenstein_hawking,
    bps_degeneracies,
    cardy_entropy,
    central_charge,
    fit_cardy,
    harmonic_radii,
    horizon_area,
    microscopic_entropy,
    moduli_spread,
    newton_five,
    normalisation,
    species,
)
from stringsim.viz.animate import animate_cardy_fit  # noqa: E402
from stringsim.viz.plots import plot_black_hole_entropy  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CHARGES = ((1, 1), (1, 2), (2, 2))


def the_effective_string() -> None:
    """What the bound state is, and how many states it has."""
    print("The effective string")
    print("-" * 70)
    print("  Q1 D1-branes and Q5 D5-branes bind into a 1+1 CFT with 4 Q1 Q5")
    print("  bosons and as many fermions, so c = 4 Q1 Q5 + (1/2) 4 Q1 Q5.")
    print()
    print("      Q1  Q5   species   c = 6 Q1 Q5")
    for q1, q5 in CHARGES:
        print(f"    {q1:4d}{q5:4d}   {species(q1, q5):7d}   {central_charge(q1, q5):11d}")
    print()
    print("  The momentum N sits in the left-movers, and the number of ways of")
    print("  putting it there is the coefficient of q^N in")
    print("      prod_n [(1 + q^n)/(1 - q^n)]^{4 Q1 Q5}")
    print()
    table = bps_degeneracies(1, 1, 60)
    print("  Q1 = Q5 = 1, the first few and one large one:")
    print(f"    d_0..d_8 = {table[:9]}")
    print(f"    d_60     = {table[60]}   ({len(str(table[60]))} digits)")
    print()


def measuring_cardy() -> None:
    """The exponent, out of the integers."""
    print("Measuring the Cardy exponent")
    print("-" * 70)
    print("  log d_N should approach 2 pi sqrt(c N / 6) = 2 pi sqrt(Q1 Q5 N).")
    print("  At any level that can be reached it is well below:")
    print()
    print("      Q1  Q5    N    log d_N     2 pi sqrt(Q1 Q5 N)   ratio")
    for q1, q5 in CHARGES:
        table = bps_degeneracies(q1, q5, 60)
        for n in (20, 60):
            micro = microscopic_entropy(q1, q5, n, table)
            macro = cardy_entropy(q1, q5, n)
            print(f"    {q1:4d}{q5:4d} {n:4d}   {micro:9.4f}   {macro:16.4f}   {micro/macro:.4f}")
    print()
    print("  So it has to be fitted.  log d_N = a sqrt(N) + b log N + c:")
    print()
    print("      Q1  Q5   n_max        a        2 pi sqrt(Q1 Q5)     error        b")
    for (q1, q5), n_max in zip(CHARGES, (400, 400, 300), strict=True):
        fit = fit_cardy(q1, q5, n_max)
        print(f"    {q1:4d}{q5:4d} {n_max:6d}   {fit.slope:9.6f}   {fit.expected:16.6f}   "
              f"{fit.error:.1e}   {fit.subleading:+7.4f}")
    print()
    print("  b is heading for -(3 + 4 Q1 Q5)/4 and gets there slowly:")
    for n_max in (200, 400, 800, 1600):
        fit = fit_cardy(1, 1, n_max)
        print(f"    n_max {n_max:5d}:  a = {fit.slope:.6f}   b = {fit.subleading:+.4f}   "
              f"(expected {fit.subleading_expected:+.2f})")
    print()
    without = fit_cardy(1, 1, 400, with_log=False)
    with_log = fit_cardy(1, 1, 400)
    print("  And the log term is not optional:")
    print(f"    two-term fit:   a = {without.slope:.6f}   error {without.error:.1e}")
    print(f"    three-term fit: a = {with_log.slope:.6f}   error {with_log.error:.1e}")
    print()


def the_horizon() -> None:
    """The other side."""
    print("The horizon")
    print("-" * 70)
    moduli = Moduli()
    r1, r5, rp = harmonic_radii(3, 7, 200, moduli)
    print(f"  Q1 = 3, Q5 = 7, N = 200 at g_s = {moduli.coupling}, V = {moduli.volume},")
    print(f"  R = {moduli.radius}, alpha' = {moduli.alpha_prime}:")
    print(f"    r_1^2 = {r1:.6f}   r_5^2 = {r5:.6f}   r_p^2 = {rp:.6f}")
    print(f"    area  = {horizon_area(3, 7, 200, moduli):.6f}   "
          f"G_5 = {newton_five(moduli):.6e}")
    print(f"    A / 4 G_5 = {bekenstein_hawking(3, 7, 200, moduli):.6f}")
    print(f"    2 pi sqrt(Q1 Q5 N) = {cardy_entropy(3, 7, 200):.6f}")
    print()
    print("  An entropy counts states, so every continuous modulus has to drop")
    print("  out.  Varying all four:")
    print()
    print("      g_s     V      R    alpha'      A / 4 G_5")
    for mod in (
        Moduli(),
        Moduli(coupling=0.05, volume=6.9, radius=0.6, alpha_prime=2.3),
        Moduli(coupling=0.9, volume=0.4, radius=5.5, alpha_prime=0.7),
        Moduli(coupling=0.31, volume=3.3, radius=1.1, alpha_prime=1.9),
    ):
        print(f"    {mod.coupling:5.2f} {mod.volume:6.2f} {mod.radius:6.2f} "
              f"{mod.alpha_prime:7.2f}   {bekenstein_hawking(3, 7, 200, mod):16.10f}")
    print()
    print("  That needs the powers in the three radii and in G_5 to be mutually")
    print("  consistent, and it is the part of this side that is derived.")
    print()


def the_convention() -> None:
    """What the match fixes, said out loud."""
    print("What the agreement fixes")
    print("-" * 70)
    print("  The moduli cancel whatever convention is used for the T^4 volume.")
    print("  What the convention changes is one overall number:")
    print()
    print("      convention        S_Cardy / S_BH        spread over 9 (charge, moduli)")
    for convention in VOLUME_CONVENTIONS:
        print(f"    {convention:<16s}  {normalisation(3, 7, 200, convention=convention):18.8f}"
              f"        {moduli_spread(convention=convention):.1e}")
    print(f"    (4 pi^2 = {4 * math.pi**2:.6f}, 16 pi^4 = {16 * math.pi**4:.6f})")
    print()
    print("  Only 'reduced' -- the dimensionless v = V/(2 pi)^4 alpha'^2 in the")
    print("  D1 and momentum radii -- makes the two calculations agree.  That is")
    print("  how the convention is chosen here, the same way the type IIB anomaly")
    print("  picks the Hirzebruch class in examples/22.  The match has seven")
    print("  parameters in it and one number to get right, so it is evidence.")
    print()
    print("      Q1   Q5     N     counted            horizon        ratio")
    for q1, q5, n in ((1, 1, 10), (3, 7, 200), (12, 5, 41), (20, 20, 1000)):
        micro = cardy_entropy(q1, q5, n)
        macro = bekenstein_hawking(q1, q5, n)
        print(f"    {q1:4d} {q5:4d} {n:5d}   {micro:14.8f}   {macro:14.8f}   {micro/macro:.12f}")
    print()


def figures() -> None:
    print("Figures")
    print("-" * 70)

    curves = []
    for q1, q5 in CHARGES:
        n_max = 120
        table = bps_degeneracies(q1, q5, n_max)
        levels = np.arange(4, n_max + 1)
        counted = np.array([math.log(table[int(n)]) for n in levels])
        cardy = np.array([cardy_entropy(q1, q5, int(n)) for n in levels])
        curves.append((f"$Q_1={q1}$, $Q_5={q5}$", np.sqrt(levels), counted, cardy))

    couplings = np.linspace(0.05, 0.95, 12)
    scans = []
    for q1, q5, n in ((1, 1, 10), (3, 7, 200)):
        values = [bekenstein_hawking(q1, q5, n, Moduli(coupling=float(g))) for g in couplings]
        scans.append((f"$({q1}, {q5}, {n})$", couplings, values))
    # The same calculation with one power of the coupling put back by hand, to
    # show that the flatness of the others is a result and not the plot.
    broken = [
        horizon_area(3, 7, 200, Moduli(coupling=float(g)))
        * float(g)
        / (4.0 * newton_five(Moduli(coupling=float(g))))
        for g in couplings
    ]
    scans.append(("one power of $g_s$ left in", couplings, broken))
    plot_black_hole_entropy(curves, scans, FIG / "black_hole_entropy.png")
    print(f"  wrote {FIG / 'black_hole_entropy.png'}")

    frames = []
    biggest = 900
    table = bps_degeneracies(1, 1, biggest)
    for n_max in (40, 60, 90, 130, 190, 280, 400, 600, 900):
        fit = fit_cardy(1, 1, n_max)
        levels = np.arange(4, n_max + 1)
        counted = np.array([math.log(table[int(n)]) for n in levels])
        design = np.column_stack(
            [np.sqrt(levels), np.log(levels), np.ones_like(levels, dtype=float)]
        )
        used = fit.levels
        rebuilt = np.column_stack(
            [np.sqrt(used), np.log(used), np.ones_like(used, dtype=float)]
        )
        coefficients, *_ = np.linalg.lstsq(
            rebuilt, np.array([math.log(table[int(n)]) for n in used]), rcond=None
        )
        frames.append((n_max, np.sqrt(levels), counted, design @ coefficients, fit.slope))
    animate_cardy_fit(frames, 2.0 * math.pi, FIG / "cardy_fit.gif")
    print(f"  wrote {FIG / 'cardy_fit.gif'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_effective_string()
    measuring_cardy()
    the_horizon()
    the_convention()
    figures()
    print("done.")
