"""Vertex operators: the Veneziano amplitude, derived rather than written down.

Run:  python examples/25_vertex_operators.py

``examples/06`` evaluates the Veneziano amplitude as a Beta function.  That is
the answer.  This is where it comes from: an integral over where the vertex
operators sit on the boundary of the disc, gauge-fixed by hand and done by
quadrature.

Three things come out that a Beta function cannot show.  The gauge choice drops
out, and it drops out *because* the external states are on shell -- nudge them
off and the invariance goes with them.  The tachyon pole is located in the
geometry: it is the region where two punctures collide.  And the same machinery
runs at five points, where there is nothing to look up.

Writes ``figures/koba_nielsen.png``, ``figures/five_point_moduli.png`` and
``figures/pole_emergence.gif``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.amplitudes.veneziano import veneziano, veneziano_residue  # noqa: E402
from stringsim.amplitudes.vertex import (  # noqa: E402
    converges,
    exponent_residual,
    exponents_from_momenta,
    fit_tachyon_pole,
    five_point_exponents,
    four_point_exponents,
    gauge_spread,
    koba_nielsen,
    mandelstam_sum,
    ordered_amplitude,
    tachyon_momenta,
)
from stringsim.units import minkowski  # noqa: E402
from stringsim.viz.animate import animate_pole_emergence  # noqa: E402
from stringsim.viz.plots import plot_five_point_moduli, plot_koba_nielsen  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
FIVE = (-2.0, -2.2, -2.4, -2.1, -2.3)
ANCHORS = (1.5, 2.0, 3.0, 6.0)


def the_correlator() -> None:
    """Exponents from the propagator, and from real momenta."""
    print("The Koba-Nielsen correlator")
    print("-" * 70)
    print("  On the disc boundary <X(y) X(y')> = -2 alpha' log|y - y'|, so the")
    print("  product of tachyon vertex operators gives")
    print("      prod_{i<j} |y_i - y_j|^{2 alpha' k_i . k_j}")
    print("  and for tachyons 2 alpha' k_i.k_j = -alpha' s_ij - 2.")
    print()
    print(f"  s + t + u = {mandelstam_sum():.1f} (the four external masses)")
    print()
    print("      s      t      e_12    e_23    e_13    row sum")
    for s, t in ((-2.0, -2.5), (-3.0, -1.6), (-2.2, -4.0)):
        m = four_point_exponents(s, t)
        print(f"    {s:5.1f}  {t:5.1f}   {m[0,1]:6.2f}  {m[1,2]:6.2f}  {m[0,2]:6.2f}   "
              f"{m[0].sum():+.1e}")
    print("  Every row sums to -2, which is 2 alpha' k_i^2 with alpha' m^2 = -1.")
    print("  Nothing imposes that here; it follows from s + t + u.")
    print()
    s, t = 6.0, -1.5
    momenta = tachyon_momenta(s, t)
    eta = minkowski(momenta.shape[1])
    print(f"  Four explicit on-shell momenta at s = {s}, t = {t}:")
    print(f"    sum of momenta   {np.max(np.abs(momenta.sum(axis=0))):.1e}")
    print(f"    k^2 (alpha' = 1) {[round(float(np.sum(eta * k * k)), 9) for k in momenta]}")
    upper = np.triu_indices(4, 1)
    same = np.allclose(
        np.sort(exponents_from_momenta(momenta)[upper]),
        np.sort(four_point_exponents(s, t)[upper]),
    )
    print(f"    their exponents match the invariant ones: {same}")
    print("  That region is physical and the integral does not converge there;")
    print("  the two facts are unrelated and both are worth knowing.")
    print()


def the_amplitude() -> list:
    """The integral against the Beta function."""
    print("The gauge-fixed integral")
    print("-" * 70)
    print("  Fix y_1 = 0, y_3 = 1, y_4 = R and integrate y_2 over (0, 1), with")
    print("  the Faddeev-Popov factor |y_1-y_3||y_1-y_4||y_3-y_4|.")
    print()
    print("      s      t     worldsheet integral    Beta function      gauge spread")
    comparison = []
    for s, t in ((-2.0, -2.5), (-3.0, -1.6), (-2.2, -4.0), (-1.5, -1.8)):
        m = four_point_exponents(s, t)
        value = ordered_amplitude(m, 3.0)
        closed = float(veneziano(s, t))
        print(f"    {s:5.1f}  {t:5.1f}   {value:18.12f}   {closed:16.12f}   "
              f"{gauge_spread(m, ANCHORS):.1e}")
        comparison.append((s, t, value, closed))
    print()
    print("  Only at R -> infinity does the integrand become the Beta")
    print("  integrand.  At R = 1.5 it looks nothing like it and agrees anyway.")
    print()
    return comparison


def on_shell_is_the_reason() -> list:
    """Break the mass-shell condition and the gauge invariance goes."""
    print("Why the gauge drops out")
    print("-" * 70)
    print("  The integrand transforms correctly only when every row of the")
    print("  exponent matrix sums to -2 -- the external mass-shell conditions.")
    print()
    print("      exponent nudged   by      row residual   gauge spread")
    series = []
    base = four_point_exponents(-2.0, -2.5)
    clean = [ordered_amplitude(base, a) for a in ANCHORS]
    series.append(("on shell", ANCHORS, clean))
    print(f"    {'none (on shell)':<17s} {0.0:5.2f}    {exponent_residual(base):.1e}"
          f"        {gauge_spread(base, ANCHORS):.1e}")
    for label, (i, j), amount in (
        ("e_12", (0, 1), 0.02),
        ("e_24", (1, 3), 0.05),
        ("e_13", (0, 2), 0.05),
    ):
        bad = four_point_exponents(-2.0, -2.5)
        bad[i, j] += amount
        bad[j, i] += amount
        values = [ordered_amplitude(bad, a) for a in ANCHORS]
        print(f"    {label:<17s} {amount:5.2f}    {exponent_residual(bad):.1e}"
              f"        {gauge_spread(bad, ANCHORS):.1e}")
        if label != "e_13":
            series.append((f"{label} off by {amount}", ANCHORS, values))
    print()
    print("  A per cent off shell is a per cent of gauge dependence -- except")
    print("  for e_13, which this gauge cannot see at all: punctures 1 and 3")
    print("  sit at 0 and 1, and 1^e = 1 for any e.  That is consistent, not a")
    print("  hole: on shell e_13 is fixed by the others, and the amplitude")
    print("  B(-alpha(s), -alpha(t)) carries no independent u either.")
    print()
    return series


def the_pole() -> list:
    """The pole is two punctures colliding."""
    print("Where the pole is")
    print("-" * 70)
    print("  As alpha(s) -> 0 the exponent of |y_2 - y_1| reaches -1 and the")
    print("  integral diverges at x = 0.  Nothing else in the integrand is")
    print("  singular, so the pole sits in the geometry.")
    print()
    for t in (-3.0, -5.0):
        fit = fit_tachyon_pole(t=t)
        print(f"    t = {t:5.1f}:  {fit}")
    print(f"  veneziano_residue(0, t) = {float(veneziano_residue(0, -3.0)):.1f}, "
          "and it does not depend on t:")
    print("  the tachyon has no spin.")
    print()
    m = four_point_exponents(-0.5, -2.5)
    print(f"  Past the pole (s = -0.5): converges = {converges(m)}, and")
    print(f"  ordered_amplitude refuses, while the Beta function continues to "
          f"{float(veneziano(-0.5, -2.5)):.6f}.")
    print("  That continuation is what the closed form is for.")
    print()
    frames = []
    xs = np.linspace(1e-4, 1.0 - 1e-9, 400)
    for offset in np.linspace(0.30, 0.02, 40):
        s = -1.0 - offset
        matrix = four_point_exponents(s, -3.0)
        values = np.array([koba_nielsen((0.0, x, 1.0, 3.0), matrix) for x in xs])
        amplitude = ordered_amplitude(matrix, 3.0)
        frames.append((-offset, xs, values, -offset * amplitude))
    return frames


def five_points() -> tuple:
    """No closed form, and the checks still work."""
    print("Five punctures")
    print("-" * 70)
    matrix = five_point_exponents(FIVE)
    print(f"  adjacent invariants {FIVE}")
    print("  the five chords are solved for, not chosen:")
    for i, j in ((0, 2), (1, 3), (2, 4), (0, 3), (1, 4)):
        print(f"    e_{i+1}{j+1} = {matrix[i, j]:+.4f}")
    print(f"  row residual {exponent_residual(matrix):.1e}, converges {converges(matrix, 5)}")
    print()
    value = ordered_amplitude(matrix, 3.0)
    spread = gauge_spread(matrix, anchors=(2.0, 4.0))
    print(f"  amplitude {value:.10f}   gauge spread {spread:.1e}")
    print()
    print("  and the disc's own symmetry, with the matrix permuted and the")
    print("  integral redone:")
    for shift in (1, 2):
        order = [(i + shift) % 5 for i in range(5)]
        rotated = ordered_amplitude(matrix[np.ix_(order, order)], 3.0)
        print(f"    cyclic by {shift}: {rotated:.10f}")
    order = list(reversed(range(5)))
    print(f"    reflected  : {ordered_amplitude(matrix[np.ix_(order, order)], 3.0):.10f}")
    print()
    print("  There is nothing to compare the number to.  What says the")
    print("  construction is right is that the gauge drops out here exactly as")
    print("  it did where a Beta function was watching.")
    print()
    return matrix, value, spread


def figures(comparison: list, series: list, frames: list, five: tuple) -> None:
    print("Figures")
    print("-" * 70)

    xs = np.linspace(1e-4, 1.0 - 1e-9, 400)
    curves = []
    for s in (-2.6, -1.6, -1.2):
        matrix = four_point_exponents(s, -3.0)
        values = [koba_nielsen((0.0, x, 1.0, 3.0), matrix) for x in xs]
        curves.append((rf"$\alpha(s) = {1.0 + s:+.1f}$", xs, values))

    scan = []
    for t in np.linspace(-4.5, -1.6, 12):
        matrix = four_point_exponents(-2.4, float(t))
        scan.append((float(t), ordered_amplitude(matrix, 3.0), float(veneziano(-2.4, t))))
    del comparison

    plot_koba_nielsen(curves, scan, series, FIG / "koba_nielsen.png")
    print(f"  wrote {FIG / 'koba_nielsen.png'}")

    matrix, value, spread = five
    size = 160
    axis = np.linspace(1e-3, 1.0 - 1e-3, size)
    grid = np.full((size, size), np.nan)
    for row, y in enumerate(axis):
        for column, x in enumerate(axis):
            if x < y:
                grid[row, column] = koba_nielsen((0.0, x, y, 1.0, 3.0), matrix)
    plot_five_point_moduli(
        grid,
        (0.0, 1.0, 0.0, 1.0),
        note=f"amplitude {value:.6f}\ngauge spread {spread:.1e}",
        path=FIG / "five_point_moduli.png",
    )
    print(f"  wrote {FIG / 'five_point_moduli.png'}")

    animate_pole_emergence(frames, FIG / "pole_emergence.gif")
    print(f"  wrote {FIG / 'pole_emergence.gif'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_correlator()
    comparison = the_amplitude()
    series = on_shell_is_the_reason()
    frames = the_pole()
    five = five_points()
    figures(comparison, series, frames, five)
    print("done.")
