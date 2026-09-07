"""The brane's own dynamics, and the eleventh dimension.

Run:  python examples/14_dbi_and_m_theory.py

``examples/05_dbranes.py`` treats a D-brane as a place where strings end.  This
one gives the brane an action, finds the string again as a spike in its shape,
and then reads the whole ten-dimensional brane list off eleven dimensions.

Writes ``figures/dbi_field.png``, ``figures/bion_spike.png``,
``figures/bion_spike.gif``, ``figures/fermion_reflection.gif`` and
``figures/twisted_string.gif``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.branes.dbi import (  # noqa: E402
    bion_charge,
    bion_flux,
    bion_profile,
    bion_tension,
    bogomolny_gap,
    critical_field,
    dbi_determinant,
    dbi_determinant_closed,
    displacement,
    electric_series,
    energy_density,
    lagrangian_density,
    reduced_displacement,
)
from stringsim.branes.mtheory import (  # noqa: E402
    dirac_residual,
    kaluza_klein_check,
    m2_tension,
    m5_tension,
    m_theory_radius,
    planck_length,
    reduction_table,
)
from stringsim.superstring.rns import Sector  # noqa: E402
from stringsim.superstring.worldsheet import evolve_open_fermion, fold, sector_twist  # noqa: E402
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.animate import (  # noqa: E402
    animate_bion_spike,
    animate_fermion_reflection,
    animate_twisted_string,
)
from stringsim.viz.plots import plot_bion_spike, plot_dbi_field  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions()
G_S = 0.4
TWO_PI = 2.0 * math.pi


def the_determinant_is_computed() -> None:
    print("=" * 72)
    print("The DBI action, with the determinant taken rather than quoted")
    print("=" * 72)
    rng = np.random.default_rng(7)
    print("  S = -T_p int sqrt(-det(eta_ab + d_a X d_b X + 2 pi alpha' F_ab))")
    print()
    print(f"  {'p':>3s} {'max |closed form - (-det M)|':>30s}")
    for p in (1, 2, 3, 4, 5):
        worst = 0.0
        for _ in range(300):
            grad = rng.normal(size=p) * 0.7
            field = rng.normal(size=p) * 0.05
            worst = max(
                worst,
                abs(
                    dbi_determinant(grad, field, CONV.alpha_prime)
                    - dbi_determinant_closed(grad, field, CONV.alpha_prime)
                ),
            )
        print(f"  {p:>3d} {worst:>30.2e}")
    print()
    print("  The closed form (1+|grad X|^2)(1-|e|^2) + (e.grad X)^2 is a test here,")
    print("  not the implementation: the matrix is built and numpy takes its")
    print("  determinant.")
    print()


def there_is_a_largest_field() -> None:
    print("=" * 72)
    print("A largest electric field, which is the string tension")
    print("=" * 72)
    crit = critical_field(CONV)
    print(f"  E_crit = 1/(2 pi alpha') = {crit:.10f}")
    print(f"  T_F1   = 1/(2 pi alpha') = {1.0 / (TWO_PI * CONV.alpha_prime):.10f}   same number")
    print()
    print("  Expanding sqrt(1 - (E/E_crit)^2), coefficients read off the function")
    print("  itself by Cauchy's formula rather than from the binomial series:")
    measured = electric_series(6)
    exact = [1.0, -0.5, -0.125, -0.0625, -5.0 / 128.0, -7.0 / 256.0]
    names = ["1", "-1/2", "-1/8", "-1/16", "-5/128", "-7/256"]
    print(f"    measured  {np.array2string(measured, precision=10, floatmode='fixed')}")
    print(f"    exact     {names}")
    print(f"    max error {max(abs(a - b) for a, b in zip(measured, exact, strict=True)):.1e}")
    print()
    print("  The first term is the brane tension, the second is Maxwell.  Every")
    print("  term after that is a correction that keeps the energy finite right up")
    print("  to E_crit, where the Lagrangian goes imaginary and the module refuses:")
    try:
        lagrangian_density(np.zeros(3), np.full(3, crit), 3, G_S, CONV)
    except ValueError as error:
        print(f"    {error}")
    print()


def the_spike_is_a_string() -> None:
    print("=" * 72)
    print("A string, seen from the brane it ends on")
    print("=" * 72)
    print("  Legendre-transforming in E gives")
    print("    H = T_p sqrt((1+|grad X|^2)(1+|D|^2) - |D x grad X|^2)")
    print("      = T_p sqrt((1 + D.grad X)^2 + |D - grad X|^2)  >=  T_p (1 + D.grad X),")
    print("  so the bound is saturated exactly at D = grad X.  Checked against the")
    print("  numerical Legendre transform of the Lagrangian:")
    rng = np.random.default_rng(3)
    worst_h, worst_gap = 0.0, float("inf")
    for _ in range(200):
        grad = rng.normal(size=3) * 0.6
        field = rng.normal(size=3) * 0.02
        conjugate = displacement(grad, field, 3, G_S, CONV)
        reduced = reduced_displacement(grad, field, 3, G_S, CONV)
        transformed = conjugate @ field - lagrangian_density(grad, field, 3, G_S, CONV)
        worst_h = max(worst_h, abs(transformed - energy_density(grad, reduced, 3, G_S, CONV)))
        worst_gap = min(worst_gap, bogomolny_gap(grad, reduced, 3, G_S, CONV))
    print(f"    max |d.E - L  -  H(D)| = {worst_h:.2e}")
    print(f"    smallest Bogomolny gap over random configurations = {worst_gap:.2e}")
    example = rng.normal(size=3) * 0.6
    print(f"    gap at D = grad X exactly: {bogomolny_gap(example, example, 3, G_S, CONV):.2e}")
    print()
    print("  For the saturating solution the energy above the flat brane is a")
    print("  boundary term, height times flux.  With the flux quantised the spike")
    print("  weighs exactly what that many fundamental strings of its height weigh:")
    print()
    print(f"  {'p':>3s} {'n':>3s} {'q in X = q/r^(p-2)':>20s} {'flux':>8s}"
          f" {'spike tension':>15s} {'n T_F1':>12s} {'rel':>9s}")
    for p in (3, 4, 5):
        for n in (1, 2, 4):
            charge = bion_charge(n, p, G_S, CONV)
            flux = bion_flux(0.31, n, p, G_S, CONV)
            tension = bion_tension(n, p, G_S, CONV)
            expected = n / (TWO_PI * CONV.alpha_prime)
            print(f"  {p:>3d} {n:>3d} {charge:>20.8f} {flux:>8.5f}"
                  f" {tension:>15.10f} {expected:>12.10f} {abs(tension / expected - 1):>9.1e}")
    print()
    print("  The spike is infinitely tall and its total energy diverges -- as a")
    print("  string of infinite length should.  What is finite is the energy per")
    print("  unit height, and that is the fundamental string tension times n.")
    print()
    print("  p = 2 is refused rather than answered: the harmonic function is a")
    print("  logarithm there, so there is no finite height to divide by.")
    try:
        bion_charge(1, 2, G_S, CONV)
    except ValueError as error:
        print(f"    {error}")
    print()


def eleven_dimensions() -> None:
    print("=" * 72)
    print("Where the branes come from: eleven dimensions")
    print("=" * 72)
    print("  D0-branes weigh 1/(g_s sqrt(alpha')) each, and n of them weigh n times")
    print("  that -- a Kaluza-Klein tower on a circle of radius R_11 = g_s sqrt(alpha').")
    print()
    print(f"  {'g_s':>6s} {'R_11':>10s} {'l_p':>10s} {'T_D0 vs 1/R_11':>16s} {'Dirac':>10s}")
    for g_s in (0.05, 0.4, 1.0, 3.0):
        print(f"  {g_s:>6.2f} {m_theory_radius(g_s, CONV):>10.5f} {planck_length(g_s, CONV):>10.5f}"
              f" {kaluza_klein_check(g_s, CONV):>16.1e} {dirac_residual(g_s, CONV):>10.1e}")
    print()
    print("  The circle grows with the coupling, so the eleventh dimension is")
    print("  invisible exactly where string perturbation theory works.")
    print()
    print(f"  With g_s = {G_S}: T_M2 = {m2_tension(G_S, CONV):.8g}, "
          f"T_M5 = {m5_tension(G_S, CONV):.8g}")
    print("  and every ten-dimensional brane is one of those two, wrapped or not:")
    print()
    for reduction in reduction_table(G_S, CONV):
        print(f"    {reduction}")
    print()
    print("  Each right-hand number came from dp_brane_tension or the string")
    print("  tension, neither of which knows about eleven dimensions.  The NS5")
    print("  goes like 1/g_s^2 rather than 1/g_s, so it is not a D-brane -- the")
    print("  reduction says that without being told.")
    print()
    print(f"  2 kappa^2 T_M2 T_M5 / 2 pi - 1 = {dirac_residual(G_S, CONV):.2e}: the membrane and")
    print("  the fivebrane are electric and magnetic sources of the same three-form,")
    print("  so their tensions were never independent.")
    print()


def figures() -> None:
    # On a flat brane D = e/sqrt(1-e^2) inverts to e = D/sqrt(1+D^2), and the
    # energy is T_p sqrt(1+D^2): linear at large charge, where Maxwell is not.
    charges = np.linspace(0.0, 6.0, 400)
    dbi = np.sqrt(1.0 + charges**2) - 1.0
    maxwell = 0.5 * charges**2
    ratio = charges / np.sqrt(1.0 + charges**2)
    plot_dbi_field(charges, dbi, maxwell, ratio, FIG / "dbi_field.png")
    print(f"  wrote {FIG / 'dbi_field.png'}")

    def spike(n: int):
        return lambda r: bion_profile(r, n, 3, G_S, CONV)

    panels = [(f"$n = {n}$ string{'s' if n > 1 else ''}", spike(n)) for n in (1, 2, 4)]
    plot_bion_spike(panels, FIG / "bion_spike.png", extent=0.6)
    print(f"  wrote {FIG / 'bion_spike.png'}")
    animate_bion_spike(
        [(f"$n = {n}$", spike(n)) for n in (1, 2, 3, 4)],
        FIG / "bion_spike.gif",
        extent=0.6,
        title="A D3-brane with strings ending on it",
    )
    print(f"  wrote {FIG / 'bion_spike.gif'}")

    n_points = 192
    sigma = np.arange(n_points) * (TWO_PI / n_points)
    bump = np.exp(-(((sigma - 0.5 * math.pi) / 0.28) ** 2))[:, None]
    runs = []
    for sector, label in ((Sector.NS, "Neveu-Schwarz"), (Sector.R, "Ramond")):
        twist = sector_twist(sector)
        minus, plus = fold(bump, twist)
        runs.append(
            (
                f"{label}\n$\\eta = {twist:+d}$",
                evolve_open_fermion(
                    minus, plus, sector=sector, n_steps=3 * n_points, courant=1.0
                ),
            )
        )
    animate_fermion_reflection(runs, FIG / "fermion_reflection.gif", stride=4)
    print(f"  wrote {FIG / 'fermion_reflection.gif'}")
    print("    The blue pulse runs to sigma = pi, hands over to the orange one,")
    print("    comes back -- and in NS it returns upside down.")

    animate_twisted_string(1.0 / 3.0, FIG / "twisted_string.gif")
    print(f"  wrote {FIG / 'twisted_string.gif'}")
    print("    A Z_3 twisted string: the two ends are 120 degrees apart about the")
    print("    fixed point, and no motion can bring them together.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_determinant_is_computed()
    there_is_a_largest_field()
    the_spike_is_a_string()
    eleven_dimensions()
    figures()
    print("done.")
