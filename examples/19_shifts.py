"""Shifts: repairing an asymmetric twist that fails on its own.

Run:  python examples/19_shifts.py

``examples/13`` found that the self-dual circle's T-duality twist misses level
matching by 1/8, and said that is why such an orbifold needs a shift.  This one
finds the shift.

Writes ``figures/shift_landscape.png``.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.compactification.asymmetric import (  # noqa: E402
    AsymmetricTwist,
    classify_automorphisms,
    shifts_that_close,
)
from stringsim.compactification.torus import (  # noqa: E402
    TorusBackground,
    factorized_duality,
    odd_metric,
)
from stringsim.viz.plots import plot_shift_landscape  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CIRCLE = TorusBackground(np.array([[1.0]]))
DUALITY = factorized_duality(1, 0)

BACKGROUNDS = [
    ("self-dual S^1", TorusBackground(np.array([[1.0]]))),
    ("half self-dual T^2", TorusBackground(np.diag([1.0, 2.3]))),
    ("self-dual T^2", TorusBackground(np.eye(2))),
]


def what_a_shift_changes() -> None:
    print("=" * 72)
    print("A twist may translate as well as turn")
    print("=" * 72)
    print("  Z -> Omega Z + v, and two things change.")
    print()
    print("  The order.  The element closes only when (1 + Omega + ...) v lands")
    print("  back on the lattice, so a shift can raise it:")
    print()
    print(f"  {'shift':>18s} {'order':>6s} {'<v,v>':>8s} {'E_L - E_R':>12s} {'defect':>9s}")
    for entries in ([0, 0], [1, 1], [1, 3], [2, 2]):
        shift = np.array(entries) / 4.0
        twist = AsymmetricTwist(CIRCLE, DUALITY, shift)
        left, right = twist.twisted_energies
        label = f"({Fraction(entries[0], 4)}, {Fraction(entries[1], 4)})"
        print(f"  {label:>18s} {twist.order:>6d} {twist.shift_norm:>8.4f}"
              f" {left - right:>12.6f} {twist.level_matching_defect:>9.4f}")
    print()
    print("  And the level-matching condition, which becomes")
    print("    N [ (a_R - a_L) + <v,v>/2 ]  in  Z.")
    print("  The <v,v> is the only way the shift enters it, and on a circle that")
    print("  is 2 n w -- so a pure momentum or pure winding shift contributes")
    print("  nothing and can never break anything:")
    print()
    print(f"  {'shift':>18s} {'order':>6s} {'<v,v>':>8s} {'defect':>9s}")
    for entries, label in (
        ([0, 1], "momentum 1/3"),
        ([1, 0], "winding 1/3"),
        ([1, 1], "both 1/3"),
    ):
        shift = np.array(entries) / 3.0
        twist = AsymmetricTwist(CIRCLE, np.eye(2), shift)
        print(f"  {label:>18s} {twist.order:>6d} {twist.shift_norm:>8.4f}"
              f" {twist.level_matching_defect:>9.4f}")
    print()


def finding_the_shift() -> None:
    print("=" * 72)
    print("The one that repairs T-duality at the self-dual radius")
    print("=" * 72)
    bare = AsymmetricTwist(CIRCLE, DUALITY)
    print(f"  on its own:  {bare}")
    print(f"    a_L = {bare.left_intercept}, a_R = {bare.right_intercept}, "
          f"and 2 (a_R - a_L) = -1/8 is not an integer.")
    print()
    print("  Searching the shifts with each denominator in turn:")
    print()
    print(f"  {'denominator':>12s} {'shifts that close it':>21s}")
    for denominator in (2, 3, 4, 5, 6):
        found = shifts_that_close(CIRCLE, DUALITY, denominator)
        names = ", ".join(
            f"({Fraction(round(t.shift[0] * denominator), denominator)}, "
            f"{Fraction(round(t.shift[1] * denominator), denominator)})"
            for t in found[:3]
        )
        print(f"  {denominator:>12d} {len(found):>6d}    {names}")
    print()
    print("  Halves and thirds do nothing at all.  Quarters work, and the answer")
    print("  is v = (1/4, 1/4):")
    print()
    for twist in shifts_that_close(CIRCLE, DUALITY, 4):
        left, right = twist.twisted_energies
        print(f"    {twist}")
        print(f"      E_L = {left:.6f}, E_R = {right:.6f}, difference {left - right:.6f}")
    print()
    print("  The first one has E_L = E_R exactly: the shift's <v,v>/2 = 1/16 is")
    print("  precisely the gap between a_L and a_R that the rotation left behind.")
    print()
    print("  And it acts freely.  (1 - Omega) x = v has no solution modulo the")
    print("  lattice, so the twisted sector is stuck to nothing -- which is how a")
    print("  shift breaks supersymmetry without leaving a fixed point behind.")
    print()


def across_the_backgrounds() -> None:
    print("=" * 72)
    print("How much of what failed can be repaired")
    print("=" * 72)
    print(f"  {'background':>20s} {'asymmetric':>11s} {'failing':>8s} {'repairable':>11s}"
          f" {'order raised':>13s} {'order kept':>11s}")
    for name, background in BACKGROUNDS:
        twists = [t for t in classify_automorphisms(background) if t.is_asymmetric]
        failing = [t for t in twists if not t.is_level_matched]
        repaired, raised, kept = 0, 0, 0
        for twist in failing:
            found = shifts_that_close(background, twist.omega, 4)
            if found:
                repaired += 1
            raised += sum(1 for fixed in found if fixed.order > twist.order)
            kept += sum(1 for fixed in found if fixed.order == twist.order)
        print(f"  {name:>20s} {len(twists):>11d} {len(failing):>8d} {repaired:>11d}"
              f" {raised:>13d} {kept:>11d}")
    print()
    print("  Not everything is rescued by quarters -- the denominator matters and")
    print("  the search says so rather than guessing.  Most repairs raise the")
    print("  order, because the shift has to work its way back onto the lattice,")
    print("  but not all of them do: on T^2 some close at the order the rotation")
    print("  already had.  The last two columns count the shifts, not the twists.")
    print()


def figures() -> None:
    grid = np.linspace(0.0, 1.0, 201)
    first, second = np.meshgrid(grid, grid)
    bare = AsymmetricTwist(CIRCLE, DUALITY)
    form = odd_metric(1)
    # E_L - E_R = (a_R - a_L) + <v,v>/2, and <v,v> = v^T eta v is smooth in v
    norms = np.zeros_like(first)
    for row in range(first.shape[0]):
        for column in range(first.shape[1]):
            shift = np.array([first[row, column], second[row, column]])
            norms[row, column] = shift @ form @ shift
    mismatch = (bare.right_intercept - bare.left_intercept) + norms / 2.0

    marked = []
    for denominator in (2, 3, 4):
        for numerator in np.ndindex(denominator, denominator):
            shift = np.array(numerator) / denominator
            try:
                twist = AsymmetricTwist(CIRCLE, DUALITY, shift)
            except ValueError:
                continue
            marked.append((shift[0], shift[1], twist.is_level_matched))
    plot_shift_landscape(first, second, mismatch, marked, FIG / "shift_landscape.png")
    print(f"  wrote {FIG / 'shift_landscape.png'}")
    print("    Every marker is a shift that closes into a finite-order element.")
    print("    The filled ones are the two that also level-match, and they are the")
    print("    reason this orbifold exists at all.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    what_a_shift_changes()
    finding_the_shift()
    across_the_backgrounds()
    figures()
    print("done.")
