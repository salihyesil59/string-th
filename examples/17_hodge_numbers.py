"""Hodge numbers of a toroidal orbifold, and what discrete torsion does to them.

Run:  python examples/17_hodge_numbers.py

``examples/13`` derived the phases the partition function may carry and said the
classic consequence -- ``(51, 3)`` becoming ``(3, 51)`` -- was not computed
there.  It is computed here, out of fixed-point counting and a character sum.

Writes ``figures/hodge_diamond.png`` and ``figures/fixed_loci.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.compactification.hodge import (  # noqa: E402
    blowup_moduli,
    euler_characteristic,
    fixed_locus,
    fixed_set,
    hodge_numbers,
    joint_euler,
    minor_gcd,
    untwisted_hodge,
    z2_z2_orbifold,
    z3_orbifold,
    z4_orbifold,
)
from stringsim.viz.plots import plot_fixed_loci, plot_hodge_diamond  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"

ORBIFOLDS = [
    ("T^6/Z_3", z3_orbifold()),
    ("T^6/Z_4", z4_orbifold()),
    ("T^6/(Z_2 x Z_2)", z2_z2_orbifold()),
]


def what_the_group_holds_still() -> None:
    print("=" * 72)
    print("Fixed sets, from the Smith invariants of 1 - g")
    print("=" * 72)
    print("  {x : (1-g) x in Z^6} / Z^6 is a union of subtori.  Its dimension is")
    print("  the nullity of 1-g, and the number of pieces is the product of the")
    print("  non-zero elementary divisors -- the gcd of the maximal minors.")
    print()
    for name, action in ORBIFOLDS:
        print(f"  {name}")
        for element in action.group.elements():
            if element == action.identity:
                continue
            locus = fixed_locus(action, element)
            print(f"    {str(element):>8s}: {str(locus):>12s}"
                  f"   contributes {locus.euler_characteristic:>3d} to chi")
        print()
    print("  A curve has Euler characteristic zero, so only the isolated points")
    print("  count -- and for Z_2 x Z_2 no single element has any.  They appear")
    print("  only when two different elements are asked to hold a point still:")
    print()
    action = z2_z2_orbifold()
    print(f"  {'g':>8s} {'h':>8s} {'chi(M^(g,h))':>14s}")
    for first in action.group.elements():
        for second in action.group.elements():
            value = joint_euler(action, first, second)
            if value:
                print(f"  {str(first):>8s} {str(second):>8s} {value:>14d}")
    print("  Six pairs, 64 points each.")
    print()


def the_euler_characteristic() -> None:
    print("=" * 72)
    print("The Euler characteristic, and the sign discrete torsion puts on it")
    print("=" * 72)
    print("  chi = (1/|G|) sum_{gh=hg} eps(g,h) chi(M^(g,h))")
    print()
    print(f"  {'orbifold':>18s} {'torsion classes':>16s} {'chi':>8s}")
    for name, action in ORBIFOLDS:
        pairings = action.group.pairings()
        values = [f"{euler_characteristic(action, p):+.0f}" for p in pairings]
        print(f"  {name:>18s} {len(pairings):>16d} {', '.join(values):>8s}")
    print()
    print("  Z_3 and Z_4 are cyclic, so there is no phase to choose and one")
    print("  answer.  Z_2 x Z_2 has two classes, and the non-trivial one is -1 on")
    print("  exactly the six pairs above -- so the whole of chi changes sign.")
    print()


def the_untwisted_forms() -> None:
    print("=" * 72)
    print("The untwisted forms, by averaging a character")
    print("=" * 72)
    print("  Tr_g on Lambda^p (x) conj(Lambda^q) is e_p(lambda) conj(e_q(lambda)),")
    print("  so the invariant dimension is its average over the group.")
    print()
    for name, action in ORBIFOLDS:
        diamond = untwisted_hodge(action)
        print(f"  {name:>18s}  h11 = {diamond[(1, 1)]}, h21 = {diamond[(2, 1)]}, "
              f"h30 = {diamond[(3, 0)]}, h00 = {diamond[(0, 0)]}")
    print()
    print("  h30 = 1 in every case: the phases multiply to 1, which is the")
    print("  Calabi-Yau condition, and the constructor refuses an action without")
    print("  it rather than producing a diamond that is not one.")
    print()


def putting_it_together() -> None:
    print("=" * 72)
    print("Two independent numbers, and the Hodge numbers between them")
    print("=" * 72)
    print("  h11 - h21 = chi/2                       (a count of lattice points)")
    print("  h11 + h21 = untwisted + blow-up moduli  (a character sum, plus one")
    print("                                           modulus per singular locus)")
    print()
    print(f"  {'orbifold':>18s} {'untwisted':>11s} {'blow-ups':>9s} {'chi':>6s}"
          f" {'(h11, h21)':>13s} {'known':>13s}")
    known = {
        ("T^6/Z_3", 0): "(36, 0)",
        ("T^6/Z_4", 0): "(31, 7)",
        ("T^6/(Z_2 x Z_2)", 0): "(51, 3)",
        ("T^6/(Z_2 x Z_2)", 1): "(3, 51)",
    }
    for name, action in ORBIFOLDS:
        for index, pairing in enumerate(action.group.pairings()):
            result = hodge_numbers(action, pairing)
            label = f"({result.h11}, {result.h21})"
            tag = name if index == 0 else f"{name} + torsion"
            print(f"  {tag:>18s} {str(result.untwisted):>11s} {result.twisted:>9d}"
                  f" {result.euler:>+6d} {label:>13s} {known[(name, index)]:>13s}")
    print()
    print("  The two sides never met: one is a gcd of integer minors, the other a")
    print("  sum of roots of unity plus a count of loci.  That they always give")
    print("  non-negative integers -- and the right ones -- is the check.")
    print()
    print("  The one thing put in by hand is that each singular locus carries one")
    print("  blow-up modulus.  It is stated in blowup_moduli and would show up as")
    print("  a parity failure if it were wrong.")
    print()


def the_mirror() -> None:
    print("=" * 72)
    print("What discrete torsion actually does")
    print("=" * 72)
    action = z2_z2_orbifold()
    trivial, torsion = action.group.pairings()
    plain, twisted = hodge_numbers(action, trivial), hodge_numbers(action, torsion)
    print(f"  without torsion: {plain}")
    print(f"  with torsion:    {twisted}")
    print()
    print(f"  Same untwisted forms {plain.untwisted}, same {plain.twisted} blow-up")
    print(f"  moduli, same total h11 + h21 = {plain.total}.  Only the sign of chi")
    print("  moves, and it swaps which side the 48 twisted moduli land on.")
    print()
    print("  Two Calabi-Yau manifolds with their Hodge numbers exchanged is what a")
    print("  mirror pair is.  A phase on the partition function -- a choice with no")
    print("  free parameter in it, one bit -- produces one.")
    print()


def figures() -> None:
    action = z2_z2_orbifold()
    panels = []
    for label, pairing in zip(
        ("no discrete torsion", "with discrete torsion"), action.group.pairings(), strict=True
    ):
        result = hodge_numbers(action, pairing)
        diamond = dict(untwisted_hodge(action))
        diamond[(1, 1)] = result.h11
        diamond[(2, 1)] = result.h21
        diamond[(1, 2)] = result.h21
        diamond[(2, 2)] = result.h11
        panels.append((f"{label}\n$\\chi = {result.euler:+d}$", diamond))
    plot_hodge_diamond(panels, FIG / "hodge_diamond.png")
    print(f"  wrote {FIG / 'hodge_diamond.png'}")

    rows = []
    for name, orbifold in ORBIFOLDS:
        entries = [
            (
                "".join(str(part) for part in element),
                fixed_locus(orbifold, element).components,
                fixed_locus(orbifold, element).dimension,
            )
            for element in orbifold.group.elements()
            if element != orbifold.identity
        ]
        rows.append((name, entries))
    plot_fixed_loci(rows, FIG / "fixed_loci.png")
    print(f"  wrote {FIG / 'fixed_loci.png'}")
    print("    Orange bars are the ones that contribute to the Euler")
    print("    characteristic.  Z_2 x Z_2 has none: its points come only from")
    print("    pairs of elements, which is why the answer is a sum over pairs.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    what_the_group_holds_still()
    the_euler_characteristic()
    the_untwisted_forms()
    putting_it_together()
    the_mirror()
    figures()
    print("  (the machinery itself: fixed_set of the 2x2 identity is "
          f"{fixed_set(np.eye(2, dtype=int))}, and minor_gcd of it at rank 2 is "
          f"{minor_gcd(np.eye(2, dtype=int), 2)})")
    print(f"  (blow-up moduli of T^6/Z_4: {blowup_moduli(z4_orbifold())})")
    print("done.")
