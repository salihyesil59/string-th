"""Non-abelian orbifold groups: what changes and what does not.

Run:  python examples/20_non_abelian.py

``examples/17`` computes Hodge numbers for abelian quotients of ``T^6``.  Two of
the three ingredients were already general and one was not, and this separates
them: the Euler characteristic and the untwisted forms need nothing from
commutativity, while the blow-up count does.

Writes ``figures/commutation.png``.
"""

from __future__ import annotations

import cmath
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.compactification.hodge import (  # noqa: E402
    block_permutation,
    close_group,
    delta27_orbifold,
    elementary_symmetric,
    euler_characteristic,
    from_abelian,
    hodge_numbers,
    lattice_rotation,
    untwisted_hodge,
    z2_z2_orbifold,
    z3_orbifold,
    z4_orbifold,
)
from stringsim.viz.plots import plot_commutation  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
ABELIAN = [("T^6/Z_3", z3_orbifold), ("T^6/Z_4", z4_orbifold), ("T^6/(Z_2 x Z_2)", z2_z2_orbifold)]


def z3_squared():
    """The abelian ``Z_3 x Z_3`` sitting inside Delta(27)."""
    rotation = np.array([[0, -1], [1, -1]], dtype=np.int64)
    unit = np.eye(2, dtype=np.int64)
    omega = cmath.exp(2j * cmath.pi / 3.0)
    return close_group(
        [
            (
                lattice_rotation(unit, rotation, rotation @ rotation),
                np.diag([1.0 + 0j, omega, omega**2]),
            ),
            (
                lattice_rotation(rotation @ rotation, unit, rotation),
                np.diag([omega**2, 1.0 + 0j, omega]),
            ),
        ]
    )


def the_general_path_agrees() -> None:
    print("=" * 72)
    print("The general code, checked against the answers already verified")
    print("=" * 72)
    print("  chi sums over commuting pairs, which for an abelian group is every")
    print("  pair; the untwisted forms average a character.  Neither needed")
    print("  commutativity, so re-presenting the abelian orbifolds as general")
    print("  groups has to give exactly the same numbers back:")
    print()
    print(f"  {'orbifold':>18s} {'|G|':>4s} {'chi (abelian)':>14s} {'chi (general)':>14s}"
          f" {'untwisted match':>16s}")
    for name, maker in ABELIAN:
        action = maker()
        general = from_abelian(action)
        first, second = untwisted_hodge(action), general.untwisted_hodge()
        agree = all(first[key] == second[key] for key in first)
        print(f"  {name:>18s} {general.size:>4d} {euler_characteristic(action):>+14.0f}"
              f" {general.euler_characteristic():>+14.0f} {str(agree):>16s}")
    print()


def what_non_abelian_costs() -> None:
    print("=" * 72)
    print("Delta(27): the smallest group that cannot be reordered")
    print("=" * 72)
    print("  a = diag(1, w, w^2) on three hexagonal tori, b the cyclic permutation")
    print("  of the three factors.  Both are in SU(3); they do not commute.")
    print()
    for name, group in (("Z_3 x Z_3", z3_squared()), ("Delta(27)", delta27_orbifold())):
        classes = group.conjugacy_classes()
        pairs = group.commuting_pairs()
        print(f"  {name}")
        print(f"    {group}")
        print(f"    class sizes: {sorted(len(entry) for entry in classes)}")
        print(f"    |G| x classes = {group.size} x {len(classes)} = "
              f"{group.size * len(classes)}, and there are {len(pairs)} commuting pairs")
        assert len(pairs) == group.size * len(classes)
    print()
    print("  That identity is not put in anywhere: the enumeration finds the")
    print("  commuting pairs one at a time and the conjugacy classes separately,")
    print("  and they agree.  It is the cheapest check that the closure is")
    print("  complete and the conjugation is right.")
    print()


def what_still_works() -> None:
    print("=" * 72)
    print("What the two general ingredients give")
    print("=" * 72)
    print(f"  {'orbifold':>18s} {'|G|':>4s} {'classes':>8s} {'untwisted (h11, h21)':>21s}"
          f" {'h30':>4s} {'chi':>6s}")
    for name, group in (
        ("T^6/Z_3", from_abelian(z3_orbifold())),
        ("T^6/(Z_3 x Z_3)", z3_squared()),
        ("T^6/Delta(27)", delta27_orbifold()),
    ):
        diamond = group.untwisted_hodge()
        print(f"  {name:>18s} {group.size:>4d} {len(group.conjugacy_classes()):>8d}"
              f" {str((diamond[(1, 1)], diamond[(2, 1)])):>21s} {diamond[(3, 0)]:>4d}"
              f" {group.euler_characteristic():>+6.0f}")
    print()
    print("  h30 = 1 all the way along: every element has holomorphic determinant")
    print("  1, including the permutation, so the quotient stays Calabi-Yau.  The")
    print("  untwisted forms are projected harder as the group grows -- 9, then 3,")
    print("  then 1 -- which is the whole effect of quotienting on that sector.")
    print()
    print("  A permutation is not diagonal, so a triple of phases cannot describe")
    print("  it.  The character is read off the principal minors instead:")
    cyclic = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
    values = [elementary_symmetric(order, cyclic) for order in range(4)]
    print(f"    e_p of the cyclic permutation: {[complex(v).real for v in values]}")
    print("    (1, 0, 0, 1: trace zero, determinant one -- and no diagonalising)")
    print()


def what_does_not_carry_over() -> None:
    print("=" * 72)
    print("And what does not carry over")
    print("=" * 72)
    print("  The blow-up count does not.  For an abelian group the twisted")
    print("  sectors are labelled by elements and the projector is the whole")
    print("  group; for a non-abelian one they are labelled by conjugacy classes")
    print("  and the projector is the centralizer, so the moduli are the")
    print("  centralizer's orbits on the fixed locus and not its components.")
    print()
    group = delta27_orbifold()
    classes = group.conjugacy_classes()
    print(f"  {'class':>7s} {'size':>5s} {'centralizer':>12s} {'|G|/|Z|':>8s}")
    for index, entry in enumerate(classes[:6]):
        representative = entry[0]
        centralizer = group.centralizer(representative)
        print(f"  {index:>7d} {len(entry):>5d} {len(centralizer):>12d}"
              f" {group.size // len(centralizer):>8d}")
    print("  ...")
    print()
    print("  The class size times the centralizer order is |G| for every one of")
    print("  them, as the orbit-stabiliser theorem requires.  Computing the")
    print("  orbits themselves needs the fixed points and not just how many there")
    print("  are, so hodge_numbers stays abelian-only and says so rather than")
    print("  returning a number it cannot stand behind:")
    print()
    for name, maker in ABELIAN:
        result = hodge_numbers(maker())
        print(f"    {name:>18s}  {result}")
    print()


def figures() -> None:
    panels = []
    for name, group in (("$Z_3 \\times Z_3$", z3_squared()), ("$\\Delta(27)$", delta27_orbifold())):
        panels.append((name, group.size, group.commuting_pairs()))
    plot_commutation(panels, FIG / "commutation.png")
    print(f"  wrote {FIG / 'commutation.png'}")
    print("    A solid block on the left: every pair of an abelian group commutes.")
    print("    On the right, 297 of 729 -- and 297 is 27 times the eleven classes.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_general_path_agrees()
    what_non_abelian_costs()
    what_still_works()
    what_does_not_carry_over()
    figures()
    print("  (a permutation of the three tori, as a lattice matrix, has trace "
          f"{np.trace(block_permutation((1, 2, 0))):.0f})")
    print("done.")
