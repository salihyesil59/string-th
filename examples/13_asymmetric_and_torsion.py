"""Asymmetric orbifolds and discrete torsion: the two freedoms a quotient has.

Run:  python examples/13_asymmetric_and_torsion.py

``examples/08_orbifold.py`` quotients a torus by a rotation of space.  Two things
were left out there, and both are choices rather than omissions: the twist need
not act the same way on left- and right-movers, and the blocks of the partition
function may be weighted by phases.  Writes
``figures/asymmetric_orbifolds.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.compactification.asymmetric import (  # noqa: E402
    AsymmetricTwist,
    classify_automorphisms,
    is_geometric,
    momentum_action,
    twist_phases,
)
from stringsim.compactification.orbifold import Orbifold  # noqa: E402
from stringsim.compactification.torsion import (  # noqa: E402
    TorsionGroup,
    projected_degeneracies,
    twisted_character,
)
from stringsim.compactification.torus import (  # noqa: E402
    TorusBackground,
    basis_change,
    factorized_duality,
    gauge_algebra,
)

FIG = Path(__file__).resolve().parents[1] / "figures"

BACKGROUNDS = [
    ("self-dual S^1", TorusBackground(np.array([[1.0]]))),
    ("generic S^1", TorusBackground(np.array([[1.7]]))),
    ("self-dual T^2", TorusBackground(np.eye(2))),
    ("hexagonal T^2", TorusBackground(np.array([[2.0, -1.0], [-1.0, 2.0]]))),
    ("half self-dual T^2", TorusBackground(np.diag([1.0, 2.3]))),
    ("self-dual T^3", TorusBackground(np.eye(3))),
]


def t_duality_is_not_a_rotation_of_space() -> None:
    print("=" * 72)
    print("A twist that no motion of the torus can produce")
    print("=" * 72)
    circle = TorusBackground(np.array([[1.0]]))
    duality = factorized_duality(1, 0)
    reflection = basis_change(np.array([[-1.0]]))
    for name, omega in (("T-duality  n <-> w", duality), ("reflection  x -> -x", reflection)):
        left, right = momentum_action(circle, omega)
        print(f"  {name:20s} Omega = {omega.astype(int).tolist()}")
        print(f"  {'':20s} R_L = {left.ravel()[0]:+.0f}, R_R = {right.ravel()[0]:+.0f}"
              f"   geometric: {is_geometric(omega)}")
    print()
    print("  With Z = (w, n) a diffeomorphism sends w -> A w and a B-shift touches")
    print("  only the momentum, so both leave the upper-right block of Omega zero.")
    print("  T-duality is exactly the statement that it does not, and the price is")
    print("  that it turns the two moving sides by different angles.")
    print()


def the_symmetry_group_is_a_lattice_question() -> None:
    print("=" * 72)
    print("Which twists exist: found by search, not by choice")
    print("=" * 72)
    print(f"  {'background':20s} {'|Aut|':>6s} {'geometric':>10s} {'asymmetric':>11s}"
          f"  {'|Aut|/|geom|':>12s}  enhanced algebra")
    for name, background in BACKGROUNDS:
        twists = classify_automorphisms(background)
        geometric = sum(twist.is_geometric for twist in twists)
        algebra = gauge_algebra(background)
        print(f"  {name:20s} {len(twists):>6d} {geometric:>10d} {len(twists) - geometric:>11d}"
              f"  {len(twists) // geometric:>12d}  {algebra.name_left}")
    print()
    print("  H is positive definite, so the group is finite and the search finishes.")
    print("  At the fully self-dual T^d it comes to 2^(2d) d!, of which 2^d d! are")
    print("  geometric -- the signed permutations, Aut of Z^d itself.  The ratio is")
    print("  the order of the Weyl group of one side's enhanced algebra, and where")
    print("  there is no enhancement (hexagonal T^2, generic S^1) the ratio is 1:")
    print("  every symmetry is a motion of the torus and nothing is asymmetric.")
    print()


def level_matching_decides() -> None:
    print("=" * 72)
    print("Level matching, and what it selects")
    print("=" * 72)
    print("  A twisted sector needs L_0 - L_0bar quantised in units of 1/N, so")
    print("  N (a_R - a_L) must be an integer.  Applied to every twist found:")
    print()
    print(f"  {'background':20s} {'asymmetric':>11s} {'matched':>8s} {'equal phases':>13s}")
    for name, background in BACKGROUNDS:
        twists = [t for t in classify_automorphisms(background) if t.is_asymmetric]
        matched = [t for t in twists if t.is_level_matched]
        equal = [t for t in twists if t.has_equal_phases]
        print(f"  {name:20s} {len(twists):>11d} {len(matched):>8d} {len(equal):>13d}")
    print()
    print("  The last two columns agree everywhere: a twist is level-matched exactly")
    print("  when the left and right rotations turn by the same angles.  They are")
    print("  still different rotations -- that is what makes them asymmetric -- but")
    print("  the two sides must share a spectrum.  This came out of the enumeration.")
    print()
    print("  The ones that survive, with an isolated fixed set:")
    print(f"  {'background':20s} {'order':>6s} {'phi_L':>26s}"
          f" {'|det(1-Omega)|':>15s} {'degeneracy':>11s}")
    for name, background in BACKGROUNDS:
        seen = set()
        for twist in classify_automorphisms(background):
            if not (twist.is_asymmetric and twist.is_level_matched):
                continue
            if twist.fixed_points < 1e-9:
                continue
            key = (twist.order, tuple(np.round(twist.left_phases, 4)))
            if key in seen:
                continue
            seen.add(key)
            print(f"  {name:20s} {twist.order:>6d} {str(np.round(twist.left_phases, 3)):>26s}"
                  f" {round(twist.fixed_points):>15d} {twist.twisted_degeneracy:>11d}")
    print()
    print("  Every one of those determinants is a perfect square.  Left- and")
    print("  right-movers each supply half of the fixed-point count on the Narain")
    print("  lattice, so the twisted sector multiplicity is its square root.")
    print()


def a_geometric_twist_cannot_fail() -> None:
    print("=" * 72)
    print("The consistency check against the symmetric machinery")
    print("=" * 72)
    orbifolds = [
        ("Z_3 hexagonal", Orbifold.z3_hexagonal()),
        ("Z_4 square", Orbifold.z4_square()),
        ("Z_6 hexagonal", Orbifold.z6_hexagonal()),
    ]
    for name, orbifold in orbifolds:
        omega = basis_change(orbifold.rotation)
        left, right = momentum_action(orbifold.background, omega)
        twist = AsymmetricTwist(orbifold.background, omega)
        print(f"  {name:16s} order {twist.order}, geometric {twist.is_geometric}, "
              f"phi_L {np.round(twist_phases(left), 4)}, phi_R {np.round(twist_phases(right), 4)}")
        print(f"  {'':16s} a_L = a_R = {twist.left_intercept:.6f}, "
              f"orbifold.intercept = {orbifold.intercept(1):.6f}, "
              f"level-matched {twist.is_level_matched}")
    print()
    print("  A geometric twist turns both sides the same way, so a_L = a_R and the")
    print("  defect is zero identically -- a symmetric orbifold can never fail level")
    print("  matching.  The intercepts also agree with the ones orbifold.py computes")
    print("  from the Hurwitz zeta function, which is the point of checking them here.")
    print()


def torsion_is_derived() -> None:
    print("=" * 72)
    print("Discrete torsion: what phases the blocks may carry")
    print("=" * 72)
    print("  Z[g,h] -> Z[g,gh] under T and Z[g,h] -> Z[h,g^-1] under S, so the")
    print("  weights must be alternating bilinear pairings on G.  Enumerated and")
    print("  tested on every element, never assumed:")
    print()
    print(f"  {'group':16s} {'|G|':>5s} {'pairings found':>15s}"
          f" {'prod gcd(N_i,N_j)':>19s} {'modular':>9s}")
    for orders in [(2,), (6,), (2, 2), (2, 4), (3, 3), (4, 4), (2, 2, 2)]:
        group = TorsionGroup(orders)
        pairings = group.pairings()
        structure = group.torsion_group()
        expected = int(np.prod(structure)) if structure else 1
        modular = all(group.is_modular_consistent(p) for p in pairings)
        name = " x ".join(f"Z_{n}" for n in orders)
        print(f"  {name:16s} {group.size:>5d} {len(pairings):>15d}"
              f" {expected:>19d} {str(modular):>9s}")
    print()
    print("  H^2(G, U(1)) = sum over i<j of Z_gcd(N_i, N_j).  A cyclic group has no")
    print("  pairs, so no torsion: every Z_N orbifold in examples/08 had no choice to")
    print("  make.  The smallest group that does is Z_2 x Z_2, with two classes.")
    print()


def torsion_swaps_the_halves() -> None:
    print("=" * 72)
    print("What the phase does to a twisted sector")
    print("=" * 72)
    group = TorsionGroup((2, 2))
    # Z_2 x Z_2 on T^4: each generator flips a different pair of directions.
    phases = {
        (0, 0): [0.0, 0.0, 0.0, 0.0],
        (1, 0): [0.5, 0.5, 0.0, 0.0],
        (0, 1): [0.0, 0.0, 0.5, 0.5],
        (1, 1): [0.5, 0.5, 0.5, 0.5],
    }
    trivial, torsion = group.pairings()
    names = ["1", "g1", "g2", "g1 g2"]
    for label, element in (("untwisted", (0, 0)), ("g1-twisted", (1, 0))):
        weights = [round(w.real) for w in group.projector_weights(torsion, element)]
        print(f"  {label:12s} projector weights with torsion: "
              f"{dict(zip(names, weights, strict=True))}")
    print()
    n_max = 10
    print(f"  counts at q^(k/2), k = 0 .. {n_max}")
    print()
    untwisted = [
        projected_degeneracies(group, phases, (0, 0), pairing, n_max)
        for pairing in (trivial, torsion)
    ]
    print(f"  untwisted    no torsion  {untwisted[0].tolist()}")
    print(f"  {'':12s} torsion     {untwisted[1].tolist()}")
    print(f"  {'':12s} identical: {np.array_equal(*untwisted)}")
    print()
    for element, label in (((1, 0), "g1-twisted"), ((1, 1), "g1g2-twisted")):
        without = projected_degeneracies(group, phases, element, trivial, n_max)
        with_it = projected_degeneracies(group, phases, element, torsion, n_max)
        # projecting with <g> alone: only the identity and g survive the average
        half = np.rint(
            0.5
            * (
                twisted_character(phases[element], phases[(0, 0)], n_max).real
                + twisted_character(phases[element], phases[element], n_max).real
            )
        ).astype(int)
        print(f"  {label:12s} no torsion  {without.tolist()}")
        print(f"  {'':12s} torsion     {with_it.tolist()}")
        print(f"  {'':12s} sum         {(without + with_it).tolist()}")
        print(f"  {'':12s} <g>-only    {half.tolist()}")
        print(f"  {'':12s} sum equals the <g> projection: "
              f"{np.array_equal(without + with_it, half)}")
        print()
    print("  The untwisted sector does not move: epsilon(1, h) = 1 for every h, so")
    print("  discrete torsion can never change the untwisted massless spectrum.")
    print("  In a twisted sector the weights become (1, 1, -1, -1) and the two")
    print("  projections keep complementary halves -- their sum is the projection by")
    print("  <g> alone, which is what 'the other half' means as a statement.")
    print()
    print("  These are oscillator counts.  The fixed-point multiplicities and the")
    print("  phases the group acts with on them are not included, so this is the")
    print("  mechanism behind (h11, h21) = (51, 3) <-> (3, 51) rather than that")
    print("  number itself.")
    print()


def figures() -> None:
    from stringsim.viz.plots import plot_twist_classification

    rows = []
    for name, background in BACKGROUNDS:
        twists = classify_automorphisms(background)
        geometric = sum(t.is_geometric for t in twists)
        asymmetric = [t for t in twists if t.is_asymmetric]
        matched = sum(t.is_level_matched for t in asymmetric)
        rows.append((name, geometric, len(asymmetric) - matched, matched))
    plot_twist_classification(rows, FIG / "asymmetric_orbifolds.png")
    print(f"  wrote {FIG / 'asymmetric_orbifolds.png'}")
    print("  Backgrounds with no enhanced symmetry have no asymmetric twists at all;")
    print("  where the symmetry is enhanced they outnumber the geometric ones, and")
    print("  level matching then throws most of them away.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    t_duality_is_not_a_rotation_of_space()
    the_symmetry_group_is_a_lattice_question()
    level_matching_decides()
    a_geometric_twist_cannot_fail()
    torsion_is_derived()
    torsion_swaps_the_halves()
    figures()
    print("done.")
