"""Mirror symmetry: a polytope, its dual, and the two Hodge numbers swapping.

Run:  python examples/29_mirror_symmetry.py

``examples/17`` computes Hodge numbers for orbifolds of a torus and finds
(51, 3) and (3, 51) exchanged by a phase.  Calabi-Yau manifolds that are
*hypersurfaces* need different machinery, and there mirror symmetry is
Batyrev's: a reflexive lattice polytope gives a Calabi-Yau, its dual gives
another, and the two Hodge numbers swap.

The exchange is not discovered here -- it is the same formula applied to the two
polytopes, so it is the shape of the combinatorics.  What has to be checked is
that the numbers are right, and the Euler characteristic from Chern classes,
which has no polytope in it, is what checks them.

Writes ``figures/mirror_hodge.png``, ``figures/reflexive_duality.png`` and
``figures/mirror_plot.gif``.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.compactification.toric import (  # noqa: E402
    batyrev_count,
    greene_plesser_group,
    greene_plesser_order,
    hodge_numbers,
    hypersurface_euler,
    projective_space,
    weighted_projective,
)
from stringsim.viz.animate import animate_mirror_plot  # noqa: E402
from stringsim.viz.plots import plot_mirror_hodge, plot_reflexive_duality  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
FAMOUS = ([1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 1, 4], [1, 1, 1, 2, 5], [1, 1, 1, 6, 9])


def the_polytope() -> None:
    """What a reflexive polytope is, and how the quintic appears as one."""
    print("The polytope")
    print("-" * 70)
    print("  P^4 has rays e_1..e_4 and -(e_1+..+e_4).  Their convex hull is")
    print("  reflexive: the dual is a lattice polytope too.")
    print()
    delta_star = projective_space(4)
    delta = delta_star.dual()
    print(f"    Delta*: {len(delta_star.vertices)} vertices, "
          f"{len(delta_star.lattice_points)} lattice points")
    print(f"    Delta : {len(delta.vertices)} vertices, "
          f"{len(delta.lattice_points)} lattice points")
    print("  The 126 are the degree-5 monomials in five variables, C(9,4).")
    print()
    print("  Faces come from grouping the lattice points by which facets they")
    print("  sit on -- no recursion over a face lattice, and the interior counts")
    print("  Batyrev's sums want are already there:")
    for name, polytope in (("Delta*", delta_star), ("Delta ", delta)):
        counts = {
            dimension: len(polytope.faces_of_dimension(dimension))
            for dimension in range(polytope.dim + 1)
        }
        carried = {d: n for d, n in counts.items() if n}
        print(f"    {name}: faces with interior points, by dimension: {carried}")
    print("  The simplex's facets are unimodular and carry nothing inside them,")
    print("  so they do not appear at all -- which is exactly what the sums want.")
    print()


def the_quintic() -> None:
    """The numbers, and a check from the other side."""
    print("The quintic and its mirror")
    print("-" * 70)
    quintic = hodge_numbers(projective_space(4))
    mirror = hodge_numbers(projective_space(4).dual())
    print(f"    from Delta*:  {quintic}")
    print(f"    from Delta :  {mirror}")
    print()
    print("  The exchange is the shape of the formula, not a discovery: the same")
    print("  function is applied to a polytope and to its dual.  What is worth")
    print("  checking is the numbers, and chi comes independently from")
    print("  c(T) = (1 + H)^5 / (1 + 5H) restricted to the hypersurface:")
    print(f"    2 (h11 - h21) = {quintic.euler}")
    print(f"    integral of c_3 = {hypersurface_euler(5, 4)}")
    print()
    print("      family              (h11, h21)         chi     mirror")
    for weights in FAMOUS:
        numbers = hodge_numbers(weighted_projective(weights))
        label = f"P{tuple(weights)}[{sum(weights)}]"
        print(f"    {label:<20s} ({numbers.h11:3d}, {numbers.h21:3d})   {numbers.euler:+6d}     "
              f"({numbers.h21:3d}, {numbers.h11:3d})")
    print()


def where_it_stops() -> None:
    """The formula in three dimensions is not computing a Hodge number."""
    print("Where the formula stops")
    print("-" * 70)
    quartic = projective_space(3)
    print("  Every K3 has h11 = 20.  For the quartic in P^3 the count gives")
    here, there = batyrev_count(quartic), batyrev_count(quartic.dual())
    print(f"    from Delta*: {here}     from Delta: {there}")
    print(f"  and the independent chi is {hypersurface_euler(4, 3)}, which for a surface")
    print("  with h20 = 1 forces b_2 = 22 and h11 = 20.  So those two numbers")
    print("  are Picard numbers -- the part of H^{1,1} the toric divisors reach.")
    print()
    print("  They sum to 20 for the simplest families and not for all:")
    print("      weights           count + mirror")
    for weights in ([1, 1, 1, 1], [1, 1, 1, 3], [1, 1, 4, 6], [1, 1, 2, 4], [1, 2, 2, 5]):
        polytope = weighted_projective(weights)
        if not polytope.is_reflexive:
            continue
        first, second = batyrev_count(polytope), batyrev_count(polytope.dual())
        flag = "" if first + second == 20 else "   <-- not 20"
        print(f"    P{tuple(weights)}[{sum(weights)}]{'':<{max(0, 8 - len(weights))}}  "
              f"{first:3d} + {second:3d} = {first + second}{flag}")
    print("  The difference is divisors no polytope sees, and it is why")
    print("  hodge_numbers refuses to run outside four dimensions.")
    print()


def greene_and_plesser() -> None:
    """The mirror as an orbifold, which is where it was found first."""
    print("Greene and Plesser")
    print("-" * 70)
    print("  Before Batyrev the mirror of the quintic was found as a quotient:")
    print("  the phase symmetries preserving the holomorphic three-form.")
    print()
    print("      family              phases   scalings   group order")
    for weights in ([1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 2, 5], [1, 1, 1, 1]):
        elements, scalings = greene_plesser_group(weights)
        label = f"P{tuple(weights)}[{sum(weights)}]"
        print(f"    {label:<20s} {len(elements):6d}   {scalings:8d}   "
              f"{greene_plesser_order(weights):11d}")
    print(f"  For the quintic that is 5^4 / 5 = {greene_plesser_order([1, 1, 1, 1, 1])} = 5^3.")
    print()
    print("  The scalings are enumerated rather than counted by a formula.  The")
    print("  obvious guess -- the smallest exponent -- is right for the quintic")
    print("  and wrong for P(1,1,1,2,5), where the exponents are (10,10,10,5,2)")
    print("  and there are 10 scalings, not 2.")
    print()
    print("  That the quotient is the same manifold as the dual polytope is a")
    print("  theorem.  It is not computed here: the orbifold cohomology of the")
    print("  quotient would need the fixed loci, and this module builds none.")
    print()


def scan() -> list:
    """Enough families for the picture."""
    print("A scan")
    print("-" * 70)
    found = []
    for extra in itertools.combinations_with_replacement(range(1, 13), 4):
        weights = [1, *extra]
        degree = sum(weights)
        if any(degree % w for w in weights):
            continue
        polytope = weighted_projective(weights)
        if not polytope.is_reflexive:
            continue
        numbers = hodge_numbers(polytope)
        found.append((tuple(weights), numbers.h11, numbers.h21))
    print(f"  {len(found)} weighted projective families with a weight of 1 and")
    print(f"  every weight dividing the degree.  h11 up to {max(a for _, a, _ in found)}, "
          f"h21 up to {max(b for _, _, b in found)}.")
    print()
    return found


def figures(found: list) -> None:
    print("Figures")
    print("-" * 70)

    points = [(a, b) for _, a, b in found]
    highlights = [
        ("quintic", 1, 101),
        (r"$P(1,1,1,6,9)$", 2, 272),
    ]
    plot_mirror_hodge(points, highlights, FIG / "mirror_hodge.png")
    print(f"  wrote {FIG / 'mirror_hodge.png'}")

    panels = []
    for weights in ([1, 1, 1], [1, 1, 2], [1, 2, 3]):
        polytope = weighted_projective(weights)
        dual = polytope.dual()
        panels.append((
            f"$P{tuple(weights)}$",
            polytope.vertices,
            polytope.lattice_points,
            dual.vertices,
            dual.lattice_points,
        ))
    plot_reflexive_duality(panels, FIG / "reflexive_duality.png")
    print(f"  wrote {FIG / 'reflexive_duality.png'}")

    frames = []
    for cut in range(1, len(found) + 1):
        weights, _, _ = found[cut - 1]
        frames.append((f"P{weights}", [(a, b) for _, a, b in found[:cut]]))
    animate_mirror_plot(frames, FIG / "mirror_plot.gif")
    print(f"  wrote {FIG / 'mirror_plot.gif'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_polytope()
    the_quintic()
    where_it_stops()
    greene_and_plesser()
    found = scan()
    figures(found)
    print("done.")
