r"""Mirror symmetry, as an exchange of a polytope with its dual.

A reflexive lattice polytope gives a Calabi-Yau hypersurface, its dual gives
another, and Batyrev's formula computes :math:`h^{1,1}` from the first and
:math:`h^{2,1}` from the second.  The exchange is therefore not a discovery
here: it is the same expression applied to the two polytopes, so mirror symmetry
is the shape of the combinatorics.

What is left to check is whether the numbers are right, and for the quintic the
Euler characteristic checks them from a side with no polytope in it at all:
:math:`c(T) = (1+H)^{5}/(1+5H)` restricted to the hypersurface integrates to
:math:`-200`, and :math:`2(h^{1,1} - h^{2,1})` is :math:`-200`.

For no other family here, and the panel says so rather than pretending.  That
formula is for a smooth hypersurface in *ordinary* projective space, where
Calabi-Yau forces degree = n + 1; the weighted families sit in singular ambient
spaces and the manifold is a resolution.  Running it on ``P(1,1,1,1,2)[6]``
returns :math:`-516` against Batyrev's :math:`-204`, which is an answer about a
different manifold rather than a disagreement about this one.

The Greene-Plesser group is enumerated beside it -- the phase symmetries that
preserve the holomorphic three-form, divided by the projective scalings -- which
is how the quintic's mirror was found before Batyrev.  That the quotient and the
dual polytope are the same manifold is a theorem, and it is not computed here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from ...compactification.toric import (
    HodgeNumbers,
    greene_plesser_group,
    greene_plesser_order,
    hodge_numbers,
    hypersurface_euler,
    weighted_projective,
)
from ...viz.plots import plot_mirror_hodge
from ..panel import Choice, Line

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["MirrorPanel", "Mirror", "FAMILIES"]

FAMILIES: dict[str, tuple[int, ...]] = {
    "P(1,1,1,1,1)[5]": (1, 1, 1, 1, 1),
    "P(1,1,1,1,2)[6]": (1, 1, 1, 1, 2),
    "P(1,1,1,1,4)[8]": (1, 1, 1, 1, 4),
    "P(1,1,1,2,5)[10]": (1, 1, 1, 2, 5),
    "P(1,1,1,6,9)[18]": (1, 1, 1, 6, 9),
    "P(1,1,2,2,2)[8]": (1, 1, 2, 2, 2),
    "P(1,1,2,2,6)[12]": (1, 1, 2, 2, 6),
}
"""The one-parameter models, plus two more with a larger Picard number."""

_SIDES = ("the polytope", "its dual")


@lru_cache(maxsize=1)
def _scan() -> tuple[tuple[int, int], ...]:
    """Every reflexive weighted projective family with a weight of 1.

    Cached: it takes about a second and does not depend on any control, and it
    is the cloud the chosen family is marked against.
    """
    found: list[tuple[int, int]] = []
    for extra in itertools.combinations_with_replacement(range(1, 13), 4):
        weights = [1, *extra]
        degree = sum(weights)
        if any(degree % w for w in weights):
            continue
        polytope = weighted_projective(weights)
        if not polytope.is_reflexive:
            continue
        # One entry per family.  ``plot_mirror_hodge`` draws each point and its
        # mirror itself, so adding the swap here would draw everything twice.
        numbers = hodge_numbers(polytope)
        found.append((numbers.h11, numbers.h21))
    return tuple(found)


@dataclass(frozen=True)
class Mirror:
    """One family, its mirror, and the Euler characteristic from two sides."""

    label: str
    weights: tuple[int, ...]
    side: str
    numbers: HodgeNumbers
    mirror: HodgeNumbers
    chern_euler: int | None
    """``int c_3``, or ``None`` where the closed form does not apply.

    :func:`~stringsim.compactification.toric.hypersurface_euler` is for a smooth
    degree-``d`` hypersurface in ordinary ``P^n``, and Calabi-Yau there means
    ``d = n + 1``.  Among these families only the quintic satisfies that: the
    rest live in *weighted* projective spaces, which are singular, and the
    manifold is a resolution the formula knows nothing about.  Applying it
    anyway would compare ``-204`` against ``-516`` and call it a failure of
    Batyrev's count.
    """
    lattice_points: int
    dual_points: int
    phases: int
    scalings: int
    group_order: int
    cloud: tuple[tuple[int, int], ...]

    @property
    def degree(self) -> int:
        return sum(self.weights)

    @property
    def has_chern_check(self) -> bool:
        """True only where the ambient space is ordinary projective space."""
        return self.chern_euler is not None

    @property
    def euler_agrees(self) -> bool:
        if self.chern_euler is None:
            return False
        return self.chern_euler in (self.numbers.euler, self.mirror.euler)

    @property
    def swapped(self) -> bool:
        return (self.mirror.h11, self.mirror.h21) == (self.numbers.h21, self.numbers.h11)


class MirrorPanel:
    """The same formula on a polytope and on its dual, checked where it can be."""

    title = "Mirror symmetry"
    blurb = (
        "A reflexive lattice polytope gives a Calabi-Yau hypersurface and its dual gives "
        "another.  Batyrev's count applied to one gives h^{1,1} and applied to the other "
        "gives h^{2,1}, so the exchange is the shape of the formula rather than something "
        "discovered in it.  What has to be checked is whether the numbers are right, and "
        "for the quintic the Euler characteristic from Chern classes -- which has no "
        "polytope in it -- does that.  For the weighted families there is no second "
        "route and the readout drops its verdict rather than inventing one.  Each family "
        "is plotted with its mirror, which is why the cloud is symmetric about chi = 0."
    )
    background = (
        "A Calabi-Yau three-fold has two independent Hodge numbers, h^{1,1} counting "
        "Kaehler deformations -- ways of changing sizes -- and h^{2,1} counting complex "
        "structure deformations, ways of changing shape.  Mirror symmetry is the claim "
        "that string theory on one manifold is the same theory as on another with the "
        "two exchanged, so a hard question about sizes becomes an easy one about shapes.",
        "Batyrev's construction makes that concrete for hypersurfaces.  Take a lattice "
        "polytope whose dual is also a lattice polytope -- reflexive -- and a "
        "Calabi-Yau lives in the toric variety it defines.  Then h^{1,1} is "
        "l(Delta*) - (d+1) - sum over facets of their interior points, plus a "
        "correction from codimension-two faces, and h^{2,1} is the identical expression "
        "with the two polytopes swapped.",
        "So the exchange is free and the numbers are not.  The check comes from the "
        "classical side: the total Chern class of a degree-d hypersurface in P^4 is "
        "(1+H)^5/(1+dH) restricted to it, and integrating c_3 gives the Euler "
        "characteristic.  For the quintic that is -200, and 2(h^{1,1} - h^{2,1}) from "
        "counting lattice points on faces is also -200.  Nothing in the polytope "
        "computation knows about Chern classes.",
        "That check exists for the quintic and for no other family here, which is worth "
        "being plain about.  The formula is for a smooth hypersurface in ordinary "
        "projective space, and Calabi-Yau there forces degree = n + 1.  The weighted "
        "families live in singular ambient spaces and the manifold is a resolution, so "
        "the same formula applied to P(1,1,1,1,2)[6] returns -516 against Batyrev's "
        "-204 -- a statement about the wrong manifold rather than a disagreement.  The "
        "readout drops its verdict there instead of reporting a failure.",
        "Greene and Plesser found the quintic's mirror before any of this, as a "
        "quotient: the phase symmetries x_i -> exp(2 pi i a_i / 5) x_i preserving the "
        "holomorphic three-form, of which there are 5^4, divided by the 5 projective "
        "scalings -- a group of order 125.  Both counts are enumerated here rather than "
        "quoted, because the obvious shortcut for the scalings is wrong in general.",
        "That the Greene-Plesser quotient and the dual polytope are the same manifold "
        "is a theorem, and it is not computed here: it would need the orbifold "
        "cohomology of the quotient, which needs the fixed loci, and this module builds "
        "none.  The panel says so rather than implying more than it has.",
    )
    suggestions = (
        "Switch between the polytope and its dual.  The two Hodge numbers swap and the "
        "Euler characteristic changes sign; the marked point jumps across the axis to "
        "its mirror.",
        "Walk through the families.  The quintic is (1, 101); P(1,1,1,6,9) is (2, 272), "
        "the largest here.  Every one of them is drawn together with its mirror, which "
        "is why the cloud is symmetric.",
        "Read the Euler line on the quintic: the left number counts lattice points on "
        "faces of a polytope, the right integrates a Chern class, and neither "
        "computation contains the other.  Switch family and the second route "
        "disappears rather than disagreeing -- it is a formula for hypersurfaces in "
        "ordinary projective space, and these ambient spaces are weighted.",
        "Look at the Greene-Plesser row for P(1,1,1,2,5).  The exponents are "
        "(10,10,10,5,2) and there are 10 scalings, not 2 -- the smallest exponent is "
        "the right answer for the quintic and wrong here, which is why they are counted.",
    )
    controls = (
        Choice("label", "family", tuple(FAMILIES), "P(1,1,1,1,1)[5]"),
        Choice("side", "computed from", _SIDES, _SIDES[0]),
    )

    def compute(
        self,
        label: str = "P(1,1,1,1,1)[5]",
        side: str = _SIDES[0],
    ) -> Mirror:
        weights = FAMILIES[label]
        polytope = weighted_projective(list(weights))
        dual = polytope.dual()
        if side == _SIDES[1]:
            polytope, dual = dual, polytope

        elements, scalings = greene_plesser_group(list(weights))
        return Mirror(
            label=label,
            weights=weights,
            side=side,
            numbers=hodge_numbers(polytope),
            mirror=hodge_numbers(dual),
            chern_euler=(
                hypersurface_euler(sum(weights), 4)
                if set(weights) == {1}
                else None
            ),
            lattice_points=len(polytope.lattice_points),
            dual_points=len(dual.lattice_points),
            phases=len(elements),
            scalings=scalings,
            group_order=greene_plesser_order(list(weights)),
            cloud=_scan(),
        )

    def draw(self, result: Mirror) -> Figure:
        """The Hodge plot over every family, with this one and its mirror marked."""
        # One highlight, not two: the plot already labels a family on both
        # sides of the axis, so naming the mirror as well puts two strings on
        # top of each other in each place.
        highlights = [(result.label, result.numbers.h11, result.numbers.h21)]
        return plot_mirror_hodge(list(result.cloud), highlights, path=None)

    def readout(self, result: Mirror) -> list[Line]:
        return [
            Line("family", f"{result.label}, degree {result.degree}",
                 f"computed from {result.side}"),
            Line("lattice points", f"{result.lattice_points} in it",
                 f"{result.dual_points} in the dual"),
            Line(
                "Hodge numbers",
                f"(h11, h21) = ({result.numbers.h11}, {result.numbers.h21})",
                f"the mirror: ({result.mirror.h11}, {result.mirror.h21})",
                ok=result.swapped,
            ),
            # The Chern-class route is for a smooth hypersurface in ordinary
            # projective space, and Calabi-Yau there means degree = n + 1.  Only
            # the quintic among these families qualifies.  Comparing anyway
            # would put -204 against -516 and blame Batyrev's count.
            Line(
                "Euler characteristic",
                f"2(h11 - h21) = {result.numbers.euler:+d}",
                f"integral of c_3 = {result.chern_euler:+d}",
                ok=result.euler_agrees,
            )
            if result.has_chern_check
            else Line(
                "Euler characteristic",
                f"2(h11 - h21) = {result.numbers.euler:+d}",
                "no independent route here: the ambient space is weighted",
            ),
            Line(
                "Greene-Plesser group",
                f"{result.phases} phases / {result.scalings} scalings = "
                f"{result.group_order}",
                "both enumerated, neither counted by a formula",
            ),
            Line(
                "families plotted",
                f"{len(result.cloud)}, each drawn with its mirror",
                "symmetric about chi = 0 by construction, not by discovery",
            ),
        ]

    def notes(self, result: Mirror) -> list[str]:
        """What this family is, and what the check does and does not establish."""
        out = [
            f"{result.label} is a degree-{result.degree} hypersurface, and from "
            f"{result.side} Batyrev's count gives ({result.numbers.h11}, "
            f"{result.numbers.h21}).  The mirror is ({result.mirror.h11}, "
            f"{result.mirror.h21}) -- but that exchange is not evidence of anything.  "
            "The same function is being applied to a polytope and to its dual, so the "
            "swap is arithmetic rather than physics."
        ]
        if result.euler_agrees:
            out.append(
                f"The evidence is the Euler characteristic.  Counting lattice points on "
                f"faces gives 2(h11 - h21) = {result.numbers.euler:+d}, and integrating "
                f"c_3 of the hypersurface -- (1+H)^5/(1+{result.degree}H) restricted to "
                f"it, with no polytope anywhere -- gives {result.chern_euler:+d}.  Those "
                "are the same number, which is what makes the Hodge numbers believable "
                "rather than merely symmetric."
            )
        else:
            out.append(
                "There is no second route to the Euler characteristic for this family, "
                "and the panel does not manufacture one.  The Chern-class computation is "
                "for a smooth degree-d hypersurface in ordinary P^n, where Calabi-Yau "
                "means d = n + 1 -- true of the quintic and of nothing else here.  These "
                "weighted ambient spaces are singular and the manifold is a resolution "
                "of the hypersurface in them, which that formula knows nothing about.  "
                "Running it anyway would compare -204 against -516 and report a failure "
                "of Batyrev's count rather than of the comparison."
            )
        out.append(
            f"Before Batyrev, the mirror of a family like this was built as a quotient.  "
            f"The phases preserving the holomorphic form number {result.phases} and the "
            f"projective scalings number {result.scalings}, leaving a group of order "
            f"{result.group_order}.  Both are enumerated: the obvious shortcut for the "
            "scalings, taking the smallest exponent, is right for the quintic and wrong "
            "for P(1,1,1,2,5), where the exponents are (10,10,10,5,2) and there are ten."
        )
        out.append(
            "What is not shown here: that the Greene-Plesser quotient and the dual "
            "polytope are the same manifold.  That is a theorem.  Checking it would "
            "need the orbifold cohomology of the quotient, which needs the fixed loci, "
            "and nothing in this package builds them.  The two constructions are "
            "presented side by side because they agree on the numbers, not because "
            "their equivalence has been demonstrated."
        )
        return out
