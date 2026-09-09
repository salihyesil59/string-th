r"""An orbifold's fixed points, and the twisted strings that live on them.

Quotient a torus by a rotation of finite order and the points it leaves alone
become singular.  A closed string can then be closed only up to that rotation --
:math:`X(\sigma + 2\pi) = \theta^k X(\sigma)` -- and such a string has nowhere
to go but a fixed point, because only there does the twisted boundary condition
have a solution.  Those are the twisted sectors, and they are not optional: the
theory is inconsistent without them.

Two numbers are computed twice each.

The fixed points are counted by :math:`|\det(1 - \theta^k)|` and, separately,
enumerated as lattice positions and drawn.  Four for :math:`Z_2`, three for
:math:`Z_3`, two for :math:`Z_4` at :math:`k=1`, one for :math:`Z_6`.

The twisted ground-state energy comes from the closed form
:math:`a_k = 1 - \tfrac14\sum_j \phi_j(1-\phi_j)` and, separately, from
:func:`~stringsim.quantum.zeta.regularised_shifted_sum`, which measures the
Hurwitz value :math:`\zeta(-1,\phi)` numerically by fitting the small-``eps``
expansion of a damped mode sum.  A boson twisted by :math:`e^{2\pi i\phi}` has
modes at :math:`n + \phi`, so its zero-point energy is
:math:`\tfrac12\zeta(-1,\phi)` rather than :math:`\tfrac12\zeta(-1,1)`, and the
two routes to :math:`a_k` share no code at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...compactification.orbifold import (
    Orbifold,
    TwistedLevel,
    twisted_spectrum,
    untwisted_degeneracy,
)
from ...compactification.torus import TorusBackground
from ...quantum.zeta import regularised_shifted_sum
from ...units import Conventions
from ...viz.plots import plot_fixed_points
from ..panel import Choice, Integer, Line

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["OrbifoldPanel", "OrbifoldResult", "ORBIFOLDS"]

ORBIFOLDS: tuple[str, ...] = (
    "T^2/Z_2",
    "T^2/Z_3 hexagonal",
    "T^2/Z_4 square",
    "T^2/Z_6 hexagonal",
)
"""Two-dimensional and crystallographic, which is the whole list at ``d = 2``.

A lattice automorphism of finite order can only have order 1, 2, 3, 4 or 6 --
the crystallographic restriction -- so this is not a selection, it is all of
them.
"""


def _build(label: str, conventions: Conventions) -> Orbifold:
    if label == "T^2/Z_2":
        return Orbifold.inversion(TorusBackground(np.eye(2), conventions=conventions))
    if label == "T^2/Z_3 hexagonal":
        return Orbifold.z3_hexagonal(conventions)
    if label == "T^2/Z_4 square":
        return Orbifold.z4_square(conventions)
    return Orbifold.z6_hexagonal(conventions)


def _intercept_from_zeta(orbifold: Orbifold, sector: int) -> float:
    r"""``a_k`` rebuilt from a numerically measured :math:`\zeta(-1,\phi)`.

    ``zeta(-1, a) = -(a^2 - a + 1/6)/2``, so a direction with phase ``phi``
    contributes ``-zeta(-1, phi)/2 - 1/24`` to the intercept and an untouched
    direction contributes nothing.  Nothing here evaluates the closed form the
    orbifold module uses: the Hurwitz value comes out of a damped mode sum.
    """
    total = 1.0
    for phase in orbifold.twist_phases(sector):
        shift = float(phase) if phase > 1e-12 else 1.0
        total += -0.5 * regularised_shifted_sum(shift).value - 1.0 / 24.0
    return total


@dataclass(frozen=True)
class OrbifoldResult:
    """One orbifold and one twisted sector, with both routes to each number."""

    label: str
    order: int
    sector: int
    requested_sector: int
    phases: tuple[float, ...]
    fixed_from_determinant: int
    fixed_from_enumeration: int
    intercept: float
    intercept_from_zeta: float
    levels: tuple[TwistedLevel, ...]
    untwisted_massless: int
    torus_massless: int
    orbifold: Orbifold

    @property
    def sector_was_clamped(self) -> bool:
        return self.sector != self.requested_sector

    @property
    def fixed_points_agree(self) -> bool:
        return self.fixed_from_determinant == self.fixed_from_enumeration

    @property
    def intercept_gap(self) -> float:
        return abs(self.intercept - self.intercept_from_zeta)

    @property
    def ground_state(self) -> TwistedLevel | None:
        return self.levels[0] if self.levels else None


class OrbifoldPanel:
    """Where a twisted string can be, and how heavy it is when it is there."""

    title = "Orbifold fixed points"
    blurb = (
        "Quotient a torus by a rotation and the points it leaves alone become singular.  "
        "A string closed only up to that rotation has nowhere to sit but one of them, so "
        "each fixed point carries its own tower of twisted states.  The picture is the "
        "fundamental cell drawn at its true angles -- 60 degrees for the hexagonal "
        "lattice, 90 for the square one -- with the fixed points of theta^k marked.  Their "
        "number is |det(1 - theta^k)|, and the readout also counts them by enumeration."
    )
    background = (
        "An orbifold is a torus with points identified under a finite rotation.  Where "
        "the rotation has a fixed point the quotient is not a manifold: the space has a "
        "conical singularity there, and a field theory on it would be ill-defined.  A "
        "string theory is not, and the reason is the twisted sectors.",
        "On a circle a closed string satisfies X(sigma + 2 pi) = X(sigma).  On an "
        "orbifold it may instead satisfy X(sigma + 2 pi) = theta^k X(sigma), which is "
        "still a closed string in the quotient because the two ends are identified "
        "there.  Such a string cannot be moved off a fixed point: away from one, the "
        "twisted boundary condition has no solution.  So each fixed point carries a "
        "tower, and the multiplicity of every twisted state is the number of points.",
        "The twisting changes the mode numbers.  A boson rotated by exp(2 pi i phi) has "
        "creation operators at n + phi rather than n, which moves the zero-point energy: "
        "instead of half of zeta(-1) = -1/12 per direction, it is half of the Hurwitz "
        "value zeta(-1, phi).  The ground state of the twisted sector is therefore "
        "higher or lower than the untwisted one by a definite amount, and that amount is "
        "what the readout computes two ways.",
        "The untwisted sector loses states rather than gaining them.  Only combinations "
        "invariant under theta survive the projection, and the count is the character "
        "sum (1/N) sum_k Tr_k(q^N_L) Tr_k(q^N_R).  That it comes out a non-negative "
        "integer is not automatic -- a sign or a conjugation error breaks exactly that "
        "-- so the value being an integer is itself a check.",
        "Only orders 1, 2, 3, 4 and 6 are possible for a lattice automorphism in two "
        "dimensions.  The crystallographic restriction is why this list is not a "
        "selection of examples but all of them.",
    )
    suggestions = (
        "Step through the four orbifolds.  Z_2 has four fixed points, Z_3 three, Z_4 two "
        "and Z_6 one -- and the cell they are drawn in changes shape with the lattice, "
        "because the rotation has to be a symmetry of it.",
        "On Z_4 and Z_6, raise the sector.  theta^2 on the square lattice is the "
        "inversion, so k = 2 has four fixed points where k = 1 had two; the extra ones "
        "are fixed by theta^2 without being fixed by theta.",
        "Watch the twisted ground state as the orbifold changes.  It is tachyonic in "
        "every one of these -- this is still the bosonic string -- but by different "
        "amounts, because the intercept depends on the twist phases.",
        "Compare the two intercept numbers.  One is a closed form in the phases; the "
        "other fits the small-epsilon expansion of a damped mode sum to extract a "
        "Hurwitz zeta value.  They agree to about 1e-8, which is the numerical route's "
        "precision rather than a disagreement.",
    )
    controls = (
        Choice("label", "orbifold", ORBIFOLDS, ORBIFOLDS[0]),
        Integer("sector", "twisted sector  k", 1, 5, 1),
        Integer("n_levels", "levels shown", 2, 6, 4),
    )

    def compute(
        self,
        label: str = ORBIFOLDS[0],
        sector: int = 1,
        n_levels: int = 4,
    ) -> OrbifoldResult:
        conv = Conventions()
        orbifold = _build(label, conv)
        # A Z_2 has no sector 3.  Rather than refuse a combination the two
        # controls can reach on their own, the panel uses the nearest sector
        # that exists and says in the readout that it did.
        used = min(int(sector), orbifold.order - 1)

        return OrbifoldResult(
            label=label,
            order=orbifold.order,
            sector=used,
            requested_sector=int(sector),
            phases=tuple(float(p) for p in orbifold.twist_phases(used)),
            fixed_from_determinant=orbifold.fixed_points(used),
            fixed_from_enumeration=len(orbifold.fixed_point_positions(used)),
            intercept=orbifold.intercept(used),
            intercept_from_zeta=_intercept_from_zeta(orbifold, used),
            levels=tuple(twisted_spectrum(orbifold, used, n_levels)),
            untwisted_massless=untwisted_degeneracy(orbifold, 1, 1),
            torus_massless=conv.transverse_dim**2,
            orbifold=orbifold,
        )

    def draw(self, result: OrbifoldResult) -> Figure:
        """The fundamental cell with every sector's fixed points on it."""
        sectors = tuple(range(1, result.order))
        return plot_fixed_points(
            result.orbifold,
            path=None,
            sectors=sectors,
            title=f"Fixed points of ${result.label.split()[0].replace('^2', '^{2}')}$",
        )

    def readout(self, result: OrbifoldResult) -> list[Line]:
        ground = result.ground_state
        lines = [
            Line(
                "orbifold",
                f"{result.label}, order {result.order}",
                f"sector k = {result.sector}"
                + (
                    f" (k = {result.requested_sector} does not exist here)"
                    if result.sector_was_clamped
                    else ""
                ),
            ),
            Line("twist phases", ", ".join(f"{p:.4f}" for p in result.phases)),
            Line(
                "fixed points",
                f"{result.fixed_from_determinant} from |det(1 - theta^k)|",
                f"{result.fixed_from_enumeration} enumerated and drawn",
                ok=result.fixed_points_agree,
            ),
            Line(
                "twisted intercept a_k",
                f"{result.intercept:.8f} in closed form",
                f"{result.intercept_from_zeta:.8f} from a measured zeta(-1, phi)",
                ok=result.intercept_gap < 1e-6,
            ),
            Line("  difference", f"{result.intercept_gap:.2e}",
                 "the numerical route's precision"),
            Line(
                "untwisted massless states",
                f"{result.untwisted_massless} survive the projection",
                f"of the torus's {result.torus_massless}",
            ),
        ]
        if ground is not None:
            lines.append(
                Line(
                    "twisted ground state",
                    f"alpha' M^2 = {ground.alpha_m2:+.6f}"
                    + ("   tachyonic" if ground.is_tachyonic else ""),
                    f"{ground.oscillator_states} oscillator state(s) at each of "
                    f"{ground.fixed_points} points",
                )
            )
            lines.append(
                Line(
                    "levels",
                    "  ".join(
                        f"N={str(level.level)}: {level.alpha_m2:+.3f} x{level.degeneracy}"
                        for level in result.levels[:4]
                    ),
                )
            )
        return lines

    def notes(self, result: OrbifoldResult) -> list[str]:
        """What this quotient does to the torus, and what has to be added back."""
        out: list[str] = []
        if result.sector_was_clamped:
            out.append(
                f"There is no sector k = {result.requested_sector} on an orbifold of "
                f"order {result.order}: theta^{result.order} is the identity, so the "
                f"sectors are k = 1 to {result.order - 1}.  The panel is showing "
                f"k = {result.sector} instead of refusing a combination the two controls "
                "can reach on their own."
            )
        out.append(
            f"{result.label} identifies points of the torus under a rotation of order "
            f"{result.order}.  theta^{result.sector} has twist phases "
            f"({', '.join(f'{p:.4f}' for p in result.phases)}) and leaves "
            f"{result.fixed_from_determinant} points alone.  Those points are singular "
            "in the quotient, and they are where a twisted string has to live: away "
            "from one there is no solution to X(sigma + 2 pi) = theta^k X(sigma), so "
            "every twisted state comes with that multiplicity."
        )
        ground = result.ground_state
        if ground is not None:
            out.append(
                f"The ground state of this sector sits at alpha' M^2 = "
                f"{ground.alpha_m2:+.4f}"
                + (
                    ", so it is tachyonic -- which the bosonic string's untwisted "
                    "ground state is too, and for the same reason: this is a "
                    "zero-point energy, not an instability the orbifold introduced."
                    if ground.is_tachyonic
                    else ", above zero, so this sector has no tachyon."
                )
                + f"  The intercept a_k = {result.intercept:.6f} is what puts it there, "
                "and it differs from the untwisted 1 because twisting moves the mode "
                "numbers off the integers."
            )
        out.append(
            f"That intercept is computed twice.  The closed form "
            f"1 - (1/4) sum phi(1 - phi) gives {result.intercept:.8f}.  Fitting the "
            "small-epsilon expansion of sum (n + phi) exp(-eps(n + phi)) to extract the "
            f"Hurwitz value zeta(-1, phi) gives {result.intercept_from_zeta:.8f}.  The "
            f"gap is {result.intercept_gap:.1e}, which is how well the numerical "
            "extraction does rather than how well the two agree."
        )
        out.append(
            f"The untwisted sector loses states: {result.untwisted_massless} of the "
            f"torus's {result.torus_massless} survive the projection onto "
            "theta-invariant combinations.  What is gained instead is the twisted "
            "sectors, and they are not decoration -- without them the one-loop "
            "amplitude is not modular invariant and the theory does not exist."
        )
        return out
