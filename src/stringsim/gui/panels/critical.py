r"""Where 26 comes from, with the intercept and the level under the controls.

Two bounds on the spacetime dimension, pointing opposite ways.

From above: the physical states at level :math:`N` have a Gram matrix, and its
signature says how many of them have negative norm.  Below 26 there are none;
above 26 there are, and a negative-norm state is a negative probability.  So
:math:`D \le 26`.

From below: the light cone, where only :math:`D-2` transverse oscillators
exist, counts the states that are really there.  The covariant construction has
to reproduce that count once null states are discarded, and below 26 it has one
too many.  So :math:`D \ge 26`.

Only at 26 are both satisfied, and neither calculation mentions a conformal
anomaly.  The panel puts the anomaly's answer beside them as the third route.

The intercept is a slider because :math:`a` is not free either: it is
:math:`(D-2)/24` from a zeta-regularised zero-point energy, which is 1 at
:math:`D = 26`.  Moving it shows that 26 and 1 stand or fall together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...quantum.virasoro import (
    Inertia,
    central_charge_from_algebra,
    ghost_scan,
    lightcone_count,
)
from ...quantum.zeta import (
    central_charge,
    critical_dimension,
    normal_ordering_constant,
    regularised_sum,
)
from ...viz.plots import plot_ghost_onset
from ..panel import Integer, Line, Slider

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["CriticalDimensionPanel", "CriticalDimension"]

_TOL = 1e-6
_D_MIN = 4


@dataclass(frozen=True)
class CriticalDimension:
    """One scan over spacetime dimensions, and the three routes to 26."""

    level: int
    intercept: float
    records: tuple[Inertia, ...]
    lightcone: tuple[int, ...]
    from_norms: int | None
    """Largest ghost-free ``D``, or ``None`` if the intercept leaves none."""
    from_counting: int | None
    """Smallest ``D`` whose physical count matches the light cone's."""
    from_anomaly: int
    """26, from ``c = D - 26 = 0``.  No Gram matrix anywhere in it."""
    measured_c: float
    measured_c_other: float
    string_intercept: float
    zeta_intercept: float
    """``a`` rebuilt from the measured ``zeta(-1)`` rather than from a formula."""

    @property
    def bounds_meet(self) -> bool:
        return (
            self.from_norms is not None
            and self.from_norms == self.from_counting == self.from_anomaly
        )

    def at(self, dim: int) -> Inertia | None:
        return next((r for r in self.records if r.dim == dim), None)


class CriticalDimensionPanel:
    """Two bounds on ``D`` that only ever meet in one place."""

    title = "D = 26, from two sides"
    blurb = (
        "Left: the smallest norm among the physical states at each spacetime dimension.  "
        "It stays non-negative up to 26 and turns over after -- a negative-norm state is "
        "a negative probability, so that is an upper bound on D.  Right: how many states "
        "the covariant construction has that the light cone does not.  The excess is 1 "
        "below 26 and 0 from 26 on -- a lower bound.  Only at 26 are both zero, and "
        "neither side mentions a conformal anomaly.  Move the intercept off 1 and the "
        "two bounds stop meeting."
    )
    background = (
        "Quantising a string covariantly keeps all D oscillator directions, and the "
        "Fock metric is the spacetime metric, so the timelike direction contributes "
        "negative norms from the start.  The Virasoro constraints are supposed to "
        "remove them.  Whether they succeed is a question about a matrix: build the "
        "physical subspace, form its Gram matrix, and count the signs of the "
        "eigenvalues.  Sylvester's law says those signs do not depend on the basis.",
        "Below 26 the constraints work and every physical norm is non-negative.  Above "
        "26 one of them goes negative and stays negative.  That is the upper bound, and "
        "it is read off eigenvalues rather than derived.",
        "The lower bound comes from counting.  Light-cone quantisation keeps only the "
        "D-2 transverse oscillators and has no ghosts to remove, so its state count is "
        "the true one.  The covariant construction must reproduce it after null states "
        "-- pure gauge, decoupled from every amplitude -- are discarded.  Below 26 it "
        "has exactly one state too many: a state that is neither null nor physical "
        "anywhere else.",
        "The third route is the conformal anomaly: c = D - 26 for the matter plus ghost "
        "system, and a two-dimensional theory with a gravitational anomaly cannot be "
        "consistent.  It has no Gram matrix and no counting in it, and it gives 26.",
        "The intercept a is not an independent knob in the real theory.  Normal "
        "ordering leaves the sum 1 + 2 + 3 + ... over each transverse direction, which "
        "zeta regularisation assigns -1/12, so a = (D-2)/24.  At D = 26 that is exactly "
        "1.  The two numbers are one statement, which is why moving the slider breaks "
        "the agreement.",
        "Level 3 gives the same answer as level 2 and takes about eighteen seconds per "
        "dimension, so the control stops at 2.  That 26 is not a level-2 accident is "
        "checked in the test suite rather than here.",
    )
    suggestions = (
        "Leave the intercept at 1 and read the two panels together.  The left curve "
        "crosses zero at 26; the right one reaches zero at 26.  They agree, and the "
        "anomaly's 26 is printed beside them from a third calculation.",
        "Set the level to 1.  The upper bound disappears -- level 1 has no ghosts in "
        "any dimension, because its physical states are just polarisations with one "
        "null direction.  26 is invisible there, which is why the default is 2.",
        "Move the intercept to 0.9 or 1.1.  The ghost-free window moves off 26 and the "
        "check against the anomaly turns red.  a = 1 and D = 26 are not two "
        "coincidences but one.",
        "Raise the largest dimension scanned and watch the smallest norm keep falling "
        "past 26.  Nothing recovers further out; the bound is one-sided.",
    )
    controls = (
        Slider("intercept", "intercept  a", 0.5, 1.5, 1.0, step=0.01),
        Integer("level", "oscillator level  N", 1, 2, 2),
        Integer("d_max", "scan D up to", 27, 32, 30),
    )

    def compute(
        self,
        intercept: float = 1.0,
        level: int = 2,
        d_max: int = 30,
    ) -> CriticalDimension:
        window = range(_D_MIN, d_max + 1)
        records = tuple(ghost_scan(window, level=level, intercept=intercept))
        lightcone = tuple(lightcone_count(level, r.dim) for r in records)

        free = [r.dim for r in records if r.ghost_free]
        matched = [
            r.dim
            for r, count in zip(records, lightcone, strict=True)
            if r.positive == count
        ]

        estimate = regularised_sum()
        return CriticalDimension(
            level=level,
            intercept=intercept,
            records=records,
            lightcone=lightcone,
            from_norms=max(free) if free else None,
            from_counting=min(matched) if matched else None,
            from_anomaly=critical_dimension("bosonic"),
            measured_c=central_charge_from_algebra(26, m=2, level=0),
            measured_c_other=central_charge_from_algebra(26, m=3, level=0),
            string_intercept=normal_ordering_constant(26, "bosonic"),
            zeta_intercept=-0.5 * (26 - 2) * estimate.value,
        )

    def draw(self, result: CriticalDimension) -> Figure:
        """The package's own two-panel figure, titled with what it is showing."""
        figure = plot_ghost_onset(
            result.records,
            result.lightcone,
            path=None,
            title=(
                f"The critical dimension at a = {result.intercept:.2f}, level "
                f"{result.level}"
            ),
        )
        return figure

    def readout(self, result: CriticalDimension) -> list[Line]:
        here = result.at(26)
        lines = [
            Line("intercept", f"a = {result.intercept:.4f}",
                 f"the string's (D-2)/24 = {result.string_intercept:.4f}"),
            Line(
                "  a from zeta(-1)",
                f"{result.zeta_intercept:.8f}",
                f"from (D-2)/24: {result.string_intercept:.8f}",
                ok=abs(result.zeta_intercept - result.string_intercept) < 1e-6,
            ),
            Line(
                "central charge at D = 26",
                f"c = {result.measured_c:.6f} from [L_2, L_-2]",
                f"{result.measured_c_other:.6f} from [L_3, L_-3]",
                ok=abs(result.measured_c - result.measured_c_other) < 1e-6,
            ),
            Line(
                "  against c = D",
                f"{central_charge(26, 'bosonic') + 26:.6f}",
                "matter plus ghosts gives c = D - 26",
                ok=abs(result.measured_c - 26.0) < 1e-6,
            ),
        ]
        # Level 1 is not a level at which these two bounds disagree with the
        # anomaly -- it is a level at which they say nothing.  Its physical
        # states are polarisations with one null direction in every dimension,
        # so the norms never go negative and the counting matches everywhere.
        # A verdict either way would be a claim about a question not asked.
        informative = result.level >= 2
        lines.append(
            Line(
                "largest ghost-free D",
                "none in this window" if result.from_norms is None
                else f"{result.from_norms}",
                f"from the anomaly: {result.from_anomaly}" if informative
                else "level 1 has no ghosts in any dimension",
                ok=(result.from_norms == result.from_anomaly) if informative else None,
                declined=not informative,
            )
        )
        lines.append(
            Line(
                "smallest D matching the light cone",
                "none in this window" if result.from_counting is None
                else f"{result.from_counting}",
                f"from the anomaly: {result.from_anomaly}" if informative
                else "level 1 matches at every dimension",
                ok=(result.from_counting == result.from_anomaly) if informative else None,
                declined=not informative,
            )
        )
        if here is not None:
            lines.append(
                Line(
                    "at D = 26",
                    f"{here.positive} positive / {here.zero} null / "
                    f"{here.negative} negative",
                    f"light cone counts {lightcone_count(result.level, 26)}",
                    ok=here.positive == lightcone_count(result.level, 26),
                )
            )
            lines.append(
                Line("  smallest physical norm there", f"{here.smallest:+.3e}")
            )
        return lines

    def notes(self, result: CriticalDimension) -> list[str]:
        """Whether the bounds are still meeting, and what it means that they are."""
        out: list[str] = []
        if result.level < 2:
            out.append(
                "At level 1 there is no upper bound to find.  The physical states are "
                "the D-1 polarisations transverse to the momentum, one of which is "
                "null, and that is true in every dimension -- so no norm ever goes "
                "negative and the left panel is flat.  26 is invisible here.  It first "
                "appears at level 2, where there are enough states for the constraints "
                "to fail to remove them all."
            )
        elif result.bounds_meet:
            out.append(
                f"Both bounds land on {result.from_anomaly}.  The left panel is an "
                "upper bound from unitarity -- past 26 a physical state has negative "
                "norm, which would be a negative probability.  The right panel is a "
                "lower bound from counting -- below 26 the covariant construction has "
                "one state the light cone does not, and that state is neither null nor "
                "genuinely there.  Two inequalities in opposite directions leave one "
                "dimension."
            )
            out.append(
                "The third number in the readout comes from neither.  c = D - 26 is "
                "the conformal anomaly of the matter fields plus the reparametrisation "
                "ghosts, and it vanishes at 26 with no Gram matrix and no state "
                "counting anywhere in the derivation.  Three routes, one answer."
            )
        else:
            norms = _or_none(result.from_norms)
            counting = _or_none(result.from_counting)
            out.append(
                f"At a = {result.intercept:.3g} the bounds no longer meet: the norms "
                f"allow up to {norms} and the counting first agrees at {counting}, "
                f"while the anomaly still says {result.from_anomaly}.  Nothing has "
                "gone wrong with the computation.  The intercept is not free: normal "
                "ordering fixes a = (D-2)/24, which is 1 exactly when D is 26, so "
                "setting a by hand describes a theory that does not exist."
            )

        here = result.at(26)
        if here is not None:
            expected = lightcone_count(result.level, 26)
            tail = (
                f"and what is left, {here.positive}, is exactly the light cone's "
                f"{expected}."
                if here.positive == expected
                else f"and what is left, {here.positive}, is not the light cone's "
                f"{expected} -- the constraints have stopped removing the right states."
            )
            norm = (
                f"The smallest norm there is {here.smallest:+.2e}, zero to machine "
                "precision, which is the boundary the upper bound is about."
                if abs(here.smallest) < 1e-9
                else f"The smallest norm there is {here.smallest:+.2e}, which is not "
                "zero and not small: at this intercept D = 26 is inside the forbidden "
                "region rather than on its edge."
            )
            out.append(
                f"At D = 26 and level {result.level} the physical subspace has "
                f"{here.total} states: {here.positive} with positive norm, {here.zero} "
                f"null and {here.negative} negative.  The null ones are pure gauge and "
                f"decouple from every amplitude, {tail}  {norm}"
            )
        return out


def _or_none(value: int | None) -> str:
    """A bound, or the fact that the scan window contains none."""
    return "nothing in this window" if value is None else str(value)
