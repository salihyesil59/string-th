r"""Green-Schwarz cancellation: a family of gauge groups, and the two that live.

Ten-dimensional :math:`N=1` supergravity coupled to a gauge group has a
gravitational anomaly, and a twelve-form that must vanish for the theory to
exist.  Three conditions come out of it, and they are not the same condition
said three ways.

The first is pure gravity.  Nothing but the gauge-group *dimension* enters the
coefficient of :math:`\mathrm{tr}\,R^6`, no counterterm can cancel a pure-gravity
term, and setting it to zero gives 496.  The panel solves for that number rather
than quoting it, and puts it beside 496 from a completely different place: the
even self-dual lattices of rank 16 have 480 roots, and 480 + 16 is the dimension
of the algebra they generate.  One is a twelve-form; the other is a count of
lattice vectors of norm two.

The second is pure gauge.  A surviving :math:`\mathrm{tr}\,F^6` cannot be
reached by any :math:`X_4 X_8` product, so it has to vanish on its own.  This is
where :math:`SO(26)\times SO(19)` dies: it has dimension 496 and fails anyway.

The third is the factorisation itself.  Each coefficient in
:math:`X_4 = \mathrm{tr}R^2 + \sum_a b_a \mathrm{Tr}F_a^2` is fixed by a single
term of the twelve-form; every other coefficient is then a prediction, and the
remainder of the division is what tests them.  It comes out :math:`b_a = 1/30`
for both survivors, which is the textbook number arrived at rather than put in.

What this is not is a uniqueness proof.  The scan covers products of
:math:`SO(N)` and :math:`E_8`; :math:`SU(N)`, :math:`Sp(N)`, the smaller
exceptional algebras and the abelian solutions are not implemented, and the
panel says so rather than letting a finite search look like a theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...heterotic.anomaly import (
    anomaly_polynomial,
    cancels,
    candidates,
    factorise,
    required_dimension,
)
from ...heterotic.lattice import heterotic_lattices
from ...viz.plots import plot_anomaly_scan
from ..panel import Integer, Line

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["AnomalyPanel", "Anomaly"]


@dataclass(frozen=True)
class Anomaly:
    """One sweep over a family of gauge groups, and what came through it."""

    max_so: int
    max_factors: int
    points: tuple[tuple[float, float, str], ...]
    """``(dimension, sixth-order residue, label)`` for every candidate."""
    survivors: tuple[str, ...]
    dimension_from_anomaly: int
    lattice_dimensions: tuple[tuple[str, int, int], ...]
    """``(name, roots, algebra dimension)`` for each even self-dual lattice."""
    coefficients: tuple[tuple[str, str], ...]
    """``b_a`` in ``X_4``, for each survivor's factors."""
    right_dimension: int
    """How many candidates have dimension 496 -- more than survive."""

    @property
    def total(self) -> int:
        return len(self.points)

    @property
    def lattice_dimension(self) -> int | None:
        values = {dim for _, _, dim in self.lattice_dimensions}
        return values.pop() if len(values) == 1 else None

    @property
    def dimensions_agree(self) -> bool:
        return self.lattice_dimension == self.dimension_from_anomaly

    @property
    def window_holds_both(self) -> bool:
        """Whether the search could find both survivors even in principle.

        ``SO(32)`` needs the scan to reach ``N = 32`` and ``E_8 x E_8`` needs
        two factors.  Outside that the scan is not disagreeing with anything;
        it has been told to look somewhere the answer is not.
        """
        return self.max_so >= 32 and self.max_factors >= 2

    @property
    def out_of_reach(self) -> tuple[str, ...]:
        missing = []
        if self.max_so < 32:
            missing.append("SO(32)")
        if self.max_factors < 2:
            missing.append("E8 x E8")
        return tuple(missing)

    @property
    def found_both(self) -> bool:
        return set(self.survivors) == {"SO(32)", "E8 x E8"}


class AnomalyPanel:
    """819 candidates, one condition each fails on, and the two that do not."""

    title = "Anomaly cancellation"
    blurb = (
        "Every product of SO(N) and E_8 up to the sizes chosen is tested against the "
        "three conditions ten dimensions imposes: the coefficient of tr R^6, which knows "
        "only the gauge group's dimension and gives 496; a surviving tr F^6, which no "
        "X_4 X_8 product can reach; and the factorisation itself.  The horizontal axis is "
        "the first condition and the vertical the second, so a survivor has to sit at "
        "496 and on the floor.  Two do."
    )
    background = (
        "An anomaly is a classical symmetry that quantum corrections break.  In ten "
        "dimensions the chiral fields of N = 1 supergravity -- a gravitino, a dilatino "
        "of the opposite chirality, and gaugini in the adjoint -- each contribute to a "
        "twelve-form, and if the total does not vanish the theory is not consistent at "
        "one loop.  For almost every gauge group it does not vanish.",
        "Green and Schwarz found that it need not vanish outright.  A two-form field "
        "with a modified transformation can cancel the part of the anomaly that "
        "factorises as X_4 X_8 -- but only that part.  So the twelve-form has to "
        "factorise, and terms that cannot appear in such a product must be zero on "
        "their own.",
        "The pure-gravity term tr R^6 is one of those.  No product of a four-form and "
        "an eight-form built from tr R^2, tr R^4 and Tr F^2 contains it, and no gauge "
        "field appears in it, so no counterterm can help.  Its coefficient depends only "
        "on how many gaugini there are, which is the dimension of the group, and it "
        "vanishes at 496.  That single number rules out every gauge group of the wrong "
        "size before any factorisation is attempted.",
        "The pure-gauge term tr F^6 is the other.  SO(N) has an independent sextic "
        "Casimir unless N = 32, where the trace identity's (N - 32) coefficient "
        "vanishes; E_8 has none at all, since its adjoint is its defining "
        "representation.  This is why SO(26) x SO(19) fails: it has exactly 496 "
        "dimensions and still leaves a tr F^6 behind.  The two conditions are separate.",
        "496 arrives from a second direction entirely.  A heterotic string needs an "
        "even self-dual lattice of rank 16, of which there are exactly two, and each "
        "has 480 vectors of norm two.  Adding the 16 Cartan directions gives 496 -- the "
        "dimension of the algebra those roots generate.  Nothing in that computation is "
        "a twelve-form, and nothing in the twelve-form is a lattice.",
        "This is a search over a family, not a uniqueness proof.  SU(N), Sp(N), the "
        "smaller exceptional algebras and the abelian solutions are not implemented "
        "here, so 'only two survive' means only two survive this scan.  The theorem is "
        "stronger than what is shown; what is shown is honest about being less.",
    )
    suggestions = (
        "Lower the largest SO(N) below 32.  SO(32) drops out of the search and only "
        "E_8 x E_8 is left -- a fact about the window, not about ten dimensions, which "
        "is why the notes say which one it is.",
        "Set the number of factors to 1.  Now only SO(32) survives, because E_8 x E_8 "
        "needs two.  The same scan, two different answers, neither of them the physics "
        "changing.",
        "Look at how many candidates have dimension exactly 496 and how few of those "
        "get through.  The first condition is cheap and the second is not: passing it "
        "means nothing on its own.",
        "Read the b_a values.  X_4 = tr R^2 - (1/30) Tr F^2 is the textbook form, and "
        "1/30 is not entered anywhere -- it is read off one coefficient of the "
        "twelve-form, and every other coefficient then has to agree with it.",
    )
    controls = (
        Integer("max_so", "largest SO(N) tried", 20, 48, 40),
        Integer("max_factors", "factors in the product", 1, 2, 2),
    )

    def compute(self, max_so: int = 40, max_factors: int = 2) -> Anomaly:
        points: list[tuple[float, float, str]] = []
        survivors: list[str] = []
        coefficients: list[tuple[str, str]] = []
        right_dimension = 0

        for group in candidates(max_so, max_factors):
            label = " x ".join(factor.name for factor in group)
            ok, report = cancels(group)
            dimension = int(report["dimension"])
            leftovers = report["sixth_order"]
            residue = float(abs(leftovers.largest())) if leftovers else 0.0
            points.append((float(dimension), residue, label))
            if dimension == 496:
                right_dimension += 1
            if ok:
                survivors.append(label)
                for name, value in factorise(
                    anomaly_polynomial(group), group
                ).coefficients:
                    coefficients.append((f"{label}: {name}", str(value)))

        lattices = tuple(
            (name, lattice.n_roots, lattice.algebra_dimension)
            for name, lattice in heterotic_lattices().items()
        )
        return Anomaly(
            max_so=max_so,
            max_factors=max_factors,
            points=tuple(points),
            survivors=tuple(survivors),
            dimension_from_anomaly=required_dimension(),
            lattice_dimensions=lattices,
            coefficients=tuple(coefficients),
            right_dimension=right_dimension,
        )

    def draw(self, result: Anomaly) -> Figure:
        """Every candidate against the two conditions, survivors marked."""
        survivors = [
            (dimension, label)
            for dimension, _, label in result.points
            if label in result.survivors
        ]
        return plot_anomaly_scan(
            list(result.points),
            survivors,
            path=None,
            title=(
                f"{result.total} candidates up to SO({result.max_so}), "
                f"{len(result.survivors)} surviving"
            ),
        )

    def readout(self, result: Anomaly) -> list[Line]:
        lines = [
            Line("candidates tested", f"{result.total}",
                 f"products of up to {result.max_factors} factor(s), SO(N) to "
                 f"N = {result.max_so}"),
            Line(
                "dimension that kills tr R^6",
                f"{result.dimension_from_anomaly} from the twelve-form",
                "  ".join(
                    f"{name}: {roots} roots + rank = {dim}"
                    for name, roots, dim in result.lattice_dimensions
                ),
                ok=result.dimensions_agree,
            ),
            Line("candidates with that dimension", f"{result.right_dimension}",
                 "passing the first condition is not passing"),
            # A verdict here only means something when the window could hold
            # both.  Below SO(32), or with one factor allowed, the scan is not
            # disagreeing -- it has been told to look where the answer is not.
            Line(
                "survivors",
                ", ".join(result.survivors) if result.survivors else "none in this window",
                "SO(32) and E8 x E8",
                ok=result.found_both,
            )
            if result.window_holds_both
            else Line(
                "survivors",
                ", ".join(result.survivors) if result.survivors else "none in this window",
                f"{', '.join(result.out_of_reach)} cannot be reached by this window",
            ),
        ]
        for name, value in result.coefficients:
            lines.append(
                Line(
                    f"  b in X_4, {name}",
                    value,
                    "read off one coefficient; the rest are predictions",
                    ok=value == "1/30",
                )
            )
        lines.append(
            Line(
                "what this is not",
                "a uniqueness proof",
                "SU(N), Sp(N) and the abelian solutions are not in the family",
            )
        )
        return lines

    def notes(self, result: Anomaly) -> list[str]:
        """Which of the two survived here, and whether that is about the window."""
        out = [
            f"{result.total} gauge groups were built and every one of them tested "
            f"against three conditions.  {result.right_dimension} of them have dimension "
            "496 and so pass the first; "
            f"{len(result.survivors)} pass all three.  The gap between those two numbers "
            "is the point of the picture: the gravitational condition is cheap and "
            "getting past it means very little.  SO(26) x SO(19) has exactly 496 "
            "dimensions and still leaves a tr F^6 that nothing can cancel."
        ]
        if result.found_both:
            out.append(
                "Both survivors are here: SO(32) and E_8 x E_8.  These are the two "
                "heterotic strings, and they were not looked for -- the scan tests a "
                "family and reports what comes through."
            )
        else:
            missing = {"SO(32)", "E8 x E8"} - set(result.survivors)
            reason = (
                f"the search stops at SO({result.max_so})"
                if "SO(32)" in missing and result.max_so < 32
                else "E_8 x E_8 needs two factors and the control allows one"
                if "E8 x E8" in missing and result.max_factors < 2
                else "the window excludes it"
            )
            out.append(
                f"Only {', '.join(result.survivors) or 'nothing'} survives here, and "
                f"{', '.join(sorted(missing))} is missing because {reason}.  That is a "
                "fact about where the scan was told to look, not about ten dimensions.  "
                "Widen the controls and it comes back."
            )
        out.append(
            f"496 is computed here and also arrives from somewhere else.  The "
            f"twelve-form's tr R^6 coefficient is affine in the gauge dimension -- "
            "checked, not assumed, by taking second differences -- and its root is "
            f"{result.dimension_from_anomaly}.  The two even self-dual lattices of rank "
            "16 each have 480 vectors of norm two, and 480 plus the 16 Cartan "
            "directions is the dimension of the algebra they generate.  A polynomial in "
            "curvature two-forms and a count of short lattice vectors, agreeing."
        )
        if result.coefficients:
            out.append(
                "The factorisation is where most of the work is.  b_a in "
                "X_4 = tr R^2 + b_a Tr F^2 is fixed by a single coefficient of the "
                "twelve-form -- the one multiplying Tr F^2 tr R^4 -- and then every "
                "other coefficient is a prediction that the division has to come out "
                "exactly on.  It does, at b = 1/30, which is the number the textbooks "
                "write and which nothing here was told."
            )
        return out
