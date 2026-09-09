r"""T-duality on a circle, with the radius under a slider.

The static figure shows the two towers crossing at :math:`R = \sqrt{\alpha'}`.
What it cannot show is that the *whole spectrum* at ``R`` and at
:math:`\alpha'/R` is the same list of masses, because that is not a picture --
it is a comparison, and it has to be made again at every radius.  So the panel
makes it again at every radius: the readout enumerates both spectra, sorts
them, and prints the largest disagreement it can find.

Two independent routes to the same statement are on screen together.  One is
that residual, computed here from two calls to
:func:`~stringsim.compactification.circle.spectrum`.  The other is
:func:`~stringsim.compactification.circle.spectrum_is_t_dual`, which the
package already had and which decides the question its own way.  Watching them
agree while the slider moves is the point of the panel.

The radius is in units of :math:`\sqrt{\alpha'}`, and the interesting places
are not only the self-dual one.  At ``R = 0.5`` the tachyon tower passes
through zero mass as well, and the panel finds those states by enumeration
rather than being told where to look.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

from ...compactification.circle import (
    CircleState,
    extra_massless_states,
    self_dual_radius,
    spectrum,
    spectrum_is_t_dual,
    t_dual_radius,
)
from ...units import Conventions
from ...viz.plots import plot_tduality
from ..panel import Integer, Line, Slider

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["TDualityPanel", "TDuality"]

_TOL = 1e-9
_TICKS = (0.25, 0.5, 1.0, 2.0, 4.0)
"""Where the grid runs, and the radii a reader is likely to stop at."""


@dataclass(frozen=True)
class TDuality:
    """Everything the panel computed at one radius."""

    radius: float
    dual_radius: float
    self_dual: float
    residual: float
    """Largest ``|alpha' M^2|`` disagreement between the two sorted spectra."""
    is_dual: bool
    """The same question, answered by the package's own predicate."""
    states: int
    lightest_here: float
    lightest_there: float
    lightest_state: CircleState | None
    """The first excitation, kept so the commentary can say what kind it is.

    At large ``R`` it is a momentum mode and at small ``R`` a winding one, and
    which of the two it is at a given radius is the whole content of T-duality
    stated as one state rather than as a spectrum.
    """
    extra_here: tuple[CircleState, ...]
    extra_there: tuple[CircleState, ...]
    grid: np.ndarray
    kaluza_klein: np.ndarray
    winding: np.ndarray

    @property
    def oscillator_extras(self) -> int:
        """Extra massless states carrying an oscillator: the gauge bosons."""
        return sum(1 for s in self.extra_here if s.level or s.level_tilde)


class TDualityPanel:
    """The first panel, and the cheapest: every control costs microseconds."""

    title = "T-duality on a circle"
    blurb = (
        "A closed string on a circle of radius R carries momentum n/R and can also "
        "wrap the circle w times at a cost wR/alpha'.  Exchanging the two towers and "
        "sending R to alpha'/R leaves the spectrum alone, so a circle has a shortest "
        "distinguishable radius and shrinking past it gets you nowhere new.  Move the "
        "slider: the whole enumerated spectrum at R is compared with the one at "
        "alpha'/R, and the largest disagreement is printed beside the package's own "
        "verdict on the same question.  Radii are in units of sqrt(alpha')."
    )
    background = (
        "Compactify one direction, X ~ X + 2 pi R.  Two things change at once.  "
        "Momentum along the circle is quantised, p = n/R, exactly as it is for a point "
        "particle -- the Kaluza-Klein tower.  But a string can also wrap the circle w "
        "times, and unwinding it costs energy proportional to the length wrapped, "
        "w R / alpha'.  A point particle has no such option, and this is the first place "
        "where a string is visibly not one.",
        "The closed-string mass is then M^2 = n^2/R^2 + w^2 R^2/alpha'^2 + (2/alpha') "
        "(N + Ntilde - 2), with level matching N - Ntilde = n w now sourced by momentum "
        "and winding rather than vanishing.",
        "Sending R to alpha'/R and n to w exchanges the first two terms and flips the "
        "sign of n w consistently with N <-> Ntilde, so the whole spectrum comes back "
        "unchanged.  A circle of radius R and one of radius alpha'/R are the same "
        "theory.  There is a shortest distinguishable radius, R = sqrt(alpha'), and "
        "shrinking past it gets you nowhere new -- it walks you back out the other side.",
        "Nothing here is fitted or asserted.  The panel enumerates every level-matched "
        "(n, w, N, Ntilde) inside the truncation at both radii, sorts the two lists of "
        "masses, and subtracts them.",
    )
    suggestions = (
        "Put the radius at 1.  Eight extra massless states appear, four of them "
        "carrying an oscillator: those four are the gauge bosons that enlarge "
        "U(1) x U(1) into SU(2) x SU(2).",
        "Put it at 0.5, then at 2.  Two extra massless states each time, and the "
        "charges swap from (+-1, 0) to (0, +-1) -- the same states seen through the "
        "duality.  This is not the self-dual radius and something is massless anyway.",
        "Drag from one end of the track to the other and watch the lightest massive "
        "state.  Its mass falls, reaches a minimum at the self-dual radius, and rises "
        "again; on the way it stops being a momentum mode and becomes a winding one.",
        "Raise the truncations.  More states are enumerated, the two spectra grow "
        "together, and the largest disagreement stays at zero.",
    )
    controls = (
        Slider("radius", "radius  R / sqrt(alpha')", 0.25, 4.0, 1.0, step=0.01, log=True),
        Integer("n_max", "momentum  |n| up to", 1, 4, 2),
        Integer("w_max", "winding  |w| up to", 1, 4, 2),
        Integer("level_max", "oscillator level up to", 0, 3, 2),
    )

    def compute(
        self,
        radius: float = 1.0,
        n_max: int = 2,
        w_max: int = 2,
        level_max: int = 2,
    ) -> TDuality:
        conv = Conventions()
        cut = {"n_max": n_max, "w_max": w_max, "level_max": level_max}
        dual = t_dual_radius(radius, conv)

        here = spectrum(radius, conv, **cut)
        there = spectrum(dual, conv, **cut)
        masses_here = sorted(s.alpha_m2 for s in here)
        masses_there = sorted(s.alpha_m2 for s in there)
        residual = max(
            (abs(a - b) for a, b in zip(masses_here, masses_there, strict=True)),
            default=0.0,
        )

        grid = np.geomspace(0.25, 4.0, 200)
        return TDuality(
            radius=radius,
            dual_radius=dual,
            self_dual=self_dual_radius(conv),
            residual=residual,
            is_dual=spectrum_is_t_dual(radius, conv, **cut),
            states=len(here),
            lightest_here=_lightest(masses_here),
            lightest_there=_lightest(masses_there),
            lightest_state=min(
                (s for s in here if s.alpha_m2 > _TOL),
                key=lambda s: s.alpha_m2,
                default=None,
            ),
            extra_here=tuple(extra_massless_states(radius, conv, **cut)),
            extra_there=tuple(extra_massless_states(dual, conv, **cut)),
            grid=grid,
            kaluza_klein=1.0 / grid,
            winding=grid / conv.alpha_prime,
        )

    def draw(self, result: TDuality) -> Figure:
        """The package's own figure, with the two radii under discussion marked.

        Three adjustments are made afterwards, all of them because a window is
        a different shape from a saved file.  A log axis draws a label on every
        minor tick, legible in a seven-inch figure and a smear in a wide short
        one, so the ticks are named at the radii the panel actually visits.  The
        two markers are labelled on the axes rather than in the legend, which
        keeps the legend to one row and says which line is which where the line
        is.  And that one row goes in the wedge under the crossing, the only
        part of these axes that no curve ever reaches.
        """
        figure = plot_tduality(
            result.grid, result.kaluza_klein, result.winding, result.self_dual, path=None
        )
        axes = figure.axes[0]
        for axis in (axes.xaxis, axes.yaxis):
            axis.set_minor_formatter(NullFormatter())
            axis.set_major_locator(FixedLocator(_TICKS))
            axis.set_major_formatter(FixedFormatter([f"{t:g}" for t in _TICKS]))
        axes.legend(loc="lower center", ncol=3, fontsize=8, framealpha=0.9)

        # The two coincide at the self-dual radius, so the labels are stacked
        # rather than drawn over one another.
        for radius, style, text, height in (
            (result.radius, "-", "$R$", 0.97),
            (result.dual_radius, ":", r"$\alpha'/R$", 0.80),
        ):
            axes.axvline(radius, color="tab:green", lw=1.4, ls=style)
            axes.annotate(
                text,
                xy=(radius, height),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=9,
                color="tab:green",
            )
        return figure

    def readout(self, result: TDuality) -> list[Line]:
        lines = [
            Line("radius", f"R = {result.radius:.4f}", f"alpha'/R = {result.dual_radius:.4f}"),
            Line("self-dual radius", f"{result.self_dual:.4f}"),
            Line("states enumerated", f"{result.states} at each radius"),
            Line(
                "lightest massive state",
                f"alpha' M^2 = {result.lightest_here:.10f}",
                f"at alpha'/R: {result.lightest_there:.10f}",
                ok=abs(result.lightest_here - result.lightest_there) < _TOL,
            ),
            Line(
                "whole spectrum, as a multiset",
                f"largest disagreement {result.residual:.2e}",
                f"spectrum_is_t_dual -> {result.is_dual}",
                ok=result.is_dual and result.residual < _TOL,
            ),
            Line(
                "extra massless states",
                f"{len(result.extra_here)} at R",
                f"{len(result.extra_there)} at alpha'/R",
                ok=len(result.extra_here) == len(result.extra_there),
            ),
        ]
        if result.extra_here:
            lines.append(
                Line(
                    "  of those, with an oscillator",
                    f"{result.oscillator_extras}"
                    + ("   the SU(2) x SU(2) gauge bosons" if result.oscillator_extras else ""),
                )
            )
            lines.append(
                Line(
                    "  their charges",
                    ", ".join(
                        f"({s.n}, {s.w})" for s in result.extra_here[:8]
                    ),
                )
            )
        return lines

    def notes(self, result: TDuality) -> list[str]:
        """What is true at this radius, and not at every radius."""
        out: list[str] = []
        if abs(result.radius - result.self_dual) < 1e-6:
            out.append(
                "This is the self-dual radius, the fixed point of R -> alpha'/R: the "
                "circle is its own dual and the two towers cross here.  Eight states "
                "that are massive at any other radius have come down to zero.  Four of "
                "them carry an oscillator and are gauge bosons -- they enlarge "
                "U(1)_L x U(1)_R to SU(2)_L x SU(2)_R, a symmetry that exists at this "
                "one radius and nowhere else.  The other four are the tachyon tower "
                "passing through zero mass on its way up."
            )
        elif result.extra_here:
            charges = ", ".join(f"({s.n}, {s.w})" for s in result.extra_here)
            out.append(
                f"Something is massless here and it is not the self-dual radius: "
                f"{len(result.extra_here)} extra states, at charges {charges}.  None of "
                "them carries an oscillator, so none is a gauge boson -- this is the "
                "bosonic string's tachyon tower crossing zero, which it does at more "
                "radii than one.  A readout that announced enhanced symmetry whenever "
                "something went massless would be wrong here, so it counts instead."
            )
        else:
            out.append(
                "A generic radius: the only massless states are the ones present at "
                "every radius -- the graviton, the B field, the dilaton, and the two "
                "U(1) gauge bosons from the metric and the B field with one leg on the "
                "circle.  Nothing extra."
            )

        out.append(
            f"The lightest massive state is {_kind(result.lightest_state)}, at "
            f"alpha' M^2 = {result.lightest_here:.4f}.  "
            + (
                "Above the self-dual radius momentum is cheap and winding is not, so "
                "the first excitation is a momentum mode."
                if result.radius > result.self_dual + 1e-9
                else "Below the self-dual radius the circle is small, momentum costs "
                "1/R and winding costs R/alpha', so the first excitation is a winding "
                "mode instead."
                if result.radius < result.self_dual - 1e-9
                else "At the fixed point the two cost the same, which is what makes it "
                "the fixed point."
            )
        )
        out.append(
            f"The dual radius alpha'/R = {result.dual_radius:.4f} is a different "
            f"circle with the same physics.  Both spectra have {result.states} states "
            "in this truncation, and sorted by mass they agree to "
            f"{result.residual:.1e} -- state by state, not merely in total."
        )
        return out


def _kind(state: CircleState | None) -> str:
    """What sort of excitation the lightest massive state is."""
    if state is None:
        return "nothing above zero mass in this truncation"
    if state.n and not state.w:
        return f"a momentum mode, (n, w) = ({state.n}, {state.w})"
    if state.w and not state.n:
        return f"a winding mode, (n, w) = ({state.n}, {state.w})"
    if state.n and state.w:
        return f"carrying both, (n, w) = ({state.n}, {state.w})"
    return "an oscillator state with no momentum and no winding"


def _lightest(masses: list[float]) -> float:
    """The first state above zero mass, or ``nan`` if the truncation has none."""
    above = [m for m in masses if m > _TOL]
    return min(above) if above else float("nan")

