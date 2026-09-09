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


def _lightest(masses: list[float]) -> float:
    """The first state above zero mass, or ``nan`` if the truncation has none."""
    above = [m for m in masses if m > _TOL]
    return min(above) if above else float("nan")

