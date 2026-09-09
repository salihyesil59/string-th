r"""Two D-branes pulled apart, and a gauge symmetry breaking as they go.

``N`` coincident branes carry ``U(N)``: the massless vectors are the strings
running from one brane to another, and there are :math:`N^2` of them because
every ordered pair counts.  Pull one away and the strings that now have to
stretch pay :math:`d/2\pi\alpha'` for the length, so they leave the massless
spectrum and :math:`U(N)` breaks to :math:`U(N-1) \times U(1)`.  The Higgs
mechanism, with the Higgs field a brane position.

Two things are checked while the slider moves.  The number of massless vectors
is counted twice -- once by grouping the branes into stacks and summing
:math:`\sum n_i^2`, once by walking every ordered pair of branes and asking
whether its two ends coincide.  And the separation at which the level-0
tachyon finally becomes massive is found twice: from the closed form
:math:`2\pi\sqrt{\alpha'}`, and by bisecting :math:`M^2` on the enumerated
spectrum, which knows no closed form at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...branes.dbrane import (
    BraneStack,
    StretchedLevel,
    gauge_group,
    stretched_spectrum,
    tachyon_free_separation,
)
from ...units import Conventions
from ...viz.plots import plot_brane_separation
from ..panel import Integer, Line, Slider

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["BranePanel", "Branes"]

_TOL = 1e-9
_SPAN = 10.0


@dataclass(frozen=True)
class Branes:
    """One separation, and the stack it leaves behind."""

    separation: float
    positions: tuple[float, ...]
    group: str
    massless_grouped: int
    """``sum n_i^2``, from the stacks."""
    massless_paired: int
    """The same number, from a walk over ordered pairs of branes."""
    threshold_closed: float
    threshold_found: float
    """The same separation, from bisecting the level-0 mass."""
    levels: tuple[StretchedLevel, ...]
    grid: np.ndarray
    spectra: tuple[tuple[StretchedLevel, ...], ...]

    @property
    def stretched(self) -> int:
        """Vectors that had to stretch, and so are no longer massless."""
        return len(self.positions) ** 2 - self.massless_paired

    @property
    def tachyonic(self) -> bool:
        return self.levels[0].mass_squared < -_TOL


class BranePanel:
    """The Higgs mechanism, with the Higgs field a distance."""

    title = "D-branes pulled apart"
    blurb = (
        "N coincident D-branes carry U(N): the massless vectors are the strings running "
        "between them, N^2 of them because every ordered pair counts.  Pull one away and "
        "those strings pay d/2 pi alpha' for their length, drop out of the massless "
        "spectrum, and U(N) breaks to U(N-1) x U(1) -- the Higgs mechanism with a brane "
        "position for a Higgs field.  The open bosonic string keeps its tachyon until the "
        "stretching outweighs the normal-ordering constant, which the readout locates two "
        "ways.  Distances are in units of sqrt(alpha')."
    )
    controls = (
        Slider("separation", "separation  d / sqrt(alpha')", 0.0, _SPAN, 2.0, step=0.05),
        Integer("branes", "branes in the stack", 2, 6, 3),
        Integer("n_max", "oscillator level up to", 1, 4, 3),
    )

    def compute(
        self,
        separation: float = 2.0,
        branes: int = 3,
        n_max: int = 3,
    ) -> Branes:
        conv = Conventions()
        positions = tuple([0.0] * (branes - 1) + [float(separation)])

        stack = BraneStack(positions)
        paired = sum(
            1 for a in positions for b in positions if abs(a - b) < 1e-6
        )

        grid = np.linspace(0.0, _SPAN, 120)
        spectra = tuple(tuple(stretched_spectrum(d, n_max, conv)) for d in grid)

        return Branes(
            separation=float(separation),
            positions=positions,
            group=gauge_group(positions),
            massless_grouped=stack.massless_vectors,
            massless_paired=paired,
            threshold_closed=tachyon_free_separation(conv),
            threshold_found=_bisect_threshold(n_max, conv),
            levels=tuple(stretched_spectrum(separation, n_max, conv)),
            grid=grid,
            spectra=spectra,
        )

    def draw(self, result: Branes) -> Figure:
        """The package's figure, with the separation and the threshold marked."""
        figure = plot_brane_separation(result.grid, [list(s) for s in result.spectra], path=None)
        axes = figure.axes[0]
        # The levels already carry a legend, and two more entries would put six
        # boxes over the curves.  A vertical line can say what it is where it
        # is, which a curve cannot.
        for where, colour, style, text, height in (
            (result.threshold_closed, "0.4", "--", r"$2\pi\sqrt{\alpha'}$", 0.97),
            (result.separation, "tab:green", "-", f"$d = {result.separation:.3g}$", 0.84),
        ):
            axes.axvline(where, color=colour, ls=style, lw=1.3)
            axes.annotate(
                text,
                xy=(where, height),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=9,
                color=colour,
            )
        axes.legend(loc="lower right", ncol=2, fontsize=8, framealpha=0.9)
        return figure

    def readout(self, result: Branes) -> list[Line]:
        return [
            Line("separation", f"d = {result.separation:.4f}",
                 f"positions {_positions(result.positions)}"),
            Line("unbroken gauge group", result.group),
            Line(
                "massless vectors",
                f"{result.massless_grouped}",
                f"{result.massless_paired} from a walk over ordered pairs",
                ok=result.massless_grouped == result.massless_paired,
            ),
            Line("  strings that had to stretch", f"{result.stretched}"),
            Line(
                "tachyon-free separation",
                f"{result.threshold_closed:.6f}",
                f"bisected on the spectrum: {result.threshold_found:.6f}",
                ok=abs(result.threshold_closed - result.threshold_found) < 1e-6,
            ),
            Line(
                "level 0 here",
                f"M^2 = {result.levels[0].mass_squared:+.6f}"
                + ("   still tachyonic" if result.tachyonic else "   massive"),
            ),
            Line(
                "levels",
                "  ".join(f"N={lv.level}: {lv.mass_squared:+.3f}" for lv in result.levels),
            ),
        ]


def _positions(positions: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{x:g}" for x in positions) + "]"


def _bisect_threshold(n_max: int, conv: Conventions) -> float:
    r"""Where the level-0 mass crosses zero, found rather than solved for.

    Bisection on :func:`stretched_spectrum`, which returns a number and no
    formula.  The closed form it is compared against was derived by hand.
    """
    def level_zero(d: float) -> float:
        return stretched_spectrum(d, n_max, conv)[0].mass_squared

    low, high = 0.0, _SPAN
    if level_zero(high) < 0.0:  # pragma: no cover - the span always reaches past it
        return float("nan")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if level_zero(middle) < 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
