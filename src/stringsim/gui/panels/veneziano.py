r"""The Veneziano amplitude, with the Regge intercept under a slider.

The amplitude has poles where :math:`\alpha(s) = a + \alpha' s` reaches a
non-negative integer, so they sit at :math:`\alpha' s = n - a`.  The open
bosonic string's mass levels sit at :math:`\alpha' M^2 = N - a` with the *same*
``a``, but for an entirely different reason: it is the normal-ordering constant
:math:`(D-2)/24`, which is 1 because :math:`D = 26`.  Nothing in
:mod:`~stringsim.amplitudes.veneziano` knows that, and nothing in
:mod:`~stringsim.quantum.spectrum` knows about gamma functions.

That the two coincide is the statement that the amplitude and the spectrum
describe one theory, and it is a coincidence that can be *broken*.  Move the
intercept off 1 and the poles slide away from the mass levels while every other
property of the amplitude survives untouched: it still factorises, its residues
are still polynomials of the right degree, it is still crossing symmetric.  The
readout says so, and it is the one place in this package where a check is meant
to be watched failing.

The residues are checked the other way round, and they hold at every intercept:
a numerical two-sided limit of :math:`(\alpha(s) - n) A(s,t)` against the closed
form :math:`-\prod_{k=1}^{n}(\alpha(t)+k)/n!`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...amplitudes.veneziano import (
    regge_alpha,
    veneziano,
    veneziano_residue,
)
from ...quantum.spectrum import open_bosonic_spectrum
from ...quantum.zeta import normal_ordering_constant
from ...units import Conventions
from ...viz.plots import plot_veneziano
from ..panel import Integer, Line, Slider

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["VenezianoPanel", "Veneziano"]

_EPS = 1e-5
"""Offset for the two-sided residue limit.  Its error cancels to ``O(eps^2)``."""

_CLIP = 25.0


@dataclass(frozen=True)
class Veneziano:
    """The amplitude on a slice of the real ``s`` axis, and what checks it."""

    intercept: float
    mandelstam_t: float
    s_values: np.ndarray
    amplitude: np.ndarray
    poles: np.ndarray
    """Where the amplitude actually diverges: ``alpha' s = n - intercept``."""
    levels: np.ndarray
    """Where the open string's states sit, from the spectrum module."""
    residues: tuple[tuple[int, float, float], ...]
    """``(n, numerical limit, closed form)`` at each pole."""
    crossing: float
    string_intercept: float
    """``a = (D-2)/24``, the value the spectrum uses."""
    t_on_shell: bool
    """``alpha(t)`` is a non-negative integer, so ``A`` is singular in ``t``.

    Reachable from the controls, and not a failure of anything: the momentum
    transfer is sitting on a t-channel resonance, and there is then no finite
    amplitude at any ``s`` for a residue limit to converge to.
    """

    @property
    def pole_offset(self) -> float:
        """Largest gap between a pole and the mass level it should sit on."""
        return float(np.max(np.abs(self.poles - self.levels)))

    @property
    def residue_error(self) -> float:
        return max((abs(a - b) for _, a, b in self.residues), default=0.0)


class VenezianoPanel:
    """Two modules that agree about where the poles are, until they are made not to."""

    title = "The Veneziano amplitude"
    blurb = (
        "A(s,t) = Gamma(-alpha_s) Gamma(-alpha_t) / Gamma(-alpha_s - alpha_t) has poles "
        "wherever alpha(s) = a + alpha' s hits a non-negative integer.  The open bosonic "
        "string's mass levels sit at alpha' M^2 = N - a with the same a, for a different "
        "reason: it is the normal-ordering constant (D-2)/24, which is 1 because D = 26.  "
        "Red lines are the amplitude's poles; green marks are the spectrum's levels.  "
        "Move the intercept off 1 and watch them part -- this is the one panel here whose "
        "check is meant to be seen failing."
    )
    controls = (
        Slider("intercept", "Regge intercept  a", 0.2, 1.8, 1.0, step=0.01),
        Slider("mandelstam_t", "momentum transfer  alpha' t", -3.0, -0.05, -0.35, step=0.01),
        Integer("n_max", "poles shown", 2, 8, 5),
    )

    def compute(
        self,
        intercept: float = 1.0,
        mandelstam_t: float = -0.35,
        n_max: int = 5,
    ) -> Veneziano:
        conv = Conventions()
        alpha_prime = conv.alpha_prime

        s_values = np.linspace(-2.0, float(n_max) + 0.5, 2000)
        amplitude = veneziano(s_values, mandelstam_t, alpha_prime, intercept)

        # Where the amplitude diverges, from the amplitude's own trajectory.
        poles = np.array([(n - intercept) / alpha_prime for n in range(n_max + 1)])
        # Where the states are, from a module that has never heard of a pole.
        levels = np.array([lv.alpha_m2 for lv in open_bosonic_spectrum(n_max, conv)])

        alpha_t = float(regge_alpha(mandelstam_t, alpha_prime, intercept))
        t_on_shell = alpha_t >= -1e-9 and abs(alpha_t - round(alpha_t)) < 1e-9
        residues = (
            ()
            if t_on_shell
            else tuple(
                (n, _limit(n, mandelstam_t, alpha_prime, intercept),
                 float(veneziano_residue(n, mandelstam_t, alpha_prime, intercept)))
                for n in range(min(n_max, 4) + 1)
            )
        )

        probe_s, probe_t = 0.37, -1.13
        crossing = abs(
            float(veneziano(probe_s, probe_t, alpha_prime, intercept))
            - float(veneziano(probe_t, probe_s, alpha_prime, intercept))
        )

        return Veneziano(
            intercept=intercept,
            mandelstam_t=mandelstam_t,
            s_values=s_values,
            amplitude=amplitude,
            poles=poles,
            levels=levels,
            residues=residues,
            crossing=crossing,
            string_intercept=normal_ordering_constant(conv.dim, "bosonic"),
            t_on_shell=t_on_shell,
        )

    def draw(self, result: Veneziano) -> Figure:
        """The package's figure, with the spectrum's levels drawn beside the poles.

        ``plot_veneziano`` marks the poles it is given.  What it cannot show,
        because it is handed one list, is that a second list computed somewhere
        else lands on the same places -- so the levels go on as their own marks,
        along the bottom, and the two agree or visibly do not.
        """
        figure = plot_veneziano(
            result.s_values, result.amplitude, result.poles, path=None, clip=_CLIP
        )
        axes = figure.axes[0]
        axes.plot(
            result.levels,
            np.full_like(result.levels, -_CLIP),
            marker="^",
            ls="none",
            color="tab:green",
            markersize=7,
            clip_on=False,
            label=r"mass levels $\alpha' M^2 = N - a$",
        )
        axes.plot([], [], color="tab:red", ls=":", label=r"poles $\alpha' s = n - a$")
        axes.set_title(
            rf"Veneziano at intercept ${result.intercept:.2f}$"
            + ("" if result.pole_offset < 1e-9 else "  — poles off the spectrum")
        )
        axes.legend(loc="upper left", fontsize=8, framealpha=0.9)
        return figure

    def readout(self, result: Veneziano) -> list[Line]:
        matched = result.pole_offset < 1e-9
        lines = [
            Line("Regge intercept", f"a = {result.intercept:.4f}",
                 f"the string's (D-2)/24 = {result.string_intercept:.4f}"),
            Line("momentum transfer", f"alpha' t = {result.mandelstam_t:.4f}"),
            Line(
                "poles against the spectrum",
                f"largest gap {result.pole_offset:.4f}",
                "amplitude vs open_bosonic_spectrum",
                ok=matched,
            ),
            Line(
                "  poles at",
                ", ".join(f"{p:+.3f}" for p in result.poles[:6]),
            ),
            Line(
                "  levels at",
                ", ".join(f"{lv:+.3f}" for lv in result.levels[:6]),
            ),
            Line(
                "residues",
                "not defined here",
                "alpha(t) is a non-negative integer",
            )
            if result.t_on_shell
            else Line(
                "residues",
                f"largest error {result.residue_error:.2e}",
                "numerical limit vs closed form",
                ok=result.residue_error < 1e-3,
            ),
            Line(
                "crossing symmetry",
                f"|A(s,t) - A(t,s)| = {result.crossing:.2e}",
                "the amplitude, twice",
                ok=result.crossing < 1e-9,
            ),
        ]
        if result.t_on_shell:
            lines.append(
                Line(
                    "the t channel",
                    "this momentum transfer is itself on a resonance,",
                )
            )
            lines.append(
                Line(
                    "",
                    "so A is singular at every s and the plot has nothing finite to draw.",
                )
            )
        if not matched:
            lines.append(
                Line(
                    "what this means",
                    "the amplitude is still fine -- residues and crossing hold.",
                )
            )
            lines.append(
                Line(
                    "",
                    "It is no longer describing the string whose states are counted next door.",
                )
            )
        return lines


def _limit(n: int, t: float, alpha_prime: float, intercept: float) -> float:
    r"""``(alpha(s) - n) A(s,t)`` as ``s`` approaches the ``n``-th pole.

    Two-sided, so the linear term in the offset cancels and what is left is
    ``O(eps^2)``.  This knows the closed form for the residue not at all: it
    evaluates the amplitude and multiplies.
    """
    where = (n - intercept) / alpha_prime
    total = 0.0
    for offset in (_EPS, -_EPS):
        s = where + offset
        total += float(
            (regge_alpha(s, alpha_prime, intercept) - n)
            * veneziano(s, t, alpha_prime, intercept)
        )
    return total / 2.0
