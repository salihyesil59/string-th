r"""The Hagedorn temperature, measured from a state count rather than quoted.

The number of open-string states at level :math:`N` grows like
:math:`\exp(2\pi\sqrt{cN/6})` with :math:`c` transverse oscillators, and
:math:`\alpha' M^2 = N` turns that into :math:`\exp(\beta_H M)`.  So there is a
temperature above which the partition function does not converge: heat a string
gas and the energy goes into making heavier strings rather than faster ones.

Two routes to :math:`\beta_H` sit side by side and neither is told the other's
answer.  One counts partitions -- literally, the coefficients of
:math:`\prod (1-q^n)^{-c}` -- and fits a straight line through their logarithms.
The other is Cardy's :math:`2\pi\sqrt{c\alpha'/6}`, which for :math:`c = 24` is
:math:`4\pi\sqrt{\alpha'}`.

They agree slowly, and that is worth watching rather than hiding.  The
subleading :math:`\log N` in the growth means the fitted slope approaches the
closed form like :math:`1/\sqrt{N}`, so the panel fits twice -- once at the
chosen truncation and once at half of it -- and reports that the gap has
shrunk.  A single fit with a loose tolerance would prove less.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...quantum.partition import (
    HagedornFit,
    fit_hagedorn,
    hagedorn_temperature,
    oscillator_degeneracies,
)
from ...units import Conventions
from ...viz.plots import plot_degeneracy_growth
from ..panel import Integer, Line

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["HagedornPanel", "Hagedorn"]


@dataclass(frozen=True)
class Hagedorn:
    """One fit, the same fit at half the truncation, and the closed form."""

    n_max: int
    n_fit: int
    n_species: int
    fit: HagedornFit
    coarse: HagedornFit
    """The same fit over half as many levels, to show the gap closing."""
    degeneracies: tuple[int, ...]
    predicted: float
    slope_levels: np.ndarray
    slope: np.ndarray
    """``d(log d_N)/d(sqrt N)`` by finite differences: a third, cruder route."""

    @property
    def gap(self) -> float:
        """Relative distance from the fitted slope to the closed form."""
        return abs(self.fit.beta_hagedorn / self.predicted - 1.0)

    @property
    def coarse_gap(self) -> float:
        return abs(self.coarse.beta_hagedorn / self.predicted - 1.0)

    @property
    def converging(self) -> bool:
        """The gap shrank when the truncation rose, which is the real check."""
        return self.gap < self.coarse_gap

    @property
    def temperature(self) -> float:
        return self.fit.temperature


class HagedornPanel:
    """A limiting temperature, read off a count of partitions."""

    title = "The Hagedorn temperature"
    blurb = (
        "The number of string states at level N grows exponentially in sqrt(N), so the "
        "partition function sum d_N exp(-beta M_N) diverges above a finite temperature.  "
        "The panel counts the states -- the coefficients of prod (1-q^n)^-c -- fits "
        "log d_N against the mass, and compares the slope with Cardy's "
        "2 pi sqrt(c alpha'/6), which is 4 pi at c = 24.  The two agree slowly, like "
        "1/sqrt(N), so the fit is also done at half the truncation and the panel reports "
        "that the gap has closed rather than that it is small."
    )
    background = (
        "Count the states of the open bosonic string at level N and you are counting "
        "the ways of writing N as a sum of positive integers, with c = D - 2 colours "
        "for each part.  The generating function is prod_n (1 - q^n)^-c, and its "
        "coefficients are what this panel plots.  Nothing is asymptotic about that: "
        "they are exact integers.",
        "Hardy and Ramanujan found how those coefficients grow, and Cardy's argument "
        "gives the same thing from modular invariance: d_N ~ exp(2 pi sqrt(cN/6)).  "
        "With alpha' M^2 = N the exponent is beta_H M, so the density of states rises "
        "exactly as fast as the Boltzmann factor falls.",
        "That is what makes T_H = 1/beta_H a limiting temperature rather than an "
        "ordinary one.  Adding energy to a string gas above it does not make the "
        "strings faster; it makes more and heavier strings.  Whether the theory truly "
        "cannot be heated past T_H, or whether something else takes over there -- a "
        "phase transition, a change of degrees of freedom -- is not settled by the "
        "counting, and this panel does not claim it is.",
        "The fit includes a b log N term.  Leaving it out is not a small sin: it biases "
        "the slope by several percent at these levels, which is larger than the gap the "
        "panel is trying to measure.  The subleading term is also why the agreement "
        "improves like 1/sqrt(N) rather than immediately.",
    )
    suggestions = (
        "Raise the number of levels and watch the fitted beta_H climb towards 4 pi = "
        "12.5664.  It approaches from below and it does not get there; the panel "
        "reports the gap closing rather than the gap being zero.",
        "Drop the number of transverse species from 24.  Both the fit and the closed "
        "form move together -- the agreement is not about the number 24, it is about "
        "the counting and Cardy's formula describing the same growth.",
        "Shrink the fit window to its minimum.  The fit gets noisier, the residual "
        "rises, and the slope wanders; the check is not free of how it is measured.",
        "Look at the second curve on the plot: the local slope by finite differences, "
        "with no fit in it at all.  It heads the same way and more slowly.",
    )
    controls = (
        Integer("n_max", "levels counted", 60, 400, 240),
        Integer("n_fit", "levels used in the fit", 20, 200, 100),
        Integer("n_species", "transverse species  c", 8, 24, 24),
    )

    def compute(
        self,
        n_max: int = 240,
        n_fit: int = 100,
        n_species: int = 24,
    ) -> Hagedorn:
        conv = Conventions()
        alpha_prime = conv.alpha_prime
        window = min(n_fit, n_max)
        half = max(30, n_max // 2)
        coarse_window = min(window, half)

        fit = fit_hagedorn(n_max, n_species, alpha_prime, window)
        coarse = fit_hagedorn(half, n_species, alpha_prime, coarse_window)
        degeneracies = oscillator_degeneracies(n_max, n_species)
        levels, slope = HagedornFit.local_slope(degeneracies, alpha_prime)

        return Hagedorn(
            n_max=n_max,
            n_fit=window,
            n_species=n_species,
            fit=fit,
            coarse=coarse,
            degeneracies=tuple(degeneracies),
            predicted=hagedorn_temperature(n_species, alpha_prime),
            slope_levels=levels,
            slope=slope,
        )

    def draw(self, result: Hagedorn) -> Figure:
        """The package's own figure: the counted degeneracies against the fit."""
        return plot_degeneracy_growth(result.fit, list(result.degeneracies), path=None)

    def readout(self, result: Hagedorn) -> list[Line]:
        return [
            Line("levels counted", f"{result.n_max}, fitted over the top {result.n_fit}"),
            Line("transverse species", f"c = {result.n_species}"),
            # No verdict on this line.  The fitted slope is not supposed to
            # equal the closed form at a finite truncation -- it approaches it
            # -- so a pass/fail against a tolerance would be a claim about how
            # many levels were counted rather than about the physics.  The
            # check that means something is the next line: the gap closing.
            Line(
                "beta_H",
                f"{result.fit.beta_hagedorn:.6f} from the count",
                f"2 pi sqrt(c/6) = {result.predicted:.6f}",
            ),
            Line("  relative gap", f"{result.gap:.4%}"),
            Line(
                "  at half the truncation",
                f"{result.coarse.beta_hagedorn:.6f}, gap {result.coarse_gap:.4%}",
                "the gap closes as the truncation rises",
                ok=result.converging,
            ),
            Line("Hagedorn temperature", f"T_H = {result.temperature:.6f}",
                 f"1 / {result.predicted:.4f} = {1.0 / result.predicted:.6f}"),
            Line("fit residual", f"{result.fit.residual:.2e} rms in log d_N"),
            Line(
                "largest degeneracy",
                f"d_{result.n_max} = {result.degeneracies[-1]:.6e}",
                "an exact integer, counted rather than estimated",
            ),
            Line(
                "local slope at the top",
                f"{result.slope[-1]:.6f}",
                "finite differences, no fit at all",
            ),
        ]

    def notes(self, result: Hagedorn) -> list[str]:
        """How close the two routes are here, and why they are not closer."""
        out = [
            f"Counting {result.n_max} levels and fitting the top {result.n_fit} gives "
            f"beta_H = {result.fit.beta_hagedorn:.5f}.  Cardy's closed form says "
            f"{result.predicted:.5f}.  The two are {result.gap:.3%} apart, and that is "
            "not a failure of either: the growth has a subleading log N in it, so the "
            "slope measured over a finite window approaches its limit like 1/sqrt(N)."
        ]
        if result.gap > 0.02:
            out.append(
                "At this truncation the fit is a long way from converged.  Counting "
                "more levels is the only cure -- the closed form is a limit, and a "
                "few hundred levels is not yet the limit."
            )
        if result.converging:
            out.append(
                f"Which is why the panel fits twice.  Over half as many levels the gap "
                f"is {result.coarse_gap:.3%}, and over the full range it is "
                f"{result.gap:.3%}: it shrank.  That is a stronger statement than "
                "'within tolerance', because a wrong closed form would not be "
                "approached by a converging sequence."
            )
        else:
            out.append(
                f"Here the gap did not shrink: {result.coarse_gap:.3%} at half the "
                f"truncation against {result.gap:.3%} at the full one.  With a narrow "
                "fit window the two fits can cover almost the same levels, in which "
                "case there is nothing for the comparison to see.  Widen the range "
                "between them and the convergence reappears."
            )
        out.append(
            f"T_H = {result.temperature:.5f} in units where alpha' = 1.  It is a "
            "limiting temperature, not an ordinary one: above it the density of states "
            "rises exactly as fast as the Boltzmann factor falls, so adding energy "
            "makes more and heavier strings rather than faster ones.  What actually "
            "happens at T_H -- a phase transition, a change of degrees of freedom, or "
            "a genuine ceiling -- is not decided by counting states, and nothing here "
            "decides it."
        )
        return out
