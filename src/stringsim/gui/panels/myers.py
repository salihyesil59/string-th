r"""The Myers effect: ``N`` D0-branes that find it cheaper to be a sphere.

A D0-brane's transverse position is a matrix rather than a number, and ``N`` of
them have three :math:`N \times N` matrices between them.  In a background flux
the potential contains a commutator term, and its minimum is not at commuting
matrices.  The matrices that minimise it are the generators of an ``N``
dimensional representation of :math:`SU(2)` -- which is to say the branes have
puffed up into a sphere with ``N`` points on it.

The two descriptions are checked against each other rather than asserted.  From
the matrix side: every partition of ``N`` into blocks is evaluated, and the
single block wins because the depth tracks :math:`\sum N_a(N_a^2-1)`.  From the
other side: a D2-brane wrapping a sphere and carrying ``N`` units of flux,
shrunk to a point, has Born-Infeld energy :math:`4\pi^2\alpha' T_2 N`, and that
is exactly :math:`N T_0`.  Nothing is fitted to make it so -- it follows from
the tension formula -- which is why the same object can be described either way.

The gap between them is exact too.  The fuzzy sphere has
:math:`\mathrm{Tr}\,J^2 = N(N^2-1)/4` where the continuum brane knows only
:math:`N^3/4`, so the ratio is :math:`1 - 1/N^2`: the price of building a sphere
out of finitely many points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...branes.dbrane import dp_brane_tension
from ...branes.myers import (
    Configuration,
    algebra_residual,
    configuration_energies,
    fuzzy_radius,
    fuzzy_sphere,
    large_n_ratio,
    myers_gradient,
    noncommutativity,
    shrunk_d2_energy,
    su2_generators,
    trace_j_squared,
)
from ...units import Conventions
from ...viz.plots import plot_myers_landscape
from ..panel import Integer, Line, Slider

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["MyersPanel", "Myers"]


@dataclass(frozen=True)
class Myers:
    """Every way of splitting ``N`` branes, and the two energies of the winner."""

    total: int
    flux: float
    configurations: tuple[Configuration, ...]
    winner: Configuration
    commuting: Configuration
    """``(1, 1, ..., 1)``: the configuration one would have called the vacuum."""
    radius: float
    gradient: float
    """Largest component of the potential's gradient at the fuzzy sphere."""
    algebra: float
    """How far the generators are from closing into ``su(2)``."""
    trace: float
    continuum_trace: float
    ratio: float
    predicted_ratio: float
    shrunk_energy: float
    d0_energy: float
    noncommutativity: float

    @property
    def sphere_wins(self) -> bool:
        return self.winner.is_irreducible

    @property
    def depth(self) -> float:
        """How far below the commuting configuration the sphere sits."""
        return self.commuting.energy - self.winner.energy


class MyersPanel:
    """Point particles that are cheaper as a sphere, described two ways."""

    title = "The Myers effect"
    blurb = (
        "N D0-branes carry three N x N matrices for their transverse position, and in a "
        "background flux the potential has a commutator term whose minimum is not at "
        "commuting matrices.  The winner is the irreducible SU(2) representation: the "
        "branes have become a fuzzy sphere.  The bars are every partition of N into "
        "blocks, sorted by energy, with the single block in orange.  The readout checks "
        "the sphere against a completely different description -- a D2-brane with N units "
        "of flux -- which weighs N D0-branes when shrunk to a point."
    )
    background = (
        "For a single D-brane the transverse position is an ordinary coordinate.  For N "
        "coincident ones it is an N x N matrix, because the open strings that measure it "
        "carry two endpoint labels.  Ordinary positions are the diagonal entries of "
        "commuting matrices; anything else is a configuration with no classical "
        "description at all.",
        "Put those branes in a background flux and the potential picks up a term cubic "
        "in the matrices alongside the commutator-squared.  Commuting matrices still "
        "sit at V = 0, but they are no longer the minimum.  What minimises it satisfies "
        "[X_i, X_j] = i eps_ijk X_k, which is the su(2) algebra: the matrices are "
        "angular momentum generators and the brane configuration is a sphere built out "
        "of N points.",
        "Which representation matters.  Splitting N into blocks gives a reducible "
        "representation -- several small spheres instead of one big one -- and the depth "
        "of the minimum tracks sum N_a (N_a^2 - 1), which one block maximises.  That is "
        "why the branes make a single sphere rather than clustering.",
        "The same object has a second description with no matrices in it: a D2-brane "
        "wrapped on a sphere carrying N units of worldvolume flux.  Its Born-Infeld "
        "energy is 4 pi T_2 sqrt(R^4 + pi^2 alpha'^2 N^2), and at R = 0 that is "
        "4 pi^2 alpha' T_2 N, which the tension formula makes exactly N T_0.  A shrunk "
        "D2-brane with N flux quanta weighs N D0-branes.  Nothing was fitted; it is why "
        "the two pictures are of one thing.",
        "The two do not agree perfectly, and the disagreement is exact rather than "
        "approximate.  Tr J^2 for the N-dimensional representation is N(N^2-1)/4, while "
        "a continuum sphere knows only N^3/4, so the ratio is 1 - 1/N^2.  That is the "
        "cost of building a sphere from finitely many points, and it goes away only as "
        "N grows.",
    )
    suggestions = (
        "Raise N and watch the orange bar pull further ahead of every way of splitting "
        "the branes up.  The gap is not marginal; one big sphere is much cheaper than "
        "several small ones.",
        "Read the last bar on the right: (1, 1, ..., 1), commuting matrices, sitting at "
        "exactly zero.  That is the configuration one would have called the vacuum "
        "before the flux was switched on.",
        "Turn the flux down towards zero.  The whole landscape flattens and the sphere "
        "stops being favoured -- the effect exists because of the background, not "
        "because of the branes.",
        "Compare the two ratio lines: Tr J^2 against the continuum's N^3/4, and the "
        "closed form 1 - 1/N^2.  They agree exactly, at every N, which is what makes "
        "the discretisation error a known quantity rather than an unknown one.",
    )
    controls = (
        Integer("total", "branes  N", 2, 9, 6),
        Slider("flux", "background flux", 0.1, 3.0, 1.0, step=0.05),
    )

    def compute(self, total: int = 6, flux: float = 1.0) -> Myers:
        conv = Conventions()
        configurations = tuple(configuration_energies(total, flux))
        winner = configurations[0]
        commuting = next(
            c for c in configurations if c.partition == tuple([1] * total)
        )

        matrices = fuzzy_sphere(total, flux)
        generators = su2_generators(total)
        trace = trace_j_squared(total)
        continuum = total**3 / 4.0

        return Myers(
            total=total,
            flux=flux,
            configurations=configurations,
            winner=winner,
            commuting=commuting,
            radius=fuzzy_radius(matrices),
            gradient=float(np.max(np.abs(myers_gradient(matrices, flux)))),
            algebra=algebra_residual(generators),
            trace=trace,
            continuum_trace=continuum,
            ratio=trace / continuum,
            predicted_ratio=large_n_ratio(total),
            shrunk_energy=shrunk_d2_energy(total, 1.0, conv),
            d0_energy=total * dp_brane_tension(0, 1.0, conv),
            noncommutativity=noncommutativity(matrices),
        )

    def draw(self, result: Myers) -> Figure:
        """The package's own landscape, with the winner already in orange."""
        figure = plot_myers_landscape(
            result.configurations,
            path=None,
            title=f"N = {result.total} at flux {result.flux:.2g}",
        )
        return figure

    def readout(self, result: Myers) -> list[Line]:
        return [
            Line("branes", f"N = {result.total}, flux {result.flux:.3g}",
                 f"{len(result.configurations)} partitions evaluated"),
            Line(
                "cheapest configuration",
                "+".join(str(n) for n in result.winner.partition),
                "the irreducible representation" if result.sphere_wins
                else "not the single block",
                ok=result.sphere_wins,
            ),
            Line("  its energy", f"V = {result.winner.energy:+.6f}",
                 f"commuting matrices sit at {result.commuting.energy:+.6f}"),
            Line("  fuzzy radius", f"{result.radius:.6f}"),
            Line(
                "is it a critical point",
                f"largest gradient {result.gradient:.2e}",
                "the potential's derivative at the sphere",
                ok=result.gradient < 1e-9,
            ),
            Line(
                "do the generators close",
                f"[X_i, X_j] - i eps X_k = {result.algebra:.2e}",
                "su(2), checked rather than assumed",
                ok=result.algebra < 1e-9,
            ),
            Line(
                "shrunk D2 with N flux quanta",
                f"{result.shrunk_energy:.8f}",
                f"N D0-branes: {result.d0_energy:.8f}",
                ok=abs(result.shrunk_energy - result.d0_energy) < 1e-9,
            ),
            Line(
                "Tr J^2 against the continuum",
                f"{result.trace:.4f} / {result.continuum_trace:.4f} = {result.ratio:.8f}",
                f"1 - 1/N^2 = {result.predicted_ratio:.8f}",
                ok=abs(result.ratio - result.predicted_ratio) < 1e-12,
            ),
            Line("noncommutativity", f"{result.noncommutativity:.6f}"),
        ]

    def notes(self, result: Myers) -> list[str]:
        """What won here, by how much, and how coarse the sphere still is."""
        out: list[str] = []
        if result.sphere_wins:
            out.append(
                f"Of the {len(result.configurations)} ways of splitting {result.total} "
                f"branes into blocks, the single block wins, at V = "
                f"{result.winner.energy:+.5f}.  Commuting matrices -- the configuration "
                f"one would have called the vacuum -- sit at "
                f"{result.commuting.energy:+.5f}, so the sphere is {result.depth:.5f} "
                "below the thing that looks like nothing happening.  The branes are not "
                "at points any more; they are the N-dimensional representation of "
                "su(2), which is a sphere made of N pieces."
            )
        else:
            out.append(
                f"The cheapest configuration here is "
                f"{'+'.join(str(n) for n in result.winner.partition)}, not the single "
                "block.  That is worth looking at rather than explaining away: at this "
                "flux the depth of the minimum no longer favours one big sphere."
            )

        out.append(
            f"A D2-brane carrying {result.total} units of flux, shrunk to a point, has "
            f"Born-Infeld energy {result.shrunk_energy:.6f}.  {result.total} D0-branes "
            f"weigh {result.d0_energy:.6f}.  Those are the same number and nothing was "
            "adjusted to make them so -- 4 pi^2 alpha' T_2 = T_0 comes out of the "
            "tension formula with no freedom in it.  It is why a matrix model and a "
            "wrapped brane are two descriptions of one object rather than two objects."
        )
        out.append(
            f"The sphere is coarse at N = {result.total}.  Tr J^2 is "
            f"{result.trace:.3f} where a continuum sphere would give "
            f"{result.continuum_trace:.3f}, a ratio of {result.ratio:.6f} -- exactly "
            f"1 - 1/{result.total}^2.  The discretisation error is not estimated here, "
            "it is known in closed form, and it disappears only as N grows."
        )
        if result.gradient < 1e-9 and result.algebra < 1e-9:
            out.append(
                "Two things were checked rather than assumed: that the generators "
                "actually close into su(2), and that the potential's gradient vanishes "
                f"there ({result.gradient:.1e}).  The fuzzy sphere is a critical point "
                "of the potential, not a configuration declared to be one."
            )
        return out
