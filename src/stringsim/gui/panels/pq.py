r"""The :math:`(p,q)` strings of type IIB, with the coupling under a slider.

The fundamental string and the D1-brane are two members of one lattice.  A
:math:`(p,q)` string has tension :math:`|p + q\tau| / 2\pi\alpha'` with
:math:`\tau = C_0 + i/g_s`, and :math:`SL(2,Z)` moves the charges around while
leaving the Einstein-frame tension alone.  Drag the coupling past 1 and the
fundamental string and the D1-brane exchange which of them is heavy.

Three independent routes are recomputed at every move.  The tension comes from
ten dimensions as :math:`|p+q\tau|/2\pi\alpha'` and from eleven as an M2-brane
wrapping a cycle of the torus, and the two share nothing but string theory.  The
Einstein-frame tension is asked to survive an :math:`SL(2,Z)` element, with the
charges transforming as the algebra says rather than as anything fitted.  And a
junction of three strings is asked whether it holds still: each leaves along the
phase of :math:`p + q\tau` pulling with :math:`|p+q\tau|`, no angle is imposed,
and the net force vanishes because the charges do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...branes.dbrane import dp_brane_tension
from ...branes.pq import (
    Junction,
    axio_dilaton,
    binding_energy,
    duality_residual,
    is_bound_state,
    junction_residual,
    membrane_residual,
    membrane_tension,
    reduce_coupling,
    tension,
    transform_charges,
)
from ...units import Conventions
from ...viz.plots import plot_pq_strings
from ..panel import Integer, Line, Slider

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["PQPanel", "PQStrings"]

_TOL = 1e-9

S = ((0, -1), (1, 0))
"""``tau -> -1/tau``: strong coupling to weak, and F1 to D1."""

T = ((1, 1), (0, 1))
"""``tau -> tau + 1``: a shift of the axion, which is a symmetry of the theory."""

_JUNCTION = ((1, 0), (0, 1), (-1, -1))


@dataclass(frozen=True)
class PQStrings:
    """One charge and one coupling, with everything that checks them."""

    p: int
    q: int
    tau: complex
    coupling: float
    axion: float
    tension: float
    membrane: float
    membrane_residual: float
    s_residual: float
    t_residual: float
    s_charges: tuple[int, int]
    junction_force: float
    junction_charge: tuple[int, int]
    binding: float
    primitive: bool
    reduced: complex
    reduction_steps: int
    d1_tension: float
    d1_from_dbrane: float
    grid: np.ndarray
    curves: tuple[tuple[str, np.ndarray], ...]

    @property
    def axion_is_zero(self) -> bool:
        return abs(self.axion) < 1e-12

    @property
    def d1_residual(self) -> float:
        return abs(self.d1_tension / self.d1_from_dbrane - 1.0)


class PQPanel:
    """One lattice of strings, checked from eleven dimensions and from a junction."""

    title = "(p,q) strings of type IIB"
    blurb = (
        "The fundamental string and the D1-brane belong to one lattice: a (p,q) string has "
        "tension |p + q tau| / 2 pi alpha' with tau = C_0 + i/g_s, and SL(2,Z) permutes the "
        "charges while the Einstein-frame tension stays put.  Drag the coupling past 1 and "
        "the two exchange which is heavy.  Left: the multiplet against g_s.  Right: three "
        "strings meeting at a point, each leaving along the phase of p + q tau -- no angle "
        "is chosen, and the junction balances because the charges add to zero."
    )
    controls = (
        Slider("coupling", "string coupling  g_s", 0.05, 5.0, 0.5, step=0.01, log=True),
        Slider("axion", "axion  C_0", -1.5, 1.5, 0.0, step=0.01),
        Integer("p", "charge  p", -3, 4, 1),
        Integer("q", "charge  q", -3, 4, 1),
    )

    def compute(
        self,
        coupling: float = 0.5,
        axion: float = 0.0,
        p: int = 1,
        q: int = 1,
    ) -> PQStrings:
        conv = Conventions()
        if p == 0 and q == 0:
            raise ValueError("(0, 0) is not a string -- give it a charge")
        tau = axio_dilaton(coupling, axion)

        grid = np.geomspace(0.05, 5.0, 200)
        wanted = [(1, 0), (0, 1), (1, 1)]
        if (p, q) not in wanted:
            wanted.append((p, q))
        curves = tuple(
            (
                f"({a}, {b})",
                np.array([tension(a, b, axio_dilaton(g, axion), conv) for g in grid]),
            )
            for a, b in wanted
        )

        junction = Junction(charges=_JUNCTION, tau=tau)
        _reduced = reduce_coupling(tau)

        return PQStrings(
            p=p,
            q=q,
            tau=tau,
            coupling=coupling,
            axion=axion,
            tension=tension(p, q, tau, conv),
            membrane=membrane_tension(p, q, tau, None, conv),
            membrane_residual=membrane_residual(p, q, tau, conv),
            s_residual=duality_residual(S, p, q, tau),
            t_residual=duality_residual(T, p, q, tau),
            s_charges=transform_charges(S, p, q),
            junction_force=junction_residual(junction, conv),
            junction_charge=junction.total_charge,
            binding=binding_energy(p, q, tau, conv),
            primitive=is_bound_state(p, q),
            reduced=_reduced.tau,
            reduction_steps=_reduced.steps,
            d1_tension=tension(0, 1, tau, conv),
            d1_from_dbrane=dp_brane_tension(1, coupling, conv),
            grid=grid,
            curves=curves,
        )

    def draw(self, result: PQStrings) -> Figure:
        """The package's figure, with the current coupling marked on the multiplet."""
        figure = plot_pq_strings(
            [(label, result.grid, values) for label, values in result.curves],
            [(f"$({result.p}, {result.q})$ at $g_s = {result.coupling:.3g}$",
              _JUNCTION, result.tau)],
            path=None,
        )
        axes = figure.axes[0]
        axes.axvline(result.coupling, color="tab:green", lw=1.4)
        axes.annotate(
            f"$g_s = {result.coupling:.3g}$",
            xy=(result.coupling, 0.97),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=9,
            color="tab:green",
        )
        # The tension curves keep the legend, in the corner the falling D1
        # leaves free; the marker names itself where a reader is already looking.
        axes.legend(loc="lower left", frameon=False, fontsize=8)
        return figure

    def readout(self, result: PQStrings) -> list[Line]:
        lines = [
            Line("charges", f"(p, q) = ({result.p}, {result.q})",
                 "a single string" if result.primitive else "several at threshold"),
            Line("axio-dilaton",
                 f"tau = {result.tau.real:+.4f} {result.tau.imag:+.4f}i",
                 f"fundamental domain: {result.reduced.real:+.4f} "
                 f"{result.reduced.imag:+.4f}i in {result.reduction_steps} step(s)"),
            Line(
                "tension",
                f"{result.tension:.8f}",
                f"M2 on a torus cycle: {result.membrane:.8f}",
                ok=result.membrane_residual < 1e-9,
            ),
            Line(
                "  relative difference",
                f"{result.membrane_residual:.2e}",
                "ten dimensions vs eleven",
                ok=result.membrane_residual < 1e-9,
            ),
            Line(
                f"S: ({result.p}, {result.q}) -> {result.s_charges}",
                f"Einstein tension moves by {result.s_residual:.2e}",
                "charges from the algebra, not fitted",
                ok=result.s_residual < 1e-9,
            ),
            Line(
                "T: an axion shift",
                f"Einstein tension moves by {result.t_residual:.2e}",
                ok=result.t_residual < 1e-9,
            ),
            Line(
                f"junction {_charges(_JUNCTION)}",
                f"net force {result.junction_force:.2e}",
                f"total charge {result.junction_charge}",
                ok=result.junction_force < 1e-9,
            ),
            Line("binding energy", f"{result.binding:+.6f}",
                 "positive: the bound state is lighter than its parts"),
        ]
        if result.axion_is_zero:
            lines.append(
                Line(
                    "the D1 at C_0 = 0",
                    f"T_(0,1) = {result.d1_tension:.8f}",
                    f"dp_brane_tension(1, g_s) = {result.d1_from_dbrane:.8f}",
                    ok=result.d1_residual < 1e-9,
                )
            )
        else:
            lines.append(
                Line(
                    "the D1 against dp_brane_tension",
                    "only at C_0 = 0",
                    f"|tau| = {abs(result.tau):.6f} rather than 1/g_s = "
                    f"{1.0 / result.coupling:.6f}",
                )
            )
        return lines


def _charges(charges: tuple[tuple[int, int], ...]) -> str:
    return " + ".join(f"({p},{q})" for p, q in charges)
