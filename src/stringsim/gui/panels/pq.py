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
    background = (
        "Type IIB has two two-form potentials: the one every string couples to, and one "
        "from the Ramond-Ramond sector that a D1-brane couples to instead.  A string can "
        "carry both charges, p units of the first and q of the second, and (1, 0) is the "
        "fundamental string while (0, 1) is the D1.  They are not two kinds of object "
        "but two members of one lattice.",
        "The coupling and the axion sit in one complex number, tau = C_0 + i/g_s, and "
        "the tension of a (p,q) string is |p + q tau| / 2 pi alpha'.  At weak coupling "
        "the D1 is heavy, at strong coupling it is light, and at g_s = 1 the two weigh "
        "the same -- which is the fixed point of the transformation that exchanges them.",
        "SL(2,Z) acts on tau by tau -> (a tau + b)/(c tau + d) and on (p, q) as a "
        "doublet.  The Einstein-frame tension is invariant under it.  How the charges "
        "have to transform is not chosen to make that work: it is read off the algebra, "
        "which is why the residual printed beside it is a check.",
        "The same tensions come out of eleven dimensions.  Type IIB on a circle is "
        "M-theory on a torus, and a (p,q) string is a single M2-brane wrapping the "
        "(p, q) cycle of that torus.  One route multiplies a membrane tension by a "
        "cycle length; the other evaluates |p + q tau|/2 pi alpha'.  They share nothing "
        "but string theory.",
        "A BPS (p,q) string cannot point wherever it likes: it must run along the phase "
        "of p + q tau, pulling with |p + q tau|.  So when three of them meet, the "
        "angles are fixed by the charges and the coupling, and the junction holds still "
        "only because the charges sum to zero.  Nothing about the geometry is imposed; "
        "the balance is what comes out.",
    )
    suggestions = (
        "Drag the coupling from one end to the other.  The blue (1,0) line is flat and "
        "the orange (0,1) line falls like 1/g_s; they cross at g_s = 1, and which "
        "string is the heavy one changes there.",
        "Set the charges to (1,0) and read the S row: the fundamental string becomes "
        "(0,1), the D1.  Set them to (0,1) and it comes back as (-1,0).  The sign is "
        "the orientation.",
        "Set (p, q) to (2, 2).  The binding energy drops to zero: a charge with a "
        "common factor is two (1,1) strings sitting at threshold rather than one bound "
        "object, and the readout says 'several at threshold' instead of 'a single "
        "string'.",
        "Move the axion off zero.  The junction's arms swing round, the net force stays "
        "at zero, and the line comparing the D1 against dp_brane_tension stops claiming "
        "an equality that only holds at C_0 = 0.",
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
                    declined=True,
                )
            )
        return lines


    def notes(self, result: PQStrings) -> list[str]:
        """Which string is heavy here, and what the coupling is doing to it."""
        out: list[str] = []
        if result.coupling < 1.0 - 1e-9:
            out.append(
                f"At g_s = {result.coupling:.3g} the theory is weakly coupled and the "
                "D1-brane is the heavy object -- its tension goes like 1/g_s while the "
                "fundamental string's does not.  This is the regime where calling the "
                "fundamental string 'the' string is harmless."
            )
        elif result.coupling > 1.0 + 1e-9:
            out.append(
                f"At g_s = {result.coupling:.3g} the D1-brane is the lighter object of the two.  "
                "Nothing distinguishes it as a brane rather than a string any more, and "
                "an S transformation relabels the strongly coupled theory as a weakly "
                "coupled one with the two swapped.  Which object is fundamental is a "
                "question about the coupling, not about the theory."
            )
        else:
            out.append(
                "g_s = 1 is the fixed point of S.  The fundamental string and the "
                "D1-brane have exactly the same tension here, and no measurement "
                "distinguishes them."
            )

        if result.primitive:
            out.append(
                f"({result.p}, {result.q}) has coprime charges, so it is a single bound "
                f"string.  It is lighter than its constituents by {result.binding:.4f} "
                "-- the triangle inequality, since p and q tau point in different "
                "directions -- and that deficit is the binding energy."
            )
        else:
            out.append(
                f"({result.p}, {result.q}) has a common factor, so it is not one object "
                "but several copies of the primitive string sitting exactly at "
                "threshold.  Its binding energy against them is zero, which is what the "
                "readout reports rather than comparing it against (1,0) and (0,1) and "
                "claiming a spurious binding."
            )

        out.append(
            f"tau = {result.tau.real:+.4f} {result.tau.imag:+.4f}i sits "
            + (
                "already inside the fundamental domain of SL(2,Z)"
                if result.reduction_steps == 0
                else f"outside the fundamental domain; {result.reduction_steps} step(s) "
                f"of SL(2,Z) walk it to {result.reduced.real:+.4f} "
                f"{result.reduced.imag:+.4f}i"
            )
            + ".  Inequivalent type IIB vacua are labelled by that domain, and the "
            "reduction reuses the very function that finds it for the one-loop "
            "worldsheet -- the same group, a different tau."
        )
        out.append(
            "The junction on the right is drawn from the charges alone.  Each arm "
            "leaves along the phase of p + q tau with a length set by its tension, and "
            f"the sum of those vectors is {result.junction_force:.1e}.  Move the "
            "coupling and the arms swing; the sum does not move off zero, because "
            "(1,0) + (0,1) + (-1,-1) = (0,0) and nothing else was ever required."
        )
        return out


def _charges(charges: tuple[tuple[int, int], ...]) -> str:
    return " + ".join(f"({p},{q})" for p, q in charges)
