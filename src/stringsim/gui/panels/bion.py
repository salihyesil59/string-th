r"""A string ending on a brane, seen as a spike the brane makes to hold it.

A fundamental string cannot simply stop.  Its tension has to be carried by
something, and when it ends on a D-brane the brane itself carries it: the
transverse position runs away near the endpoint as the harmonic function
:math:`X = q/r^{p-2}`, and that funnel *is* the string.  The picture is a slice
of the brane; what varies with the number of strings is not the height -- every
spike is infinitely tall -- but the width of the mouth.

Three things are checked while the controls move, and none of them is the
formula being restated.

The flux :math:`2\pi\alpha' T_p \oint \nabla X \cdot dS` through a sphere is
evaluated at several radii.  It comes out equal to the number of strings and
independent of the radius, which it must be because :math:`X` is harmonic.

The energy of the spike above the flat brane, :math:`T_p\int|\nabla X|^2`, is
integrated numerically and divided by its height.  The ratio is
:math:`n/2\pi\alpha'` -- the tension of that many fundamental strings -- so the
brane is holding exactly what it should be.

And the Bogomolny bound: the energy density is never below
:math:`T_p(1 + D\cdot\nabla X)`, with equality only at :math:`D = \nabla X`.
``D`` here is not written down twice; it is the Legendre transform
:math:`\partial L/\partial E` taken by central differences on the Lagrangian, so
the two sides of the comparison are computed differently on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...branes.dbi import (
    bion_charge,
    bion_flux,
    bion_tension,
    bogomolny_gap,
    critical_field,
    dbi_determinant,
    dbi_determinant_closed,
    energy_density,
    reduced_displacement,
)
from ...units import Conventions
from ...viz.plots import plot_bion_spike
from ..panel import Integer, Line, Slider

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["BIonPanel", "BIon"]

_RADII = (0.05, 0.3, 1.7)
"""Where the flux is measured.  Three radii, because one would prove nothing."""


@dataclass(frozen=True)
class BIon:
    """One spike, and the flat brane it rises out of."""

    n_strings: int
    p: int
    charge: float
    flux: tuple[float, ...]
    tension: float
    string_tension: float
    """``n / 2 pi alpha'``: what ``n`` fundamental strings weigh per unit length."""
    slope: float
    gradient: np.ndarray
    displacement: np.ndarray
    """``D = dL/dE``, by central differences rather than from a formula."""
    gap: float
    energy: float
    bound: float
    determinant_matrix: float
    determinant_closed: float
    critical: float
    profiles: tuple[tuple[str, float], ...]
    """``(label, charge)`` for each string number drawn."""
    probe_fraction: tuple[float, float]
    """The probe field, as a fraction of ``E_crit`` in each direction."""

    @property
    def flux_spread(self) -> float:
        """How much the measured flux varies with the radius.  It should not."""
        return float(max(self.flux) - min(self.flux))

    @property
    def flux_error(self) -> float:
        return max(abs(value - self.n_strings) for value in self.flux)

    @property
    def tension_ratio(self) -> float:
        return self.tension / self.string_tension

    @property
    def bps(self) -> bool:
        return abs(self.gap) < 1e-12

    @property
    def determinant_gap(self) -> float:
        return abs(self.determinant_matrix - self.determinant_closed)


class BIonPanel:
    """The brane holding a string up, and the three ways of checking that it is."""

    title = "A string ending on a brane"
    blurb = (
        "A fundamental string cannot just stop: its tension has to go somewhere.  When it "
        "ends on a D-brane the brane carries it, rising into a spike X = q / r^(p-2) whose "
        "funnel is the string.  Every spike is infinitely tall, so what more strings "
        "change is the width of the mouth, not the depth -- the panels are clipped at a "
        "common height to show that.  The readout measures the flux through spheres of "
        "three different radii, integrates the spike's energy, and divides by its height."
    )
    background = (
        "The Dirac-Born-Infeld action is the low-energy action of a D-brane: "
        "-T_p sqrt(-det(g + 2 pi alpha' F)) with the induced metric g.  Expanded to "
        "quadratic order it is Maxwell plus a scalar for each transverse direction; kept "
        "whole it knows about strong fields, and the two things this panel shows are "
        "both invisible in the quadratic approximation.",
        "The first is that a static solution with a spike exists.  Take the brane flat "
        "except for one transverse coordinate X(r), and switch on a radial electric "
        "field.  When 2 pi alpha' E equals grad X the square root collapses to a perfect "
        "square, the energy saturates a bound, and X is harmonic -- q / r^(p-2) in p "
        "spatial dimensions.  That is a BPS solution: it costs the least energy for its "
        "charge, so nothing makes it decay.",
        "The second is what the spike weighs.  The energy above the flat brane is "
        "T_p integral |grad X|^2, which converges, and the height is X at the inner "
        "radius.  Their ratio is n / 2 pi alpha' -- exactly n times the tension of a "
        "fundamental string.  The brane is not merely shaped like a string is pulling on "
        "it; it is carrying that pull, in the right amount.",
        "The electric field cannot be raised past E_crit = 1 / 2 pi alpha'.  There the "
        "determinant changes sign and the Lagrangian would be imaginary.  That number is "
        "also the fundamental string's tension, and the coincidence is the physics: at "
        "the critical field the force on a string endpoint cancels the string's own "
        "tension and the brane can no longer hold it.",
        "At p = 2 there is no spike at all.  The harmonic function in two spatial "
        "dimensions is a logarithm, so the funnel has no finite height and the charge is "
        "not defined; the module refuses rather than returning a number, which is why "
        "the control starts at 3.",
    )
    suggestions = (
        "Raise the number of strings.  The mouth of the funnel widens and the depth does "
        "not, because q is proportional to n and every spike is infinitely tall anyway.",
        "Read the three flux numbers.  They are measured through spheres of radius 0.05, "
        "0.3 and 1.7, and they are the same number -- the string count -- because X is "
        "harmonic and the flux does not know where you put the surface.",
        "Watch the tension ratio as p changes.  The integral, the sphere area and the "
        "brane tension all change with the dimension and the ratio does not: the spike "
        "always weighs n fundamental strings per unit height.",
        "Move the slope.  The Legendre transform D is taken numerically from the "
        "Lagrangian, and it lands on grad X at every value, which is what makes the "
        "Bogomolny gap vanish rather than being arranged to.",
    )
    controls = (
        Integer("n_strings", "strings ending here  n", 1, 6, 3),
        Integer("p", "brane dimension  p", 3, 6, 3),
        Slider("slope", "slope  |grad X|", 0.05, 2.0, 0.30, step=0.01),
    )

    def compute(
        self,
        n_strings: int = 3,
        p: int = 3,
        slope: float = 0.30,
    ) -> BIon:
        conv = Conventions()
        charge = bion_charge(n_strings, p, 1.0, conv)

        # The flat-brane part, in two worldvolume directions.  At
        # 2 pi alpha' E = grad X the configuration is BPS, and D is *not* set
        # to grad X -- it is differentiated out of the Lagrangian.
        gradient = np.array([float(slope), 0.0])
        bps_field = gradient / (2.0 * math.pi * conv.alpha_prime)
        displacement = reduced_displacement(gradient, bps_field, p, 1.0, conv)

        # A sub-critical probe field, so the determinant compared below is one
        # the Lagrangian is real at.  My first choice was past E_crit, where
        # both routes still agree -- on a number describing nothing.
        probe = np.array([0.4, 0.2]) * critical_field(conv)
        return BIon(
            n_strings=n_strings,
            p=p,
            charge=charge,
            flux=tuple(bion_flux(r, n_strings, p, 1.0, conv) for r in _RADII),
            tension=bion_tension(n_strings, p, 1.0, conv),
            string_tension=n_strings / (2.0 * math.pi * conv.alpha_prime),
            slope=float(slope),
            gradient=gradient,
            displacement=displacement,
            gap=bogomolny_gap(gradient, displacement, p, 1.0, conv),
            energy=energy_density(gradient, displacement, p, 1.0, conv),
            bound=energy_density(gradient, displacement, p, 1.0, conv)
            - bogomolny_gap(gradient, displacement, p, 1.0, conv),
            determinant_matrix=dbi_determinant(gradient, probe, conv.alpha_prime),
            determinant_closed=dbi_determinant_closed(gradient, probe, conv.alpha_prime),
            critical=critical_field(conv),
            profiles=tuple(
                (f"n = {n}", bion_charge(n, p, 1.0, conv))
                for n in range(1, n_strings + 1)
            ),
            probe_fraction=(0.4, 0.2),
        )

    def draw(self, result: BIon) -> Figure:
        """One panel per string number, all clipped at the same depth."""
        power = result.p - 2

        def funnel(charge: float):
            return lambda radius: charge / np.asarray(radius, dtype=float) ** power

        return plot_bion_spike(
            [(label, funnel(charge)) for label, charge in result.profiles],
            path=None,
            extent=1.0,
            n_grid=90,
            title=f"A string ending on a D{result.p}-brane",
        )

    def readout(self, result: BIon) -> list[Line]:
        return [
            Line("configuration", f"{result.n_strings} string(s) on a D{result.p}-brane",
                 f"charge q = {result.charge:.6f} in X = q / r^{result.p - 2}"),
            Line(
                "flux through a sphere",
                ", ".join(f"{value:.6f}" for value in result.flux),
                f"at r = {', '.join(str(r) for r in _RADII)}; it counts the strings",
                ok=result.flux_error < 1e-9 and result.flux_spread < 1e-12,
            ),
            Line(
                "energy per unit height",
                f"{result.tension:.8f} by integration",
                f"n / 2 pi alpha' = {result.string_tension:.8f}",
                ok=abs(result.tension_ratio - 1.0) < 1e-6,
            ),
            Line("  ratio", f"{result.tension_ratio:.8f}"),
            Line(
                "Legendre transform at the BPS field",
                f"D = ({result.displacement[0]:.8f}, {result.displacement[1]:.8f})",
                f"grad X = ({result.gradient[0]:.8f}, {result.gradient[1]:.8f})",
                ok=bool(np.allclose(result.displacement, result.gradient, atol=1e-7)),
            ),
            Line(
                "Bogomolny gap",
                f"{result.gap:+.2e}",
                f"energy {result.energy:.6f} against the bound {result.bound:.6f}",
                ok=result.bps,
            ),
            Line(
                "DBI determinant",
                f"{result.determinant_matrix:.10f} from the matrix, at E = "
                f"({result.probe_fraction[0]:g}, {result.probe_fraction[1]:g}) E_crit",
                f"{result.determinant_closed:.10f} from the closed form",
                ok=result.determinant_gap < 1e-12,
            ),
            Line(
                "critical field",
                f"E_crit = {result.critical:.8f}",
                f"the fundamental string tension 1/2 pi alpha' = "
                f"{1.0 / (2.0 * math.pi):.8f}",
                ok=abs(result.critical - 1.0 / (2.0 * math.pi)) < 1e-12,
            ),
        ]

    def notes(self, result: BIon) -> list[str]:
        """What the spike is carrying, and how that was established."""
        out = [
            f"{result.n_strings} fundamental string(s) end on a D{result.p}-brane here.  "
            f"The brane rises as X = {result.charge:.4f} / r^{result.p - 2}, which is "
            "harmonic away from the endpoint -- so the funnel is not put in by hand, it "
            "is what the equations give once the electric field is switched on.  Every "
            "such spike is infinitely tall; more strings widen the mouth rather than "
            "deepening it, because q is proportional to n."
        ]
        out.append(
            f"The flux through spheres of radius {', '.join(str(r) for r in _RADII)} is "
            f"{', '.join(f'{value:.4f}' for value in result.flux)} -- the same number "
            f"three times, and that number is {result.n_strings}.  It has to be "
            "independent of the radius because X is harmonic, and it has to be an "
            "integer because it counts strings.  Neither was imposed."
        )
        out.append(
            f"The spike's energy above the flat brane, integrated numerically, is "
            f"{result.tension:.6f} per unit height.  {result.n_strings} fundamental "
            f"strings weigh {result.string_tension:.6f} per unit length.  The ratio is "
            f"{result.tension_ratio:.8f}.  So the brane is not merely shaped as though "
            "something were pulling on it -- it is carrying that pull, in the right "
            "amount, and the amount was measured rather than assumed."
        )
        if result.bps:
            out.append(
                f"On the flat part, with |grad X| = {result.slope:.3g} and the field at "
                "2 pi alpha' E = grad X, the Legendre transform D = dL/dE comes out equal "
                "to grad X to machine precision -- and D was differentiated out of the "
                "Lagrangian by central differences, not written down.  The Bogomolny gap "
                f"is {result.gap:+.1e}: the configuration saturates its bound, which is "
                "what makes it BPS and what stops it decaying."
            )
        out.append(
            f"The field cannot be pushed past E_crit = {result.critical:.6f}, where the "
            "determinant changes sign and the Lagrangian would be imaginary.  That is "
            "also the fundamental string's tension, and the coincidence is the point: at "
            "the critical field the pull on a string endpoint cancels the string's own "
            "tension, and the brane can no longer hold it."
        )
        return out
