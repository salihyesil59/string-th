r"""D-branes: where open strings end, and the gauge theory that lives there.

Dirichlet boundary conditions pin an open-string endpoint to a surface.  A
``Dp``-brane is a ``p``-spatial-dimensional such surface, and it is not a
background choice imposed by hand -- T-duality forces it.  Dualising a
direction exchanges Neumann and Dirichlet conditions, so a theory of ordinary
open strings, viewed in the dual variables, is a theory of strings ending on a
brane.

Two results are implemented here.

**Tension.**

.. math::
   T_p = \frac{1}{(2\pi)^p g_s \, \alpha'^{(p+1)/2}} .

The :math:`1/g_s` is the point: a D-brane is heavy at weak coupling, so it is
invisible in perturbation theory yet becomes light -- and dominant -- at strong
coupling.  That single power of ``g_s`` is why D-branes carry so much of the
non-perturbative story.

**Stretched strings.**  A string running from a brane at :math:`x_a` to one at
:math:`x_b` has a minimum length, and stretching costs tension times length:

.. math::
   M^2 = \left(\frac{|x_a - x_b|}{2\pi\alpha'}\right)^2 + \frac{N - 1}{\alpha'} .

When the branes coincide the level-1 states become massless, and ``N``
coincident branes give :math:`N^2` massless vectors -- a ``U(N)`` gauge field.
Separating the branes gives mass to the off-diagonal ones: the Higgs mechanism,
realised geometrically, with the Higgs vev literally a distance.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from ..units import Conventions

__all__ = [
    "dp_brane_tension",
    "StretchedLevel",
    "stretched_spectrum",
    "BraneStack",
    "gauge_group",
]


def dp_brane_tension(p: int, g_s: float = 1.0, conventions: Conventions | None = None) -> float:
    r"""``T_p = 1 / ((2 pi)^p g_s alpha'^{(p+1)/2})``, mass per unit ``p``-volume."""
    c = conventions or Conventions()
    if p < 0:
        raise ValueError("p must be non-negative")
    if g_s <= 0:
        raise ValueError("g_s must be positive")
    return 1.0 / ((2.0 * math.pi) ** p * g_s * c.alpha_prime ** ((p + 1) / 2.0))


@dataclass(frozen=True)
class StretchedLevel:
    """One mass level of a string stretched between two branes."""

    level: int
    separation: float
    mass_squared: float
    degeneracy: int

    @property
    def mass(self) -> float:
        """``sqrt(M^2)``, or ``-sqrt(-M^2)`` for the tachyonic level."""
        return math.copysign(math.sqrt(abs(self.mass_squared)), self.mass_squared)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"N={self.level}  M^2={self.mass_squared:+.5f}  x{self.degeneracy}"


def stretched_spectrum(
    separation: float,
    n_max: int = 3,
    conventions: Conventions | None = None,
) -> list[StretchedLevel]:
    r"""Mass levels of an open string between two parallel ``Dp``-branes.

    Parameters
    ----------
    separation:
        Distance ``|x_a - x_b|`` between the branes.
    n_max:
        Highest oscillator level.
    conventions:
        Sets ``alpha'`` and ``D``.

    Notes
    -----
    The stretching energy adds in quadrature and never cancels the
    normal-ordering constant's sign at level 0: the open bosonic string keeps
    its tachyon at small separation, and is stabilised only once
    :math:`|x_a - x_b| > 2\pi\sqrt{\alpha'}`.  That threshold is returned by
    :func:`tachyon_free_separation`.
    """
    c = conventions or Conventions()
    if separation < 0:
        raise ValueError("separation must be non-negative")
    from ..quantum.partition import oscillator_degeneracies

    degen = oscillator_degeneracies(n_max, c.transverse_dim)
    stretch = (separation / (2.0 * math.pi * c.alpha_prime)) ** 2
    return [
        StretchedLevel(
            level=n,
            separation=separation,
            mass_squared=stretch + (n - 1) / c.alpha_prime,
            degeneracy=degen[n],
        )
        for n in range(n_max + 1)
    ]


def tachyon_free_separation(conventions: Conventions | None = None) -> float:
    r"""Smallest separation at which the level-0 state is no longer tachyonic.

    ``M^2 >= 0`` at level 0 requires ``d >= 2 pi sqrt(alpha')``.
    """
    c = conventions or Conventions()
    return 2.0 * math.pi * c.string_length


@dataclass(frozen=True)
class BraneStack:
    """A set of parallel ``Dp``-branes at given transverse positions."""

    positions: tuple[float, ...]
    tolerance: float = 1e-9

    @property
    def multiplicities(self) -> list[int]:
        """How many branes sit at each distinct location, largest stack first."""
        rounded = [round(x / self.tolerance) * self.tolerance for x in self.positions]
        return sorted(Counter(rounded).values(), reverse=True)

    @property
    def massless_vectors(self) -> int:
        r"""``sum n_i^2``: the dimension of the unbroken gauge group."""
        return sum(n * n for n in self.multiplicities)

    def group_name(self) -> str:
        """``'U(2) x U(1)'`` and so on."""
        return " x ".join(f"U({n})" for n in self.multiplicities)


def gauge_group(positions) -> str:
    r"""Unbroken gauge group of a stack of parallel branes.

    ``N`` coincident branes give ``U(N)``; pulling one away breaks
    ``U(N) -> U(N-1) x U(1)``, and the vectors that lost their masslessness are
    exactly the strings that now have to stretch.
    """
    return BraneStack(tuple(float(x) for x in positions)).group_name()


__all__ += ["tachyon_free_separation"]
