r"""Eleven dimensions: M2, M5, and where the ten-dimensional branes come from.

Type IIA at strong coupling grows a dimension.  The evidence is a mass formula
that the package already computes: a D0-brane weighs

.. math::  T_0 = \frac{1}{g_s\sqrt{\alpha'}} ,

and ``n`` of them weigh ``n`` times that -- a tower with spacing ``1/R`` for
:math:`R = g_s\sqrt{\alpha'}`, which is what Kaluza-Klein momentum on a circle
of that radius looks like.  :func:`m_theory_radius` is that ``R``, and
:func:`kaluza_klein_check` compares the two numbers instead of asserting they
agree.  The circle grows with :math:`g_s`, so the eleventh dimension is
invisible exactly where string perturbation theory works.

The dictionary is fixed by two relations,

.. math::  R_{11} = g_s \sqrt{\alpha'}, \qquad
           \ell_p^3 = g_s\,\alpha'^{3/2} ,

after which eleven-dimensional supergravity has only two branes,

.. math::  T_{M2} = \frac{1}{(2\pi)^2 \ell_p^3}, \qquad
           T_{M5} = \frac{1}{(2\pi)^5 \ell_p^6} ,

and *everything else in ten dimensions is one of them, wrapped or not*:

=========================  ======================================  ==================
eleven dimensions          reduces to                              tension
=========================  ======================================  ==================
M2 wrapped on the circle   fundamental string                      :math:`1/2\pi\alpha'`
M2 transverse              D2-brane                                :math:`T_2`
M5 wrapped                 D4-brane                                :math:`T_4`
M5 transverse              NS5-brane                               :math:`1/(2\pi)^5 g_s^2\alpha'^3`
momentum on the circle     D0-brane                                :math:`1/g_s\sqrt{\alpha'}`
=========================  ======================================  ==================

:func:`reduction_table` produces those five numbers from the eleven-dimensional
side and checks each against :func:`stringsim.branes.dbrane.dp_brane_tension`
or the string tension, which is the whole content of the claim: **the same four
constants, arrived at two ways, agree exactly.**  Note the NS5 goes like
:math:`1/g_s^2`, not :math:`1/g_s` -- it is not a D-brane, and the reduction
says so without being told.

**Charge quantisation.**  The M2 and M5 are electric and magnetic sources for
the same three-form, so a Dirac condition ties their tensions to Newton's
constant.  With :math:`2\kappa_{11}^2 = (2\pi)^8 \ell_p^9`,

.. math::  2\kappa_{11}^2\, T_{M2}\, T_{M5} = 2\pi ,

which :func:`dirac_residual` evaluates; it comes out at zero to machine
precision for every ``g_s`` and ``alpha'``, and that is a statement about the
normalisation of :math:`\ell_p` being consistent, not a fit.

**Not here.**  The eleven-dimensional supergravity fields, the M5 worldvolume
theory (which has no Lagrangian description of the usual kind), and the
Matrix-model definition.  This module is the tension dictionary and its
consistency conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..units import Conventions
from .dbrane import dp_brane_tension

__all__ = [
    "m_theory_radius",
    "planck_length",
    "m2_tension",
    "m5_tension",
    "gravitational_coupling",
    "dirac_residual",
    "kaluza_klein_check",
    "Reduction",
    "reduction_table",
    "string_coupling_from_radius",
]


def m_theory_radius(g_s: float = 1.0, conventions: Conventions | None = None) -> float:
    r"""``R_11 = g_s sqrt(alpha')``: the radius of the eleventh dimension."""
    if g_s <= 0:
        raise ValueError("g_s must be positive")
    conv = conventions or Conventions()
    return g_s * conv.string_length


def string_coupling_from_radius(radius: float, conventions: Conventions | None = None) -> float:
    """Invert :func:`m_theory_radius`: strong coupling is a large circle."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    conv = conventions or Conventions()
    return radius / conv.string_length


def planck_length(g_s: float = 1.0, conventions: Conventions | None = None) -> float:
    r"""``l_p = g_s^{1/3} sqrt(alpha')``, from ``l_p^3 = g_s alpha'^{3/2}``."""
    if g_s <= 0:
        raise ValueError("g_s must be positive")
    conv = conventions or Conventions()
    return g_s ** (1.0 / 3.0) * conv.string_length


def m2_tension(g_s: float = 1.0, conventions: Conventions | None = None) -> float:
    r"""``T_M2 = 1 / ((2 pi)^2 l_p^3)``, mass per unit area of the membrane."""
    return 1.0 / ((2.0 * math.pi) ** 2 * planck_length(g_s, conventions) ** 3)


def m5_tension(g_s: float = 1.0, conventions: Conventions | None = None) -> float:
    r"""``T_M5 = 1 / ((2 pi)^5 l_p^6)``."""
    return 1.0 / ((2.0 * math.pi) ** 5 * planck_length(g_s, conventions) ** 6)


def gravitational_coupling(g_s: float = 1.0, conventions: Conventions | None = None) -> float:
    r"""``2 kappa_11^2 = (2 pi)^8 l_p^9``, the eleven-dimensional Newton constant."""
    return (2.0 * math.pi) ** 8 * planck_length(g_s, conventions) ** 9


def dirac_residual(g_s: float = 1.0, conventions: Conventions | None = None) -> float:
    r"""``2 kappa^2 T_M2 T_M5 / (2 pi) - 1``: zero when the charges are quantised.

    The membrane and the fivebrane are the electric and magnetic sources of the
    same three-form potential, so their tensions cannot be chosen
    independently.  This is that condition, and it holds for every ``g_s``
    because both tensions are powers of the same :math:`\ell_p`.
    """
    product = (
        gravitational_coupling(g_s, conventions)
        * m2_tension(g_s, conventions)
        * m5_tension(g_s, conventions)
    )
    return product / (2.0 * math.pi) - 1.0


def kaluza_klein_check(g_s: float = 1.0, conventions: Conventions | None = None) -> float:
    r"""``T_D0 - 1/R_11``, the relative difference; zero is the eleventh dimension.

    The D0-brane mass and the first Kaluza-Klein momentum on a circle of radius
    :math:`R_{11}` are computed from different places -- one from the D-brane
    tension formula, one from the radius -- and their agreement is the reason to
    believe the circle is there at all.
    """
    conv = conventions or Conventions()
    mass = dp_brane_tension(0, g_s, conv)
    momentum = 1.0 / m_theory_radius(g_s, conv)
    return (mass - momentum) / momentum


@dataclass(frozen=True)
class Reduction:
    """One eleven-dimensional object and the ten-dimensional one it becomes."""

    source: str
    wrapped: bool
    result: str
    tension: float
    expected: float

    @property
    def residual(self) -> float:
        """Relative difference between the two routes to the same number."""
        return abs(self.tension - self.expected) / abs(self.expected)

    def __str__(self) -> str:  # pragma: no cover - display only
        how = "wrapped" if self.wrapped else "transverse"
        return (
            f"{self.source:9s} {how:10s} -> {self.result:8s} "
            f"T = {self.tension:.8g}   (expected {self.expected:.8g}, "
            f"residual {self.residual:.1e})"
        )


def reduction_table(g_s: float = 1.0, conventions: Conventions | None = None) -> list[Reduction]:
    r"""The five reductions, each computed from eleven dimensions and checked.

    A brane wrapped on the circle contributes :math:`2\pi R_{11}` times its
    tension per unit of the remaining volume; a transverse one keeps its own
    tension.  The expected values come from
    :func:`stringsim.branes.dbrane.dp_brane_tension`, the fundamental string
    tension and the NS5 tension, none of which know anything about eleven
    dimensions.
    """
    conv = conventions or Conventions()
    radius = m_theory_radius(g_s, conv)
    circumference = 2.0 * math.pi * radius
    membrane, fivebrane = m2_tension(g_s, conv), m5_tension(g_s, conv)
    return [
        Reduction(
            "M2", True, "F1", circumference * membrane, 1.0 / (2.0 * math.pi * conv.alpha_prime)
        ),
        Reduction("M2", False, "D2", membrane, dp_brane_tension(2, g_s, conv)),
        Reduction("M5", True, "D4", circumference * fivebrane, dp_brane_tension(4, g_s, conv)),
        Reduction(
            "M5",
            False,
            "NS5",
            fivebrane,
            1.0 / ((2.0 * math.pi) ** 5 * g_s**2 * conv.alpha_prime**3),
        ),
        Reduction(
            "momentum", True, "D0", 1.0 / radius, dp_brane_tension(0, g_s, conv)
        ),
    ]
