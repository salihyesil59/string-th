r"""``(p, q)`` strings: type IIB's ``SL(2,Z)``, and a junction that balances itself.

Type IIB has a fundamental string and a D1-brane, and they are not different
kinds of object.  A duality group ``SL(2,Z)`` acts on the axio-dilaton

.. math::  \tau = C_0 + \frac{i}{g_s}

and rotates one into the other, so what exists is a lattice of strings labelled
by coprime integers :math:`(p, q)` -- ``p`` units of fundamental charge and
``q`` of D1 charge.  Their tension is

.. math::  T_{p,q} = \frac{|p + q\tau|}{2\pi\alpha'} ,

which at :math:`(1,0)` is the fundamental string and at :math:`(0,1)` is
:math:`1/2\pi\alpha' g_s` -- what
:func:`stringsim.branes.dbrane.dp_brane_tension` returns for a D1, with no
mention of duality anywhere in it.

**The formula is not put in; it comes from eleven dimensions.**  Type IIB on a
circle is M-theory on a torus, and a :math:`(p,q)` string is an M2-brane
wrapping the :math:`(p,q)` cycle of that torus.  A cycle of a torus with
modulus :math:`\tau` and side :math:`L` has length :math:`L|p+q\tau|`, so the
wrapped membrane has tension :math:`T_{M2} L |p+q\tau|` -- and matching the
:math:`(1,0)` case to the fundamental string forces

.. math::  L = \frac{1}{2\pi\alpha'\,T_{M2}} = 2\pi R_{11} ,

the circumference of the M-theory circle.  Both sides of that come from
:mod:`stringsim.branes.mtheory` and neither knows about ``SL(2,Z)``.
:func:`membrane_tension` does the calculation and
:func:`membrane_residual` compares it.

**The invariant statement is in Einstein frame.**  Under
:math:`\tau \to (a\tau+b)/(c\tau+d)` the charges go to
:math:`(p, q) \to (pd + qb,\; pc + qa)`, and

.. math::  \frac{|p + q\tau|}{\sqrt{\mathrm{Im}\,\tau}}

is unchanged.  So the string tension is not invariant but the *spectrum* is: a
different duality frame relabels which string is called fundamental.
:func:`einstein_tension` and :func:`duality_residual` check it, and
:func:`transform_charges` is where the transformation lives.

**A junction balances because charge is conserved, and for no other reason.**
Strings of charge :math:`(p_i, q_i)` meeting at a point pull with force
:math:`T_i` along their own direction, and the direction a BPS
:math:`(p,q)` string takes is the phase of :math:`p + q\tau`.  So the total
force is

.. math::
   \sum_i T_i \hat n_i = \frac{1}{2\pi\alpha'}\sum_i (p_i + q_i \tau)
   = \frac{1}{2\pi\alpha'}\Bigl[\sum_i p_i + \tau \sum_i q_i\Bigr] ,

which vanishes exactly when the charges do.  Mechanical equilibrium and charge
conservation are the same equation.  :func:`junction_residual` computes it, and
it is zero to the last bit for a conserved junction and of order one otherwise.

**Which charges are a bound state.**  :math:`|p + q\tau| < |p| + |q||\tau|`
unless the two terms are parallel, so a :math:`(p,q)` string is lighter than the
:math:`p` fundamental strings and :math:`q` D1-branes it is made of -- it binds.
The binding is real only for :math:`\gcd(p,q) = 1`; otherwise the state is
:math:`\gcd` copies of a lighter one sitting at threshold, and
:func:`binding_energy` returns zero for it.

Reference: J. H. Schwarz, *An SL(2,Z) multiplet of type IIB superstrings*,
Phys. Lett. B **360** (1995) 13; J. Polchinski, *String Theory* Vol. II,
section 14.1.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from math import gcd

import numpy as np

from ..amplitudes.oneloop import fundamental_domain_representative
from ..units import Conventions
from .mtheory import m2_tension, m_theory_radius

__all__ = [
    "axio_dilaton",
    "coupling_from_tau",
    "tension",
    "einstein_tension",
    "transform_charges",
    "act_on_tau",
    "duality_residual",
    "is_bound_state",
    "binding_energy",
    "Junction",
    "junction_residual",
    "junction_angles",
    "membrane_tension",
    "membrane_residual",
    "reduce_coupling",
]


def axio_dilaton(g_s: float = 1.0, axion: float = 0.0) -> complex:
    r""":math:`\tau = C_0 + i/g_s`.  Weak coupling is high in the upper half plane."""
    if g_s <= 0:
        raise ValueError("the string coupling must be positive")
    return complex(axion, 1.0 / g_s)


def coupling_from_tau(tau: complex) -> float:
    r"""``g_s = 1 / Im tau``, inverting :func:`axio_dilaton`."""
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half plane")
    return 1.0 / tau.imag


def _check_charges(p: int, q: int) -> None:
    if p == 0 and q == 0:
        raise ValueError("(0, 0) is not a string")


def tension(p: int, q: int, tau: complex, conventions: Conventions | None = None) -> float:
    r""":math:`|p + q\tau| / 2\pi\alpha'`, the string-frame tension.

    ``(1, 0)`` is the fundamental string and ``(0, 1)`` the D1-brane, and at
    :math:`C_0 = 0` the second is :math:`1/2\pi\alpha' g_s` -- which is what
    :func:`stringsim.branes.dbrane.dp_brane_tension` gives for ``p = 1``.
    """
    _check_charges(p, q)
    conv = conventions or Conventions()
    return abs(p + q * tau) / (2.0 * math.pi * conv.alpha_prime)


def einstein_tension(
    p: int, q: int, tau: complex, conventions: Conventions | None = None
) -> float:
    r""":math:`|p + q\tau| / (\sqrt{\mathrm{Im}\,\tau}\, 2\pi\alpha')`.

    The combination that ``SL(2,Z)`` leaves alone once the charges are
    transformed with it.  The string-frame tension is not invariant, and it
    should not be: changing frame changes which string is being called
    fundamental.
    """
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half plane")
    return tension(p, q, tau, conventions) / math.sqrt(tau.imag)


def act_on_tau(matrix, tau: complex) -> complex:
    r""":math:`\tau \to (a\tau + b)/(c\tau + d)`."""
    (a, b), (c, d) = np.asarray(matrix, dtype=float)
    lower = c * tau + d
    if lower == 0:
        raise ValueError("the transformation is singular at this tau")
    return (a * tau + b) / lower


def transform_charges(matrix, p: int, q: int) -> tuple[int, int]:
    r""":math:`(p, q) \to (pd + qb,\; pc + qa)`.

    Read off the requirement that :func:`einstein_tension` be invariant:
    :math:`p + q\tau'` equals :math:`[(pc+qa)\tau + (pd+qb)]/(c\tau+d)`, and the
    denominator is exactly what :math:`\sqrt{\mathrm{Im}\,\tau'}` supplies.  So
    the charges are a doublet, and this is which doublet.

    Under ``S`` the fundamental string becomes the D1-brane and back, with a
    sign that is the orientation.
    """
    (a, b), (c, d) = np.asarray(matrix, dtype=np.int64)
    return int(p * d + q * b), int(p * c + q * a)


def duality_residual(matrix, p: int, q: int, tau: complex) -> float:
    """Relative change of :func:`einstein_tension` under one duality element.

    Zero.  The transformation of the charges is not fitted to make it so -- it
    is read off the algebra -- so this is a check and not a definition.
    """
    moved = einstein_tension(p, q, act_on_tau(matrix, tau))
    here = einstein_tension(*transform_charges(matrix, p, q), tau)
    return abs(moved / here - 1.0)


def is_bound_state(p: int, q: int) -> bool:
    r"""``gcd(p, q) == 1``: a single string rather than several at threshold."""
    _check_charges(p, q)
    return gcd(abs(p), abs(q)) == 1


def binding_energy(
    p: int, q: int, tau: complex, conventions: Conventions | None = None
) -> float:
    r"""How much lighter the bound state is than its constituents.

    :math:`|p| T_{1,0} + |q| T_{0,1} - T_{p,q}`, which is positive by the
    triangle inequality unless :math:`p` and :math:`q\tau` point the same way.
    A :math:`(p, q)` with a common factor is that many copies of a lighter
    string sitting exactly at threshold, and the binding energy of *those*
    against each other is zero -- which this reports by comparing against the
    primitive charges rather than against the constituents.
    """
    _check_charges(p, q)
    divisor = gcd(abs(p), abs(q))
    if divisor > 1:
        primitive = divisor * tension(p // divisor, q // divisor, tau, conventions)
        return primitive - tension(p, q, tau, conventions)
    apart = abs(p) * tension(1, 0, tau, conventions) + abs(q) * tension(
        0, 1, tau, conventions
    )
    return apart - tension(p, q, tau, conventions)


@dataclass(frozen=True)
class Junction:
    """Strings of given charges meeting at a point."""

    charges: tuple[tuple[int, int], ...]
    tau: complex

    def __post_init__(self) -> None:
        if len(self.charges) < 3:
            raise ValueError("a junction needs at least three strings")

    @property
    def total_charge(self) -> tuple[int, int]:
        """``(sum p, sum q)``.  Zero for a junction that can exist."""
        return (
            sum(p for p, _ in self.charges),
            sum(q for _, q in self.charges),
        )

    @property
    def conserved(self) -> bool:
        return self.total_charge == (0, 0)

    def __str__(self) -> str:  # pragma: no cover - display only
        listed = ", ".join(f"({p},{q})" for p, q in self.charges)
        return f"[{listed}] total {self.total_charge}"


def junction_residual(junction: Junction, conventions: Conventions | None = None) -> float:
    r""":math:`|\sum_i T_i \hat n_i|`, the net force on the meeting point.

    Each string pulls along the phase of :math:`p + q\tau` with strength
    :math:`|p + q\tau|`, so the sum is
    :math:`\sum_i (p_i + q_i\tau)` and vanishes exactly when the charges do.
    Nothing about angles is imposed: the directions come from the charges and
    the balance comes out.
    """
    conv = conventions or Conventions()
    total = sum(p + q * junction.tau for p, q in junction.charges)
    return abs(total) / (2.0 * math.pi * conv.alpha_prime)


def junction_angles(junction: Junction) -> np.ndarray:
    r"""The direction each string leaves in, in radians.

    :math:`\arg(p + q\tau)`.  A BPS string of charge :math:`(p,q)` is not free
    to point anywhere -- the coupling fixes its angle, which is why a junction
    is rigid and why changing :math:`\tau` deforms it.
    """
    return np.array(
        [cmath.phase(p + q * junction.tau) for p, q in junction.charges]
    )


def membrane_tension(
    p: int, q: int, tau: complex, g_s: float | None = None,
    conventions: Conventions | None = None,
) -> float:
    r"""The same tension from an M2-brane wrapping the :math:`(p,q)` cycle.

    :math:`T_{M2} \times 2\pi R_{11} \times |p + q\tau|`, with both factors from
    :mod:`stringsim.branes.mtheory`.  The side length is *not* fitted: matching
    the :math:`(1,0)` string to the fundamental one forces
    :math:`L = 1/2\pi\alpha' T_{M2}`, and that is :math:`2\pi R_{11}` because
    :math:`\ell_p^3 = g_s\alpha'^{3/2}`.

    ``g_s`` defaults to the one :math:`\tau` encodes.
    """
    _check_charges(p, q)
    conv = conventions or Conventions()
    coupling = coupling_from_tau(tau) if g_s is None else g_s
    side = 2.0 * math.pi * m_theory_radius(coupling, conv)
    return m2_tension(coupling, conv) * side * abs(p + q * tau)


def membrane_residual(
    p: int, q: int, tau: complex, conventions: Conventions | None = None
) -> float:
    """Relative difference between the eleven- and ten-dimensional routes.

    Zero for every charge and every coupling.  One side multiplies a membrane
    tension by a cycle length, the other evaluates :math:`|p+q\\tau|/2\\pi\\alpha'`;
    the only thing they share is that both are string theory.
    """
    return abs(
        membrane_tension(p, q, tau, None, conventions) / tension(p, q, tau, conventions)
        - 1.0
    )


def reduce_coupling(tau: complex):
    r"""Walk the axio-dilaton into the fundamental domain of ``SL(2,Z)``.

    Reuses :func:`stringsim.amplitudes.oneloop.fundamental_domain_representative`
    -- the same function, the same group, a different :math:`\tau`.  There it
    says the string has no ultraviolet region; here it says that inequivalent
    type IIB vacua are labelled by the fundamental domain, and that a
    strongly-coupled one is a weakly-coupled one with the strings relabelled.

    Returns the :class:`~stringsim.amplitudes.oneloop.Domain` it produces, whose
    ``matrix`` is what :func:`transform_charges` should be given.
    """
    return fundamental_domain_representative(tau)
