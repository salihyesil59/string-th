r"""The Dirac-Born-Infeld action, and what a string looks like from the brane.

:mod:`stringsim.branes.dbrane` treats a D-brane as a place where strings end and
counts the states.  This module gives the brane its own dynamics.  Its action is

.. math::
   S = -T_p \int d^{p+1}\xi \;
       \sqrt{-\det\left(\eta_{ab} + \partial_a X \partial_b X
                        + 2\pi\alpha' F_{ab}\right)} ,

with :math:`X` one transverse position and :math:`F` the gauge field living on
the brane.  Nothing here is quoted: :func:`dbi_matrix` builds the matrix and
:func:`dbi_determinant` takes its determinant, and the closed form

.. math::  -\det M = (1 + |\nabla X|^2)(1 - |e|^2) + (e \cdot \nabla X)^2,
           \qquad e = 2\pi\alpha' E,

is a *test* rather than the implementation.

**There is a largest electric field.**  With no transverse deformation the
determinant is :math:`1 - |e|^2`, so the Lagrangian goes imaginary at

.. math::  E_{\text{crit}} = \frac{1}{2\pi\alpha'} ,

which is the fundamental string tension.  Pulling on a string with that force
cancels its tension, and :func:`critical_field` is where the brane stops making
sense.  Below it the expansion of the square root reproduces Maxwell with a
definite first correction, and :func:`electric_series` measures the
coefficients from the function instead of writing them down.

**The spike is a string.**  Legendre-transforming in the electric field --
numerically, in :func:`displacement`, and in closed form in
:func:`energy_density` -- gives

.. math::
   \mathcal{H} = T_p \sqrt{(1+|\nabla X|^2)(1+|D|^2) - |D \times \nabla X|^2}
               = T_p \sqrt{(1 + D\cdot\nabla X)^2 + |D - \nabla X|^2} ,

so :math:`\mathcal{H} \geq T_p (1 + D\cdot\nabla X)` with equality exactly when
:math:`D = \nabla X`.  That is the BPS condition, and
:func:`bogomolny_gap` measures how far a configuration is from it.

For the saturating solution the energy above the flat brane is

.. math::
   T_p \int D\cdot\nabla X \, d^p x = T_p \int \nabla\!\cdot\!(X D)
   = T_p\, X(r_0) \oint D \cdot dS ,

a boundary term: the spike's energy is proportional to its *height*, which is
what a string of that length would weigh.  With the flux quantised,
:math:`\oint (2\pi\alpha' T_p D)\cdot dS = n`, the constant of proportionality
is exactly :math:`n/(2\pi\alpha')` -- ``n`` fundamental strings.
:func:`bion_tension` integrates it numerically and the tests require it to come
out at :math:`n\,T_{F1}` to a part in :math:`10^{10}`.

The profile itself is the harmonic function :math:`X = q/r^{p-2}`, so the brane
is flat far away and stretches into a spike where the strings end --
:func:`bion_profile`.  ``p >= 3`` only: at ``p = 2`` the harmonic function is a
logarithm and the spike has no finite height, which :func:`bion_charge` refuses
rather than papering over.
"""

from __future__ import annotations

import math

import numpy as np

from ..units import Conventions
from .dbrane import dp_brane_tension

__all__ = [
    "dbi_matrix",
    "dbi_determinant",
    "dbi_determinant_closed",
    "lagrangian_density",
    "critical_field",
    "electric_series",
    "displacement",
    "reduced_displacement",
    "energy_density",
    "bogomolny_gap",
    "sphere_area",
    "bion_charge",
    "bion_profile",
    "bion_flux",
    "bion_tension",
]


def dbi_matrix(grad_x, electric, alpha_prime: float = 1.0) -> np.ndarray:
    r"""``M_ab = eta_ab + d_a X d_b X + 2 pi alpha' F_ab`` for a static configuration.

    ``grad_x`` and ``electric`` are ``p``-vectors: the gradient of the
    transverse scalar and the electric field, both along the brane.  The result
    is ``(p+1) x (p+1)`` with mostly-plus signature and ``F_{0i} = E_i``.
    """
    grad = np.asarray(grad_x, dtype=float).reshape(-1)
    field = np.asarray(electric, dtype=float).reshape(-1)
    if grad.shape != field.shape:
        raise ValueError(f"grad_x and electric must match: {grad.shape} against {field.shape}")
    scaled = 2.0 * math.pi * alpha_prime * field
    size = grad.size
    matrix = np.eye(size + 1)
    matrix[0, 0] = -1.0
    matrix[1:, 1:] += np.outer(grad, grad)
    matrix[0, 1:] += scaled
    matrix[1:, 0] -= scaled
    return matrix


def dbi_determinant(grad_x, electric, alpha_prime: float = 1.0) -> float:
    """``-det M``, computed from the matrix itself."""
    return float(-np.linalg.det(dbi_matrix(grad_x, electric, alpha_prime)))


def dbi_determinant_closed(grad_x, electric, alpha_prime: float = 1.0) -> float:
    r"""``(1 + |grad X|^2)(1 - |e|^2) + (e . grad X)^2`` with ``e = 2 pi alpha' E``.

    The closed form the literature writes down.  It is here to be compared with
    :func:`dbi_determinant`, not to be used in its place.
    """
    grad = np.asarray(grad_x, dtype=float).reshape(-1)
    scaled = 2.0 * math.pi * alpha_prime * np.asarray(electric, dtype=float).reshape(-1)
    return float((1.0 + grad @ grad) * (1.0 - scaled @ scaled) + (scaled @ grad) ** 2)


def lagrangian_density(
    grad_x, electric, p: int, g_s: float = 1.0, conventions: Conventions | None = None
) -> float:
    """``-T_p sqrt(-det M)``.  Raises past the critical field, where it is imaginary."""
    conv = conventions or Conventions()
    determinant = dbi_determinant(grad_x, electric, conv.alpha_prime)
    if determinant <= 0.0:
        raise ValueError(
            f"the determinant is {determinant:.3e}: the field is at or beyond critical, "
            "where the Lagrangian is imaginary"
        )
    return -dp_brane_tension(p, g_s, conv) * math.sqrt(determinant)


def critical_field(conventions: Conventions | None = None) -> float:
    r"""``E_crit = 1 / (2 pi alpha')``, which is also the fundamental string tension.

    At this field the square root vanishes for a flat brane: the force on a
    string endpoint cancels the string's own tension, and the brane can no
    longer hold it.  Nothing in the action is special about this number except
    that it is where the determinant changes sign.
    """
    conv = conventions or Conventions()
    return 1.0 / (2.0 * math.pi * conv.alpha_prime)


def electric_series(n_terms: int = 4) -> np.ndarray:
    r"""Coefficients of ``sqrt(1 - (E/E_crit)^2)`` in powers of ``(E/E_crit)^2``.

    The series is in ``E/E_crit``, so ``alpha'`` drops out of it entirely.
    Extracted by Cauchy's formula -- the coefficients are Fourier modes of the
    function on a circle inside its branch cut -- so they come out of
    :math:`\sqrt{1-x}` itself at machine precision rather than from the
    binomial series.  The first two are ``1`` and ``-1/2``, the constant brane
    tension and Maxwell; the rest are the corrections that keep the energy
    finite right up to :func:`critical_field`.
    """
    if n_terms < 2:
        raise ValueError("ask for at least two terms")
    # Cauchy's formula on a circle inside the branch cut at 1: a_k is a discrete
    # Fourier coefficient of the function sampled on |z| = radius.
    radius, samples = 0.5, 512
    angles = 2.0 * math.pi * np.arange(samples) / samples
    values = np.sqrt(1.0 + 0j - radius * np.exp(1j * angles))
    transform = np.fft.fft(values) / samples
    return np.real(transform[:n_terms] / radius ** np.arange(n_terms))


def energy_density(
    grad_x, reduced_displacement, p: int, g_s: float = 1.0, conventions: Conventions | None = None
) -> float:
    r"""``T_p sqrt((1+|grad X|^2)(1+|D|^2) - |D x grad X|^2)``.

    ``reduced_displacement`` is ``D = d / T_p`` with ``d = dL/dE`` the momentum
    conjugate to the electric field, so it is dimensionless like ``grad X``.
    The cross-product norm is written as ``|D|^2|grad X|^2 - (D . grad X)^2``,
    which is what it means in any number of dimensions.
    """
    conv = conventions or Conventions()
    grad = np.asarray(grad_x, dtype=float).reshape(-1)
    disp = np.asarray(reduced_displacement, dtype=float).reshape(-1)
    if grad.shape != disp.shape:
        raise ValueError(f"shapes must match: {grad.shape} against {disp.shape}")
    cross = (disp @ disp) * (grad @ grad) - (disp @ grad) ** 2
    inside = (1.0 + grad @ grad) * (1.0 + disp @ disp) - cross
    return dp_brane_tension(p, g_s, conv) * math.sqrt(inside)


def displacement(
    grad_x, electric, p: int, g_s: float = 1.0, conventions: Conventions | None = None,
    step: float = 1e-6,
) -> np.ndarray:
    """``d = dL/dE``, by central differences on :func:`lagrangian_density`.

    Doing the Legendre transform numerically means :func:`energy_density` can be
    checked against it instead of both being written from the same source.
    """
    field = np.asarray(electric, dtype=float).reshape(-1)
    out = np.empty_like(field)
    for index in range(field.size):
        shift = np.zeros_like(field)
        shift[index] = step
        forward = lagrangian_density(grad_x, field + shift, p, g_s, conventions)
        backward = lagrangian_density(grad_x, field - shift, p, g_s, conventions)
        out[index] = (forward - backward) / (2.0 * step)
    return out


def reduced_displacement(
    grad_x, electric, p: int, g_s: float = 1.0, conventions: Conventions | None = None,
    step: float = 1e-6,
) -> np.ndarray:
    r"""``D = d / (2 pi alpha' T_p)``, the dimensionless partner of ``grad X``.

    :func:`displacement` returns ``d = dL/dE``, which carries the units of the
    gauge field.  Dividing by ``2 pi alpha' T_p`` -- the same factor that turns
    ``E`` into the dimensionless ``e`` in :func:`dbi_matrix` -- gives the
    variable :func:`energy_density` and :func:`bogomolny_gap` want, and the one
    whose flux is quantised in integers.  Getting this factor wrong leaves every
    formula shaped correctly and every number wrong, so it lives here rather
    than in the caller.
    """
    conv = conventions or Conventions()
    scale = 2.0 * math.pi * conv.alpha_prime * dp_brane_tension(p, g_s, conv)
    return displacement(grad_x, electric, p, g_s, conv, step) / scale


def bogomolny_gap(
    grad_x, reduced_displacement, p: int, g_s: float = 1.0, conventions: Conventions | None = None
) -> float:
    r"""``H - T_p (1 + D . grad X)``: non-negative, and zero exactly at ``D = grad X``.

    Follows from the identity

    .. math::
       (1+|a|^2)(1+|b|^2) - |a \times b|^2 = (1 + a\cdot b)^2 + |a - b|^2 ,

    so the gap vanishes only when the two vectors coincide.  That is the BPS
    condition, and the number this returns is how far a configuration misses it.
    """
    conv = conventions or Conventions()
    grad = np.asarray(grad_x, dtype=float).reshape(-1)
    disp = np.asarray(reduced_displacement, dtype=float).reshape(-1)
    bound = dp_brane_tension(p, g_s, conv) * (1.0 + disp @ grad)
    return energy_density(grad, disp, p, g_s, conv) - bound


# ---------------------------------------------------------------------------
# the BIon spike
# ---------------------------------------------------------------------------


def sphere_area(dim: int) -> float:
    r"""Area of the unit ``S^{d-1}``: ``2 pi^{d/2} / Gamma(d/2)``."""
    if dim < 1:
        raise ValueError("dim must be at least 1")
    return 2.0 * math.pi ** (dim / 2.0) / math.gamma(dim / 2.0)


def bion_charge(
    n_strings: int, p: int, g_s: float = 1.0, conventions: Conventions | None = None
) -> float:
    r"""``q`` in ``X = q / r^{p-2}`` for a spike carrying ``n`` fundamental strings.

    Fixed by the flux: :math:`\oint 2\pi\alpha' T_p D \cdot dS = n` with
    :math:`D = \nabla X`, which gives
    :math:`2\pi\alpha' T_p \Omega_{p-1}(p-2)\,q = n`.

    ``p >= 3``: at ``p = 2`` the harmonic function is a logarithm, so the spike
    has no finite height and no well-defined charge, and this refuses instead of
    returning something wrong.
    """
    if p < 3:
        raise ValueError(f"a spike needs p >= 3; at p = {p} the harmonic function is a logarithm")
    if n_strings == 0:
        return 0.0
    conv = conventions or Conventions()
    tension = dp_brane_tension(p, g_s, conv)
    return n_strings / (
        2.0 * math.pi * conv.alpha_prime * tension * sphere_area(p) * (p - 2)
    )


def bion_profile(
    radii, n_strings: int, p: int, g_s: float = 1.0, conventions: Conventions | None = None
) -> np.ndarray:
    """``X(r) = q / r^(p-2)``: flat far away, a spike at the origin."""
    radius = np.asarray(radii, dtype=float)
    if np.any(radius <= 0.0):
        raise ValueError("radii must be positive; the spike is singular at r = 0")
    return bion_charge(n_strings, p, g_s, conventions) / radius ** (p - 2)


def bion_flux(
    radius: float, n_strings: int, p: int, g_s: float = 1.0, conventions: Conventions | None = None
) -> float:
    r"""``2 pi alpha' T_p \oint D . dS`` through a sphere of the given radius.

    Independent of the radius, because ``X`` is harmonic, and equal to
    ``n_strings`` by construction -- which is the point of checking it.
    """
    conv = conventions or Conventions()
    charge = bion_charge(n_strings, p, g_s, conv)
    slope = (p - 2) * charge / radius ** (p - 1)  # |dX/dr|
    return (
        2.0
        * math.pi
        * conv.alpha_prime
        * dp_brane_tension(p, g_s, conv)
        * sphere_area(p)
        * radius ** (p - 1)
        * slope
    )


def bion_tension(
    n_strings: int,
    p: int,
    g_s: float = 1.0,
    conventions: Conventions | None = None,
    inner: float = 1e-3,
    n_points: int = 400_000,
) -> float:
    r"""Energy of the spike per unit height, integrated numerically.

    The BPS energy above the flat brane is
    :math:`T_p \int |\nabla X|^2 d^p x` over ``r >= inner``, and the height is
    ``X(inner)``; the ratio should be ``n_strings / (2 pi alpha')``, the tension
    of that many fundamental strings.  The integral converges, so the answer
    does not depend on ``inner`` -- which the tests check by varying it.
    """
    conv = conventions or Conventions()
    charge = bion_charge(n_strings, p, g_s, conv)
    if charge == 0.0:
        return 0.0
    outer = inner * 1e12
    radius = np.geomspace(inner, outer, n_points)
    slope = (p - 2) * charge / radius ** (p - 1)
    integrand = slope**2 * sphere_area(p) * radius ** (p - 1)
    energy = dp_brane_tension(p, g_s, conv) * float(np.trapezoid(integrand, radius))
    height = charge / inner ** (p - 2)
    return energy / height
