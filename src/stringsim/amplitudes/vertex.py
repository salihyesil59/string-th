r"""Vertex operators: the Veneziano amplitude derived instead of written down.

:mod:`stringsim.amplitudes.veneziano` evaluates

.. math::  A(s,t) = B(-\alpha(s), -\alpha(t))

as a Beta function.  That is the answer, not the derivation.  Here it comes out
of the worldsheet: an integral over where the vertex operators sit on the
boundary of the disc.

**The correlator.**  A tachyon of momentum ``k`` is the operator
:math:`\,:\!e^{ik\cdot X}\!:\,`, and on the boundary of the disc
:math:`\langle X(y) X(y')\rangle = -2\alpha' \ln|y - y'|`, so

.. math::
   \Bigl\langle \prod_i :\!e^{ik_i \cdot X(y_i)}\!: \Bigr\rangle
   \;\propto\; \prod_{i<j} |y_i - y_j|^{\,2\alpha' k_i \cdot k_j} .

The exponents depend on the momenta only through the invariants, and for
tachyons :math:`k_i^2 = 1/\alpha'` they are
:math:`2\alpha' k_i\!\cdot\! k_j = -\alpha' s_{ij} - 2` with
:math:`s_{ij} = -(k_i + k_j)^2`.  :func:`four_point_exponents` builds them.

**The gauge fixing, and why it can be checked.**  The disc has an ``SL(2,R)``
of conformal maps, so three punctures can be put anywhere and the rest
integrated.  Fixing :math:`y_1 = 0`, :math:`y_{n-1} = 1`, :math:`y_n = R` and
including the Faddeev-Popov factor
:math:`|y_1 - y_{n-1}||y_1 - y_n||y_{n-1} - y_n|` leaves an ordinary integral
over the remaining punctures.  **The answer must not depend on ``R``**, and
:func:`gauge_spread` measures whether it does.  At four points and
:math:`R \to \infty` the integral collapses to the Beta function; at finite
``R`` the integrand looks nothing like it and gives the same number.

**And the invariance is a mass-shell condition.**  What makes the integrand
transform correctly is

.. math::  \sum_{j \neq i} 2\alpha' k_i \cdot k_j = -2\alpha' k_i^2 = -2 ,

one equation per puncture, which is exactly :math:`\alpha' m^2 = -1`.  So the
gauge invariance is not a formal property of the construction -- it holds only
on shell.  :func:`exponent_residual` measures those row sums, and detuning them
by a per cent breaks the ``R``-independence by a per cent, which is the check
that the invariance is doing work.

**The pole is two punctures colliding.**  As :math:`\alpha(s) \to 0` the
exponent of :math:`y_2 - y_1` reaches :math:`-1` and the integral diverges at
its endpoint.  Nothing else in the integrand is singular, so the tachyon pole
comes from the region where two vertex operators meet.  Fitting the divergence
gives the residue :math:`-1`, which is what
:func:`stringsim.amplitudes.veneziano.veneziano_residue` returns at ``n = 0``.
Higher poles need the analytic continuation the Beta function performs and are
not reachable from the convergent integral -- see :func:`converges`.

**Five points.**  Two punctures are left free instead of one, and there is no
closed form to compare against.  What can still be checked is everything
structural: gauge independence, cyclic relabelling, and the same mass-shell
condition.  That the machinery keeps working where there is nothing to look up
is the point of having built it.

Reference: J. Polchinski, *String Theory* Vol. I, sections 6.2 and 6.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate

from ..units import minkowski

__all__ = [
    "TACHYON_ALPHA_M2",
    "mandelstam_sum",
    "four_point_exponents",
    "five_point_exponents",
    "exponent_residual",
    "koba_nielsen",
    "faddeev_popov",
    "converges",
    "ordered_amplitude",
    "gauge_spread",
    "PoleFit",
    "fit_tachyon_pole",
    "tachyon_momenta",
    "exponents_from_momenta",
]

TACHYON_ALPHA_M2 = -1.0
"""``alpha' M^2`` of the open bosonic tachyon, so ``k^2 = 1/alpha'``."""


def mandelstam_sum(n_points: int = 4, alpha_prime: float = 1.0) -> float:
    r"""``s + t + u = sum of the external masses`` -- ``-4/alpha'`` at four points.

    From :math:`\sum_i k_i = 0` and :math:`k_i^2 = -m_i^2`; only the four-point
    case has three invariants, so ``n_points`` other than 4 raises.
    """
    if n_points != 4:
        raise ValueError("only the four-point case has a single Mandelstam sum")
    return -4.0 / alpha_prime


def _exponent(invariant: float, alpha_prime: float) -> float:
    r""":math:`2\alpha' k_i\cdot k_j = -\alpha' s_{ij} - 2` for two tachyons."""
    return -alpha_prime * invariant - 2.0


def four_point_exponents(s: float, t: float, alpha_prime: float = 1.0) -> np.ndarray:
    r"""The ``4 x 4`` matrix of :math:`2\alpha' k_i\cdot k_j`.

    Labelled so that the cyclic order on the boundary is ``1, 2, 3, 4`` with
    :math:`s_{12} = s` and :math:`s_{23} = t`; then ``u`` is fixed by
    :func:`mandelstam_sum` and the opposite pairs repeat, since
    :math:`k_3 + k_4 = -(k_1 + k_2)`.

    This is the ordering for which the amplitude is
    :math:`B(-\alpha(s), -\alpha(t))`; a different labelling of the same four
    particles gives one of the other two channel pairings.
    """
    u = mandelstam_sum(4, alpha_prime) - s - t
    e_s, e_t, e_u = (_exponent(x, alpha_prime) for x in (s, t, u))
    matrix = np.zeros((4, 4))
    for i, j, value in ((0, 1, e_s), (2, 3, e_s), (1, 2, e_t), (0, 3, e_t),
                        (0, 2, e_u), (1, 3, e_u)):
        matrix[i, j] = matrix[j, i] = value
    return matrix


def five_point_exponents(adjacent, alpha_prime: float = 1.0) -> np.ndarray:
    r"""Exponents for five tachyons, from the five adjacent invariants.

    ``adjacent`` is :math:`(s_{12}, s_{23}, s_{34}, s_{45}, s_{51})`.  The five
    non-adjacent invariants are not free: the mass-shell conditions
    :math:`\sum_{j\neq i} e_{ij} = -2` are five linear equations for exactly
    those five unknowns, and this solves them rather than quoting a
    parameterisation.

    Raises
    ------
    ValueError
        If the linear system is singular, which would mean the labelling is not
        what this function assumes.
    """
    adjacent = np.asarray(adjacent, dtype=float).reshape(5)
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    chord = [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]
    matrix = np.zeros((5, 5))
    for (i, j), value in zip(ring, adjacent, strict=True):
        matrix[i, j] = matrix[j, i] = _exponent(float(value), alpha_prime)

    system = np.zeros((5, 5))
    target = np.full(5, -2.0)
    for column, (i, j) in enumerate(chord):
        system[i, column] += 1.0
        system[j, column] += 1.0
    for vertex in range(5):
        target[vertex] -= sum(matrix[vertex, other] for other in range(5))
    if abs(np.linalg.det(system)) < 1e-12:
        raise ValueError("the mass-shell conditions do not determine the chords")
    solved = np.linalg.solve(system, target)
    for (i, j), value in zip(chord, solved, strict=True):
        matrix[i, j] = matrix[j, i] = value
    return matrix


def exponent_residual(matrix: np.ndarray) -> float:
    r"""How far the rows are from summing to ``-2``.

    That sum is :math:`-2\alpha' k_i^2`, so it is the mass-shell condition, and
    it is what the gauge invariance of the integral rests on.  Zero for any
    honest set of tachyon momenta; detune it and :func:`gauge_spread` notices.
    """
    matrix = np.asarray(matrix, dtype=float)
    return float(np.max(np.abs(matrix.sum(axis=1) + 2.0)))


def koba_nielsen(points, matrix: np.ndarray) -> float:
    r""":math:`\prod_{i<j} |y_i - y_j|^{e_{ij}}`, the boundary correlator.

    Returned through logarithms: the factors span many orders of magnitude
    once a puncture is far away, and multiplying them directly loses digits
    that the gauge-independence check would then be measuring instead of the
    physics.
    """
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if points.shape[0] != matrix.shape[0]:
        raise ValueError(f"{points.shape[0]} punctures against a {matrix.shape} matrix")
    total = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            gap = abs(points[i] - points[j])
            if gap <= 0.0:
                return math.inf
            total += matrix[i, j] * math.log(gap)
    return math.exp(total)


def faddeev_popov(fixed) -> float:
    r"""``|y_a - y_b| |y_b - y_c| |y_a - y_c|`` for the three fixed punctures.

    The Jacobian of dividing out ``SL(2,R)``.  Without it the "amplitude" would
    depend on where the three were nailed down.
    """
    a, b, c = (float(x) for x in fixed)
    return abs(a - b) * abs(b - c) * abs(a - c)


def converges(matrix: np.ndarray, n_points: int = 4) -> bool:
    r"""Whether the ordered integral converges as written.

    Two adjacent punctures may collide, and the integral survives that only
    when the exponent between them exceeds ``-1``.  Outside this region the
    amplitude is *defined* by analytic continuation -- which is what the Beta
    function in :mod:`~stringsim.amplitudes.veneziano` performs -- and no
    quadrature can reach it.
    """
    matrix = np.asarray(matrix, dtype=float)
    return all(
        matrix[i, (i + 1) % n_points] > -1.0 + 1e-12 for i in range(n_points)
    )


def ordered_amplitude(
    matrix: np.ndarray,
    anchor: float = 3.0,
    limit: int = 200,
    tolerance: float = 1e-11,
) -> float:
    r"""The gauge-fixed disc integral for one cyclic ordering.

    Punctures sit at :math:`y_1 = 0`, then the free ones in increasing order
    inside ``(0, 1)``, then :math:`y_{n-1} = 1` and :math:`y_n =` ``anchor``.
    Four points leave one integration, five leave two.

    ``anchor`` is the gauge choice.  Any value above 1 is legitimate and they
    must all agree; large ones lose digits, because the Faddeev-Popov factor
    grows while the integral shrinks and the two are multiplied.  Values
    between 1.5 and 10 keep the product balanced -- see :func:`gauge_spread`,
    which is what says so rather than the docstring.
    """
    matrix = np.asarray(matrix, dtype=float)
    n_points = matrix.shape[0]
    if n_points not in (4, 5):
        raise ValueError("only four and five punctures are implemented")
    if anchor <= 1.0:
        raise ValueError("the anchor must lie beyond the puncture fixed at 1")
    if not converges(matrix, n_points):
        raise ValueError(
            "an adjacent exponent is at or below -1: the ordered integral diverges, "
            "and the amplitude there is defined by continuation instead"
        )
    prefactor = faddeev_popov((0.0, 1.0, anchor))

    if n_points == 4:
        def integrand(x: float) -> float:
            return koba_nielsen((0.0, x, 1.0, anchor), matrix)

        value, _ = integrate.quad(integrand, 0.0, 1.0, limit=limit, epsabs=tolerance)
        return prefactor * value

    def inner(x: float, y: float) -> float:
        return koba_nielsen((0.0, x, y, 1.0, anchor), matrix)

    value, _ = integrate.dblquad(
        inner, 0.0, 1.0, lambda y: 0.0, lambda y: y, epsabs=tolerance, epsrel=tolerance
    )
    return prefactor * value


def gauge_spread(matrix: np.ndarray, anchors=(1.5, 2.0, 3.0, 6.0), **kwargs) -> float:
    r"""Relative spread of the amplitude over different ``SL(2,R)`` gauges.

    Zero is the statement that the three punctures really can be put anywhere.
    It is the only handle on whether the construction is right, since at five
    points there is nothing to compare the answer to.
    """
    values = [ordered_amplitude(matrix, float(a), **kwargs) for a in anchors]
    middle = float(np.mean(values))
    if middle == 0.0:  # pragma: no cover - would mean a vanishing amplitude
        return float(np.max(np.abs(values)))
    return float(np.max(np.abs(np.array(values) - middle)) / abs(middle))


@dataclass(frozen=True)
class PoleFit:
    """A fit of ``A ~ residue / alpha(s)`` as the first pole is approached."""

    residue: float
    expected: float
    samples: np.ndarray
    amplitudes: np.ndarray

    @property
    def error(self) -> float:
        return abs(self.residue - self.expected)

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"residue {self.residue:+.8f} against {self.expected:+.1f} "
            f"(error {self.error:.2e})"
        )


def fit_tachyon_pole(
    t: float = -3.0,
    alpha_prime: float = 1.0,
    offsets=(0.05, 0.02, 0.01, 0.005),
    anchor: float = 3.0,
) -> PoleFit:
    r"""Watch the tachyon pole appear as two punctures collide.

    Approaches :math:`\alpha(s) = 0` from below, where the exponent between the
    first two punctures tends to :math:`-1` and the integral diverges at
    ``x = 0``.  Multiplying by :math:`\alpha(s)` and extrapolating gives the
    residue, which is :math:`-1` -- the value
    :func:`stringsim.amplitudes.veneziano.veneziano_residue` returns at the
    zeroth pole, independent of ``t`` because the tachyon has no spin.

    Nothing else in the integrand is singular there, so this locates the pole
    in the *geometry*: it is the region where two vertex operators meet.
    """
    offsets = np.asarray(offsets, dtype=float)
    if np.any(offsets <= 0):
        raise ValueError("offsets must be positive: the pole is approached from below")
    values = []
    for offset in offsets:
        s = (-offset - 1.0) / alpha_prime
        matrix = four_point_exponents(s, t, alpha_prime)
        values.append(ordered_amplitude(matrix, anchor))
    amplitudes = np.array(values)
    residues = -offsets * amplitudes
    # alpha(s) A is analytic at the pole -- it is Gamma(1+eps) times a ratio --
    # so the samples are a smooth function of the offset and a quadratic fit
    # extrapolates to the residue.  A linear one leaves an error of 1e-4 at
    # these offsets; the quadratic is three orders better.
    order = min(2, len(offsets) - 1)
    intercept = float(np.polyval(np.polyfit(offsets, residues, order), 0.0))
    return PoleFit(
        residue=float(intercept),
        expected=-1.0,
        samples=offsets,
        amplitudes=amplitudes,
    )


def tachyon_momenta(
    s: float, t: float, dim: int = 26, alpha_prime: float = 1.0
) -> np.ndarray:
    r"""Four explicit on-shell momenta with the given ``s`` and ``t``.

    All incoming, so :math:`\sum_i k_i = 0`, each with
    :math:`k_i^2 = 1/\alpha'`.  Built in the centre-of-mass frame: the energy
    comes from ``s`` and the scattering angle from ``t``.

    Only exists in the physical region -- ``s`` above threshold and
    :math:`|\cos\theta| \leq 1` -- which is *not* where the Koba-Nielsen
    integral converges.  That is the usual state of affairs and the reason
    :func:`four_point_exponents` works with invariants instead.  What this is
    for is checking that those invariants are the ones real momenta give:
    :func:`exponents_from_momenta` recomputes the matrix from these vectors.
    """
    if dim < 4:
        raise ValueError("need at least four spacetime dimensions")
    k_squared = 1.0 / alpha_prime
    if s <= 0.0:
        raise ValueError("the centre-of-mass energy needs s > 0")
    energy = math.sqrt(s) / 2.0
    momentum_sq = energy**2 + k_squared
    cosine = (t + 2.0 * k_squared + 2.0 * energy**2) / (2.0 * momentum_sq)
    if abs(cosine) > 1.0 + 1e-12:
        raise ValueError(f"cos(theta) = {cosine:.4f} is outside the physical region")
    cosine = float(np.clip(cosine, -1.0, 1.0))
    sine = math.sqrt(max(1.0 - cosine**2, 0.0))
    p = math.sqrt(momentum_sq)

    out = np.zeros((4, dim))
    out[0, 0] = out[1, 0] = energy
    out[0, 3] = p
    out[1, 3] = -p
    out[2, 0] = out[3, 0] = -energy
    out[2, 1], out[2, 3] = -p * sine, -p * cosine
    out[3, 1], out[3, 3] = p * sine, p * cosine
    return out


def exponents_from_momenta(momenta: np.ndarray, alpha_prime: float = 1.0) -> np.ndarray:
    r"""``2 alpha' k_i . k_j`` straight from the vectors, mostly-plus metric."""
    momenta = np.asarray(momenta, dtype=float)
    eta = minkowski(momenta.shape[1])
    products = (momenta * eta) @ momenta.T
    matrix = 2.0 * alpha_prime * products
    np.fill_diagonal(matrix, 0.0)
    return matrix
