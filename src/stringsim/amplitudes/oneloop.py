r"""One loop: the moduli-space integral, and why there is no ultraviolet.

:mod:`stringsim.amplitudes.veneziano` is a tree amplitude in closed form.  At
one loop the worldsheet is a torus, and the amplitude is an integral over its
shape:

.. math::
   Z = \int_{\mathcal{F}} \frac{d^2\tau}{\tau_2^2}\;
       \left(\tau_2^{1/2}\,|\eta(\tau)|^2\right)^{-(D-2)} ,

with :math:`\mathcal{F}` the fundamental domain of :math:`SL(2,\mathbb{Z})`.
Both factors are modular invariant on their own -- the measure by inspection,
the integrand because :math:`\tau_2^{1/2}|\eta|^2` is, which
:func:`modular_residual` checks at machine precision under both generators
rather than deducing from :math:`\eta(-1/\tau) = \sqrt{-i\tau}\,\eta(\tau)`.

**That invariance is what removes the ultraviolet.**  A field theory would
integrate the Schwinger parameter down to zero and diverge there.  Here
:math:`\tau_2 \to 0` is not a separate region at all: it is the image of large
:math:`\tau_2` under :math:`S`, and :func:`fundamental_domain_representative`
walks any point into :math:`\mathcal{F}`, where

.. math::  \tau_2 \geq \frac{\sqrt3}{2} \approx 0.866 .

Hand it :math:`\tau = -2.7 + 0.011i`, as deep into the would-be ultraviolet as
you like, and it comes back at :math:`\tau_2 \approx 0.99`.  There is nothing to
regulate because there is nowhere to regulate.

**What does diverge is the infrared, and it is the tachyon.**  As
:math:`\tau_2 \to \infty` the integrand grows like :math:`e^{-\pi\alpha' M^2
\tau_2}` for the lightest state, so the bosonic string diverges like
:math:`e^{4\pi\tau_2}`.  :func:`tachyon_alpha_m2` measures the rate off the
integrand and returns :math:`-4`, which is
:func:`stringsim.quantum.spectrum.closed_bosonic_spectrum` at ``N = 0`` --
arrived at from the amplitude rather than from the spectrum.

**The superstring integrand vanishes.**  Its numerator is
:math:`\theta_3^4 - \theta_4^4 - \theta_2^4`, zero by the abstruse identity, so
the one-loop cosmological constant is zero point by point on the upper half
plane and not merely after integration -- :func:`superstring_torus_integrand`.
That is the same identity :func:`stringsim.quantum.partition.jacobi_identity_residual`
proves on the ``q``-series, here as a statement about functions.

**The cylinder is two theories at once.**  Between two D\ ``p``-branes the same
diagram is a loop of *open* strings in the modulus ``t`` and a tree exchange of
*closed* strings in :math:`s = 1/t`; :math:`\eta(i/t) = \sqrt{t}\,\eta(it)`
turns one integrand into the other, and :func:`channel_duality_residual`
verifies that instead of asserting it.  For the superstring the cylinder
integrand vanishes too: **parallel BPS branes exert no force**, which is the
same cancellation between NS-NS attraction and R-R repulsion that the Jacobi
identity encodes.

**Not here.**  Higher genus, amplitudes with vertex operators inserted, and the
open-string annulus with different branes at the two ends.  This module is the
vacuum diagram and the structure of its moduli space.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

from ..quantum.partition import dedekind_eta
from ..units import Conventions

__all__ = [
    "jacobi_theta",
    "abstruse_residual",
    "modular_t",
    "modular_s",
    "in_fundamental_domain",
    "fundamental_domain_representative",
    "FUNDAMENTAL_DOMAIN_FLOOR",
    "torus_integrand",
    "modular_residual",
    "superstring_torus_integrand",
    "torus_amplitude",
    "large_tau_behaviour",
    "LargeTauFit",
    "tachyon_alpha_m2",
    "annulus_integrand",
    "closed_channel_integrand",
    "channel_duality_residual",
    "superstring_annulus_integrand",
    "Domain",
]

#: The smallest ``Im tau`` anywhere in the fundamental domain, at its corners.
FUNDAMENTAL_DOMAIN_FLOOR = math.sqrt(3.0) / 2.0

_TOL = 1e-9


# ---------------------------------------------------------------------------
# theta functions
# ---------------------------------------------------------------------------


def jacobi_theta(which: int, tau: complex, n_terms: int = 120) -> complex:
    r"""``theta_2``, ``theta_3`` or ``theta_4`` at zero argument, as products.

    .. math::
       \theta_2 = 2q^{1/8}\prod (1-q^n)(1+q^n)^2, \quad
       \theta_3 = \prod (1-q^n)(1+q^{n-1/2})^2, \quad
       \theta_4 = \prod (1-q^n)(1-q^{n-1/2})^2,

    with :math:`q = e^{2\pi i\tau}`.  The product converges quickly for
    ``Im tau`` not too small; below about ``0.05`` ask for more terms.
    """
    if which not in (2, 3, 4):
        raise ValueError(f"which must be 2, 3 or 4, got {which}")
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half plane")
    q = cmath.exp(2j * cmath.pi * tau)
    product = 1.0 + 0.0j
    for n in range(1, n_terms + 1):
        common = 1.0 - q**n
        if which == 2:
            product *= common * (1.0 + q**n) ** 2
        elif which == 3:
            product *= common * (1.0 + q ** (n - 0.5)) ** 2
        else:
            product *= common * (1.0 - q ** (n - 0.5)) ** 2
    return 2.0 * q**0.125 * product if which == 2 else product


def abstruse_residual(tau: complex, n_terms: int = 120) -> float:
    r"""``|theta_3^4 - theta_2^4 - theta_4^4|``, zero by Jacobi's identity.

    :func:`stringsim.quantum.partition.jacobi_identity_residual` proves this on
    the integer ``q``-series; this is the same statement about the functions
    themselves, and it is what makes the superstring's one-loop amplitude
    vanish pointwise.
    """
    values = [jacobi_theta(which, tau, n_terms) ** 4 for which in (3, 2, 4)]
    return abs(values[0] - values[1] - values[2])


# ---------------------------------------------------------------------------
# the modular group
# ---------------------------------------------------------------------------


def modular_t(tau: complex) -> complex:
    """``tau -> tau + 1``."""
    return tau + 1.0


def modular_s(tau: complex) -> complex:
    """``tau -> -1/tau``."""
    if tau == 0:
        raise ValueError("S is singular at tau = 0")
    return -1.0 / tau


def in_fundamental_domain(tau: complex, tol: float = 1e-9) -> bool:
    r"""``|Re tau| <= 1/2`` and ``|tau| >= 1``, the standard domain."""
    return abs(tau.real) <= 0.5 + tol and abs(tau) >= 1.0 - tol


@dataclass(frozen=True)
class Domain:
    """Where a point in the upper half plane really lives.

    Attributes
    ----------
    tau:
        The representative inside the fundamental domain.
    matrix:
        The integer ``SL(2,Z)`` element taking the original point to it, as
        ``[[a, b], [c, d]]`` acting by ``(a tau + b)/(c tau + d)``.
    steps:
        How many generators were applied.
    """

    tau: complex
    matrix: np.ndarray
    steps: int

    @property
    def determinant(self) -> int:
        """Always ``1``; the search only ever multiplies by ``T`` and ``S``."""
        return int(round(float(np.linalg.det(self.matrix))))

    def apply(self, point: complex) -> complex:
        """Act with :attr:`matrix` on any point, to check it reproduces :attr:`tau`."""
        a, b = self.matrix[0]
        c, d = self.matrix[1]
        return (a * point + b) / (c * point + d)


def fundamental_domain_representative(tau: complex, max_steps: int = 400) -> Domain:
    r"""Walk ``tau`` into the fundamental domain with ``T`` and ``S``.

    Shift by an integer until :math:`|\mathrm{Re}\,\tau| \leq 1/2`, then invert
    if :math:`|\tau| < 1`, and repeat.  Each inversion strictly increases
    :math:`\tau_2` when :math:`|\tau| < 1`, so the walk terminates.

    The point of it: a point with tiny :math:`\tau_2` -- the region a field
    theory would call ultraviolet -- comes back with
    :math:`\tau_2 \geq \sqrt3/2`.  The ultraviolet of the string is not a region
    of moduli space that has been cut off; it is not there.
    """
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half plane")
    matrix = np.eye(2, dtype=object)
    steps = 0
    for _ in range(max_steps):
        shift = math.floor(tau.real + 0.5)
        if shift:
            tau = tau - shift
            matrix = np.array([[1, -shift], [0, 1]], dtype=object) @ matrix
            steps += 1
        if abs(tau) < 1.0 - _TOL:
            tau = -1.0 / tau
            matrix = np.array([[0, -1], [1, 0]], dtype=object) @ matrix
            steps += 1
        else:
            return Domain(tau=tau, matrix=matrix.astype(np.int64), steps=steps)
    raise RuntimeError(f"no representative found in {max_steps} steps")


# ---------------------------------------------------------------------------
# the torus
# ---------------------------------------------------------------------------


def torus_integrand(tau: complex, dim: int = 26, n_terms: int = 200) -> float:
    r"""``(tau_2^{1/2} |eta(tau)|^2)^{-(D-2)}``, the modular-invariant part.

    The measure :math:`d^2\tau/\tau_2^2` is invariant on its own, so the whole
    integrand is.  Physically this is a trace over the transverse oscillators
    of the closed string, weighted by :math:`q^{L_0}\bar q^{\bar L_0}`.

    ``n_terms`` matters at small :math:`\tau_2`.  The product for :math:`\eta`
    converges in powers of :math:`|q| = e^{-2\pi\tau_2}`, which is close to 1
    there, so a point at :math:`\tau_2 = 0.01` needs thousands of terms rather
    than hundreds.  Modular invariance then holds to a part in :math:`10^{11}`
    instead of a part in :math:`10^{3}`; the gap is truncation, not physics, and
    the tests check that it closes.
    """
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half plane")
    if dim < 3:
        raise ValueError("dim must exceed 2")
    eta = dedekind_eta(tau, n_terms)
    return float((math.sqrt(tau.imag) * abs(eta) ** 2) ** (-(dim - 2)))


def modular_residual(tau: complex, dim: int = 26, n_terms: int = 200) -> tuple[float, float]:
    """Relative change of :func:`torus_integrand` under ``T`` and ``S``.

    Both must vanish.  They are computed from the integrand directly, so this
    tests the implementation and not the identity it was derived from.
    """
    base = torus_integrand(tau, dim, n_terms)
    shifted = torus_integrand(modular_t(tau), dim, n_terms)
    inverted = torus_integrand(modular_s(tau), dim, n_terms)
    return abs(shifted / base - 1.0), abs(inverted / base - 1.0)


def superstring_torus_integrand(tau: complex, n_terms: int = 120) -> float:
    r"""The type II one-loop integrand: zero, pointwise.

    Its numerator is :math:`|\theta_3^4 - \theta_4^4 - \theta_2^4|^2`, which the
    abstruse identity kills.  So the one-loop vacuum energy vanishes before any
    integration -- the cancellation is between states, not between regions of
    moduli space.
    """
    numerator = abstruse_residual(tau, n_terms) ** 2
    eta = dedekind_eta(tau, max(n_terms, 200))
    return float(numerator / (tau.imag**4 * abs(eta) ** 24))


def torus_amplitude(
    dim: int = 26, tau2_max: float = 4.0, n_tau1: int = 60, n_tau2: int = 240
) -> float:
    r"""``int_F d^2tau / tau_2^2`` times the integrand, cut off at ``tau2_max``.

    The domain is :math:`|\mathrm{Re}\,\tau| \leq 1/2`, :math:`|\tau| \geq 1`,
    :math:`\tau_2 \leq` ``tau2_max``.  The cut-off is not a regulator for a
    divergence of principle: it is there because the integral genuinely diverges
    at large :math:`\tau_2`, and how fast is :func:`tachyon_alpha_m2`.
    """
    if tau2_max <= FUNDAMENTAL_DOMAIN_FLOOR:
        raise ValueError(f"tau2_max must exceed {FUNDAMENTAL_DOMAIN_FLOOR:.4f}")
    tau2 = np.linspace(FUNDAMENTAL_DOMAIN_FLOOR, tau2_max, n_tau2)
    tau1 = np.linspace(-0.5, 0.5, n_tau1)
    total = np.zeros_like(tau2)
    for index, height in enumerate(tau2):
        row = 0.0
        for real in tau1:
            point = complex(real, height)
            if abs(point) < 1.0:
                continue
            row += torus_integrand(point, dim) / height**2
        total[index] = row * (tau1[1] - tau1[0])
    return float(np.trapezoid(total, tau2))


@dataclass(frozen=True)
class LargeTauFit:
    """What the torus integrand looks like far out in the infrared.

    Attributes
    ----------
    alpha_m2:
        ``alpha' M^2`` of the lightest state, from the exponential rate.
    log_power:
        The power of ``tau_2`` multiplying it, which must be ``-(D-2)/2``.
    residual:
        Largest absolute deviation of the fit from the sampled logarithm.
    """

    alpha_m2: float
    log_power: float
    residual: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"alpha' M^2 = {self.alpha_m2:+.6f}, tau_2 power {self.log_power:+.4f}, "
            f"residual {self.residual:.1e}"
        )


def large_tau_behaviour(dim: int = 26, low: float = 4.0, high: float = 10.0) -> LargeTauFit:
    r"""Fit ``log I = -pi alpha' M^2 tau_2 + b log tau_2 + c`` on the integrand.

    Both fitted numbers are checks.  The exponential rate is the mass of the
    lightest closed string, and the power of :math:`\tau_2` has to come out at
    :math:`-(D-2)/2` -- the transverse momentum integral -- because the
    integrand is :math:`\tau_2^{-(D-2)/2}|\eta|^{-2(D-2)}`.  Fitting only the
    exponential and ignoring the power gives ``-3.13`` instead of ``-4``, which
    is how one learns the subleading term is not optional.
    """
    if low <= FUNDAMENTAL_DOMAIN_FLOOR or high <= low:
        raise ValueError("need FUNDAMENTAL_DOMAIN_FLOOR < low < high")
    heights = np.linspace(low, high, 30)
    values = np.array([math.log(torus_integrand(complex(0.0, h), dim)) for h in heights])
    design = np.column_stack([heights, np.log(heights), np.ones_like(heights)])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = float(np.max(np.abs(design @ coefficients - values)))
    return LargeTauFit(
        alpha_m2=float(-coefficients[0] / math.pi),
        log_power=float(coefficients[1]),
        residual=residual,
    )


def tachyon_alpha_m2(dim: int = 26, low: float = 4.0, high: float = 10.0) -> float:
    r"""``alpha' M^2`` of the lightest state, measured from the integrand's growth.

    For the closed bosonic string the answer is :math:`-4`, the tachyon, which
    is :func:`stringsim.quantum.spectrum.closed_bosonic_spectrum` at ``N = 0``
    reached from the amplitude side instead of the spectrum side.
    """
    return large_tau_behaviour(dim, low, high).alpha_m2


# ---------------------------------------------------------------------------
# the cylinder
# ---------------------------------------------------------------------------


def annulus_integrand(
    modulus: float,
    p: int,
    separation: float = 1.0,
    conventions: Conventions | None = None,
    n_terms: int = 200,
) -> float:
    r"""The open-string loop between two D\ ``p``-branes, per unit ``dt/t``.

    .. math::
       t^{-(p+1)/2}\,
       e^{-\,y^2 t / 2\pi\alpha'}\;\eta(it)^{-24} ,

    a trace over open strings stretched between the branes: the exponential is
    their tension times length, the power of ``t`` the momentum integral along
    the brane, and :math:`\eta^{-24}` the oscillators.
    """
    if modulus <= 0:
        raise ValueError("the modulus must be positive")
    conv = conventions or Conventions()
    eta = dedekind_eta(complex(0.0, modulus), n_terms)
    exponent = -(separation**2) * modulus / (2.0 * math.pi * conv.alpha_prime)
    return float(
        modulus ** (-(p + 1) / 2.0) * math.exp(exponent) * abs(eta) ** (-24)
    )


def closed_channel_integrand(
    modulus: float,
    p: int,
    separation: float = 1.0,
    conventions: Conventions | None = None,
    n_terms: int = 200,
) -> float:
    r"""The same diagram read as closed strings exchanged, per unit ``ds/s``.

    Substituting :math:`t = 1/s` and using
    :math:`\eta(i/s) = \sqrt{s}\,\eta(is)` turns the open-channel integrand into

    .. math::
       s^{(p+1)/2 - 12}\, e^{-\,y^2 / 2\pi\alpha' s}\;\eta(is)^{-24} ,

    where the exponential is now a propagator: a closed string emitted by one
    brane and absorbed by the other.  Long cylinders, large ``s``, are dominated
    by the lightest closed string.
    """
    if modulus <= 0:
        raise ValueError("the modulus must be positive")
    conv = conventions or Conventions()
    eta = dedekind_eta(complex(0.0, modulus), n_terms)
    exponent = -(separation**2) / (2.0 * math.pi * conv.alpha_prime * modulus)
    return float(
        modulus ** ((p + 1) / 2.0 - 12.0) * math.exp(exponent) * abs(eta) ** (-24)
    )


def channel_duality_residual(
    modulus: float,
    p: int,
    separation: float = 1.0,
    conventions: Conventions | None = None,
    n_terms: int = 200,
) -> float:
    """Relative difference between the two channels at ``t`` and ``s = 1/t``.

    Zero: they are the same integral in different variables, and the equality is
    the modular transform of ``eta`` rather than a coincidence.  This is what
    makes a one-loop open-string diagram a statement about closed-string
    exchange, and hence about gravity.
    """
    open_channel = annulus_integrand(modulus, p, separation, conventions, n_terms)
    closed = closed_channel_integrand(1.0 / modulus, p, separation, conventions, n_terms)
    return abs(open_channel / closed - 1.0)


def superstring_annulus_integrand(
    modulus: float, p: int, separation: float = 1.0, n_terms: int = 120
) -> float:
    r"""The supersymmetric cylinder: zero, so parallel BPS branes do not attract.

    The bosonic and fermionic open strings between the branes cancel level by
    level, which is again :math:`\theta_3^4 - \theta_4^4 - \theta_2^4 = 0`.  In
    the closed channel the same zero is the NS-NS attraction cancelling the R-R
    repulsion, so the statement about forces and the statement about the
    spectrum are one identity seen twice.
    """
    if modulus <= 0:
        raise ValueError("the modulus must be positive")
    del p, separation  # the prefactors are finite; only the numerator matters
    tau = complex(0.0, modulus)
    eta = dedekind_eta(tau, max(n_terms, 200))
    return float(abstruse_residual(tau, n_terms) / abs(eta) ** 12)
