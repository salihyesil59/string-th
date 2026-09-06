r"""Veneziano and Virasoro-Shapiro amplitudes.

Veneziano wrote his amplitude in 1968 as a guess that fitted hadronic data; the
string was found afterwards, as the thing that produces it.  For four
open-string tachyons,

.. math::
   A(s,t) = B(-\alpha(s), -\alpha(t))
          = \frac{\Gamma(-\alpha(s))\Gamma(-\alpha(t))}{\Gamma(-\alpha(s)-\alpha(t))},
   \qquad \alpha(x) = 1 + \alpha' x .

Two features make it a string amplitude rather than a clever function.

**The poles are the spectrum.**  :math:`\Gamma(-\alpha(s))` has poles at
:math:`\alpha(s) = 0, 1, 2, \dots`, i.e. at :math:`\alpha' s = -1, 0, 1, \dots`
-- precisely the open-string masses :math:`\alpha' M^2 = N - 1` of
:mod:`stringsim.quantum.spectrum`.  The residue at the ``n``-th pole is a
polynomial of degree ``n`` in ``t``, so the exchanged states run up to spin
``n``, and no higher: the leading Regge trajectory, arrived at from a third
direction.

**The high-energy behaviour is soft.**  At fixed ``t`` and large ``|s|`` the
amplitude behaves as :math:`\Gamma(-\alpha(t))\,(-\alpha(s))^{\alpha(t)}`: the
power slides with ``t`` instead of being fixed, which is Regge behaviour rather
than a field-theory exchange.  Push both invariants large together and the
fall-off becomes exponential -- an extended object has no point-like hard core.

The closed-string counterpart is the Virasoro-Shapiro amplitude, whose poles
sit at :math:`\alpha' M^2 = 4(N-1)`, matching the closed spectrum.

Numerical note: these amplitudes overflow double precision quickly, so
everything goes through ``gammaln``/``gammasgn`` rather than ``gamma``.
:func:`veneziano_log_abs` returns the logarithm directly, which is the only way
to study the high-energy regime without the answer underflowing to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, gammasgn

__all__ = [
    "regge_alpha",
    "veneziano",
    "veneziano_log_abs",
    "veneziano_residue",
    "veneziano_pole_positions",
    "regge_asymptotic",
    "virasoro_shapiro",
    "virasoro_shapiro_pole_positions",
    "closed_mandelstam_sum",
    "HardScattering",
    "hard_scattering",
]


def regge_alpha(x, alpha_prime: float = 1.0, intercept: float = 1.0):
    r"""The Regge trajectory ``alpha(x) = intercept + alpha' x``.

    ``intercept = 1`` is the open bosonic string, where ``alpha(M^2) = J``
    reproduces :func:`stringsim.quantum.spectrum.leading_trajectory_spin`.
    """
    return intercept + alpha_prime * np.asarray(x, dtype=float)


def veneziano_log_abs(s, t, alpha_prime: float = 1.0, intercept: float = 1.0):
    r"""``log |A(s,t)|`` for the Veneziano amplitude.

    Poles come back as ``+inf`` and zeros as ``-inf``.  Use this rather than
    :func:`veneziano` whenever ``|alpha(s)|`` is large: the amplitude itself
    underflows long before its logarithm stops being informative.
    """
    a_s = regge_alpha(s, alpha_prime, intercept)
    a_t = regge_alpha(t, alpha_prime, intercept)
    with np.errstate(divide="ignore", invalid="ignore"):
        return gammaln(-a_s) + gammaln(-a_t) - gammaln(-a_s - a_t)


def _veneziano_sign(s, t, alpha_prime: float = 1.0, intercept: float = 1.0):
    a_s = regge_alpha(s, alpha_prime, intercept)
    a_t = regge_alpha(t, alpha_prime, intercept)
    return gammasgn(-a_s) * gammasgn(-a_t) * gammasgn(-a_s - a_t)


def _assemble(sign, log_abs):
    """``sign * exp(log_abs)``, with an exact zero where the log has gone to ``-inf``.

    A gamma function in the *denominator* hitting a pole makes the amplitude
    vanish; ``gammasgn`` returns a non-finite sign there, so the product alone
    would give ``nan`` for what is really a clean zero.
    """
    log_abs = np.asarray(log_abs, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = np.asarray(sign, dtype=float) * np.exp(log_abs)
    return np.where(np.isneginf(log_abs), 0.0, value)[()]


def veneziano(s, t, alpha_prime: float = 1.0, intercept: float = 1.0):
    r"""``A(s,t) = Gamma(-alpha_s) Gamma(-alpha_t) / Gamma(-alpha_s - alpha_t)``.

    Accepts scalars or arrays.  The function has genuine poles, so ``inf`` at an
    on-shell resonance is the correct answer rather than a failure; far from the
    poles at large ``|s|`` the true value underflows, and
    :func:`veneziano_log_abs` should be used instead.
    """
    return _assemble(
        _veneziano_sign(s, t, alpha_prime, intercept),
        veneziano_log_abs(s, t, alpha_prime, intercept),
    )


def veneziano_pole_positions(n_max: int = 5, alpha_prime: float = 1.0) -> np.ndarray:
    r"""Values of ``s`` at which ``A`` has a pole: ``alpha' s = n - 1``.

    Identical to the open-string mass levels, which is the check that the
    amplitude and the spectrum describe the same theory.
    """
    return np.array([(n - 1) / alpha_prime for n in range(n_max + 1)])


def veneziano_residue(n: int, t, alpha_prime: float = 1.0, intercept: float = 1.0):
    r"""Residue of ``A`` at the ``n``-th pole, as a function of ``t``.

    .. math::
       \lim_{\alpha(s)\to n} (\alpha(s) - n)\, A(s,t)
       = -\frac{1}{n!} \prod_{k=1}^{n} (\alpha(t) + k) .

    A polynomial of degree ``n``: level ``n`` exchanges states of spin up to
    ``n``, and no higher.  Take the limit numerically to see it -- that is what
    the test suite does.
    """
    if n < 0:
        raise ValueError("pole index must be non-negative")
    a_t = regge_alpha(t, alpha_prime, intercept)
    out = np.ones_like(np.asarray(a_t, dtype=float))
    for k in range(1, n + 1):
        out = out * (a_t + k)
    return -out / math.factorial(n)


def regge_asymptotic(s, t, alpha_prime: float = 1.0, intercept: float = 1.0):
    r"""``Gamma(-alpha_t) (-alpha_s)^{alpha_t}``, the fixed-``t`` Regge limit.

    Valid for large ``|alpha(s)|``.  Evaluate at large *negative* ``s`` to stay
    on the real axis; on the physical (positive ``s``) side the same formula
    carries the usual signature phase.
    """
    a_s = regge_alpha(s, alpha_prime, intercept)
    a_t = regge_alpha(t, alpha_prime, intercept)
    sign = gammasgn(-a_t)
    with np.errstate(over="ignore", under="ignore"):
        return sign * np.exp(gammaln(-a_t) + a_t * np.log(-a_s))


def closed_mandelstam_sum(alpha_prime: float = 1.0) -> float:
    r"""``s + t + u`` for four closed-string tachyons: ``-16/alpha'``.

    Four external masses of :math:`M^2 = -4/\alpha'` give
    :math:`s+t+u = \sum M_i^2 = -16/\alpha'`.
    """
    return -16.0 / alpha_prime


def virasoro_shapiro(s, t, u=None, alpha_prime: float = 1.0):
    r"""The closed-string four-tachyon amplitude.

    .. math::
       A = \frac{\Gamma(a)\Gamma(b)\Gamma(c)}
                {\Gamma(a+b)\Gamma(b+c)\Gamma(c+a)},
       \quad a = -1 - \frac{\alpha' s}{4}, \ \text{etc.}

    If ``u`` is omitted it is fixed by :func:`closed_mandelstam_sum`, which
    makes ``a + b + c = 1``.
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    u = closed_mandelstam_sum(alpha_prime) - s - t if u is None else np.asarray(u, dtype=float)
    a = -1.0 - alpha_prime * s / 4.0
    b = -1.0 - alpha_prime * t / 4.0
    c = -1.0 - alpha_prime * u / 4.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        log_abs = (
            gammaln(a) + gammaln(b) + gammaln(c)
            - gammaln(a + b) - gammaln(b + c) - gammaln(c + a)
        )
        sign = (
            gammasgn(a) * gammasgn(b) * gammasgn(c)
            * gammasgn(a + b) * gammasgn(b + c) * gammasgn(c + a)
        )
    return _assemble(sign, log_abs)


def virasoro_shapiro_pole_positions(n_max: int = 4, alpha_prime: float = 1.0) -> np.ndarray:
    r"""Values of ``s`` at which the closed amplitude has a pole: ``alpha' s = 4(n-1)``."""
    return np.array([4.0 * (n - 1) / alpha_prime for n in range(n_max + 1)])


@dataclass(frozen=True)
class HardScattering:
    """Large-``|s|`` behaviour of the Veneziano amplitude at fixed ``t/s``.

    Sampled at ``s < 0`` with ``t = ratio * s``, so both invariants are
    spacelike: the amplitude stays real and away from the resonance poles,
    which is the regime in which "exponential fall-off" is a clean statement.
    """

    s: np.ndarray
    log_amplitude: np.ndarray
    ratio: float
    alpha_prime: float

    @property
    def log_slope(self) -> float:
        """Fitted ``d log|A| / ds``.  Positive here, i.e. decay as ``s -> -inf``."""
        good = np.isfinite(self.log_amplitude)
        if good.sum() < 2:
            raise ValueError("not enough finite points to fit")
        return float(np.polyfit(self.s[good], self.log_amplitude[good], 1)[0])

    @property
    def predicted_log_slope(self) -> float:
        r"""Stirling prediction, ``-alpha' [rho log rho - (1+rho) log(1+rho)]``.

        With :math:`x = -\alpha(s) \simeq \lambda` and
        :math:`y = -\alpha(t) \simeq \rho\lambda`, Stirling applied to
        :math:`B(x,y) = \Gamma(x)\Gamma(y)/\Gamma(x+y)` gives

        .. math::
           \log B \simeq \lambda\left[\rho\log\rho - (1+\rho)\log(1+\rho)\right],

        the :math:`\lambda\log\lambda` terms cancelling exactly.  The bracket is
        negative, so the amplitude dies exponentially in ``|s|``: no power law,
        which is what an extended scatterer looks like at short distance.
        """
        rho = self.ratio
        return -self.alpha_prime * (rho * math.log(rho) - (1.0 + rho) * math.log(1.0 + rho))


def hard_scattering(s_values, ratio: float = 1.0, alpha_prime: float = 1.0) -> HardScattering:
    r"""Sample ``log|A|`` along ``t = ratio * s`` with ``s < 0``.

    Compare :attr:`HardScattering.log_slope` with
    :attr:`HardScattering.predicted_log_slope`.
    """
    s = np.asarray(s_values, dtype=float)
    if np.any(s >= 0):
        raise ValueError("sample at negative (spacelike) s to avoid the resonance poles")
    if ratio <= 0:
        raise ValueError("ratio must be positive so that t is spacelike too")
    return HardScattering(
        s=s,
        log_amplitude=np.asarray(veneziano_log_abs(s, ratio * s, alpha_prime)),
        ratio=float(ratio),
        alpha_prime=float(alpha_prime),
    )
