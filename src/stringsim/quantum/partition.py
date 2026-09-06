r"""Counting string states: oscillator partition functions and their asymptotics.

Every physical state of the light-cone string is a product of transverse
oscillators acting on the ground state, so the number of states at level ``N``
is the number of ways of writing ``N`` as an ordered sum over ``D - 2``
independent towers.  Its generating function is

.. math::
   \prod_{n=1}^{\infty} \frac{1}{(1 - q^n)^{D-2}}
   = \sum_{N \geq 0} d_N \, q^N ,

which is :math:`q^{(D-2)/24}/\eta(\tau)^{D-2}` up to the ground-state energy.
The coefficients are computed here in exact integer arithmetic -- they grow
fast (``d_10`` already exceeds ten million in ``D = 26``) and floating point
would quietly lose digits.

The growth is the physics: :math:`d_N \sim \exp(4\pi\sqrt N)` for ``D = 26``
means the density of states rises exponentially in the mass, and the canonical
partition function :math:`\sum_N d_N e^{-\beta M_N}` diverges above the
**Hagedorn temperature**.  :func:`fit_hagedorn` measures that exponent from the
computed degeneracies instead of quoting it.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "oscillator_degeneracies",
    "dedekind_eta",
    "theta_series",
    "jacobi_identity_residual",
    "superstring_degeneracies",
    "HagedornFit",
    "fit_hagedorn",
    "hagedorn_temperature",
]


def oscillator_degeneracies(n_max: int, n_species: int = 24) -> list[int]:
    r"""Coefficients of :math:`\prod_{n\geq1}(1-q^n)^{-c}` up to ``q^{n_max}``.

    Parameters
    ----------
    n_max:
        Highest level to compute.
    n_species:
        ``c``, the number of independent oscillator towers.  For the bosonic
        string in ``D`` dimensions this is the transverse count ``D - 2``.

    Returns
    -------
    list[int]
        ``[d_0, d_1, ..., d_{n_max}]``, exact integers.  In ``D = 26``:
        ``1, 24, 324, 3200, 25650, ...``
    """
    if n_max < 0:
        raise ValueError("n_max must be non-negative")
    if n_species < 1:
        raise ValueError("n_species must be positive")
    coeffs = [0] * (n_max + 1)
    coeffs[0] = 1
    for n in range(1, n_max + 1):
        for _ in range(n_species):
            # multiply by 1/(1 - q^n) == 1 + q^n + q^{2n} + ...
            for k in range(n, n_max + 1):
                coeffs[k] += coeffs[k - n]
    return coeffs


def _poly_pow(series: list[int], power: int, n_max: int) -> list[int]:
    """``series ** power`` truncated at ``q^{n_max}``, exact integers."""
    result = [0] * (n_max + 1)
    result[0] = 1
    for _ in range(power):
        new = [0] * (n_max + 1)
        for i, a in enumerate(series):
            if a == 0:
                continue
            for j in range(0, n_max + 1 - i):
                if result[j]:
                    new[i + j] += a * result[j]
        result = new
    return result


def theta_series(which: int, n_max: int) -> list[int]:
    r"""Integer ``q``-series of a Jacobi theta constant.

    ``which = 2`` returns the coefficients of :math:`\theta_2(q)/(2q^{1/4})`,
    ``which = 3`` those of :math:`\theta_3(q)`, ``which = 4`` those of
    :math:`\theta_4(q)`, with

    .. math::
       \theta_3 = 1 + 2\sum_{n\geq1} q^{n^2}, \quad
       \theta_4 = 1 + 2\sum_{n\geq1} (-1)^n q^{n^2}, \quad
       \theta_2 = 2q^{1/4}\sum_{n\geq0} q^{n(n+1)} .

    Pulling the :math:`2q^{1/4}` out of :math:`\theta_2` keeps every series in
    integer powers of ``q``, which is what makes
    :func:`jacobi_identity_residual` an exact integer statement.
    """
    out = [0] * (n_max + 1)
    if which == 3:
        out[0] = 1
        n = 1
        while n * n <= n_max:
            out[n * n] += 2
            n += 1
    elif which == 4:
        out[0] = 1
        n = 1
        while n * n <= n_max:
            out[n * n] += 2 * (-1) ** n
            n += 1
    elif which == 2:
        n = 0
        while n * (n + 1) <= n_max:
            out[n * (n + 1)] += 1
            n += 1
    else:
        raise ValueError("which must be 2, 3 or 4")
    return out


def jacobi_identity_residual(n_max: int = 40) -> list[int]:
    r"""Coefficients of :math:`\theta_3^4 - \theta_2^4 - \theta_4^4`, which vanish.

    Jacobi called this the *aequatio identica satis abstrusa*.  In the
    superstring it is the statement that the GSO-projected NS sector (spacetime
    bosons) and the R sector (spacetime fermions) contain the **same number of
    states at every mass level**, so the one-loop vacuum amplitude vanishes --
    the first sign of spacetime supersymmetry.

    Returns a list of exact integers; every entry is 0.
    """
    t3 = _poly_pow(theta_series(3, n_max), 4, n_max)
    t4 = _poly_pow(theta_series(4, n_max), 4, n_max)
    t2h = _poly_pow(theta_series(2, n_max), 4, n_max)  # theta_2^4 / (16 q)
    # theta_2^4 = 16 q * (that), so shift by one power of q.
    t2 = [0] * (n_max + 1)
    for k in range(n_max):
        t2[k + 1] = 16 * t2h[k]
    return [t3[k] - t2[k] - t4[k] for k in range(n_max + 1)]


def superstring_degeneracies(n_max: int = 8) -> list[int]:
    r"""GSO-projected open-superstring degeneracies, ``alpha' M^2 = 0, 1, 2, ...``.

    The NS-sector generating function after the GSO projection is

    .. math::
       \frac{1}{2\prod_n (1-q^n)^8}
       \left[ \prod_n (1+q^{n-1/2})^8 - \prod_n (1-q^{n-1/2})^8 \right]
       = 8 + 128\,q + 1152\,q^2 + \dots

    The 8 massless states are the transverse polarisations of the gauge boson;
    the R sector supplies 8 fermionic ones, and the counts agree at every level
    (see :func:`jacobi_identity_residual`).
    """
    if n_max < 0:
        raise ValueError("n_max must be non-negative")
    # Work in u = q^{1/2} so half-integer modes have integer exponents.
    m = 2 * n_max + 2
    plus = [0] * (m + 1)
    plus[0] = 1
    minus = [0] * (m + 1)
    minus[0] = 1
    bose = [0] * (m + 1)
    bose[0] = 1
    for r in range(1, m + 1, 2):  # u^r with r odd == q^{r/2}, half-integer
        for _ in range(8):
            for k in range(m, r - 1, -1):
                plus[k] += plus[k - r]
                minus[k] -= minus[k - r]
    for n in range(2, m + 1, 2):  # u^{2n'} == q^{n'}, integer modes
        for _ in range(8):
            for k in range(n, m + 1):
                bose[k] += bose[k - n]
    num = [(plus[k] - minus[k]) // 2 for k in range(m + 1)]
    total = [0] * (m + 1)
    for i, a in enumerate(num):
        if not a:
            continue
        for j in range(0, m + 1 - i):
            if bose[j]:
                total[i + j] += a * bose[j]
    # The series starts at u^1 = q^{1/2}: alpha' M^2 = level - 1/2 with the
    # tachyon projected out, so the physical levels sit at odd powers of u.
    return [total[2 * k + 1] for k in range(n_max + 1)]


def dedekind_eta(tau: complex, n_terms: int = 200) -> complex:
    r""":math:`\eta(\tau) = q^{1/24}\prod_{n\geq1}(1-q^n)`, ``q = e^{2\pi i \tau}``.

    Requires ``Im(tau) > 0``.  The modular weight-1/2 behaviour
    :math:`\eta(-1/\tau) = \sqrt{-i\tau}\,\eta(\tau)` is what makes the one-loop
    torus amplitude well defined, and is checked in the test suite.
    """
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half plane")
    q = cmath.exp(2j * cmath.pi * tau)
    prod = 1.0 + 0.0j
    for n in range(1, n_terms + 1):
        prod *= 1.0 - q**n
    return cmath.exp(2j * cmath.pi * tau / 24.0) * prod


@dataclass(frozen=True)
class HagedornFit:
    """Result of fitting ``log d_N ~ beta_H * M_N + const + power * log N``."""

    beta_hagedorn: float
    temperature: float
    predicted_beta: float
    levels: np.ndarray
    residual: float
    coefficients: np.ndarray
    alpha_prime: float = 1.0

    def model(self, levels: np.ndarray) -> np.ndarray:
        r"""The fitted ``log d_N`` at the given levels, all three terms included.

        Plotting ``beta_H * sqrt(N)`` alone would sit far above the data: the
        ``b log N + c`` piece is not a small correction at these levels.  Use
        this when drawing the fit, and :attr:`local_slope` when comparing the
        *slope* against ``4 pi``.
        """
        levels = np.asarray(levels, dtype=float)
        design = np.column_stack(
            [np.sqrt(levels / self.alpha_prime), np.log(levels), np.ones_like(levels)]
        )
        return design @ self.coefficients

    @staticmethod
    def local_slope(degeneracies: list[int], alpha_prime: float = 1.0):
        r"""``d(log d_N) / d(sqrt(N/alpha'))`` by finite differences.

        This is the quantity that tends to :math:`\beta_H` as ``N`` grows, and
        it approaches it slowly -- like ``1/sqrt(N)`` -- because of the
        subleading ``log N``.  Returns ``(levels, slope)``.
        """
        n = np.arange(1, len(degeneracies))
        logd = np.array([math.log(d) for d in degeneracies[1:]])
        mass = np.sqrt(n / alpha_prime)
        return n[:-1], np.diff(logd) / np.diff(mass)

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"beta_H = {self.beta_hagedorn:.6f} (expected {self.predicted_beta:.6f}), "
            f"T_H = {self.temperature:.6f}, rms residual {self.residual:.2e}"
        )


def hagedorn_temperature(n_species: int = 24, alpha_prime: float = 1.0) -> float:
    r"""The predicted Hagedorn temperature of the open bosonic string.

    With ``c`` transverse bosons the Cardy growth is
    :math:`d_N \sim \exp(2\pi\sqrt{cN/6})`, and ``alpha' M^2 = N`` turns that
    into :math:`\exp(\beta_H M)` with
    :math:`\beta_H = 2\pi\sqrt{c\alpha'/6}`.  For ``c = 24`` this is
    :math:`4\pi\sqrt{\alpha'}`.
    """
    return 2.0 * math.pi * math.sqrt(n_species * alpha_prime / 6.0)


def fit_hagedorn(
    n_max: int = 160, n_species: int = 24, alpha_prime: float = 1.0, n_fit: int = 60
) -> HagedornFit:
    r"""Measure ``beta_H`` from the computed degeneracies.

    Fits :math:`\log d_N = \beta_H M_N + b \log N + c` over the highest
    ``n_fit`` levels, with :math:`M_N = \sqrt{N/\alpha'}`.  The subleading
    ``log N`` term matters: leaving it out biases the slope by several percent
    at these levels.
    """
    if n_fit < 3 or n_fit > n_max:
        raise ValueError("need 3 <= n_fit <= n_max")
    degen = oscillator_degeneracies(n_max, n_species)
    levels = np.arange(n_max - n_fit + 1, n_max + 1)
    logd = np.array([math.log(degen[int(n)]) for n in levels])
    mass = np.sqrt(levels / alpha_prime)
    design = np.column_stack([mass, np.log(levels), np.ones_like(mass)])
    coef, *_ = np.linalg.lstsq(design, logd, rcond=None)
    resid = float(np.sqrt(np.mean((design @ coef - logd) ** 2)))
    beta = float(coef[0])
    return HagedornFit(
        beta_hagedorn=beta,
        temperature=1.0 / beta,
        predicted_beta=hagedorn_temperature(n_species, alpha_prime),
        levels=levels,
        residual=resid,
        coefficients=coef,
        alpha_prime=alpha_prime,
    )
