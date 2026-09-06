r"""The rigidly rotating open string and the leading Regge trajectory.

A straight open string spinning about its midpoint is an exact solution of the
constraints:

.. math::
   X^0 = A\tau, \quad
   X^1 = A \cos\tau \cos\sigma, \quad
   X^2 = A \sin\tau \cos\sigma, \quad \sigma \in [0, \pi],

with every other component zero.  Both conditions hold identically:
:math:`\dot X \cdot X' = 0` and :math:`\dot X^2 + X'^2 = 0`.  The endpoints
trace a circle of radius ``A`` at :math:`|d\vec X/dX^0| = 1` -- exactly the
speed of light, which is what stops the string from being stretched further.

Integrating the Noether charges gives

.. math::
   M = \frac{A}{2\alpha'}, \qquad J = \frac{A^2}{4\alpha'}
   \quad \Longrightarrow \quad J = \alpha' M^2 ,

the leading Regge trajectory, with :math:`\alpha'` appearing as its slope.
That is where the Regge slope gets its name, and it is the same ``alpha'`` that
sets the spacing of the quantum spectrum and the poles of the Veneziano
amplitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..units import Conventions
from .constraints import angular_momentum, total_momentum
from .modes import OpenString

__all__ = ["rigid_rotator", "ReggePoint", "regge_trajectory"]


def rigid_rotator(conventions: Conventions | None = None, amplitude: float = 1.0) -> OpenString:
    """Build the rotating solution as an ordinary mode expansion.

    It is a pure ``n = 1`` excitation in two transverse directions with a
    quarter-period relative phase: the same amplitudes that
    :func:`~stringsim.classical.lightcone.oscillator_amplitudes` produces for
    the state ``alpha_{-1}^1 alpha_{-1}^2 |0>``, seen in the rest frame.

    Parameters
    ----------
    conventions:
        Sets ``alpha'`` and ``D``.
    amplitude:
        ``A`` above: the endpoints orbit at radius ``A`` and the mass is
        ``A / (2 alpha')``.
    """
    c = conventions or Conventions()
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    ap = c.alpha_prime
    scale = amplitude / (2.0 * math.sqrt(2.0 * ap))

    p = np.zeros(c.dim)
    p[0] = amplitude / (2.0 * ap)

    a1 = np.zeros(c.dim, dtype=complex)
    a1[1] = -1j * scale  # X^1 = A cos(tau) cos(sigma)
    a1[2] = scale  # X^2 = A sin(tau) cos(sigma)
    return OpenString(conventions=c, p=p, alphas={1: a1})


@dataclass(frozen=True)
class ReggePoint:
    """One point on the trajectory, measured rather than assumed.

    ``mass_squared`` comes from ``P^mu`` and ``spin`` from ``J^{12}``, both
    integrated numerically over the string.
    """

    amplitude: float
    mass_squared: float
    spin: float

    alpha_prime: float = 1.0

    @property
    def alpha_m2(self) -> float:
        """``alpha' M^2``, which should equal :attr:`spin` on the leading trajectory."""
        return self.alpha_prime * self.mass_squared


def regge_trajectory(
    amplitudes: np.ndarray | None = None,
    conventions: Conventions | None = None,
    n_sigma: int = 2049,
) -> list[ReggePoint]:
    """Measure ``(M^2, J)`` for a family of rotating strings.

    Both charges are obtained by integrating the worldsheet currents, so a fit
    of ``J`` against ``alpha' M^2`` recovers the slope 1 (and intercept 0) as a
    *result*, not as an input.
    """
    c = conventions or Conventions()
    amps = np.asarray([0.5, 1.0, 1.5, 2.0, 3.0] if amplitudes is None else amplitudes, float)
    points: list[ReggePoint] = []
    for a in amps:
        s = rigid_rotator(c, float(a))
        p = total_momentum(s, tau=0.0, n_sigma=n_sigma)
        j = angular_momentum(s, tau=0.0, n_sigma=n_sigma)
        m2 = float(p[0] ** 2 - np.dot(p[1:], p[1:]))
        points.append(
            ReggePoint(
                amplitude=float(a),
                mass_squared=m2,
                spin=float(j[1, 2]),
                alpha_prime=c.alpha_prime,
            )
        )
    return points
