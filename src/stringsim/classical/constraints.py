r"""Virasoro constraints and worldsheet conserved charges.

Conformal gauge does not fix the gauge completely: the vanishing of the
worldsheet stress tensor survives as a constraint on the solution,

.. math::
   \dot X \cdot X' = 0, \qquad \dot X^2 + X'^2 = 0,

equivalently :math:`(\dot X \pm X')^2 = 0`.  A generic set of Fourier
amplitudes solves the wave equation but violates these; the functions here
measure by how much, which is the honest way to tell a physical solution from
an arbitrary one.

The conserved charges follow from the Noether currents of the Polyakov action,

.. math::
   P^\mu = T \int d\sigma\, \dot X^\mu, \qquad
   J^{\mu\nu} = T \int d\sigma\, (X^\mu \dot X^\nu - X^\nu \dot X^\mu),

with :math:`T = 1/(2\pi\alpha')`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import dot

__all__ = ["ConstraintReport", "virasoro_residual", "total_momentum", "angular_momentum"]


@dataclass(frozen=True)
class ConstraintReport:
    """Worst-case Virasoro violation over a sampled patch of the worldsheet.

    Attributes
    ----------
    orthogonality:
        ``max |Xdot . Xprime|``.
    normalisation:
        ``max |Xdot^2 + Xprime^2|``.
    scale:
        ``max (Xdot^2 - Xprime^2)`` in absolute value, a natural size for the
        problem.  Dividing by it makes the residuals dimensionless.
    """

    orthogonality: float
    normalisation: float
    scale: float

    @property
    def relative(self) -> float:
        """The larger of the two residuals, divided by ``scale``."""
        if self.scale == 0.0:
            return max(self.orthogonality, self.normalisation)
        return max(self.orthogonality, self.normalisation) / self.scale

    def satisfied(self, tol: float = 1e-10) -> bool:
        """True when the relative residual is below ``tol``."""
        return self.relative < tol

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"|Xdot.Xprime| <= {self.orthogonality:.3e}, "
            f"|Xdot^2+Xprime^2| <= {self.normalisation:.3e}, "
            f"relative {self.relative:.3e}"
        )


def _sample(string, n_tau: int, n_sigma: int, tau_max: float):
    tau = np.linspace(0.0, tau_max, n_tau)[:, None]
    sigma = np.linspace(0.0, string.sigma_max, n_sigma)[None, :]
    return tau, sigma


def virasoro_residual(string, n_tau: int = 41, n_sigma: int = 61, tau_max: float = 2 * np.pi
                      ) -> ConstraintReport:
    """Sample ``(Xdot . Xprime)`` and ``(Xdot^2 + Xprime^2)`` on a worldsheet grid."""
    tau, sigma = _sample(string, n_tau, n_sigma, tau_max)
    v = string.velocity(tau, sigma)
    s = string.slope(tau, sigma)
    vv = dot(v, v)
    ss = dot(s, s)
    vs = dot(v, s)
    return ConstraintReport(
        orthogonality=float(np.max(np.abs(vs))),
        normalisation=float(np.max(np.abs(vv + ss))),
        scale=float(np.max(np.abs(vv - ss))),
    )


def total_momentum(string, tau: float = 0.0, n_sigma: int = 4097) -> np.ndarray:
    """``P^mu`` by trapezoidal integration of ``T * dX/dtau`` over ``sigma``.

    For any solution this must reproduce the centre-of-mass momentum stored on
    the object, which is a cheap end-to-end check of the mode expansion.
    """
    sigma = np.linspace(0.0, string.sigma_max, n_sigma)
    v = string.velocity(np.full_like(sigma, tau), sigma)
    return string.conventions.tension * np.trapezoid(v, sigma, axis=0)


def angular_momentum(string, tau: float = 0.0, n_sigma: int = 4097) -> np.ndarray:
    """``J^{mu nu}``, an antisymmetric ``(D, D)`` array.

    The spatial block gives the spin; for the rigidly rotating string it
    saturates the leading Regge trajectory ``J = alpha' M^2``.
    """
    sigma = np.linspace(0.0, string.sigma_max, n_sigma)
    tau_arr = np.full_like(sigma, tau)
    x = string.position(tau_arr, sigma)
    v = string.velocity(tau_arr, sigma)
    integrand = x[:, :, None] * v[:, None, :] - x[:, None, :] * v[:, :, None]
    return string.conventions.tension * np.trapezoid(integrand, sigma, axis=0)
