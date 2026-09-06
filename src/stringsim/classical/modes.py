r"""Analytic mode expansions for the classical relativistic string.

In conformal gauge the worldsheet embedding obeys the free wave equation

.. math::  (\partial_\tau^2 - \partial_\sigma^2) X^\mu = 0,

so every solution is a superposition of normal modes.  The boundary condition
picks the family:

* **open string**, Neumann at both ends, :math:`\sigma \in [0, \pi]`

  .. math::
     X^\mu = x^\mu + 2\alpha' p^\mu \tau
              + i\sqrt{2\alpha'} \sum_{n \neq 0} \frac{1}{n}
                \alpha_n^\mu e^{-in\tau} \cos n\sigma

* **closed string**, periodic, :math:`\sigma \in [0, 2\pi)`

  .. math::
     X^\mu = x^\mu + \alpha' p^\mu \tau
              + i\sqrt{\alpha'/2} \sum_{n \neq 0} \frac{1}{n}
                \left[ \alpha_n^\mu e^{-in(\tau-\sigma)}
                      + \tilde\alpha_n^\mu e^{-in(\tau+\sigma)} \right]

Reality of :math:`X^\mu` fixes :math:`\alpha_{-n} = \alpha_n^{*}`, so only the
``n > 0`` coefficients are stored.  Nothing here imposes the Virasoro
constraints -- a generic set of amplitudes solves the wave equation but is not a
physical string.  Use :mod:`stringsim.classical.lightcone` for constrained
solutions and :mod:`stringsim.classical.constraints` to measure the violation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..units import Conventions, dot, minkowski

__all__ = ["OpenString", "ClosedString"]


def _as_mode_array(modes: Mapping[int, np.ndarray] | None, dim: int) -> dict[int, np.ndarray]:
    """Validate ``{n: alpha_n}`` with ``n >= 1`` and complex vectors of length ``dim``."""
    out: dict[int, np.ndarray] = {}
    for n, amp in (modes or {}).items():
        n = int(n)
        if n < 1:
            raise ValueError(
                f"mode numbers must be >= 1 (got {n}); alpha_-n is the conjugate of alpha_n"
            )
        vec = np.asarray(amp, dtype=complex).reshape(-1)
        if vec.size != dim:
            raise ValueError(f"mode {n}: expected {dim} components, got {vec.size}")
        out[n] = vec
    return out


@dataclass
class OpenString:
    """An open string with Neumann boundary conditions at both ends.

    Parameters
    ----------
    conventions:
        Sets ``alpha'`` and ``D``.
    x0:
        Centre-of-mass position ``x^mu``, shape ``(D,)``.
    p:
        Centre-of-mass momentum ``p^mu``, shape ``(D,)``.
    alphas:
        ``{n: alpha_n}`` for ``n >= 1``; each value a complex ``(D,)`` array.
        Modes not listed are zero.
    """

    conventions: Conventions = field(default_factory=Conventions)
    x0: np.ndarray | None = None
    p: np.ndarray | None = None
    alphas: Mapping[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        d = self.conventions.dim
        self.x0 = np.zeros(d) if self.x0 is None else np.asarray(self.x0, float).reshape(d)
        self.p = np.zeros(d) if self.p is None else np.asarray(self.p, float).reshape(d)
        self.alphas = _as_mode_array(self.alphas, d)

    @property
    def sigma_max(self) -> float:
        """The string runs over ``sigma`` in ``[0, pi]``."""
        return math.pi

    @staticmethod
    def _grid(tau, sigma):
        return np.broadcast_arrays(np.asarray(tau, float), np.asarray(sigma, float))

    def position(self, tau, sigma) -> np.ndarray:
        """``X^mu(tau, sigma)``, shape ``broadcast(tau, sigma) + (D,)``."""
        tau, sigma = self._grid(tau, sigma)
        ap = self.conventions.alpha_prime
        pref = 2.0 * math.sqrt(2.0 * ap)
        X = self.x0 + 2.0 * ap * tau[..., None] * self.p
        for n, a in self.alphas.items():
            phase = np.exp(-1j * n * tau)[..., None] * a
            X = X - (pref / n) * np.imag(phase) * np.cos(n * sigma)[..., None]
        return X

    def velocity(self, tau, sigma) -> np.ndarray:
        """``dX^mu/dtau``."""
        tau, sigma = self._grid(tau, sigma)
        ap = self.conventions.alpha_prime
        pref = 2.0 * math.sqrt(2.0 * ap)
        V = np.broadcast_to(2.0 * ap * self.p, tau.shape + (self.conventions.dim,)).copy()
        for n, a in self.alphas.items():
            phase = np.exp(-1j * n * tau)[..., None] * a
            V = V + pref * np.real(phase) * np.cos(n * sigma)[..., None]
        return V

    def slope(self, tau, sigma) -> np.ndarray:
        """``dX^mu/dsigma``.  Vanishes at ``sigma = 0`` and ``sigma = pi`` (Neumann)."""
        tau, sigma = self._grid(tau, sigma)
        ap = self.conventions.alpha_prime
        pref = 2.0 * math.sqrt(2.0 * ap)
        S = np.zeros(tau.shape + (self.conventions.dim,))
        for n, a in self.alphas.items():
            phase = np.exp(-1j * n * tau)[..., None] * a
            S = S + pref * np.imag(phase) * np.sin(n * sigma)[..., None]
        return S

    def level(self) -> float:
        """Classical level ``N = sum_{n>=1} alpha_{-n} . alpha_n``, contracted with ``eta``.

        For a light-cone solution the transverse part of this equals
        ``alpha' M^2`` classically.  For unconstrained amplitudes it is only
        bookkeeping, and can be negative because the timelike component enters
        with a minus sign.
        """
        eta = minkowski(self.conventions.dim)
        return float(sum(np.sum(eta * np.abs(a) ** 2).real for a in self.alphas.values()))

    def mass_squared(self) -> float:
        """``M^2 = -p . p`` from the centre-of-mass momentum."""
        return float(-dot(self.p, self.p))


@dataclass
class ClosedString:
    """A closed string: independent left- and right-moving oscillators.

    ``alphas`` are right-movers (functions of ``tau - sigma``) and
    ``alphas_tilde`` left-movers (functions of ``tau + sigma``).
    """

    conventions: Conventions = field(default_factory=Conventions)
    x0: np.ndarray | None = None
    p: np.ndarray | None = None
    alphas: Mapping[int, np.ndarray] | None = None
    alphas_tilde: Mapping[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        d = self.conventions.dim
        self.x0 = np.zeros(d) if self.x0 is None else np.asarray(self.x0, float).reshape(d)
        self.p = np.zeros(d) if self.p is None else np.asarray(self.p, float).reshape(d)
        self.alphas = _as_mode_array(self.alphas, d)
        self.alphas_tilde = _as_mode_array(self.alphas_tilde, d)

    @property
    def sigma_max(self) -> float:
        """The string runs over ``sigma`` in ``[0, 2 pi)``."""
        return 2.0 * math.pi

    @staticmethod
    def _grid(tau, sigma):
        return np.broadcast_arrays(np.asarray(tau, float), np.asarray(sigma, float))

    def position(self, tau, sigma) -> np.ndarray:
        """``X^mu(tau, sigma)``."""
        tau, sigma = self._grid(tau, sigma)
        ap = self.conventions.alpha_prime
        pref = 2.0 * math.sqrt(ap / 2.0)
        X = self.x0 + ap * tau[..., None] * self.p
        for n, a in self.alphas.items():
            X = X - (pref / n) * np.imag(np.exp(-1j * n * (tau - sigma))[..., None] * a)
        for n, a in self.alphas_tilde.items():
            X = X - (pref / n) * np.imag(np.exp(-1j * n * (tau + sigma))[..., None] * a)
        return X

    def velocity(self, tau, sigma) -> np.ndarray:
        """``dX^mu/dtau``."""
        tau, sigma = self._grid(tau, sigma)
        ap = self.conventions.alpha_prime
        pref = 2.0 * math.sqrt(ap / 2.0)
        V = np.broadcast_to(ap * self.p, tau.shape + (self.conventions.dim,)).copy()
        for n, a in self.alphas.items():
            V = V + pref * np.real(np.exp(-1j * n * (tau - sigma))[..., None] * a)
        for n, a in self.alphas_tilde.items():
            V = V + pref * np.real(np.exp(-1j * n * (tau + sigma))[..., None] * a)
        return V

    def slope(self, tau, sigma) -> np.ndarray:
        """``dX^mu/dsigma``.  Right-movers contribute with the opposite sign."""
        tau, sigma = self._grid(tau, sigma)
        ap = self.conventions.alpha_prime
        pref = 2.0 * math.sqrt(ap / 2.0)
        S = np.zeros(tau.shape + (self.conventions.dim,))
        for n, a in self.alphas.items():
            S = S - pref * np.real(np.exp(-1j * n * (tau - sigma))[..., None] * a)
        for n, a in self.alphas_tilde.items():
            S = S + pref * np.real(np.exp(-1j * n * (tau + sigma))[..., None] * a)
        return S

    def levels(self) -> tuple[float, float]:
        """``(N, N_tilde)``, each contracted with ``eta``."""
        eta = minkowski(self.conventions.dim)
        n = float(sum(np.sum(eta * np.abs(a) ** 2).real for a in self.alphas.values()))
        nt = float(sum(np.sum(eta * np.abs(a) ** 2).real for a in self.alphas_tilde.values()))
        return n, nt

    def mass_squared(self) -> float:
        """``M^2 = -p . p``."""
        return float(-dot(self.p, self.p))
