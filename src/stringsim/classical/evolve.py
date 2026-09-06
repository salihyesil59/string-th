r"""Finite-difference evolution of the worldsheet wave equation.

The mode expansion is exact but assumes you already know the amplitudes.  This
module goes the other way: give it an initial shape and velocity, and it
integrates

.. math::  \partial_\tau^2 X = \partial_\sigma^2 X

forward in ``tau`` with a standard second-order leapfrog, subject to whichever
boundary condition defines the string:

``"neumann"``
    Free open-string ends, ``X' = 0`` at both.  Momentum flows nowhere.
``"dirichlet"``
    Both ends pinned -- an open string with both endpoints on a D-brane.
``"mixed"``
    Neumann at ``sigma = 0``, Dirichlet at ``sigma = sigma_max``: one free end,
    one stuck to a brane.
``"periodic"``
    A closed string.

Being able to *pluck* a string and watch it, rather than prescribing its
Fourier content, is what makes the mode expansion an observation instead of an
assumption -- :func:`mode_spectrum` reads the amplitudes back off a snapshot.

Stability: the scheme needs the Courant number ``lambda = dtau/h <= 1``.  At
exactly ``lambda = 1`` it reproduces d'Alembert's solution to round-off, which
is a useful sanity check but hides the discretisation error; the default 0.5
shows the honest second-order convergence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Evolution", "evolve", "mode_spectrum"]

_BOUNDARIES = ("neumann", "dirichlet", "mixed", "periodic")


@dataclass(frozen=True)
class Evolution:
    """The result of :func:`evolve`.

    Attributes
    ----------
    tau:
        Times, shape ``(n_steps + 1,)``.
    sigma:
        Worldsheet coordinate of each grid point, shape ``(n_points,)``.
    X:
        Embedding, shape ``(n_steps + 1, n_points, n_target)``.  Only the
        directions the string actually moves in need be included, so
        ``n_target`` is whatever the initial data had -- typically 2 or 3 for
        something you want to draw.
    boundary:
        The boundary condition used.
    """

    tau: np.ndarray
    sigma: np.ndarray
    X: np.ndarray
    boundary: str

    @property
    def courant(self) -> float:
        """``dtau / dsigma``."""
        return float((self.tau[1] - self.tau[0]) / (self.sigma[1] - self.sigma[0]))

    def at(self, index: int) -> np.ndarray:
        """Snapshot ``X(tau_index, sigma)``, shape ``(n_points, n_target)``."""
        return self.X[index]


def _laplacian(x: np.ndarray, h: float, boundary: str) -> np.ndarray:
    """Second ``sigma``-derivative with the ghost points the boundary implies."""
    if boundary == "periodic":
        return (np.roll(x, -1, axis=0) - 2.0 * x + np.roll(x, 1, axis=0)) / h**2
    lap = np.empty_like(x)
    lap[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / h**2
    # Neumann end: X_{-1} = X_{1} makes the stencil one-sided.
    lap[0] = 2.0 * (x[1] - x[0]) / h**2 if boundary in ("neumann", "mixed") else 0.0
    lap[-1] = 2.0 * (x[-2] - x[-1]) / h**2 if boundary == "neumann" else 0.0
    return lap


def _apply_dirichlet(x: np.ndarray, x_ref: np.ndarray, boundary: str) -> None:
    """Pin whichever ends the boundary condition holds fixed."""
    if boundary in ("dirichlet",):
        x[0] = x_ref[0]
    if boundary in ("dirichlet", "mixed"):
        x[-1] = x_ref[-1]


def evolve(
    X0: np.ndarray,
    V0: np.ndarray | None = None,
    *,
    boundary: str = "neumann",
    sigma_max: float = np.pi,
    n_steps: int = 400,
    courant: float = 0.5,
) -> Evolution:
    """Integrate the wave equation from initial data.

    Parameters
    ----------
    X0:
        Initial shape, shape ``(n_points, n_target)``.  For ``"periodic"`` the
        grid is taken to be ``n_points`` points spanning ``[0, sigma_max)``
        without repeating the first; otherwise it spans ``[0, sigma_max]``
        inclusive.
    V0:
        Initial ``dX/dtau``, same shape.  Defaults to zero (a plucked string
        released from rest).
    boundary:
        One of ``"neumann"``, ``"dirichlet"``, ``"mixed"``, ``"periodic"``.
    sigma_max:
        ``pi`` for the open string, ``2 pi`` for the closed one.
    n_steps:
        Number of time steps.
    courant:
        ``dtau / dsigma``; must not exceed 1.

    Returns
    -------
    Evolution
    """
    if boundary not in _BOUNDARIES:
        raise ValueError(f"boundary must be one of {_BOUNDARIES}, got {boundary!r}")
    if not 0 < courant <= 1.0:
        raise ValueError(f"courant must lie in (0, 1], got {courant}")

    X0 = np.asarray(X0, float)
    if X0.ndim != 2:
        raise ValueError(f"X0 must have shape (n_points, n_target), got {X0.shape}")
    V0 = np.zeros_like(X0) if V0 is None else np.asarray(V0, float).reshape(X0.shape)

    n_points = X0.shape[0]
    h = sigma_max / n_points if boundary == "periodic" else sigma_max / (n_points - 1)
    dt = courant * h
    sigma = np.arange(n_points) * h

    out = np.empty((n_steps + 1,) + X0.shape)
    out[0] = X0
    prev = X0
    curr = X0 + dt * V0 + 0.5 * dt**2 * _laplacian(X0, h, boundary)
    _apply_dirichlet(curr, X0, boundary)
    out[1] = curr

    lam2 = courant**2
    for k in range(2, n_steps + 1):
        nxt = 2.0 * curr - prev + lam2 * (h**2) * _laplacian(curr, h, boundary)
        _apply_dirichlet(nxt, X0, boundary)
        out[k] = nxt
        prev, curr = curr, nxt

    return Evolution(tau=np.arange(n_steps + 1) * dt, sigma=sigma, X=out, boundary=boundary)


def mode_spectrum(
    X: np.ndarray, sigma_max: float = np.pi, boundary: str = "neumann", n_modes: int = 16
) -> np.ndarray:
    r"""Project a snapshot onto the boundary condition's normal modes.

    ``"neumann"`` uses :math:`\cos n\sigma` (``n = 0`` is the centre of mass),
    ``"dirichlet"`` and ``"mixed"`` use :math:`\sin` with the appropriate
    half-integer shift, and ``"periodic"`` uses the complex Fourier basis.

    Returns
    -------
    ndarray
        Shape ``(n_modes, n_target)``.  A string plucked into a triangle has
        coefficients falling like ``1/n^2`` with every other one vanishing --
        the same result as for a classical guitar string, because the equation
        is the same one.
    """
    X = np.asarray(X, float)
    n_points = X.shape[0]
    if boundary == "periodic":
        sigma = np.arange(n_points) * (sigma_max / n_points)
        basis = np.exp(-2j * np.pi * np.outer(np.arange(n_modes), sigma) / sigma_max)
        return (basis @ X) / n_points
    sigma = np.linspace(0.0, sigma_max, n_points)
    coeffs = np.empty((n_modes, X.shape[1]))
    for n in range(n_modes):
        if boundary == "neumann":
            f = np.cos(n * np.pi * sigma / sigma_max)
            norm = sigma_max if n == 0 else sigma_max / 2.0
        elif boundary == "dirichlet":
            f = np.sin((n + 1) * np.pi * sigma / sigma_max)
            norm = sigma_max / 2.0
        else:  # mixed: Neumann at 0, Dirichlet at sigma_max -> half-integer modes
            f = np.cos((n + 0.5) * np.pi * sigma / sigma_max)
            norm = sigma_max / 2.0
        coeffs[n] = np.trapezoid(f[:, None] * X, sigma, axis=0) / norm
    return coeffs
