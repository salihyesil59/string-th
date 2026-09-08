r"""The one-loop partition function on a torus, and why the lattice has to be what it is.

Two parts of this package have so far not spoken to each other.
:mod:`stringsim.compactification.torus` builds the Narain lattice
:math:`\Gamma_{d,d}` and reads a spectrum off it;
:mod:`stringsim.amplitudes.oneloop` integrates a modular-invariant density over
the fundamental domain.  The object that joins them is the lattice theta
series,

.. math::
   \Theta_\Gamma(\tau, \bar\tau)
   = \sum_{(w, n) \in \mathbb{Z}^{2d}}
     q^{\ell_L^2/2}\, \bar q^{\ell_R^2/2} ,
   \qquad q = e^{2\pi i \tau} ,

summed over the same charges that give the spectrum.  It converges for any
:math:`\tau_2 > 0` because :math:`\ell_L^2 + \ell_R^2` is *positive definite* --
the generalized metric of
:meth:`~stringsim.compactification.torus.TorusBackground.generalized_metric` --
even though :math:`\ell_L^2 - \ell_R^2` is not.  That is what makes the sum a
finite computation rather than a formal one.

**The full integrand.**  Of the ``D - 2`` transverse directions, ``d`` are
compact and contribute the lattice sum; the rest contribute momentum integrals:

.. math::
   I(\tau) = \tau_2^{-(D-2-d)/2}\,|\eta(\tau)|^{-2(D-2)}\,
             \Theta_\Gamma(\tau,\bar\tau) .

At ``d = 0`` the theta series is 1 and this is exactly
:func:`stringsim.amplitudes.oneloop.torus_integrand`, which is checked rather
than assumed.

**Modular invariance is the claim, and it is a claim about the lattice.**
:func:`stringsim.compactification.torus.narain_gram_matrix` says that even
self-duality "is what makes the one-loop amplitude modular invariant".  This
module computes it:

* :math:`T:\ \tau \to \tau + 1` multiplies each term by
  :math:`e^{i\pi(\ell_L^2 - \ell_R^2)} = e^{2\pi i\, n\cdot w}`, which is 1
  because the lattice is **even**.  Nothing else is needed.
* :math:`S:\ \tau \to -1/\tau` sends :math:`\Theta_\Gamma \to |\tau|^d
  \Theta_{\Gamma^*}` by Poisson resummation, and that is the original series
  only because the lattice is **self-dual**.

The two conditions are therefore visible separately, and the module makes them
fail separately.  Restricting the momenta to multiples of ``s`` gives a
sublattice that is still even -- :math:`\ell_L^2 - \ell_R^2 = 2 s\, n\cdot w` --
but has index :math:`s^d` and is not self-dual.  ``T`` still holds to
round-off; ``S`` breaks.  See :func:`modular_residual` and the
``momentum_scale`` argument.

**And it is invariant under the duality group too.**  ``O(d,d;Z)`` acts on the
moduli and on the charges together, leaving both quadratic forms alone, so the
partition function of two T-dual backgrounds is the same number.
:func:`duality_residual` checks that against the generators already in
:mod:`~stringsim.compactification.torus`.  The spectrum-level version of the
same statement is
:func:`~stringsim.compactification.torus.spectrum_is_dual`; this is the
generating function of it.

**Truncation.**  The sum is cut at :math:`\ell_L^2 + \ell_R^2 \leq` ``cutoff``,
and the tail is bounded by :math:`e^{-\pi \tau_2 \cdot \text{cutoff}}` times a
lattice-point count.  :func:`truncation_gap` measures what raising the cutoff
changes, so the residuals below can be told apart from the truncation that
produced them.
"""

from __future__ import annotations

import math

import numpy as np

from ..compactification.torus import (
    TorusBackground,
    integer_points_in_ball,
    odd_metric,
    transform,
)
from ..quantum.partition import dedekind_eta

__all__ = [
    "charge_vectors",
    "lattice_norms",
    "theta_series",
    "lattice_partition",
    "compactified_integrand",
    "modular_residual",
    "duality_residual",
    "evenness",
    "level_degeneracies",
    "root_multiplicity",
    "truncation_gap",
]


def _scaling(dim: int, momentum_scale: int) -> np.ndarray:
    """``diag(I, s I)`` in the ``(w, n)`` basis: momenta in multiples of ``s``."""
    if momentum_scale < 1:
        raise ValueError("momentum_scale must be a positive integer")
    return np.diag([1.0] * dim + [float(momentum_scale)] * dim)


def charge_vectors(
    background: TorusBackground, cutoff: float, momentum_scale: int = 1
) -> np.ndarray:
    r"""Every ``(w, n)`` with :math:`\ell_L^2 + \ell_R^2 \leq` ``cutoff``.

    Returned as an integer array of shape ``(count, 2d)`` in the ``(w, n)``
    ordering the rest of the package uses.  The enumeration is exact -- the
    ellipsoid is walked by Fincke-Pohst in
    :func:`~stringsim.compactification.torus.integer_points_in_ball`, not
    approximated by a box.

    ``momentum_scale`` restricts the momenta to multiples of ``s``.  That is a
    sublattice of index :math:`s^d`: still even, no longer self-dual, and it is
    how the two conditions are separated below.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    dim = background.dim
    scale = _scaling(dim, momentum_scale)
    gram = scale.T @ background.generalized_metric() @ scale
    reduced = integer_points_in_ball(gram, np.zeros(2 * dim), float(cutoff))
    # ``reduced`` is (count, 2d).  At d = 0 that is (1, 0) -- one charge vector,
    # the empty one -- whose ``size`` is also 0, so the emptiness test has to be
    # on the row count.  Getting this wrong drops the lattice's only point and
    # makes Theta zero instead of one.
    if reduced.shape[0] == 0:  # pragma: no cover - cutoff below the origin
        return np.zeros((0, 2 * dim), dtype=int)
    return np.rint(reduced @ scale.T).astype(int).reshape(reduced.shape[0], 2 * dim)


def lattice_norms(
    background: TorusBackground, charges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""``(l_L^2, l_R^2)`` for a stack of ``(w, n)`` charge vectors.

    From the two quadratic forms rather than from the momenta themselves:

    .. math::
       \ell_L^2 + \ell_R^2 = Z^{T} H Z, \qquad
       \ell_L^2 - \ell_R^2 = Z^{T} \eta Z ,

    with ``H`` the generalized metric and ``eta`` the ``O(d,d)`` form.  Adding
    and subtracting gives both.  This is a different route from
    :func:`~stringsim.compactification.torus.narain_momenta`, which builds
    :math:`\ell_L` and :math:`\ell_R` explicitly through
    :math:`G^{-1/2}`; the test suite checks the two agree, and the theta series
    needs the vectorised one because it sums over thousands of charges.
    """
    charges = np.atleast_2d(np.asarray(charges, dtype=float))
    metric = background.generalized_metric()
    eta = odd_metric(background.dim)
    total = np.sum((charges @ metric) * charges, axis=1)
    signed = np.sum((charges @ eta) * charges, axis=1)
    return (total + signed) / 2.0, (total - signed) / 2.0


def theta_series(
    background: TorusBackground,
    tau: complex,
    cutoff: float = 24.0,
    momentum_scale: int = 1,
) -> complex:
    r""":math:`\Theta_\Gamma(\tau,\bar\tau) = \sum q^{\ell_L^2/2}\bar q^{\ell_R^2/2}`.

    Each term has modulus :math:`e^{-\pi\tau_2(\ell_L^2+\ell_R^2)}`, so the sum
    converges absolutely and the ``cutoff`` is a precision knob rather than an
    approximation of unknown quality.
    """
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half plane")
    charges = charge_vectors(background, cutoff, momentum_scale)
    left, right = lattice_norms(background, charges)
    phase = 1j * math.pi * (tau * left - tau.conjugate() * right)
    return complex(np.sum(np.exp(phase)))


def lattice_partition(
    background: TorusBackground,
    tau: complex,
    cutoff: float = 24.0,
    momentum_scale: int = 1,
    n_terms: int = 200,
) -> complex:
    r""":math:`\Theta_\Gamma / |\eta|^{2d}` -- the compact directions' contribution.

    Weight zero on its own: :math:`\Theta` picks up :math:`|\tau|^d` under
    ``S`` and :math:`|\eta|^{2d}` picks up exactly the same, so the ratio is
    invariant.  That cancellation is the whole content of even self-duality.
    """
    eta = dedekind_eta(tau, n_terms)
    return theta_series(background, tau, cutoff, momentum_scale) / abs(eta) ** (
        2 * background.dim
    )


def compactified_integrand(
    background: TorusBackground,
    tau: complex,
    cutoff: float = 24.0,
    dim: int | None = None,
    momentum_scale: int = 1,
    n_terms: int = 200,
) -> complex:
    r"""The modular-invariant one-loop density on :math:`\mathbb{R}^{D-d}\times T^d`.

    .. math::
       \tau_2^{-(D-2-d)/2}\,|\eta(\tau)|^{-2(D-2)}\,\Theta_\Gamma(\tau,\bar\tau)

    ``dim`` defaults to the background's own ``D``.  At ``d = 0`` the theta
    series is 1 and this collapses onto
    :func:`stringsim.amplitudes.oneloop.torus_integrand`; the test suite checks
    the two agree rather than trusting the algebra.
    """
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half plane")
    total = background.conventions.dim if dim is None else dim
    compact = background.dim
    if total - 2 - compact < 0:
        raise ValueError(f"{compact} compact directions do not fit in D = {total}")
    eta = dedekind_eta(tau, n_terms)
    measure = tau.imag ** (-(total - 2 - compact) / 2.0)
    return (
        measure
        * abs(eta) ** (-2 * (total - 2))
        * theta_series(background, tau, cutoff, momentum_scale)
    )


def modular_residual(
    background: TorusBackground,
    tau: complex,
    cutoff: float = 24.0,
    momentum_scale: int = 1,
    n_terms: int = 200,
) -> tuple[float, float]:
    r"""Relative change of :func:`compactified_integrand` under ``T`` and ``S``.

    Returns ``(T residual, S residual)``.  Both vanish for the Narain lattice.
    With ``momentum_scale > 1`` the first still vanishes -- the sublattice is
    still even -- and the second does not, which is the separation the module
    exists to show.
    """
    base = compactified_integrand(background, tau, cutoff, None, momentum_scale, n_terms)
    shifted = compactified_integrand(
        background, tau + 1.0, cutoff, None, momentum_scale, n_terms
    )
    inverted = compactified_integrand(
        background, -1.0 / tau, cutoff, None, momentum_scale, n_terms
    )
    return abs(shifted / base - 1.0), abs(inverted / base - 1.0)


def duality_residual(
    background: TorusBackground,
    omega: np.ndarray,
    tau: complex,
    cutoff: float = 24.0,
    n_terms: int = 200,
) -> float:
    r"""Relative difference between a background and its ``O(d,d;Z)`` image.

    ``omega`` acts on the moduli through
    :func:`~stringsim.compactification.torus.transform` and on the charges
    through :func:`~stringsim.compactification.torus.transform_charges`, leaving
    both :math:`Z^T H Z` and :math:`Z^T \eta Z` alone -- so the theta series,
    which depends on nothing else, cannot move.

    Comparing partition functions rather than spectra sidesteps the trap that
    :func:`~stringsim.compactification.torus.spectrum_is_dual` was written to
    avoid: a truncated charge box is sheared by the duality and states leave it.
    Here the truncation is a *ball* in the invariant form ``Z^T H Z``, so the
    same set of terms is summed on both sides by construction.
    """
    dual = transform(background, omega)
    here = compactified_integrand(background, tau, cutoff, None, 1, n_terms)
    there = compactified_integrand(dual, tau, cutoff, None, 1, n_terms)
    return abs(there / here - 1.0)


def evenness(
    background: TorusBackground, cutoff: float = 24.0, momentum_scale: int = 1
) -> float:
    r"""How far :math:`\ell_L^2 - \ell_R^2` is from an even integer, at worst.

    It equals :math:`2\,n\cdot w` identically, so this is zero to round-off --
    and it is the only thing ``T`` invariance needs, which is why a sublattice
    keeps it.
    """
    charges = charge_vectors(background, cutoff, momentum_scale)
    left, right = lattice_norms(background, charges)
    difference = left - right
    return float(np.max(np.abs(difference - 2.0 * np.rint(difference / 2.0))))


def level_degeneracies(
    background: TorusBackground, cutoff: float = 24.0, momentum_scale: int = 1
) -> dict[tuple[float, float], int]:
    r"""How many charges sit at each :math:`(\ell_L^2/2,\ \ell_R^2/2)`.

    The coefficients of the theta series, grouped.  Keys are rounded to six
    decimals so that numerically equal levels collect together; on a generic
    background most levels hold a single charge, and at a symmetry-enhanced
    point they pile up.
    """
    charges = charge_vectors(background, cutoff, momentum_scale)
    left, right = lattice_norms(background, charges)
    counts: dict[tuple[float, float], int] = {}
    for a, b in zip(left, right, strict=True):
        key = (round(a / 2.0, 6), round(b / 2.0, 6))
        counts[key] = counts.get(key, 0) + 1
    return counts


def root_multiplicity(background: TorusBackground, cutoff: float = 24.0) -> int:
    r"""Charges with one side of squared length 2 and the other zero.

    These are the gauge bosons, and
    :func:`~stringsim.compactification.torus.root_vectors` finds them by solving
    the condition exactly rather than enumerating.  The two counts agreeing is a
    check on the enumeration used for the theta series.
    """
    counts = level_degeneracies(background, cutoff)
    return counts.get((1.0, 0.0), 0) + counts.get((0.0, 1.0), 0)


def truncation_gap(
    background: TorusBackground,
    tau: complex,
    cutoff: float = 24.0,
    extra: float = 12.0,
    momentum_scale: int = 1,
) -> float:
    r"""Relative change in :math:`\Theta` when the cutoff is raised by ``extra``.

    An honest error bar for every number in this module.  A residual larger than
    this is physics; one comparable to it is truncation, and the test suite
    insists on the difference.
    """
    tight = theta_series(background, tau, cutoff, momentum_scale)
    wide = theta_series(background, tau, cutoff + extra, momentum_scale)
    return abs(wide / tight - 1.0)

