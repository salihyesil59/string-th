r"""Where ``D = 26`` comes from: the zero-point energy and the conformal anomaly.

Quantising the transverse oscillators leaves a divergent zero-point energy

.. math::  E_0 = \frac{D-2}{2} \sum_{n=1}^{\infty} n ,

and the sum has to be regularised.  Analytic continuation of the Riemann zeta
function gives :math:`\zeta(-1) = -1/12`, so the normal-ordering constant is
:math:`a = (D-2)/24`.  Nothing about that is a formal trick: the *same* finite
part appears in the Casimir energy of a field between plates, and
:func:`regularised_sum` extracts it here from an honest convergent sum with a
smooth cutoff, by removing the cutoff-dependent piece.

Lorentz invariance then fixes ``D``.  The massless vector at level 1 has
:math:`\alpha' M^2 = 1 - a`, and a massive vector cannot have only ``D - 2``
polarisations, so consistency demands ``a = 1`` and hence ``D = 26``.  The same
number appears as the vanishing of the total conformal anomaly,
:math:`c = D - 26 = 0`, once the ``bc`` ghosts contribute ``-26``.  Both routes
are implemented, and they agree.

For the superstring the fermionic modes shift the count: ``a_NS = (D-2)/16``,
and ``c = D + D/2 - 15 = 0`` gives ``D = 10``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ZetaEstimate",
    "regularised_sum",
    "normal_ordering_constant",
    "critical_dimension",
    "central_charge",
]


@dataclass(frozen=True)
class ZetaEstimate:
    """A numerical estimate of a regularised sum.

    Attributes
    ----------
    value:
        The extracted finite part.
    exact:
        The analytic value it should reproduce.
    epsilons:
        Cutoffs used in the fit.
    """

    value: float
    exact: float
    epsilons: np.ndarray

    @property
    def error(self) -> float:
        """``|value - exact|``."""
        return abs(self.value - self.exact)


def regularised_sum(epsilons: np.ndarray | None = None, n_orders: int = 3) -> ZetaEstimate:
    r"""Extract ``zeta(-1) = -1/12`` from a smoothly cut-off sum.

    With an exponential cutoff the sum converges,

    .. math::
       \sum_{n\geq1} n\, e^{-\epsilon n} = \frac{e^{-\epsilon}}{(1-e^{-\epsilon})^2}
       = \frac{1}{\epsilon^2} - \frac{1}{12} + \frac{\epsilon^2}{240} - \dots

    The :math:`1/\epsilon^2` piece is the cutoff-dependent divergence, which in
    a physical setting is absorbed by a counterterm; the constant left behind is
    universal, and it is ``-1/12``.  This function subtracts the divergence and
    fits the remainder in powers of :math:`\epsilon^2` to read off that
    constant.

    Parameters
    ----------
    epsilons:
        Cutoffs to use.  They must be small enough for the expansion to hold
        but large enough that ``1/eps**2`` does not swamp the constant in
        double precision; the default range is chosen with that in mind.
    n_orders:
        Number of terms ``eps^{2k}`` kept in the fit.
    """
    eps = np.geomspace(0.02, 0.2, 24) if epsilons is None else np.asarray(epsilons, float)
    if np.any(eps <= 0):
        raise ValueError("cutoffs must be positive")
    summed = np.exp(-eps) / (1.0 - np.exp(-eps)) ** 2
    remainder = summed - 1.0 / eps**2
    design = np.column_stack([eps ** (2 * k) for k in range(n_orders)])
    coef, *_ = np.linalg.lstsq(design, remainder, rcond=None)
    return ZetaEstimate(value=float(coef[0]), exact=-1.0 / 12.0, epsilons=eps)


def normal_ordering_constant(dim: int = 26, theory: str = "bosonic") -> float:
    r"""``a`` in ``alpha' M^2 = N - a``.

    ``bosonic``: :math:`a = (D-2)/24`, which is 1 at ``D = 26``.
    ``superstring``: the NS-sector value :math:`a = (D-2)/16`, which is 1/2 at
    ``D = 10``.
    """
    if theory == "bosonic":
        return (dim - 2) / 24.0
    if theory == "superstring":
        return (dim - 2) / 16.0
    raise ValueError("theory must be 'bosonic' or 'superstring'")


def critical_dimension(theory: str = "bosonic") -> int:
    """``26`` for the bosonic string, ``10`` for the superstring.

    Solved from ``normal_ordering_constant(D) == 1`` (bosonic) or ``== 1/2``
    (superstring), rather than returned as a literal.
    """
    target = {"bosonic": 1.0, "superstring": 0.5}
    if theory not in target:
        raise ValueError("theory must be 'bosonic' or 'superstring'")
    for dim in range(3, 200):
        if abs(normal_ordering_constant(dim, theory) - target[theory]) < 1e-12:
            return dim
    raise RuntimeError("no critical dimension found")  # pragma: no cover


def central_charge(dim: int, theory: str = "bosonic") -> float:
    r"""Total worldsheet central charge, matter plus ghosts.

    ``bosonic``: ``D`` free scalars give ``c = D``, the ``bc`` ghost system
    gives ``-26``.

    ``superstring``: ``D`` scalars and ``D`` Majorana fermions give
    ``D + D/2``, the ``bc`` and ``beta gamma`` ghosts give ``-26 + 11 = -15``.

    Vanishing central charge is the same condition as ``a = 1`` (resp. 1/2), so
    ``central_charge(critical_dimension(t), t)`` is zero for both theories.
    """
    if theory == "bosonic":
        return dim - 26.0
    if theory == "superstring":
        return 1.5 * dim - 15.0
    raise ValueError("theory must be 'bosonic' or 'superstring'")
