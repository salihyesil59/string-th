r"""The heterotic string on :math:`T^d`: :math:`\Gamma_{16+d,d}` and Wilson lines.

Compactify ``d`` of the ten spacetime directions and the two constructions the
package already has merge into one.  The left-movers now carry ``16 + d``
compact directions -- the gauge lattice plus the torus -- and the right-movers
carry ``d``, so the charge lattice is even, self-dual and of signature

.. math::  (16 + d,\ d).

It is simply :math:`\Gamma_{16} \oplus \Gamma_{d,d}` as a lattice: the gauge
lattice of :mod:`stringsim.heterotic.lattice` and the Narain lattice of
:mod:`stringsim.compactification.torus`, side by side.  What is *new* is a third
kind of modulus.

**Wilson lines.**  Besides ``G`` and ``B`` there is now
:math:`A_i^{\,I}`, a gauge-lattice vector for each compact direction --
``16 d`` more moduli, so

.. math::  d\frac{d+1}{2} + d\frac{d-1}{2} + 16 d = d(d + 16)

in all, which is the dimension of :math:`O(16+d,d)/(O(16+d)\times O(d))`.

A Wilson line acts on charges as a shift, :math:`\pi \to \pi + A w` with a
compensating shift of the momentum, and that shift is an :math:`O(16+d,d)`
rotation -- :func:`wilson_boost` builds it and the test suite checks it
preserves the lattice form.  The moduli therefore enter exactly as they did for
the torus: through a positive-definite :math:`\mathcal{H}`, with

.. math::
   p_L^2 + p_R^2 = Z^{T}\mathcal{H}Z, \qquad
   p_L^2 - p_R^2 = Z^{T}\eta Z = \pi^2 + 2 n_i w^i .

Those two determine :math:`p_L^2` and :math:`p_R^2` separately, which is all the
mass formula needs:

.. math::
   \frac{\alpha' M^2}{4} = N_L - 1 + \frac{p_L^2}{2}
                         = N_R - a_R + \frac{p_R^2}{2}.

**Wilson lines break the gauge group.**  A gauge boson needs
:math:`p_R = 0` and :math:`p_L^2 = 2`.  With no winding that forces
:math:`n_i = A_i \cdot \pi`, and ``n`` is an *integer*, so a root survives only
when

.. math::  A_i \cdot \pi \in \mathbb{Z} \quad\text{for every } i .

That single condition is the whole mechanism.  Switching on
:math:`A = (1, 0^7; 0^8)` in :math:`E_8 \times E_8` keeps the 112 roots
:math:`\pm e_i \pm e_j` and discards the 128 half-integer ones, leaving
``so(16) + e8``.  :func:`unbroken_roots` applies the condition and
:func:`gauge_algebra` names what is left, reusing the root decomposition written
for the torus.

**Reductions, which the tests enforce.**  At ``d = 0`` this must reproduce
:mod:`stringsim.heterotic.spectrum` exactly -- every root has
:math:`(p_L^2, p_R^2) = (2, 0)` and the mass formula loses its ``p_R`` term.  At
zero gauge charge and zero Wilson line the compact part must reproduce
:mod:`stringsim.compactification.torus` momentum for momentum.

**Not here.**  Enhanced symmetry from winding states -- roots with
:math:`w \neq 0`, which appear at special radii -- is not enumerated; that would
need lattice vectors of gauge norm above 2 and is a search of a different size.
:func:`unbroken_roots` says so in its docstring rather than quietly returning a
partial answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..compactification.torus import TorusBackground, odd_metric
from .lattice import RootLattice, e8_squared

__all__ = [
    "GAUGE_RANK",
    "wilson_boost",
    "narain_form",
    "HeteroticBackground",
    "unbroken_roots",
    "gauge_algebra",
    "moduli_count",
    "gauge_rank",
    "narain_signature",
]

GAUGE_RANK = 16
"""The internal left-moving rank, the same ``26 - 10`` for every heterotic string."""


def moduli_count(dim: int) -> int:
    r"""``d(d + 16)``: the dimension of ``O(16+d,d)/(O(16+d) x O(d))``.

    ``d(d+1)/2`` from the metric, ``d(d-1)/2`` from the ``B`` field and ``16d``
    from the Wilson lines.
    """
    if dim < 0:
        raise ValueError("dim must be non-negative")
    return dim * (dim + GAUGE_RANK)


def gauge_rank(dim: int) -> int:
    """``16 + 2d``: the rank of the gauge group after compactifying ``d`` directions.

    Sixteen from the internal lattice, and one each from the metric and the
    ``B`` field per compact direction.
    """
    if dim < 0:
        raise ValueError("dim must be non-negative")
    return GAUGE_RANK + 2 * dim


def narain_signature(dim: int) -> tuple[int, int]:
    """``(16 + d, d)``, the signature of the charge lattice."""
    if dim < 0:
        raise ValueError("dim must be non-negative")
    return GAUGE_RANK + dim, dim


def narain_form(dim: int, gauge_rank_: int = GAUGE_RANK) -> np.ndarray:
    r"""``eta`` on the charge basis ``Z = (pi; w, n)``.

    Block diagonal: the identity on the gauge directions -- the gauge lattice is
    written in Cartesian coordinates, where its form *is* the identity -- and
    :func:`stringsim.compactification.torus.odd_metric` on the rest.  It is
    moduli independent, which is what makes ``p_L^2 - p_R^2`` a topological
    quantity.
    """
    if dim < 0 or gauge_rank_ < 0:
        raise ValueError("dimensions must be non-negative")
    top = np.hstack([np.eye(gauge_rank_), np.zeros((gauge_rank_, 2 * dim))])
    bottom = np.hstack([np.zeros((2 * dim, gauge_rank_)), odd_metric(dim)])
    return np.vstack([top, bottom])


def wilson_boost(wilson_lines: np.ndarray) -> np.ndarray:
    r"""The ``O(16+d,d)`` rotation implementing a Wilson line.

    ``wilson_lines`` is ``A``, of shape ``(16, d)``: one gauge vector per compact
    direction.  The charge shift is

    .. math::
       \pi \to \pi + A w, \qquad w \to w, \qquad
       n \to n - A^{T}\pi - \tfrac12 (A^{T}A) w ,

    and the momentum shift is exactly what keeps :math:`\pi^2 + 2 n\cdot w`
    invariant -- the cross terms cancel between the two.  The test suite checks
    that rather than taking it on trust.
    """
    a = np.asarray(wilson_lines, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"wilson lines must be a 2-D array (16, d), got {a.shape}")
    gauge, dim = a.shape
    return np.block(
        [
            [np.eye(gauge), a, np.zeros((gauge, dim))],
            [np.zeros((dim, gauge)), np.eye(dim), np.zeros((dim, dim))],
            [-a.T, -0.5 * (a.T @ a), np.eye(dim)],
        ]
    )


@dataclass(frozen=True)
class HeteroticBackground:
    """A heterotic string on ``T^d``: a gauge lattice, a torus, and Wilson lines.

    Parameters
    ----------
    gauge_lattice:
        ``E8 + E8`` or ``D16+`` from :mod:`stringsim.heterotic.lattice`.
    torus:
        The compact ``T^d`` carrying ``G`` and ``B``.  ``None`` means ``d = 0``,
        the uncompactified heterotic string.
    wilson_lines:
        ``A``, shape ``(16, d)``.  Defaults to zero.
    """

    gauge_lattice: RootLattice = field(default_factory=e8_squared)
    torus: TorusBackground | None = None
    wilson_lines: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.gauge_lattice.dim != GAUGE_RANK:
            raise ValueError(
                f"gauge lattice must be {GAUGE_RANK}-dimensional, "
                f"got {self.gauge_lattice.dim}"
            )
        dim = 0 if self.torus is None else self.torus.dim
        lines = (
            np.zeros((GAUGE_RANK, dim))
            if self.wilson_lines is None
            else np.asarray(self.wilson_lines, dtype=float).reshape(GAUGE_RANK, dim)
        )
        object.__setattr__(self, "wilson_lines", lines)

    # -- shape --------------------------------------------------------------

    @property
    def dim(self) -> int:
        """``d``, the number of compact spacetime directions."""
        return 0 if self.torus is None else self.torus.dim

    @property
    def spacetime_dimension(self) -> int:
        """``10 - d``: what is left non-compact."""
        return 10 - self.dim

    @property
    def rank(self) -> int:
        """``16 + 2d``."""
        return gauge_rank(self.dim)

    @property
    def moduli_count(self) -> int:
        """``d(d + 16)``."""
        return moduli_count(self.dim)

    @property
    def signature(self) -> tuple[int, int]:
        """``(16 + d, d)``."""
        return narain_signature(self.dim)

    # -- the two forms ------------------------------------------------------

    def narain_form(self) -> np.ndarray:
        """``eta``, moduli independent."""
        return narain_form(self.dim)

    def generalized_metric(self) -> np.ndarray:
        r"""``H``, positive definite and in ``O(16+d,d)``.

        Built as :math:`\Omega_A^{T} \mathcal{H}_0 \Omega_A`, where
        :math:`\mathcal{H}_0` is the identity on the gauge block beside the
        torus module's own generalized metric.  Conjugating by an element of
        the group keeps ``H`` in it, which the tests verify.
        """
        dim = self.dim
        base = np.eye(GAUGE_RANK)
        if dim:
            torus_block = self.torus.generalized_metric()
            base = np.block(
                [
                    [base, np.zeros((GAUGE_RANK, 2 * dim))],
                    [np.zeros((2 * dim, GAUGE_RANK)), torus_block],
                ]
            )
        boost = wilson_boost(self.wilson_lines)
        return boost.T @ base @ boost

    def charge_vector(self, gauge, winding=None, momentum=None) -> np.ndarray:
        """Assemble ``Z = (pi; w, n)`` from its three pieces."""
        dim = self.dim
        gauge = np.asarray(gauge, dtype=float).reshape(GAUGE_RANK)
        winding = np.zeros(dim) if winding is None else np.asarray(winding, float).reshape(dim)
        momentum = np.zeros(dim) if momentum is None else np.asarray(momentum, float).reshape(dim)
        return np.concatenate([gauge, winding, momentum])

    def momenta_squared(self, charge) -> tuple[float, float]:
        r"""``(p_L^2, p_R^2)`` from the two quadratic forms.

        Since :math:`p_L^2 + p_R^2 = Z^T H Z` and
        :math:`p_L^2 - p_R^2 = Z^T \eta Z`, the halves and sums give each
        separately -- the individual vectors are never needed.
        """
        z = np.asarray(charge, dtype=float)
        total = float(z @ self.generalized_metric() @ z)
        difference = float(z @ self.narain_form() @ z)
        return 0.5 * (total + difference), 0.5 * (total - difference)

    def alpha_m2(self, charge, left_level: int = 0) -> float:
        r"""``alpha' M^2`` from the left-moving side, ``4(N_L - 1 + p_L^2/2)``.

        Only the left side is needed: for a physical state the right side gives
        the same number, and :meth:`level_matching_defect` is what checks that.
        """
        left, _ = self.momenta_squared(charge)
        return 4.0 * (left_level - 1.0 + 0.5 * left)

    def level_matching_defect(
        self, charge, left_level: int = 0, right_level: float = 0.0, right_intercept: float = 0.5
    ) -> float:
        r"""Left mass minus right mass; zero for a physical state.

        ``right_intercept`` is ``1/2`` in the NS sector and ``0`` in R, from
        :func:`stringsim.superstring.rns.intercept`.
        """
        left, right = self.momenta_squared(charge)
        return (left_level - 1.0 + 0.5 * left) - (right_level - right_intercept + 0.5 * right)

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"heterotic {self.gauge_lattice.name} on T^{self.dim} "
            f"in {self.spacetime_dimension}d: rank {self.rank}, "
            f"{self.moduli_count} moduli, signature {self.signature}"
        )


# ---------------------------------------------------------------------------
# what survives
# ---------------------------------------------------------------------------


def unbroken_roots(background: HeteroticBackground, tol: float = 1e-8) -> np.ndarray:
    r"""Gauge roots that stay massless once the Wilson lines are switched on.

    A massless vector needs :math:`p_R = 0` and :math:`p_L^2 = 2`.  With no
    winding the first condition sets :math:`n_i = A_i \cdot \pi`, and ``n`` has
    to be an integer, so the surviving roots are exactly

    .. math::  \{\pi : \pi^2 = 2,\ A_i \cdot \pi \in \mathbb{Z}\ \forall i\}.

    At ``A = 0`` every root survives and the group is unbroken.

    **This counts the roots with zero winding only.**  At special radii there
    are further massless vectors carrying winding, which enhance the group
    again; finding those means searching gauge charges of norm greater than 2
    and is not attempted here.  For the pure-torus case that enhancement is
    what :func:`stringsim.compactification.torus.root_vectors` computes.
    """
    roots = background.gauge_lattice.roots
    if background.dim == 0:
        return roots
    products = roots @ background.wilson_lines  # (n_roots, d)
    integral = np.all(np.abs(products - np.rint(products)) < tol, axis=1)
    return roots[integral]


def gauge_algebra(background: HeteroticBackground) -> str:
    """Name the unbroken gauge algebra, ``u(1)`` factors included.

    Uses :func:`stringsim.compactification.torus.decompose_roots`, so an
    ambiguous root count is settled by geometry rather than by counting -- the
    same machinery that separates ``e8 + e8`` from ``so(32)``.
    """
    from ..compactification.torus import _name_from_roots

    return _name_from_roots(unbroken_roots(background), GAUGE_RANK)
