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

**Winding states enhance it again.**  :func:`unbroken_roots` only sees charges
with :math:`w = 0`.  The complete answer is :func:`massless_vectors`, which
undoes the Wilson line to split the two conditions into

.. math::
   |\pi + Aw|^2 = 2 - 2\,w^{T}Gw, \qquad
   Ew + A^{T}\pi + \tfrac12 (A^{T}A)w \in \mathbb{Z}^d ,

and that search is **finite and exhaustive**: the left side cannot be negative,
so :math:`w^{T}Gw \leq 1` bounds the winding, and each ``w`` leaves a ball of
radius at most :math:`\sqrt2` for the gauge charge.  Nothing is truncated.

On a circle the first equation *solves* for ``G`` instead of being tested at it,
so :func:`enhancement_radii` returns the special radii themselves rather than
finding them by scanning.  What they show is that a Wilson line is not by itself
gauge-invariant information: :math:`A = (\tfrac12, 0^{15})` breaks
:math:`E_8 \times E_8` to :math:`e_8 + so(14)` at a generic radius, and at
:math:`G = 1/8` all 480 roots are back, 156 of them carrying winding.  Two
points of the same moduli space, related by :math:`O(17,1;\mathbb{Z})`.

And the two ten-dimensional theories meet.  :math:`E_8 \times E_8` with
:math:`A = (1, 0^7; 1, 0^7)` and :math:`Spin(32)/\mathbb{Z}_2` with
:math:`A = (\tfrac12^8; 0^8)` both give :math:`so(16) + so(16) + u(1)^2` at
*every* radius -- neither has an enhancement point anywhere -- which is what one
expects of two descriptions of a single nine-dimensional theory.  The lattice
statement behind it is that :math:`\Gamma_{17,1}` is unique up to isomorphism.
"""

from __future__ import annotations

import itertools
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

    **This counts the roots with zero winding only**, which is the whole answer
    at a generic point in the moduli space and not at a special one.
    :func:`massless_vectors` adds the winding states and is complete; the two
    agree exactly when :func:`enhancement_radii` says there is nothing extra
    at this radius.
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


# ---------------------------------------------------------------------------
# the complete answer, winding included
# ---------------------------------------------------------------------------


def _points_in_ball(
    gram: np.ndarray, offset: np.ndarray, radius_sq: float, tol: float = 1e-9
) -> np.ndarray:
    r"""Integer ``m`` with ``(m - offset)^T Q (m - offset) <= radius_sq``.

    Fincke-Pohst: factor ``Q = R^{T}R`` with ``R`` upper triangular, so the form
    is :math:`\sum_i (\sum_{j \geq i} R_{ij}(m_j - y_j))^2`.  Choosing the last
    coordinate first leaves an interval for each earlier one, and the recursion
    visits exactly the points inside the ellipsoid -- no box, no truncation.
    """
    size = gram.shape[0]
    triangular = np.linalg.cholesky(gram).T
    offset = np.asarray(offset, dtype=float).reshape(size)
    found: list[np.ndarray] = []
    current = np.zeros(size)

    def descend(index: int, budget: float) -> None:
        if index < 0:
            found.append(current.copy())
            return
        tail = float(triangular[index, index + 1 :] @ (current[index + 1 :] - offset[index + 1 :]))
        diagonal = triangular[index, index]
        half_width = np.sqrt(max(budget, 0.0)) / diagonal
        centre = offset[index] - tail / diagonal
        for value in range(
            int(np.ceil(centre - half_width - tol)), int(np.floor(centre + half_width + tol)) + 1
        ):
            current[index] = value
            term = diagonal * (value - offset[index]) + tail
            descend(index - 1, budget - term * term)

    descend(size - 1, radius_sq + tol)
    return np.array(found) if found else np.zeros((0, size))


def gauge_vectors_near(
    lattice: RootLattice, centre, radius_sq: float, tol: float = 1e-9
) -> np.ndarray:
    """Every lattice vector within ``radius_sq`` of ``centre``, in Cartesian coordinates.

    Complete, not sampled: the ellipsoid is finite because the Gram matrix is
    positive definite.  At ``centre = 0`` and ``radius_sq = 2`` this returns the
    roots and the origin -- 481 vectors for either heterotic lattice.
    """
    if radius_sq < -tol:
        raise ValueError(f"radius_sq must be non-negative, got {radius_sq}")
    basis = lattice.basis()
    centre = np.asarray(centre, dtype=float).reshape(lattice.dim)
    offset = np.linalg.solve(basis.T, centre)
    return _points_in_ball(lattice.gram(), offset, max(radius_sq, 0.0), tol) @ basis


def massless_vectors(background: HeteroticBackground, tol: float = 1e-8) -> np.ndarray:
    r"""Every charge carrying a massless gauge boson.  Complete, winding included.

    A massless vector needs :math:`p_R = 0` and :math:`p_L^2 = 2`.  Undoing the
    Wilson line with :func:`wilson_boost` splits both conditions in two, because
    at :math:`A = 0` the generalized metric is block diagonal:

    .. math::
       p_R^2 = \ell_R^2, \qquad
       p_L^2 = |\pi + Aw|^2 + \ell_L^2 ,

    with :math:`\ell` the torus momenta of the shifted charges
    :math:`\tilde n = n - A^{T}\pi - \tfrac12 (A^{T}A) w`.  Then
    :math:`\ell_R = 0` forces :math:`\tilde n = Ew` and leaves
    :math:`\ell_L^2 = 2\,w^{T}Gw`, so the whole condition is

    .. math::
       |\pi + Aw|^2 = 2 - 2\,w^{T}Gw, \qquad
       Ew + A^{T}\pi + \tfrac12 (A^{T}A)w \in \mathbb{Z}^d .

    **The search is finite and exhaustive.**  The left side of the first
    equation cannot be negative, so :math:`w^{T}Gw \leq 1` bounds the winding by
    :math:`|w|^2 \leq 1/\lambda_{\min}(G)`, and for each ``w`` the gauge charge
    lies in a ball of radius at most :math:`\sqrt 2`, which
    :func:`gauge_vectors_near` enumerates exactly.  Nothing is truncated.

    At ``w = 0`` this reduces to :func:`unbroken_roots`: the ball of radius
    :math:`\sqrt2` about the origin holds the roots, and the integrality
    condition becomes :math:`A_i \cdot \pi \in \mathbb{Z}`.  Every other ``w``
    is a state that only becomes massless at particular moduli -- the
    enhancement :func:`unbroken_roots` was documented as not finding.

    Returns
    -------
    ndarray
        Shape ``(n_roots, 16 + 2d)``, the charges ``Z = (pi; w, n)``, sorted.
    """
    dim = background.dim
    lattice = background.gauge_lattice
    if dim == 0:
        return lattice.roots.copy()

    metric = background.torus.metric
    e_matrix = background.torus.e_matrix
    lines = background.wilson_lines
    bound = int(np.floor(np.sqrt(1.0 / float(np.linalg.eigvalsh(metric)[0])) + 1e-9))

    found: list[np.ndarray] = []
    for winding in itertools.product(range(-bound, bound + 1), repeat=dim):
        w = np.array(winding, dtype=float)
        radius_sq = 2.0 - 2.0 * float(w @ metric @ w)
        if radius_sq < -tol:
            continue
        shift = lines @ w
        for gauge in gauge_vectors_near(lattice, -shift, max(radius_sq, 0.0), tol):
            shifted = gauge + shift
            if abs(float(shifted @ shifted) - radius_sq) > tol:
                continue
            momentum = e_matrix @ w + lines.T @ gauge + 0.5 * (lines.T @ lines) @ w
            if np.all(np.abs(momentum - np.rint(momentum)) < tol):
                found.append(background.charge_vector(gauge, w, np.rint(momentum)))
    if not found:
        return np.zeros((0, background.rank))
    charges = np.array(found)
    return charges[np.lexsort(charges.T[::-1])]


def left_momenta(background: HeteroticBackground, charges) -> np.ndarray:
    r"""The ``(16 + d)``-dimensional left-moving momenta of a set of charges.

    :meth:`HeteroticBackground.momenta_squared` only ever needs the norms, but
    naming an algebra needs the vectors: :func:`decompose_roots` splits a root
    system by which roots are *not* orthogonal.  The map is

    .. math::  Z \mapsto \big(\pi + Aw,\ \ell_L(\tilde n, w)\big),

    linear in ``Z``, and it reproduces the norms and inner products the two
    quadratic forms give -- which is what the test suite checks rather than the
    formula itself.
    """
    from ..compactification.torus import narain_momenta

    charges = np.atleast_2d(np.asarray(charges, dtype=float))
    dim = background.dim
    lines = background.wilson_lines
    gauge = charges[:, :GAUGE_RANK]
    if dim == 0:
        return gauge
    winding = charges[:, GAUGE_RANK : GAUGE_RANK + dim]
    momentum = charges[:, GAUGE_RANK + dim :]
    tilde = momentum - gauge @ lines - 0.5 * winding @ (lines.T @ lines)
    compact = np.array(
        [narain_momenta(background.torus, t, w)[0] for t, w in zip(tilde, winding, strict=True)]
    )
    return np.hstack([gauge + winding @ lines.T, compact])


def enhanced_algebra(background: HeteroticBackground) -> str:
    """Name the full gauge algebra, winding states included.

    Equal to :func:`gauge_algebra` at a generic point in the moduli space, and
    larger exactly where extra states come down to zero mass.
    """
    from ..compactification.torus import _name_from_roots

    return _name_from_roots(left_momenta(background, massless_vectors(background)), background.rank)





def charge_lattice_gram(background: HeteroticBackground) -> np.ndarray:
    r"""Gram matrix of :math:`\Gamma_{16+d,d}` in an integral basis.

    Block diagonal: the gauge lattice's own Gram matrix beside
    :func:`stringsim.compactification.torus.odd_metric`, because the charge
    lattice *is* :math:`\Gamma_{16} \oplus \Gamma_{d,d}` -- the Wilson lines
    move the moduli, not the lattice.  Its determinant and diagonal are what
    make the self-duality and evenness checkable rather than quoted.
    """
    gauge = background.gauge_lattice.gram()
    dim = background.dim
    if dim == 0:
        return gauge
    return np.block(
        [
            [gauge, np.zeros((GAUGE_RANK, 2 * dim))],
            [np.zeros((2 * dim, GAUGE_RANK)), odd_metric(dim)],
        ]
    )


def enhancement_radii(
    lattice: RootLattice, wilson_lines, winding_max: int = 4, tol: float = 1e-8
) -> list[float]:
    r"""The circle metrics ``G`` at which winding states become massless vectors.

    On :math:`S^1` there is no ``B`` field, so :math:`E = G` and the two
    conditions of :func:`massless_vectors` read

    .. math::  |\pi + Aw|^2 = 2 - 2Gw^2, \qquad Gw + A\cdot\pi + \tfrac12 A^2 w
               \in \mathbb{Z}.

    The first **solves** for ``G`` rather than being tested at it, so the
    enhancement points come out of a finite enumeration instead of a scan of the
    moduli space: for each ``w`` collect the lattice vectors within
    :math:`\sqrt2` of :math:`-Aw`, read off
    :math:`G = (2 - |\pi + Aw|^2) / 2w^2`, and keep the ones whose momentum is
    an integer.

    ``winding_max`` truncates the enumeration, but not blindly: enhancement
    needs :math:`Gw^2 \leq 1`, so **every** enhancement point with
    :math:`G > 1/\texttt{winding\_max}^2` is found.  Points below that need a
    larger ``winding_max``.

    Returns
    -------
    list[float]
        The metric values, ascending.  ``G = 1`` is the self-dual point, where
        the pure-torus :math:`su(2)` appears whatever the gauge lattice is.
    """
    if winding_max < 1:
        raise ValueError(f"winding_max must be at least 1, got {winding_max}")
    lines = np.asarray(wilson_lines, dtype=float).reshape(GAUGE_RANK)
    found: set[float] = set()
    for winding in range(1, winding_max + 1):
        shift = lines * winding
        for gauge in gauge_vectors_near(lattice, -shift, 2.0, tol):
            shifted = gauge + shift
            norm = float(shifted @ shifted)
            if norm > 2.0 - tol:
                continue
            metric = (2.0 - norm) / (2.0 * winding * winding)
            momentum = (
                metric * winding + float(lines @ gauge) + 0.5 * float(lines @ lines) * winding
            )
            if abs(momentum - round(momentum)) < 1e-7:
                found.add(round(metric, 10))
    return sorted(found)


__all__ += [
    "charge_lattice_gram",
    "gauge_vectors_near",
    "massless_vectors",
    "left_momenta",
    "enhanced_algebra",
    "enhancement_radii",
]
