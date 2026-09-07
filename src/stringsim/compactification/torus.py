r"""Toroidal compactification on :math:`T^d`, the Narain lattice, and ``O(d,d;Z)``.

The circle of :mod:`stringsim.compactification.circle` generalises in a way that
is more than bookkeeping.  Compactifying ``d`` directions brings in a constant
metric :math:`G_{ij}` **and** a constant antisymmetric field :math:`B_{ij}` --
:math:`d(d+1)/2 + d(d-1)/2 = d^2` moduli in all -- and the discrete symmetry
grows from :math:`R \to \alpha'/R` to the full :math:`O(d,d;\mathbb{Z})`.

**Charges and the mass formula.**  A state carries winding :math:`w^i` and
momentum :math:`n_i`, collected into :math:`Z = (w, n) \in \mathbb{Z}^{2d}`.
Writing :math:`E = G + B`, the Narain momenta are

.. math::
   \ell_L = \tfrac{1}{\sqrt2} G^{-1/2}\,(n + E^{T} w), \qquad
   \ell_R = \tfrac{1}{\sqrt2} G^{-1/2}\,(n - E w),

and then

.. math::
   \alpha' M^2 = \ell_L^2 + \ell_R^2 + 2(N + \tilde N - 2), \qquad
   \ell_L^2 - \ell_R^2 = 2\, n_i w^i = 2 (N - \tilde N).

``G`` here is **dimensionless**, measured in units of :math:`\alpha'`; use
:meth:`TorusBackground.from_radii` to build it from physical radii.  At ``d = 1``
with ``B = 0`` this collapses onto the circle formula exactly, which the tests
check rather than assume.

**The generalized metric.**  The charge-dependent part is a quadratic form,

.. math::
   \ell_L^2 + \ell_R^2 = Z^{T} \mathcal{H} Z, \qquad
   \mathcal{H} = \begin{pmatrix} G - BG^{-1}B & BG^{-1} \\
                                 -G^{-1}B     & G^{-1}  \end{pmatrix},

while level matching is the *indefinite* form :math:`Z^{T}\eta Z = 2 n_i w^i`
with :math:`\eta = \bigl(\begin{smallmatrix}0&I\\I&0\end{smallmatrix}\bigr)`.
Two forms on the same lattice: one positive definite and moduli-dependent, one
fixed and of signature ``(d,d)``.  That pairing is the whole structure.

**T-duality.**  Any integer :math:`\Omega` preserving :math:`\eta` maps the
theory to itself, acting as :math:`Z \to \Omega Z` and
:math:`\mathcal{H} \to \Omega^{-T} \mathcal{H} \Omega^{-1}`.  Three kinds of
generator span the group: a change of lattice basis (:func:`basis_change`), an
integer shift of ``B`` (:func:`b_shift`, so ``B`` is periodic), and the
factorized duality that exchanges :math:`w^k \leftrightarrow n_k`
(:func:`factorized_duality`), which is the old ``R -> alpha'/R`` in one
direction.

**Gauge symmetry is a root system.**  A massless vector needs
:math:`(\ell_L^2, \ell_R^2) = (2, 0)` or ``(0, 2)`` -- vectors of squared
length 2, which is exactly the normalisation of the roots of a simply laced Lie
algebra.  :func:`root_vectors` finds them by a *complete* search (the condition
:math:`w^{T} G w = 1` bounds ``w`` outright, so nothing is truncated), and
:func:`gauge_algebra` names the algebra by decomposing that root system into
irreducible pieces -- geometry, not just counting, because rank 3 with six roots
is ``su(2)^3`` or ``su(3) + u(1)`` and the two numbers alone cannot say which.
At the self-dual point of :math:`T^d` this returns :math:`su(2)^d` on each side;
at the :math:`A_2` point of :math:`T^2` it returns :math:`su(3)`.

Reference: Narain, Phys. Lett. B **169** (1986) 41; Giveon, Porrati and
Rabinovici, Phys. Rept. **244** (1994) 77.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..quantum.partition import oscillator_degeneracies
from ..units import Conventions

__all__ = [
    "TorusBackground",
    "TorusState",
    "narain_momenta",
    "spectrum",
    "massless_states",
    "extra_massless_states",
    "root_vectors",
    "GaugeAlgebra",
    "gauge_algebra",
    "identify_algebra",
    "decompose_roots",
    "integer_points_in_ball",
    "odd_metric",
    "is_odd_integer",
    "basis_change",
    "b_shift",
    "factorized_duality",
    "transform",
    "transform_charges",
    "spectrum_is_dual",
    "narain_gram_matrix",
]

_TOL = 1e-9


# ---------------------------------------------------------------------------
# background
# ---------------------------------------------------------------------------


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    """``M^{-1/2}`` for a symmetric positive-definite ``M``, via its eigenbasis."""
    values, vectors = np.linalg.eigh(matrix)
    if np.any(values <= 0):
        raise ValueError("matrix is not positive definite")
    return (vectors / np.sqrt(values)) @ vectors.T


@dataclass(frozen=True)
class TorusBackground:
    r"""Constant metric and ``B``-field on :math:`T^d`.

    Parameters
    ----------
    metric:
        ``G_ij``, symmetric positive definite, **dimensionless**: it is the
        torus metric divided by ``alpha'``, so that the self-dual point is
        ``G = I``.  Build it from physical radii with :meth:`from_radii`.
    b_field:
        ``B_ij``, antisymmetric.  Defaults to zero.  Only its value modulo
        integers matters -- see :func:`b_shift`.
    conventions:
        Supplies ``alpha'`` and ``D``.  ``D`` fixes the oscillator degeneracies,
        which still run over all ``D - 2`` transverse directions, compact ones
        included.
    """

    metric: np.ndarray
    b_field: np.ndarray | None = None
    conventions: Conventions = field(default_factory=Conventions)

    def __post_init__(self) -> None:
        g = np.asarray(self.metric, dtype=float)
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError(f"metric must be square, got shape {g.shape}")
        if not np.allclose(g, g.T, atol=_TOL):
            raise ValueError("metric must be symmetric")
        if np.any(np.linalg.eigvalsh(g) <= _TOL):
            raise ValueError("metric must be positive definite")
        d = g.shape[0]
        b = np.zeros((d, d)) if self.b_field is None else np.asarray(self.b_field, dtype=float)
        if b.shape != (d, d):
            raise ValueError(f"b_field must be {d}x{d}, got {b.shape}")
        if not np.allclose(b, -b.T, atol=_TOL):
            raise ValueError("b_field must be antisymmetric")
        if d >= self.conventions.dim - 1:
            raise ValueError(
                f"cannot compactify {d} directions out of D = {self.conventions.dim}"
            )
        object.__setattr__(self, "metric", g)
        object.__setattr__(self, "b_field", b)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_radii(cls, radii, conventions: Conventions | None = None) -> TorusBackground:
        r"""A rectangular torus of physical radii ``R_i``: ``G = diag((R_i/sqrt(alpha'))^2)``."""
        c = conventions or Conventions()
        r = np.asarray(radii, dtype=float).reshape(-1)
        if np.any(r <= 0):
            raise ValueError("radii must be positive")
        return cls(metric=np.diag((r / c.string_length) ** 2), conventions=c)

    @classmethod
    def self_dual(cls, dim: int, conventions: Conventions | None = None) -> TorusBackground:
        r"""``G = I``, ``B = 0``: every radius at :math:`\sqrt{\alpha'}`.

        The gauge symmetry here is :math:`su(2)^d` on each side, which
        :func:`gauge_algebra` recovers by counting roots.
        """
        return cls(metric=np.eye(int(dim)), conventions=conventions or Conventions())

    @classmethod
    def su3_point(cls, conventions: Conventions | None = None) -> TorusBackground:
        r"""The :math:`A_2` point of :math:`T^2`, where the symmetry is :math:`su(3)`.

        ``G`` is the Gram matrix of the :math:`A_2` root lattice scaled so that
        ``w^T G w = 1`` has six solutions, and ``B`` is chosen to make
        ``E = G + B`` an integer matrix -- which is what makes ``n = E w``
        an allowed momentum for each of them:

        .. math::
           G = \begin{pmatrix} 1 & -1/2 \\ -1/2 & 1 \end{pmatrix}, \qquad
           B = \begin{pmatrix} 0 & 1/2 \\ -1/2 & 0 \end{pmatrix} .
        """
        return cls(
            metric=np.array([[1.0, -0.5], [-0.5, 1.0]]),
            b_field=np.array([[0.0, 0.5], [-0.5, 0.0]]),
            conventions=conventions or Conventions(),
        )

    @classmethod
    def from_generalized_metric(
        cls, generalized: np.ndarray, conventions: Conventions | None = None
    ) -> TorusBackground:
        r"""Read ``(G, B)`` back off a generalized metric.

        The lower-right block is :math:`G^{-1}` and the lower-left is
        :math:`-G^{-1}B`, so both are determined; the upper blocks are then
        *checked* rather than used, which catches a matrix that is not of the
        required form.
        """
        h = np.asarray(generalized, dtype=float)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] % 2:
            raise ValueError(f"expected a square matrix of even size, got {h.shape}")
        d = h.shape[0] // 2
        g_inv = h[d:, d:]
        g = np.linalg.inv(g_inv)
        b = -g @ h[d:, :d]
        candidate = cls(metric=0.5 * (g + g.T), b_field=0.5 * (b - b.T),
                        conventions=conventions or Conventions())
        if not np.allclose(candidate.generalized_metric(), h, atol=1e-8):
            raise ValueError("matrix is not the generalized metric of any (G, B)")
        return candidate

    # -- derived ------------------------------------------------------------

    @property
    def dim(self) -> int:
        """``d``, the number of compact directions."""
        return self.metric.shape[0]

    @property
    def moduli_count(self) -> int:
        r"""``d^2``: the dimension of the Narain moduli space ``O(d,d)/(O(d) x O(d))``."""
        return self.dim**2

    @property
    def volume(self) -> float:
        r"""``sqrt(det G)``, the torus volume in units of ``(2 pi sqrt(alpha'))^d``."""
        return float(np.sqrt(np.linalg.det(self.metric)))

    @property
    def e_matrix(self) -> np.ndarray:
        """``E = G + B``.  Integer ``E`` is what allows roots at all."""
        return self.metric + self.b_field

    def generalized_metric(self) -> np.ndarray:
        r"""The ``2d x 2d`` positive-definite form ``H`` with ``Z^T H Z = l_L^2 + l_R^2``."""
        g, b = self.metric, self.b_field
        g_inv = np.linalg.inv(g)
        return np.block([[g - b @ g_inv @ b, b @ g_inv], [-g_inv @ b, g_inv]])


# ---------------------------------------------------------------------------
# momenta and spectrum
# ---------------------------------------------------------------------------


def narain_momenta(background: TorusBackground, n, w) -> tuple[np.ndarray, np.ndarray]:
    r"""``(l_L, l_R)`` for momentum ``n`` and winding ``w``.

    Normalised so that :math:`\ell_L^2 - \ell_R^2 = 2 n_i w^i`; roots therefore
    have squared length 2, matching the Lie-algebra convention.
    """
    d = background.dim
    n = np.asarray(n, dtype=float).reshape(d)
    w = np.asarray(w, dtype=float).reshape(d)
    root_inv = _inverse_sqrt(background.metric)
    e = background.e_matrix
    left = root_inv @ (n + e.T @ w) / np.sqrt(2.0)
    right = root_inv @ (n - e @ w) / np.sqrt(2.0)
    return left, right


@dataclass(frozen=True)
class TorusState:
    """One level-matched state on the torus."""

    momentum: tuple[int, ...]
    winding: tuple[int, ...]
    level: int
    level_tilde: int
    alpha_m2: float
    p_left_sq: float
    p_right_sq: float
    degeneracy: int

    @property
    def is_massless(self) -> bool:
        return abs(self.alpha_m2) < 1e-9

    @property
    def is_root(self) -> bool:
        """True for a vector of squared length 2 on one side and 0 on the other."""
        left, right = self.p_left_sq, self.p_right_sq
        return (abs(left - 2) < _TOL and abs(right) < _TOL) or (
            abs(right - 2) < _TOL and abs(left) < _TOL
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"(n={list(self.momentum)}, w={list(self.winding)}, "
            f"N={self.level}, Nt={self.level_tilde}) "
            f"alpha'M^2 = {self.alpha_m2:+.4f}  "
            f"l_L^2={self.p_left_sq:.3f} l_R^2={self.p_right_sq:.3f}  x{self.degeneracy}"
        )


def spectrum(
    background: TorusBackground,
    charge_max: int = 1,
    level_max: int = 1,
) -> list[TorusState]:
    """Enumerate level-matched states, sorted by mass.

    Parameters
    ----------
    background:
        The torus.
    charge_max:
        Each component of ``n`` and ``w`` runs over ``-charge_max .. charge_max``.
        The cost grows as ``(2 charge_max + 1)^{2d}``, so keep it small for
        ``d >= 3``.
    level_max:
        Highest oscillator level on either side.

    Notes
    -----
    This is a *truncated* enumeration and is meant for inspecting the light
    states.  :func:`root_vectors` does not truncate -- it solves the root
    condition exactly -- so use that when the gauge group is the question.
    """
    if charge_max < 0 or level_max < 0:
        raise ValueError("charge_max and level_max must be non-negative")
    d = background.dim
    degen = oscillator_degeneracies(level_max, background.conventions.transverse_dim)
    grid = range(-charge_max, charge_max + 1)
    out: list[TorusState] = []
    for n in itertools.product(grid, repeat=d):
        for w in itertools.product(grid, repeat=d):
            left, right = narain_momenta(background, n, w)
            l2, r2 = float(left @ left), float(right @ right)
            for lvl in range(level_max + 1):
                lvl_t = lvl - int(np.dot(n, w))
                if not 0 <= lvl_t <= level_max:
                    continue
                out.append(
                    TorusState(
                        momentum=tuple(n),
                        winding=tuple(w),
                        level=lvl,
                        level_tilde=lvl_t,
                        alpha_m2=l2 + r2 + 2.0 * (lvl + lvl_t - 2),
                        p_left_sq=l2,
                        p_right_sq=r2,
                        degeneracy=degen[lvl] * degen[lvl_t],
                    )
                )
    return sorted(out, key=lambda s: (round(s.alpha_m2, 10), s.momentum, s.winding))


def massless_states(background: TorusBackground, **kw) -> list[TorusState]:
    """Every enumerated state with ``alpha' M^2 = 0``."""
    return [s for s in spectrum(background, **kw) if s.is_massless]


def extra_massless_states(background: TorusBackground, **kw) -> list[TorusState]:
    """Massless states carrying momentum or winding.

    As on the circle, these come in two kinds: the roots (squared length 2 on
    one side, 0 on the other), which are the enhanced gauge bosons, and states
    with ``(2, 2)`` and no oscillators, which are the bosonic string's tachyon
    tower crossing zero mass.
    """
    zero = (0,) * background.dim
    return [
        s
        for s in massless_states(background, **kw)
        if s.momentum != zero or s.winding != zero
    ]


# ---------------------------------------------------------------------------
# roots and the gauge algebra
# ---------------------------------------------------------------------------


def root_vectors(background: TorusBackground) -> tuple[np.ndarray, np.ndarray]:
    r"""All charge vectors giving a massless gauge boson, found exactly.

    A root needs one side to vanish.  Setting :math:`\ell_R = 0` forces
    :math:`n = E w`, and then :math:`\ell_L^2 = 2\, w^{T} G w`, so the condition
    is

    .. math::  w^{T} G w = 1, \qquad E w \in \mathbb{Z}^d ,

    and the left half likewise with :math:`n = -E^{T} w`.  Because ``G`` is
    positive definite the first condition bounds ``w`` by
    :math:`|w|^2 \leq 1/\lambda_{\min}(G)`, so the search below is complete --
    there is no truncation to worry about.

    Returns
    -------
    (left, right):
        Arrays of shape ``(n_roots, d)`` holding the ``l_L`` vectors of the
        left-moving roots and the ``l_R`` vectors of the right-moving ones.
    """
    g = background.metric
    e = background.e_matrix
    lambda_min = float(np.linalg.eigvalsh(g)[0])
    bound = int(np.floor(np.sqrt(1.0 / lambda_min) + 1e-9))

    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for w in itertools.product(range(-bound, bound + 1), repeat=background.dim):
        w_arr = np.array(w, dtype=float)
        if abs(w_arr @ g @ w_arr - 1.0) > _TOL:
            continue
        n_left = e @ w_arr  # l_R = 0
        if np.all(np.abs(n_left - np.rint(n_left)) < _TOL):
            vec, _ = narain_momenta(background, np.rint(n_left), w_arr)
            left.append(vec)
        n_right = -e.T @ w_arr  # l_L = 0
        if np.all(np.abs(n_right - np.rint(n_right)) < _TOL):
            _, vec = narain_momenta(background, np.rint(n_right), w_arr)
            right.append(vec)
    empty = np.zeros((0, background.dim))
    return (np.array(left) if left else empty, np.array(right) if right else empty)


def _simply_laced(max_rank: int) -> list[tuple[str, int, int]]:
    """``(name, rank, number of roots)`` for the simply-laced simple algebras.

    Only ``A``, ``D`` and ``E`` appear, and that is a physical statement rather
    than a simplification: every gauge boson here comes from a lattice vector of
    squared length 2, so all roots are the same length and the algebra cannot be
    ``B_n``, ``C_n``, ``G_2`` or ``F_4``.  (Non-simply-laced groups do arise in
    string theory, but from asymmetric orbifolds and Wilson lines, not from a
    plain torus.)

    Restricting to ``ADE`` also makes the identification unique: ``D_n`` is only
    listed from ``n = 4``, since ``D_2 = A_1 + A_1`` and ``D_3 = A_3``, and past
    that ``A_n`` and ``D_n`` of equal rank have different numbers of roots.
    """
    out: list[tuple[str, int, int]] = [
        (f"su({n + 1})", n, n * (n + 1)) for n in range(1, max_rank + 1)
    ]
    out += [(f"so({2 * n})", n, 2 * n * (n - 1)) for n in range(4, max_rank + 1)]
    out += [(name, rank, roots) for name, rank, roots in
            (("e6", 6, 72), ("e7", 7, 126), ("e8", 8, 240)) if rank <= max_rank]
    return out


def _irreducible_name(rank: int, n_roots: int) -> str | None:
    """The one simply-laced simple algebra with this rank and root count, if any."""
    for name, simple_rank, simple_roots in _simply_laced(max(rank, 8)):
        if simple_rank == rank and simple_roots == n_roots:
            return name
    return None


def identify_algebra(rank: int, n_roots: int) -> tuple[str, ...]:
    r"""Name every simply-laced algebra with this rank and this many roots.

    This is the *counting-only* identification, and it is not always unique: at
    rank 3 with 6 roots, ``su(2) + su(2) + su(2)`` and ``su(3) + u(1)`` have the
    same numbers.  :func:`decompose_roots` settles such cases from the geometry
    of the roots themselves, and :func:`gauge_algebra` uses it, so this function
    is here for when only the two numbers are known.

    Returns a tuple of candidate names, e.g. ``('su(2) + su(2)',)``; an empty
    tuple means nothing has those numbers, which would point to a bug upstream.
    """
    if rank < 0 or n_roots < 0:
        raise ValueError("rank and n_roots must be non-negative")
    simples = _simply_laced(rank)
    found: list[str] = []

    def recurse(start: int, rank_left: int, roots_left: int, chosen: list[str]) -> None:
        if roots_left == 0:
            names = list(chosen) + ["u(1)"] * rank_left
            found.append(" + ".join(names) if names else "trivial")
            return
        for index in range(start, len(simples)):
            name, simple_rank, simple_roots = simples[index]
            if simple_rank <= rank_left and simple_roots <= roots_left:
                recurse(index, rank_left - simple_rank, roots_left - simple_roots,
                        [*chosen, name])

    recurse(0, rank, n_roots, [])
    return tuple(sorted(set(found)))


def decompose_roots(roots: np.ndarray, tol: float = 1e-8) -> list[tuple[int, int]]:
    r"""Split a root system into its irreducible pieces.

    A root system decomposes into mutually orthogonal irreducible components,
    and those components are exactly the connected pieces of the graph in which
    two roots are joined when they are **not** orthogonal.  (A root is always
    joined to its negative, so no root is ever stranded on its own.)

    This is what distinguishes ``su(2)^3`` from ``su(3) + u(1)``: both have rank
    3 and six roots, but the first has three orthogonal pairs spanning three
    dimensions while the second has all six roots in one plane.

    Returns ``[(rank, n_roots), ...]``, one entry per component, largest first.
    """
    roots = np.asarray(roots, dtype=float)
    count = len(roots)
    if count == 0:
        return []

    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(count):
        for j in range(i + 1, count):
            if abs(float(roots[i] @ roots[j])) > tol:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)

    pieces = [
        (int(np.linalg.matrix_rank(roots[members], tol=tol)), len(members))
        for members in groups.values()
    ]
    return sorted(pieces, reverse=True)


def _name_from_roots(roots: np.ndarray, rank: int) -> str:
    """Full algebra name from the root vectors, ``u(1)`` factors included."""
    pieces = decompose_roots(roots)
    names: list[str] = []
    used = 0
    for piece_rank, piece_roots in pieces:
        name = _irreducible_name(piece_rank, piece_roots)
        names.append(name if name else f"<rank {piece_rank}, {piece_roots} roots>")
        used += piece_rank
    names += ["u(1)"] * (rank - used)
    return " + ".join(names) if names else "trivial"


@dataclass(frozen=True)
class GaugeAlgebra:
    """The gauge symmetry of a toroidal compactification, read off the lattice."""

    rank: int
    roots_left: int
    roots_right: int
    name_left: str
    name_right: str
    components_left: tuple[tuple[int, int], ...]
    components_right: tuple[tuple[int, int], ...]

    @property
    def name(self) -> str:
        """``'(su(3))_L x (su(3))_R'`` and so on."""
        return f"({self.name_left})_L x ({self.name_right})_R"

    @property
    def dimension(self) -> int:
        """Total number of gauge bosons: roots on both sides plus ``2d`` Cartans."""
        return self.roots_left + self.roots_right + 2 * self.rank

    @property
    def is_enhanced(self) -> bool:
        """True when anything beyond ``u(1)^{2d}`` is present."""
        return self.roots_left + self.roots_right > 0

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.name}   rank {self.rank} each side, "
            f"{self.roots_left}+{self.roots_right} roots, {self.dimension} gauge bosons"
        )


def gauge_algebra(background: TorusBackground) -> GaugeAlgebra:
    r"""Identify the enhanced gauge symmetry from the root vectors.

    The rank is ``d`` on each side -- those Cartan generators are always
    present, coming from the metric and ``B`` field with one leg on the torus.
    Everything beyond that is a root, so a generic torus gives ``u(1)^d`` on
    both sides and special points give something larger.

    The name comes from :func:`decompose_roots`, i.e. from the geometry of the
    roots rather than from counting them, so rank-3 cases like ``su(2)^3``
    versus ``su(3) + u(1)`` come out unambiguously.
    """
    left, right = root_vectors(background)
    d = background.dim
    return GaugeAlgebra(
        rank=d,
        roots_left=len(left),
        roots_right=len(right),
        name_left=_name_from_roots(left, d),
        name_right=_name_from_roots(right, d),
        components_left=tuple(decompose_roots(left)),
        components_right=tuple(decompose_roots(right)),
    )


# ---------------------------------------------------------------------------
# O(d,d;Z)
# ---------------------------------------------------------------------------


def integer_points_in_ball(
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


def odd_metric(dim: int) -> np.ndarray:
    r"""``eta``, the ``O(d,d)`` metric ``[[0, I], [I, 0]]`` in the ``(w, n)`` basis."""
    d = int(dim)
    return np.block([[np.zeros((d, d)), np.eye(d)], [np.eye(d), np.zeros((d, d))]])


def is_odd_integer(omega: np.ndarray, tol: float = _TOL) -> bool:
    r"""True when ``omega`` is an integer matrix with ``omega^T eta omega = eta``."""
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 2 or omega.shape[0] != omega.shape[1] or omega.shape[0] % 2:
        return False
    if np.any(np.abs(omega - np.rint(omega)) > tol):
        return False
    eta = odd_metric(omega.shape[0] // 2)
    return bool(np.allclose(omega.T @ eta @ omega, eta, atol=tol))


def basis_change(unimodular: np.ndarray) -> np.ndarray:
    r"""``[[A, 0], [0, A^{-T}]]``: relabel the lattice basis.

    Requires ``A`` integer with ``det A = +/-1``, so that ``A^{-T}`` is integer
    too.  This is the ``GL(d,Z)`` subgroup of ``O(d,d;Z)``: it says nothing
    physical, only that the choice of basis vectors was arbitrary.
    """
    a = np.asarray(unimodular, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be square")
    if np.any(np.abs(a - np.rint(a)) > _TOL):
        raise ValueError("A must be an integer matrix")
    if abs(abs(np.linalg.det(a)) - 1.0) > 1e-8:
        raise ValueError("A must be unimodular (det = +/-1)")
    d = a.shape[0]
    zero = np.zeros((d, d))
    return np.block([[a, zero], [zero, np.linalg.inv(a).T]])


def b_shift(theta: np.ndarray) -> np.ndarray:
    r"""``[[I, 0], [Theta, I]]`` for integer antisymmetric ``Theta``: ``B -> B + Theta``.

    This is why ``B`` is only defined modulo integers -- shifting it by an
    integer antisymmetric matrix is a relabelling of momenta, not a new theory.
    """
    t = np.asarray(theta, dtype=float)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("Theta must be square")
    if not np.allclose(t, -t.T, atol=_TOL):
        raise ValueError("Theta must be antisymmetric")
    if np.any(np.abs(t - np.rint(t)) > _TOL):
        raise ValueError("Theta must be an integer matrix")
    d = t.shape[0]
    return np.block([[np.eye(d), np.zeros((d, d))], [t, np.eye(d)]])


def factorized_duality(dim: int, direction: int) -> np.ndarray:
    r"""Exchange ``w^k`` with ``n_k`` in one direction.

    At ``d = 1`` this is precisely ``R -> alpha'/R`` with ``n <-> w``, so the
    circle's T-duality is one generator of a much larger group.
    """
    d = int(dim)
    k = int(direction)
    if not 0 <= k < d:
        raise ValueError(f"direction must lie in [0, {d})")
    omega = np.eye(2 * d)
    omega[[k, d + k]] = omega[[d + k, k]]
    return omega


def transform_charges(omega: np.ndarray, n, w) -> tuple[np.ndarray, np.ndarray]:
    """Map ``(n, w)`` through ``Z -> omega Z`` with ``Z = (w, n)``."""
    omega = np.asarray(omega, dtype=float)
    d = omega.shape[0] // 2
    z = np.concatenate([np.asarray(w, float).reshape(d), np.asarray(n, float).reshape(d)])
    z_new = omega @ z
    return np.rint(z_new[d:]).astype(int), np.rint(z_new[:d]).astype(int)


def transform(background: TorusBackground, omega: np.ndarray) -> TorusBackground:
    r"""Act with ``omega`` on the moduli: ``H -> omega^{-T} H omega^{-1}``.

    Paired with :func:`transform_charges` this leaves both ``Z^T H Z`` and
    ``Z^T eta Z`` alone, so the whole spectrum is unchanged -- the two theories
    are the same theory in different variables.  The new ``(G, B)`` is recovered
    from the transformed generalized metric, which also verifies that the result
    really is of the required block form.
    """
    if not is_odd_integer(omega):
        raise ValueError("omega must be an integer matrix preserving eta")
    omega = np.asarray(omega, dtype=float)
    inv = np.linalg.inv(omega)
    return TorusBackground.from_generalized_metric(
        inv.T @ background.generalized_metric() @ inv, background.conventions
    )


def spectrum_is_dual(
    background: TorusBackground,
    omega: np.ndarray,
    charge_max: int = 2,
    level_max: int = 1,
    tol: float = 1e-9,
) -> bool:
    r"""Check that ``omega`` maps the spectrum onto itself, state by state.

    Every enumerated state is mapped with :func:`transform_charges` and its mass
    recomputed in the transformed background; the test is that each mass comes
    back unchanged.

    **Why not simply compare the two spectra as multisets** (which is what
    :func:`stringsim.compactification.circle.spectrum_is_t_dual` does)?  Because
    the enumeration is truncated to a box ``|n|, |w| <= charge_max``, and only
    some elements of ``O(d,d;Z)`` map that box to itself.  A factorized duality
    permutes the components and is fine; a basis change like
    ``[[1,1],[0,1]]`` shears the box, so states near its edge leave the window
    and the two truncated multisets differ for a reason that has nothing to do
    with the physics.  Following individual states avoids the whole issue.
    """
    if not is_odd_integer(omega):
        raise ValueError("omega must be an integer matrix preserving eta")
    dual = transform(background, omega)
    for state in spectrum(background, charge_max=charge_max, level_max=level_max):
        n_new, w_new = transform_charges(omega, state.momentum, state.winding)
        left, right = narain_momenta(dual, n_new, w_new)
        moved = float(left @ left + right @ right) + 2.0 * (
            state.level + state.level_tilde - 2
        )
        if abs(moved - state.alpha_m2) > tol:
            return False
    return True


def narain_gram_matrix(dim: int) -> np.ndarray:
    r"""Gram matrix of the Narain lattice :math:`\Gamma_{d,d}` in the ``(w, n)`` basis.

    The bilinear form :math:`\ell_L \cdot \ell'_L - \ell_R \cdot \ell'_R` is
    moduli-independent and equals :math:`Z^{T} \eta Z'`, so the Gram matrix is
    ``eta``: determinant ``+/-1`` (self-dual), zero diagonal and integer entries
    with even norm ``2 n_i w^i`` (even).  Even self-duality is what makes the
    one-loop amplitude modular invariant, and it is why the moduli can be varied
    continuously without the lattice ceasing to exist.
    """
    return odd_metric(dim)
