r"""Mirror symmetry, as an exchange of a polytope with its dual.

:mod:`stringsim.compactification.hodge` computes Hodge numbers for orbifolds of
a torus, and finds the pair ``(51, 3)`` and ``(3, 51)`` swapped by a phase.
Calabi-Yau manifolds that are *hypersurfaces* live somewhere else, and mirror
symmetry there is the statement of Batyrev's: a reflexive lattice polytope
:math:`\Delta^*` gives a Calabi-Yau, its dual :math:`\Delta` gives another, and

.. math::
   h^{1,1}(X_{\Delta^*}) = h^{d-2,1}(X_{\Delta}) , \qquad
   h^{d-2,1}(X_{\Delta^*}) = h^{1,1}(X_{\Delta}) .

**The exchange is not checked; it is the same formula read twice.**  Batyrev's
count is

.. math::
   h^{1,1} = \ell(\Delta^*) - (d+1)
   - \sum_{\text{facets } \Gamma^*} \ell^*(\Gamma^*)
   + \sum_{\text{codim-2 } \Gamma^*} \ell^*(\Gamma^*)\,\ell^*(\Gamma) ,

with :math:`\ell` the lattice points of a face, :math:`\ell^*` those in its
relative interior, and :math:`\Gamma` the face of :math:`\Delta` dual to
:math:`\Gamma^*`.  The other Hodge number is the same expression with
:math:`\Delta` and :math:`\Delta^*` swapped, so the mirror is built into the
combinatorics rather than discovered in it.  What has to be checked is that the
numbers are right, and :func:`hypersurface_euler` does that from the other side.

**The quintic.**  :math:`\mathbb{P}^4` gives the reflexive simplex with 6
lattice points and its dual with 126 -- the 126 quintic monomials.  Out comes
:math:`(h^{1,1}, h^{2,1}) = (1, 101)` and :math:`\chi = -200`, and the dual
polytope gives :math:`(101, 1)` with :math:`\chi = +200`.  The Euler
characteristic is checked against
:math:`\int c_3` for a degree-5 hypersurface, computed from
:math:`c(T) = (1+H)^5/(1+5H)` with no polytope in sight.

**Where the formula stops.**  In :math:`d = 4` it computes the Hodge numbers of
a Calabi-Yau threefold.  In :math:`d = 3` it does *not* compute
:math:`h^{1,1}` of a K3, which is 20 for every K3: it computes the Picard
number of the generic member of the family, the part of :math:`H^{1,1}` the
toric divisors reach.  The module shows this rather than warning about it --
the quartic gives 1 and its mirror 19, while :func:`hypersurface_euler` gives
:math:`\chi = 24`, which for a surface with :math:`h^{2,0} = 1` forces
:math:`h^{1,1} = 20`.  The two numbers sum to 20 for the simplest families and
not for all of them, and the difference is divisors no polytope sees.

**Greene and Plesser.**  For a Fermat hypersurface the mirror was found first as
an orbifold: the quintic divided by the phase symmetries that preserve the
holomorphic three-form, a group of order :math:`5^3 = 125`.
:func:`greene_plesser_order` counts it.  That the toric dual and that quotient
are the same manifold is a theorem, not something computed here -- the orbifold
cohomology of the quotient would need the fixed loci, which this module does not
build.

Reference: V. Batyrev, *Dual polyhedra and mirror symmetry for Calabi-Yau
hypersurfaces in toric varieties*, J. Alg. Geom. **3** (1994) 493;
B. Greene and M. Plesser, *Duality in Calabi-Yau moduli space*, Nucl. Phys. B
**338** (1990) 15.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import comb

import numpy as np
from scipy.spatial import ConvexHull

__all__ = [
    "Face",
    "Polytope",
    "projective_space",
    "weighted_projective",
    "hypersurface_euler",
    "HodgeNumbers",
    "hodge_numbers",
    "batyrev_count",
    "greene_plesser_group",
    "greene_plesser_order",
]

_TOL = 1e-9


@dataclass(frozen=True)
class Face:
    """One face of a polytope, found by which facets its points sit on."""

    active: tuple[int, ...]
    dimension: int
    interior: tuple[tuple[int, ...], ...]

    @property
    def interior_count(self) -> int:
        r""":math:`\ell^*`: lattice points in the relative interior."""
        return len(self.interior)


@dataclass(frozen=True)
class Polytope:
    r"""A lattice polytope containing the origin in its interior.

    Stored by its vertices.  Everything else -- the facets, the lattice points,
    the face structure -- is derived, and the face structure is derived in the
    one way that needs no convex-hull recursion: two lattice points lie in the
    relative interior of the same face exactly when they sit on the same set of
    facets.  Grouping the points by that set *is* the face lattice, with the
    interior counts already in hand.
    """

    vertices: np.ndarray
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[0] <= vertices.shape[1]:
            raise ValueError(f"need at least dim+1 vertices, got shape {vertices.shape}")
        object.__setattr__(self, "vertices", vertices)

    @property
    def dim(self) -> int:
        return int(self.vertices.shape[1])

    @property
    def facets(self) -> np.ndarray:
        r"""Normals ``n`` with :math:`\langle n, x\rangle \leq 1` on the polytope.

        For a reflexive polytope these are integral, and they are the vertices
        of the dual.
        """
        if "facets" not in self._cache:
            hull = ConvexHull(self.vertices.astype(float))
            rows: list[np.ndarray] = []
            for equation in hull.equations:
                normal, offset = equation[:-1], equation[-1]
                if abs(offset) < _TOL:
                    raise ValueError("the origin lies on a facet; the polytope is not reflexive")
                candidate = normal / (-offset)
                if not any(np.allclose(candidate, seen, atol=_TOL) for seen in rows):
                    rows.append(candidate)
            self._cache["facets"] = np.array(rows)
        return self._cache["facets"]

    @property
    def is_reflexive(self) -> bool:
        """Whether the dual is a lattice polytope too."""
        try:
            normals = self.facets
        except ValueError:  # pragma: no cover - degenerate input
            return False
        return bool(np.allclose(normals, np.rint(normals), atol=_TOL))

    def dual(self) -> Polytope:
        r""":math:`\Delta^{\vee} = \{y : \langle y, x\rangle \leq 1\ \forall x\}`.

        Its vertices are this polytope's facet normals, which is why reflexivity
        is exactly the statement that they are integral.
        """
        if not self.is_reflexive:
            raise ValueError("the dual of a non-reflexive polytope is not a lattice polytope")
        return Polytope(np.rint(self.facets).astype(np.int64))

    @property
    def lattice_points(self) -> np.ndarray:
        """Every integer point in the polytope, by walking its bounding box."""
        if "points" not in self._cache:
            normals = self.facets
            low = np.floor(self.vertices.min(axis=0)).astype(int)
            high = np.ceil(self.vertices.max(axis=0)).astype(int)
            spans = [range(a, b + 1) for a, b in zip(low, high, strict=True)]
            found = [
                point
                for point in itertools.product(*spans)
                if np.all(normals @ np.array(point, dtype=float) <= 1.0 + _TOL)
            ]
            self._cache["points"] = np.array(found, dtype=np.int64)
        return self._cache["points"]

    @property
    def faces(self) -> dict[tuple[int, ...], Face]:
        """Faces keyed by the set of facets they lie on.

        The whole polytope is the empty key; a vertex has the largest key.  The
        dimension comes from the rank of the active normals, so no explicit
        recursion over the face lattice is needed.
        """
        if "faces" not in self._cache:
            normals = self.facets
            grouped: dict[tuple[int, ...], list] = {}
            for point in self.lattice_points:
                active = tuple(
                    index
                    for index, row in enumerate(normals)
                    if abs(float(row @ point) - 1.0) < _TOL
                )
                grouped.setdefault(active, []).append(tuple(int(x) for x in point))
            built = {}
            for active, members in grouped.items():
                rank = 0 if not active else int(np.linalg.matrix_rank(normals[list(active)]))
                built[active] = Face(
                    active=active,
                    dimension=self.dim - rank,
                    interior=tuple(members),
                )
            self._cache["faces"] = built
        return self._cache["faces"]

    def faces_of_dimension(self, dimension: int) -> list[Face]:
        """Those faces that carry a lattice point in their relative interior.

        A face with none simply does not appear, which is what the sums in
        Batyrev's formula want: they run over :math:`\\ell^*`.
        """
        return [face for face in self.faces.values() if face.dimension == dimension]


def projective_space(dim: int) -> Polytope:
    r"""The reflexive simplex whose Calabi-Yau is the degree-``dim+1`` hypersurface.

    Rays :math:`e_1, \dots, e_d` and :math:`-\sum e_i`: the fan of
    :math:`\mathbb{P}^d`.
    """
    if dim < 2:
        raise ValueError("need at least two dimensions")
    basis = np.eye(dim, dtype=np.int64)
    return Polytope(np.vstack([basis, -basis.sum(axis=0)]))


def weighted_projective(weights) -> Polytope:
    r"""The fan polytope of :math:`\mathbb{P}(w_0, \dots, w_d)`.

    The rays satisfy :math:`\sum_i w_i v_i = 0`.  One ray of weight 1 is made
    the dependent one and the rest are a basis, which keeps every ray integral;
    without a weight of 1 the relation cannot be solved inside
    :math:`\mathbb{Z}^d` this way and the function says so rather than rounding.
    """
    weights = [int(w) for w in weights]
    if min(weights) < 1:
        raise ValueError("weights must be positive")
    if 1 not in weights:
        raise ValueError("this construction needs a weight equal to 1")
    dimension = len(weights) - 1
    basis = list(np.eye(dimension, dtype=np.int64))
    spot = weights.index(1)
    rays: list = [None] * len(weights)
    cursor = 0
    for index in range(len(weights)):
        if index == spot:
            continue
        rays[index] = basis[cursor]
        cursor += 1
    rays[spot] = -sum(
        weights[index] * rays[index] for index in range(len(weights)) if index != spot
    )
    return Polytope(np.array(rays, dtype=np.int64))


def hypersurface_euler(degree: int, ambient: int) -> int:
    r""":math:`\chi` of a smooth degree-``d`` hypersurface in :math:`\mathbb{P}^n`.

    From :math:`c(T_X) = (1+H)^{n+1}/(1+dH)` restricted to ``X``, whose top
    Chern class integrates to :math:`d` times the relevant coefficient.  No
    polytope anywhere: this is the classical computation, and it is what checks
    the combinatorial one.

    Gives ``24`` for the quartic K3, ``-200`` for the quintic, ``0`` for a plane
    cubic.
    """
    if degree < 1 or ambient < 2:
        raise ValueError("need degree >= 1 and ambient >= 2")
    top = ambient - 1
    coefficient = sum(
        comb(ambient + 1, i) * ((-degree) ** (top - i)) for i in range(top + 1)
    )
    return degree * coefficient


def batyrev_count(polytope: Polytope) -> int:
    r"""Batyrev's count for a polytope and its dual.

    .. math::
       \ell(\Delta^*) - (d+1) - \sum_{\text{facets}} \ell^*
       + \sum_{\text{codim } 2} \ell^*(\Gamma^*)\,\ell^*(\Gamma)

    In :math:`d = 4` this is :math:`h^{1,1}` of the Calabi-Yau threefold; in
    :math:`d = 3` it is the Picard number of the generic K3, which is not
    :math:`h^{1,1} = 20`.  The function computes the count and
    :func:`hodge_numbers` is where the interpretation lives.
    """
    if not polytope.is_reflexive:
        raise ValueError("Batyrev's formula needs a reflexive polytope")
    dual = polytope.dual()
    dimension = polytope.dim

    # Each facet of the polytope is a vertex of the dual; the face of the dual
    # spanned by a set of them is the dual face, and its own active set is the
    # intersection of theirs.
    vertex_active = {}
    for index, vertex in enumerate(dual.vertices):
        vertex_active[index] = frozenset(
            slot
            for slot, row in enumerate(dual.facets)
            if abs(float(row @ vertex) - 1.0) < _TOL
        )

    total = len(polytope.lattice_points) - (dimension + 1)
    for face in polytope.faces_of_dimension(dimension - 1):
        total -= face.interior_count
    for face in polytope.faces_of_dimension(dimension - 2):
        if not face.active:  # pragma: no cover - only the whole polytope
            continue
        shared = frozenset.intersection(*[vertex_active[i] for i in face.active])
        partner = dual.faces.get(tuple(sorted(shared)))
        if partner is not None:
            total += face.interior_count * partner.interior_count
    return total


@dataclass(frozen=True)
class HodgeNumbers:
    """The two Hodge numbers of a Calabi-Yau threefold, and what they imply."""

    h11: int
    h21: int

    @property
    def euler(self) -> int:
        r""":math:`\chi = 2(h^{1,1} - h^{2,1})`."""
        return 2 * (self.h11 - self.h21)

    @property
    def mirror(self) -> HodgeNumbers:
        """The pair with the two numbers exchanged."""
        return HodgeNumbers(h11=self.h21, h21=self.h11)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"(h11, h21) = ({self.h11}, {self.h21}), chi = {self.euler:+d}"


def hodge_numbers(polytope: Polytope) -> HodgeNumbers:
    r"""Both Hodge numbers of the Calabi-Yau threefold from a reflexive 4-polytope.

    :math:`h^{1,1}` from the polytope and :math:`h^{2,1}` from its dual -- the
    *same* function of a polytope, applied to the two of them.  Mirror symmetry
    is then not a result but the shape of the formula, and what is left to check
    is whether the numbers are right.

    Raises
    ------
    ValueError
        Outside four dimensions.  In three the count is a Picard number rather
        than :math:`h^{1,1}`, and calling this would put the wrong name on it;
        use :func:`batyrev_count` there and read the module docstring.
    """
    if polytope.dim != 4:
        raise ValueError(
            f"a Calabi-Yau threefold needs a four-dimensional polytope, got {polytope.dim}; "
            "batyrev_count still works, but in three dimensions it is a Picard number"
        )
    return HodgeNumbers(
        h11=batyrev_count(polytope),
        h21=batyrev_count(polytope.dual()),
    )


def greene_plesser_group(weights) -> tuple[tuple[tuple[int, ...], ...], int]:
    r"""The phase symmetries that give the mirror, and the scalings divided out.

    For :math:`\sum_i x_i^{n_i}` in :math:`\mathbb{P}(w)` with
    :math:`n_i = d/w_i`, a phase :math:`x_i \to e^{2\pi i a_i/n_i} x_i`
    preserves the holomorphic form when :math:`\sum_i a_i/n_i \in \mathbb{Z}`,
    which is :math:`\sum_i a_i w_i \equiv 0 \pmod d`.  The projective scalings
    :math:`x_i \to \lambda^{w_i} x_i` act trivially and are quotiented out.

    Returns ``(elements, scalings)`` -- the tuples satisfying the condition and
    how many of them are scalings.  The scalings are *enumerated* rather than
    counted by a formula: the obvious guess that there are as many as the
    smallest :math:`n_i` is right for the quintic and wrong in general, which is
    the sort of thing that only shows up when the two are compared.
    """
    weights = [int(w) for w in weights]
    if min(weights) < 1:
        raise ValueError("weights must be positive")
    degree = sum(weights)
    exponents = []
    for weight in weights:
        if degree % weight:
            raise ValueError(f"weight {weight} does not divide the degree {degree}")
        exponents.append(degree // weight)

    elements = tuple(
        phases
        for phases in itertools.product(*[range(n) for n in exponents])
        if sum(phases[i] * weights[i] for i in range(len(weights))) % degree == 0
    )
    scalings = {
        tuple(k % n for n in exponents) for k in range(degree)
    }
    return elements, len(scalings)


def greene_plesser_order(weights) -> int:
    r"""Order of the group whose quotient is the mirror of a Fermat hypersurface.

    For the quintic it is :math:`5^3 = 125`, and the group is
    :math:`(\mathbb{Z}_5)^3`.  That the quotient by it is the same manifold as
    the toric dual polytope is a theorem, not something computed here: the
    orbifold cohomology of the quotient would need the fixed loci, which this
    module does not build.
    """
    elements, scalings = greene_plesser_group(weights)
    if len(elements) % scalings:  # pragma: no cover - would mean a bug above
        raise RuntimeError("the scalings are not a subgroup of the phase symmetries")
    return len(elements) // scalings
